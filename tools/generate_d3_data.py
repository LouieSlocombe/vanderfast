"""Generate the bundled D3 reference tables from pinned upstream sources.

The generator reads the exact upstream files listed in ``data/sources.toml``
(downloaded by commit SHA or taken from a local checkout), verifies every
sha256, parses the Fortran data statements strictly, and writes three
artifacts into ``src/vanderfast/data``:

``d3_reference.bin``
    Packed little-endian tables (format v1, see ``docs/data-provenance.md``).
``d3bj_methods.json``
    D3(BJ) damping parameters and method-name aliases.
``manifest.json``
    Provenance, per-table hashes, units, conventions, and anomalies.

Outputs are deterministic so ``--check`` can require byte-identical
regeneration. ``--verify-binary`` additionally searches the compiled upstream
shared libraries for the parsed arrays, proving that parsing and constant
folding reproduce the reference build bit for bit.

Usage::

    python tools/generate_d3_data.py [--source-dir NAME=DIR ...] [--cache DIR]
                                     [--check] [--verify-binary LIBDIR]
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datagen_common import (
    DTYPE_F64,
    DTYPE_U8,
    HEADER_BYTES,
    REAL_LITERAL,
    SECTION_ALIGN,
    Constants,
    Generated,
    GenerationError,
    Section,
    SourcePin,
    SourceReader,
    build_container,
    check_convention_lines,
    dumps,
    find_one,
    fortran_statements,
    int_parameter,
    manifest_outputs,
    manifest_sources,
    pack_f64,
    parse_constants,
    parse_int_list,
    parse_real_list,
    run_cli,
    squash,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_TOML = REPO_ROOT / "data" / "sources.toml"

GENERATOR_VERSION = 1
FORMAT_VERSION = 1
BYTE_ORDER_MARK = 0x01020304
FLAG_RADII_D3_SCALED = 1 << 0
FLAG_ATOMIC_UNITS = 1 << 1

SECTION_NREF = 1
SECTION_REF_CN = 2
SECTION_RCOV_D3 = 3
SECTION_R4R2 = 4
SECTION_C6 = 5
SECTION_RVDW = 6

Z_MAX = 103
MAX_REF = 7
N_PAIRS = Z_MAX * (Z_MAX + 1) // 2

SQRT_Z_EXPR = "sqrt(0.5_wp*(r4_over_r2*[(sqrt(real(idum,wp)),idum=1,max_elem)]))"

# Lines whose presence pins a numerical convention that vanderfast reproduces.
# They are compared after removing all whitespace.
CONVENTION_LINES: dict[str, list[tuple[str, str]]] = {
    "src/dftd3/ncoord.f90": [
        ("cn.kcn", "real(wp), parameter :: default_kcn = 16.0_wp"),
        ("cn.exclusion", "if (r2 > cutoff2 .or. r2 < 1.0e-12_wp) cycle"),
        (
            "cn.counting",
            "countf = 1.0_wp/(1.0_wp + exp(-default_kcn*(rc/r1 - 1.0_wp)))",
        ),
        ("cn.derivative", "dcf = -cf*(1.0_wp - cf)*default_kcn*rc/r2"),
    ],
    "src/dftd3/model.f90": [
        ("weights.wf", "real(wp), parameter :: wf_default = 4.0_wp"),
        ("weights.gaussian", "cngw = exp ( -wf * ( cn - cnref )**2 )"),
        (
            "weights.exceptional",
            "is_exceptional = ieee_is_nan(val) .or. abs(val) > huge(val)",
        ),
        (
            "weights.fallback",
            "if (maxval(self%cn(:self%ref(izp), izp)) == self%cn(iref, izp)) then",
        ),
        ("weights.dgw", "dgwk = expd * norm - expw * dnorm * norm**2"),
    ],
    "src/dftd3/damping/rational.f90": [
        ("atm.rs9", "real(wp), parameter :: rs9 = 4.0_wp/3.0_wp"),
        ("atm.alp_shift", "self%s9, rs9, self%alp+2,"),
        ("atm.enabled", "has = abs(self%s9) >= epsilon(1.0_wp)"),
        ("pair.kappa", "rrij = 3*r4r2(izp)*r4r2(jzp)"),
        ("pair.r0", "r0ij = self%a1 * sqrt(rrij) + self%a2"),
        ("pair.exclusion", "if (r2 > cutoff2 .or. r2 < epsilon(1.0_wp)) cycle"),
        (
            "pair.skip",
            "if (abs(self%s6) < epsilon(1.0_wp) .and. abs(self%s8) < epsilon(1.0_wp)) return",
        ),
    ],
    "src/dftd3/damping/atm.f90": [
        ("atm.fdmp", "fdmp = 1.0_wp / (1.0_wp + 6.0_wp * (r0 / r1)**alp3)"),
        ("atm.c9", "c9 = -s9 * sqrt(abs(c6ij*c6ik*c6jk))"),
        ("atm.third_edge", "vjk(:) = vik(:) - vij(:)"),
        (
            "atm.exclusion",
            "if (r2ij > cutoff2 .or. r2ij < epsilon(1.0_wp)) cycle",
        ),
    ],
    "src/dftd3/cutoff.f90": [
        ("cutoff.cn", "real(wp), parameter :: cn_default = 40.0_wp"),
        ("cutoff.disp2", "real(wp), parameter :: disp2_default = 60.0_wp"),
        ("cutoff.disp3", "real(wp), parameter :: disp3_default = 40.0_wp"),
    ],
}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tables:
    """Parsed upstream reference tables (Fortran layout, before packing)."""

    nref: list[int]  # index Z-1
    reference_cn_padded: list[float]  # column-major (max_ref, max_elem)
    c6_padded: list[float]  # column-major (max_ref, max_ref, n_pairs)
    r4r2_all: list[float]  # sqrt_z_r4_over_r2, Z = 1..118
    rcov_d3_all: list[float]  # covalent_rad_d3, Z = 1..118 (bohr)
    vdwrad: list[float]  # aatoau * vdwrad, packed pairs (bohr)


def parse_reference(reader: SourceReader) -> tuple[list[int], list[float], list[float]]:
    """Parse reference counts, reference CNs, and the padded C6 table."""
    statements = fortran_statements(reader.text("simple-dftd3", "src/dftd3/reference.f90"))
    if int_parameter(statements, "max_elem") != Z_MAX:
        msg = "reference.f90 max_elem changed"
        raise GenerationError(msg)
    if int_parameter(statements, "max_ref") != MAX_REF:
        msg = "reference.f90 max_ref changed"
        raise GenerationError(msg)

    nref_match = find_one(
        statements,
        r"integer, parameter :: number_of_references\(max_elem\) = \[(.*)\]",
    )
    nref = parse_int_list(nref_match.group(1))
    if len(nref) != Z_MAX:
        msg = f"expected {Z_MAX} reference counts, got {len(nref)}"
        raise GenerationError(msg)

    cn_match = find_one(
        statements,
        r"real\(wp\), parameter :: reference_cn\(max_ref, max_elem\) = "
        r"reshape\(\[(.*)\], ?\[max_ref, max_elem\]\)",
    )
    reference_cn = parse_real_list(cn_match.group(1))
    if len(reference_cn) != MAX_REF * Z_MAX:
        msg = f"expected {MAX_REF * Z_MAX} reference CNs, got {len(reference_cn)}"
        raise GenerationError(msg)

    total = MAX_REF * MAX_REF * N_PAIRS
    c6 = [math.nan] * total
    chunk = re.compile(r"c6ab_view\((\d+):(\d+)\) = \[(.*)\]")
    covered = 0
    for statement in statements:
        match = chunk.fullmatch(statement)
        if not match:
            continue
        lo, hi = int(match.group(1)), int(match.group(2))
        values = parse_real_list(match.group(3))
        if len(values) != hi - lo + 1 or lo != covered + 1:
            msg = f"non-contiguous C6 chunk {lo}:{hi}"
            raise GenerationError(msg)
        c6[lo - 1 : hi] = values
        covered = hi
    if covered != total:
        msg = f"C6 chunks cover {covered} of {total} entries"
        raise GenerationError(msg)
    return nref, reference_cn, c6


def parse_r4r2(reader: SourceReader) -> list[float]:
    """Return ``sqrt_z_r4_over_r2`` evaluated in upstream operation order."""
    statements = fortran_statements(reader.text("simple-dftd3", "src/dftd3/data/r4r2.f90"))
    max_elem = int_parameter(statements, "max_elem")
    match = find_one(
        statements,
        r"real\(wp\), parameter :: r4_over_r2\(max_elem\) = \[(.*)\]",
    )
    r4_over_r2 = parse_real_list(match.group(1))
    if len(r4_over_r2) != max_elem:
        msg = "r4_over_r2 length mismatch"
        raise GenerationError(msg)
    expr = find_one(
        statements,
        r"real\(wp\), parameter :: sqrt_z_r4_over_r2\(max_elem\) = (.*)",
    )
    if squash(expr.group(1)) != SQRT_Z_EXPR:
        msg = f"sqrt_z_r4_over_r2 expression changed: {expr.group(1)!r}"
        raise GenerationError(msg)
    return [
        math.sqrt(0.5 * (value * math.sqrt(float(z))))
        for z, value in enumerate(r4_over_r2, start=1)
    ]


def parse_vdwrad(reader: SourceReader, constants: Constants) -> list[float]:
    """Return the pairwise van der Waals radii in bohr."""
    statements = fortran_statements(reader.text("simple-dftd3", "src/dftd3/data/vdwrad.f90"))
    if int_parameter(statements, "max_elem") != Z_MAX:
        msg = "vdwrad.f90 max_elem changed"
        raise GenerationError(msg)
    match = find_one(
        statements,
        r"real\(wp\), parameter :: vdwrad\(max_elem\*\(1\+max_elem\)/2\) = "
        r"aatoau \* \[(.*)\]",
    )
    values = parse_real_list(match.group(1))
    if len(values) != N_PAIRS:
        msg = f"expected {N_PAIRS} vdW radii, got {len(values)}"
        raise GenerationError(msg)
    lookup = squash(
        "rad = vdwrad(num2 + num1*(num1-1)/2)",
    )
    if lookup not in {squash(s) for s in statements}:
        msg = "vdwrad.f90 pair indexing changed"
        raise GenerationError(msg)
    return [constants.aatoau * value for value in values]


def parse_covrad(reader: SourceReader, constants: Constants) -> list[float]:
    """Return mctc-lib's ``covalent_rad_d3`` (already D3-scaled, bohr)."""
    statements = fortran_statements(reader.text("mctc-lib", "src/mctc/data/covrad.f90"))
    max_elem = int_parameter(statements, "max_elem")
    match = find_one(
        statements,
        r"real\(wp\), parameter :: covalent_rad_2009\(max_elem\) = aatoau \* \[(.*)\]",
    )
    radii = parse_real_list(match.group(1))
    if len(radii) != max_elem:
        msg = "covalent_rad_2009 length mismatch"
        raise GenerationError(msg)
    scaled = find_one(
        statements,
        r"real\(wp\), parameter :: covalent_rad_d3\(max_elem\) = (.*)",
    )
    if squash(scaled.group(1)) != "4.0_wp/3.0_wp*covalent_rad_2009":
        msg = f"covalent_rad_d3 expression changed: {scaled.group(1)!r}"
        raise GenerationError(msg)
    if "rad=covalent_rad_d3(num)" not in {squash(s) for s in statements}:
        msg = "get_covalent_rad no longer returns covalent_rad_d3"
        raise GenerationError(msg)
    return [(4.0 / 3.0) * (constants.aatoau * value) for value in radii]


def check_conventions(reader: SourceReader) -> dict[str, str]:
    """Assert convention-bearing source lines and return them by key."""
    return check_convention_lines(
        reader, {("simple-dftd3", relpath): lines for relpath, lines in CONVENTION_LINES.items()}
    )


# --------------------------------------------------------------------------
# Method parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodTable:
    """Parsed D3(BJ) parameter table and alias map."""

    defaults: dict[str, float]
    aliases: dict[str, str]  # alias -> method id
    bj: dict[str, dict[str, Any]]  # method id -> parameters
    first_alias: dict[str, str]  # method id -> canonical alias


_PARAM_KEYS = ("s6", "s8", "s9", "a1", "a2", "alp")


def _parse_keyword_args(body: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in body.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key in result:
            msg = f"duplicate keyword {key} in {body!r}"
            raise GenerationError(msg)
        result[key] = parse_real_list(value.strip())[0]
    return result


def _parse_defaults(statements: list[str]) -> dict[str, float]:
    start = statements.index("type :: d3_param")
    end = statements.index("end type d3_param", start)
    defaults: dict[str, float] = {}
    default_re = re.compile(r"real\(wp\) :: (\w+) = (" + REAL_LITERAL.pattern + ")")
    for statement in statements[start + 1 : end]:
        match = default_re.fullmatch(statement)
        if not match:
            msg = f"unexpected d3_param member {statement!r}"
            raise GenerationError(msg)
        defaults[match.group(1)] = parse_real_list(match.group(2))[0]
    return defaults


def _parse_method_ids(statements: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return alias -> method id and method id -> first alias from ``get_method_id``."""
    start = statements.index("function get_method_id(method) result(id)")
    end = statements.index("end function get_method_id", start)
    case_re = re.compile(r'case\(("[^"]*"(?:, ?"[^"]*")*)\)')
    id_re = re.compile(r"id = (p_\w+)")
    aliases: dict[str, str] = {}
    first_alias: dict[str, str] = {}
    pending: list[str] = []
    for statement in statements[start:end]:
        match = case_re.fullmatch(statement)
        if match:
            pending = re.findall(r'"([^"]*)"', match.group(1))
            continue
        id_match = id_re.fullmatch(statement)
        if not id_match:
            continue
        method_id = id_match.group(1)
        for alias in pending:
            if alias in aliases:
                msg = f"duplicate method alias {alias!r}"
                raise GenerationError(msg)
            aliases[alias] = method_id
        if pending:
            first_alias.setdefault(method_id, pending[0])
        pending = []
    return aliases, first_alias


_DEFAULT_BRANCH = [
    "case default",
    'call fatal_error(error, "No entry for \'"//method//"\' present")',
    "return",
]


def _parse_bj_entries(
    statements: list[str], defaults: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Return method id -> parameters from the ``get_rational_damping`` select block."""
    start = statements.index("subroutine get_rational_damping(param, method, error, s9, citation)")
    end = statements.index("end subroutine get_rational_damping", start)
    select = statements.index("select case(get_method_id(method))", start)
    end_select = statements.index("end select", select)
    body = statements[select + 1 : end_select]
    if body[:3] != _DEFAULT_BRANCH:
        msg = "unexpected default branch in get_rational_damping"
        raise GenerationError(msg)
    ids_re = re.compile(r"case\((p_\w+(?:, ?p_\w+)*)\)")
    param_re = re.compile(r"param = d3_param\((.*)\)")
    doi_re = re.compile(r"doi = (doi_\w+)")
    entries: list[tuple[list[str], dict[str, float], str]] = []
    for statement in body[3:]:
        if match := ids_re.fullmatch(statement):
            entries.append(([p.strip() for p in match.group(1).split(",")], {}, "doi_dftd3_bj"))
        elif (match := param_re.fullmatch(statement)) and entries and not entries[-1][1]:
            entries[-1][1].update(_parse_keyword_args(match.group(1)))
        elif (match := doi_re.fullmatch(statement)) and entries:
            entries[-1] = (entries[-1][0], entries[-1][1], match.group(1))
        else:
            msg = f"unexpected statement in get_rational_damping: {statement!r}"
            raise GenerationError(msg)
    bj: dict[str, dict[str, Any]] = {}
    for ids, values, doi in entries:
        if not values or set(values) - {"s6", "s8", "a1", "a2"}:
            msg = f"BJ entry for {ids} sets unexpected keys {sorted(values)}"
            raise GenerationError(msg)
        for method_id in ids:
            if method_id in bj:
                msg = f"duplicate BJ entry for {method_id}"
                raise GenerationError(msg)
            bj[method_id] = {**{k: defaults[k] for k in _PARAM_KEYS}, **values, "citation": doi}
    tail = statements[end_select + 1 : end]
    for required in ["if (present(s9)) then", "param%s9 = s9"]:
        if required not in tail:
            msg = f"get_rational_damping no longer contains {required!r}"
            raise GenerationError(msg)
    return bj


def parse_methods(reader: SourceReader) -> MethodTable:
    """Parse ``get_method_id`` and ``get_rational_damping`` from ``param.f90``."""
    statements = fortran_statements(reader.text("simple-dftd3", "src/dftd3/param.f90"))
    defaults = _parse_defaults(statements)
    aliases, first_alias = _parse_method_ids(statements)
    bj = _parse_bj_entries(statements, defaults)
    unknown = set(bj) - set(aliases.values())
    if unknown:
        msg = f"BJ entries without a method name: {sorted(unknown)}"
        raise GenerationError(msg)
    return MethodTable(defaults=defaults, aliases=aliases, bj=bj, first_alias=first_alias)


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def _pair_index(zi: int, zj: int) -> int:
    """Zero-based packed pair index for 1-based ``zi >= zj``."""
    return zi * (zi - 1) // 2 + (zj - 1)


def _padded_c6(c6: Sequence[float], ic: int, a: int, b: int) -> float:
    """Fortran ``reference_c6(a+1, b+1, ic+1)`` from the column-major layout."""
    return c6[a + MAX_REF * b + MAX_REF * MAX_REF * ic]


@dataclass(frozen=True)
class Packed:
    """Packed section payloads and validation findings."""

    nref: list[int]  # length Z_MAX + 1, slot 0 = 0
    ref_cn: list[float]
    rcov: list[float]  # length Z_MAX + 1, slot 0 = NaN
    r4r2: list[float]
    c6: list[float]
    rvdw: list[float]
    anomalies: list[dict[str, Any]]
    zero_valid_refs: list[dict[str, int]]


def _pack_reference_cn(tables: Tables, nref: list[int]) -> tuple[list[float], list[dict[str, Any]]]:
    ref_cn: list[float] = []
    anomalies: list[dict[str, Any]] = []
    for z in range(1, Z_MAX + 1):
        column = tables.reference_cn_padded[(z - 1) * MAX_REF : z * MAX_REF]
        ref_cn.extend(column[: nref[z]])
        anomalies.extend(
            {
                "Z": z,
                "kind": "populated_padding_reference_cn",
                "reference": a + 1,
                "value": column[a],
            }
            for a in range(nref[z], MAX_REF)
            if column[a] != -1.0
        )
    return ref_cn, anomalies


def _scan_c6_block(tables: Tables, nref: list[int], zi: int, zj: int) -> list[tuple[int, int]]:
    """Validate one padded block; return owners ``(Z, ref)`` of nonzero padding."""
    ic = _pair_index(zi, zj)
    owners: list[tuple[int, int]] = []
    for a in range(MAX_REF):
        for b in range(MAX_REF):
            value = _padded_c6(tables.c6_padded, ic, a, b)
            if not math.isfinite(value):
                msg = f"non-finite C6 for Z={zi},{zj}"
                raise GenerationError(msg)
            if a < nref[zi] and b < nref[zj]:
                if zi == zj and value != _padded_c6(tables.c6_padded, ic, b, a):
                    msg = f"asymmetric diagonal C6 block for Z={zi}"
                    raise GenerationError(msg)
            elif value != 0.0:
                owners.append((zi, a + 1) if a >= nref[zi] else (zj, b + 1))
    return owners


def _pack_c6(tables: Tables, nref: list[int]) -> tuple[list[float], list[dict[str, Any]]]:
    c6: list[float] = []
    padded_nonzero: dict[tuple[int, int], int] = {}
    for zi in range(1, Z_MAX + 1):
        for zj in range(1, zi + 1):
            for owner in _scan_c6_block(tables, nref, zi, zj):
                padded_nonzero[owner] = padded_nonzero.get(owner, 0) + 1
            ic = _pair_index(zi, zj)
            for a in range(nref[zi]):
                c6.extend(_padded_c6(tables.c6_padded, ic, a, b) for b in range(nref[zj]))
    anomalies = [
        {"Z": z, "kind": "populated_padding_c6", "reference": ref, "nonzero_entries": count}
        for (z, ref), count in sorted(padded_nonzero.items())
    ]
    return c6, anomalies


def _zero_valid_refs(tables: Tables, nref: list[int]) -> list[dict[str, int]]:
    """Return valid references whose C6 against every partner is exactly zero."""
    found: list[dict[str, int]] = []
    for z in range(1, Z_MAX + 1):
        for a in range(nref[z]):
            values = []
            for partner in range(1, Z_MAX + 1):
                ic = _pair_index(max(z, partner), min(z, partner))
                for b in range(nref[partner]):
                    row, col = (a, b) if z >= partner else (b, a)
                    values.append(_padded_c6(tables.c6_padded, ic, row, col))
            if not any(values):
                found.append({"Z": z, "reference": a + 1})
    return found


def pack_tables(tables: Tables) -> Packed:
    """Pack the valid reference blocks and validate every padding entry."""
    nref = [0, *tables.nref]
    if not all(1 <= n <= MAX_REF for n in tables.nref):
        msg = "reference count out of range"
        raise GenerationError(msg)
    ref_cn, cn_anomalies = _pack_reference_cn(tables, nref)
    c6, c6_anomalies = _pack_c6(tables, nref)
    rcov = [math.nan, *tables.rcov_d3_all[:Z_MAX]]
    r4r2 = [math.nan, *tables.r4r2_all[:Z_MAX]]
    if not all(v > 0.0 for v in rcov[1:] + r4r2[1:] + tables.vdwrad):
        msg = "non-positive radius"
        raise GenerationError(msg)
    return Packed(
        nref=nref,
        ref_cn=ref_cn,
        rcov=rcov,
        r4r2=r4r2,
        c6=c6,
        rvdw=list(tables.vdwrad),
        anomalies=cn_anomalies + c6_anomalies,
        zero_valid_refs=_zero_valid_refs(tables, nref),
    )


EXPECTED_ANOMALIES = [
    {"Z": 96, "kind": "populated_padding_reference_cn", "reference": 7},
    {"Z": 96, "kind": "populated_padding_c6", "reference": 7},
]
EXPECTED_ZERO_VALID_REFS = [{"Z": 101, "reference": 6}]


def validate_anomalies(packed: Packed) -> None:
    """Require the anomaly list to be exactly the documented one."""
    seen = [{k: a[k] for k in ("Z", "kind", "reference")} for a in packed.anomalies]
    if seen != EXPECTED_ANOMALIES:
        msg = f"unexpected reference-data anomalies: {packed.anomalies}"
        raise GenerationError(msg)
    if packed.zero_valid_refs != EXPECTED_ZERO_VALID_REFS:
        msg = f"unexpected all-zero valid references: {packed.zero_valid_refs}"
        raise GenerationError(msg)


def tied_maxima(packed: Packed) -> list[int]:
    """Return atomic numbers whose maximum reference CN is tied."""
    tied: list[int] = []
    offset = 0
    for z in range(1, Z_MAX + 1):
        refs = packed.ref_cn[offset : offset + packed.nref[z]]
        offset += packed.nref[z]
        if refs.count(max(refs)) > 1:
            tied.append(z)
    return tied


def build_binary(packed: Packed, source_id: str) -> tuple[bytes, dict[str, str]]:
    """Serialise the packed tables into format v1 and return per-section hashes."""
    sections = [
        Section(
            SECTION_NREF, DTYPE_U8, bytes(packed.nref), len(packed.nref), "number_of_references"
        ),
        Section(
            SECTION_REF_CN, DTYPE_F64, pack_f64(packed.ref_cn), len(packed.ref_cn), "reference_cn"
        ),
        Section(
            SECTION_RCOV_D3, DTYPE_F64, pack_f64(packed.rcov), len(packed.rcov), "covalent_radii_d3"
        ),
        Section(SECTION_R4R2, DTYPE_F64, pack_f64(packed.r4r2), len(packed.r4r2), "r4r2"),
        Section(SECTION_C6, DTYPE_F64, pack_f64(packed.c6), len(packed.c6), "reference_c6_packed"),
        Section(SECTION_RVDW, DTYPE_F64, pack_f64(packed.rvdw), len(packed.rvdw), "vdw_radii"),
    ]
    prefix = struct.pack(
        "<8sIIIIIIIIQ",
        b"VDFD3TBL",
        FORMAT_VERSION,
        HEADER_BYTES,
        BYTE_ORDER_MARK,
        FLAG_RADII_D3_SCALED | FLAG_ATOMIC_UNITS,
        Z_MAX,
        MAX_REF,
        len(packed.ref_cn),
        N_PAIRS,
        len(packed.c6),
    )
    return build_container(prefix, source_id, sections)


# --------------------------------------------------------------------------
# JSON outputs
# --------------------------------------------------------------------------


def build_methods(methods: MethodTable, pin: SourcePin) -> dict[str, Any]:
    """Build the D3(BJ) method table document."""
    canonical = {
        method_id: methods.first_alias[method_id] for method_id in sorted(methods.first_alias)
    }
    entries = {
        canonical[method_id]: {
            **{key: params[key] for key in _PARAM_KEYS},
            "citation": params["citation"],
            "upstream_id": method_id,
        }
        for method_id, params in methods.bj.items()
    }
    known_without_bj = sorted(
        canonical[method_id] for method_id in canonical if method_id not in methods.bj
    )
    aliases = {alias: canonical[method_id] for alias, method_id in methods.aliases.items()}
    return {
        "schema": "vanderfast.d3bj-methods/1",
        "source": f"simple-dftd3 {pin.meta['tag']} ({pin.commit}) src/dftd3/param.f90",
        "defaults": {key: methods.defaults[key] for key in _PARAM_KEYS},
        "atm_rule": (
            "atm=True sets s9=1.0; atm=False keeps s9=0.0; pair coefficients "
            "s6, s8, a1, a2 are identical in both cases"
        ),
        "normalisation": (
            "UTF-8 bytes; remove '-'; lowercase ASCII A-Z outside quotes; "
            "trim trailing blanks; leading blanks are significant"
        ),
        "methods": entries,
        "aliases": aliases,
        "known_without_bj": known_without_bj,
    }


def build_manifest(
    pins: dict[str, SourcePin],
    constants: Constants,
    packed: Packed,
    *,
    table_hashes: dict[str, str],
    outputs: dict[str, bytes],
    conventions: dict[str, str],
    methods_doc: dict[str, Any],
) -> dict[str, Any]:
    """Build the provenance manifest."""
    return {
        "schema": "vanderfast.reference-manifest/1",
        "generator": {"script": "tools/generate_d3_data.py", "version": GENERATOR_VERSION},
        "format": {
            "name": "d3_reference.bin",
            "version": FORMAT_VERSION,
            "header_bytes": HEADER_BYTES,
            "section_alignment": SECTION_ALIGN,
            "layout": (
                "C6 pairs ordered by ic = Zi(Zi-1)/2 + (Zj-1) for Zi >= Zj; each block "
                "row-major nref(Zi) x nref(Zj) with rows for the larger Z; per-element "
                "arrays indexed by Z with slot 0 a sentinel"
            ),
        },
        "sources": manifest_sources(pins),
        "outputs": manifest_outputs(outputs),
        "tables": {name: {"sha256": digest} for name, digest in sorted(table_hashes.items())}
        | {"bj_parameters": {"sha256": hashlib.sha256(dumps(methods_doc)).hexdigest()}},
        "supported_atomic_numbers": {"min": 1, "max": Z_MAX},
        "counts": {
            "n_refs_total": len(packed.ref_cn),
            "n_c6": len(packed.c6),
            "n_pairs": N_PAIRS,
            "bj_methods": len(methods_doc["methods"]),
            "aliases": len(methods_doc["aliases"]),
        },
        "units": {
            "internal": "bohr, hartree",
            "table_aatoau": {
                "value": constants.aatoau,
                "hex": constants.aatoau.hex(),
                "source": ("mctc_io_convert: aatoau = 1/(hbar/(me*c*alpha)*1e10), CODATA 2018"),
            },
            "boundary": {
                "bohr_angstrom": 0.5291772105638411,
                "hartree_ev": 27.211386024367243,
                "source": "ase.units (CODATA 2014), identical to dftd3.ase",
                "recipes": {
                    "positions": "x * (1/Bohr)",
                    "cutoffs": "x / Bohr",
                    "energy": "e * Hartree",
                    "forces": "g * -(Hartree/Bohr)",
                    "stress": "G * (Hartree/V) -> flat[[0, 4, 8, 5, 2, 1]]",
                },
            },
        },
        "conventions": {
            "rcov": "mctc covalent_rad_d3 = (4/3)*covalent_rad_2009, already scaled, bohr",
            "cn": {
                "kcn": 16.0,
                "exclude_r2_below": 1e-12,
                "cutoff_rule": "r2 > rc^2 excluded (r2 == rc^2 included)",
                "default_cutoff_bohr": 40.0,
            },
            "weights": {
                "wf": 4.0,
                "fallback": (
                    "non-finite normalised weight -> 1.0 for every reference whose CN "
                    "equals the maximum valid reference CN, else 0.0; non-finite "
                    "derivative -> 0.0"
                ),
                "tied_maxima": tied_maxima(packed),
            },
            "pair": {
                "kappa": "3*r4r2_i*r4r2_j",
                "r0": "a1*sqrt(kappa) + a2",
                "exclude_r2_below": "DBL_EPSILON",
                "skip_if": "|s6| < eps and |s8| < eps",
                "default_cutoff_bohr": 60.0,
            },
            "atm": {
                "rs9": "4/3",
                "alp_effective": "alp + 2",
                "enabled_if": "|s9| >= DBL_EPSILON",
                "c9": "-s9*sqrt(|c6ij*c6ik*c6jk|)",
                "edge_exclusion": "each of the three edges: r2 > rc^2 or r2 < DBL_EPSILON",
                "third_edge": "vjk = vik - vij (no re-imaging)",
                "default_cutoff_bohr": 40.0,
            },
            "source_lines": dict(sorted(conventions.items())),
        },
        "anomalies": packed.anomalies,
        "zero_c6_valid_references": packed.zero_valid_refs,
        "notices": "THIRD_PARTY_NOTICES.md",
    }


# --------------------------------------------------------------------------
# Binary verification against compiled upstream libraries
# --------------------------------------------------------------------------


def verify_binary(libdir: Path, tables: Tables, constants: Constants) -> list[str]:
    """Search compiled upstream libraries for the parsed arrays; return failures."""
    dftd3 = b"".join(p.read_bytes() for p in sorted(libdir.glob("libs-dftd3.so*")) if p.is_file())
    mctc = b"".join(p.read_bytes() for p in sorted(libdir.glob("libmctc-lib.so*")) if p.is_file())
    if not dftd3 or not mctc:
        return [f"libs-dftd3/libmctc-lib not found in {libdir}"]
    checks: list[tuple[str, bytes, bytes]] = [
        ("reference_cn", pack_f64(tables.reference_cn_padded), dftd3),
        ("sqrt_z_r4_over_r2", pack_f64(tables.r4r2_all), dftd3),
        ("vdwrad", pack_f64(tables.vdwrad), dftd3),
        ("covalent_rad_d3", pack_f64(tables.rcov_d3_all), mctc),
        ("aatoau", pack_f64([constants.aatoau]), mctc),
    ]
    chunk = 2000
    for start in range(0, len(tables.c6_padded), chunk):
        values = tables.c6_padded[start : start + chunk]
        checks.append((f"c6[{start + 1}:{start + len(values)}]", pack_f64(values), dftd3))
    return [name for name, needle, haystack in checks if needle not in haystack]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def generate(reader: SourceReader, pins: dict[str, SourcePin]) -> Generated:
    """Run the full generation and return output bytes keyed by file name."""
    constants = parse_constants(reader)
    nref, reference_cn, c6 = parse_reference(reader)
    tables = Tables(
        nref=nref,
        reference_cn_padded=reference_cn,
        c6_padded=c6,
        r4r2_all=parse_r4r2(reader),
        rcov_d3_all=parse_covrad(reader, constants),
        vdwrad=parse_vdwrad(reader, constants),
    )
    conventions = check_conventions(reader)
    wrap = reader.text("simple-dftd3", "subprojects/mctc-lib.wrap")
    if "revision = " + pins["mctc-lib"].commit[:7] not in wrap:
        msg = "simple-dftd3 mctc-lib.wrap does not pin the configured mctc-lib commit"
        raise GenerationError(msg)
    methods = parse_methods(reader)

    packed = pack_tables(tables)
    validate_anomalies(packed)
    pin = pins["simple-dftd3"]
    source_id = f"simple-dftd3 {pin.meta['tag']} {pin.commit[:12]}"
    binary, table_hashes = build_binary(packed, source_id)
    methods_doc = build_methods(methods, pin)
    methods_bytes = dumps(methods_doc)
    outputs = {"d3_reference.bin": binary, "d3bj_methods.json": methods_bytes}
    manifest = build_manifest(
        pins,
        constants,
        packed,
        table_hashes=table_hashes,
        outputs=outputs,
        conventions=conventions,
        methods_doc=methods_doc,
    )
    outputs["manifest.json"] = dumps(manifest)
    return Generated(outputs, lambda libdir: verify_binary(libdir, tables, constants))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return run_cli(
        argv,
        description=__doc__.splitlines()[0],
        script="tools/generate_d3_data.py",
        sources_toml=SOURCES_TOML,
        generate=generate,
        verify_help="search libs-dftd3/libmctc-lib in LIBDIR for the parsed arrays",
    )


if __name__ == "__main__":
    raise SystemExit(main())
