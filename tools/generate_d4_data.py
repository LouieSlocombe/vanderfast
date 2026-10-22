"""Generate the bundled D4 reference tables from pinned upstream sources.

The generator reads the exact upstream files listed in ``data/sources-d4.toml``
(dftd4, multicharge, and mctc-lib, downloaded by commit SHA or taken from local
checkouts), verifies every sha256, parses the Fortran data statements
strictly, and writes three artifacts into ``src/vanderfast/data``:

``d4_reference.bin``
    Packed little-endian raw tables (format v1, see ``docs/data-provenance.md``):
    reference counts, reference CNs and charges, dynamic polarizabilities of
    the reference and secondary systems, per-element data for the D4 model and
    the EEQ charge model, and the Casimir-Polder quadrature.
``d4_methods.json``
    D4 rational-damping parameters and functional-name aliases.
``d4_manifest.json``
    Provenance, per-table hashes, units, conventions, anomalies, and hashes of
    the reference polarizabilities and C6 coefficients derived at the default
    ``ga``/``gc`` (so every reimplementation of that derivation is checked).

The tables are stored raw because the reference polarizabilities depend on the
charge-scaling parameters ``ga`` and ``gc``, which users may change; dftd4 also
derives them at model construction. Outputs are deterministic so ``--check``
can require byte-identical regeneration. ``--verify-binary`` searches the
compiled upstream libraries for the parsed arrays.

Usage::

    python -I tools/generate_d4_data.py [--source-dir NAME=DIR ...] [--cache DIR]
                                        [--check] [--verify-binary LIBDIR]
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
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
    check_licence_copies,
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
SOURCES_TOML = REPO_ROOT / "data" / "sources-d4.toml"

GENERATOR_VERSION = 1
FORMAT_VERSION = 1
MAGIC = b"VDFD4TBL"
BYTE_ORDER_MARK = 0x01020304
FLAG_RCOV_SCALED = 1 << 0
FLAG_ATOMIC_UNITS = 1 << 1
FLAG_RAW_POLARIZABILITIES = 1 << 2
FLAG_EEQ2019 = 1 << 3

SECTION_NREF = 1
SECTION_NGW = 2
SECTION_REFSYS = 3
SECTION_REF_SCALARS = 4
SECTION_ALPHAIW = 5
SECTION_SECONDARY = 6
SECTION_ELEMENT = 7
SECTION_FREQ = 8

Z_MAX = 103  # the EEQ 2019 parameters end at Lr
UPSTREAM_MAX_ELEM = 118
MAX_REF = 7
N_FREQ = 23
N_SECONDARY = 17
MAX_CN_BIN = 19

GA_DEFAULT = 3.0
GC_DEFAULT = 2.0
WF_DEFAULT = 6.0

REF_SCALAR_COLUMNS = ("refcovcn", "refcn", "clsq", "clsh", "hcount", "ascale")
ELEMENT_COLUMNS = (
    "zeff",
    "hardness",
    "en",
    "rcov_d4",
    "r4r2",
    "eeq_chi",
    "eeq_eta",
    "eeq_kcnchi",
    "eeq_rad",
    "eeq_rcov",
)

# Arrays declared in reference.f90. Only the EEQ-relevant ones are emitted; the
# others are parsed and shape-checked so that an upstream layout change fails.
PER_REF_REAL = (
    "refq",
    "refh",
    "dftq",
    "pbcq",
    "gffq",
    "clsq",
    "eeqbcq",
    "dfth",
    "pbch",
    "gffh",
    "clsh",
    "eeqbch",
    "hcount",
    "ascale",
    "refcovcn",
    "refcn",
)
PER_REF_INT = ("refsys",)
SECONDARY_REAL = ("secq", "sscale", "seccn", "seccnd3")
EMITTED_PER_REF = (*REF_SCALAR_COLUMNS, "refsys", "alphaiw")

# References counted by refn but never given data statements. gfortran zero-fills
# them, so they carry zero polarizability and a reference CN of zero, but still
# take part in the weight normalisation and the Gaussian-count histogram.
EXPECTED_PHANTOMS = [(92, 3), (95, 5), (98, 6)]
EXPECTED_UNINITIALISED_SECONDARY = [3, 4, 5, 12, 13, 14, 15, 16]

SQRT_Z_EXPR = "sqrt(0.5_wp*(r4_over_r2*[(sqrt(real(idum,wp)),idum=1,max_elem)]))"
TRAPZD_WEIGHTS_EXPR = (
    "real(wp),parameter::weights(23)=0.5_wp*[(freq(2)-freq(1)),"
    + ",".join(f"(freq({k})-freq({k - 1}))+(freq({k + 1})-freq({k}))" for k in range(2, 23))
    + ",(freq(23)-freq(22))]"
)

# Lines whose presence pins a numerical convention that vanderfast reproduces,
# keyed by (source, path). They are compared after removing all whitespace.
CONVENTION_LINES: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("dftd4", "src/dftd4/reference.f90"): [
        ("reference.max_cn_bin", "integer, parameter :: max_cn = 19"),
        ("reference.ngw_histogram_start", "cnc(:) = [1, spread(0, 1, max_cn)]"),
        ("reference.ngw_bin", "icn = min(nint(refcn(ir, num)), max_cn)"),
        ("reference.ngw", "ngw(ir) = icn*(icn+1)/2"),
        ("reference.cn", "cn(:ref) = refcovcn(:ref, num)"),
        ("reference.charge", "q(:ref) = clsq(:ref, num)"),
        ("reference.refsys_skip", "if (abs(is) < 1e-12_wp) cycle"),
        ("reference.secondary_zeff", "iz = get_effective_charge(is)"),
        (
            "reference.secondary_alpha",
            "aiw = sscale(is)*secaiw(:, is) * zeta(ga, get_hardness(is)*gc, iz, clsh(ir, num)+iz)",
        ),
        (
            "reference.alpha",
            (
                "alpha(:, ir) = max(ascale(ir, num)*(alphaiw(:, ir, num) "
                "- hcount(ir, num)*aiw), 0.0_wp)"
            ),
        ),
    ],
    ("dftd4", "src/dftd4/model/d4.f90"): [
        ("model.ga", "real(wp), parameter :: ga_default = 3.0_wp"),
        ("model.gc", "real(wp), parameter :: gc_default = 2.0_wp"),
        ("model.wf", "real(wp), parameter :: wf_default = 6.0_wp"),
        ("model.thopi", "real(wp), parameter :: thopi = 3.0_wp/pi"),
        ("model.c6_integrand", "aiw(:) = d4%aiw(:, iref, isp) * d4%aiw(:, jref, jsp)"),
        ("model.c6", "c6 = thopi * trapzd(aiw)"),
        ("model.unsupported", "if (mol%num(isp) > 103 .and. mol%num(isp) < 112) then"),
        ("weights.eps_norm", "real(wp), parameter :: eps_norm = tiny(1._wp)**0.5_wp"),
        ("weights.wf", "wf = igw * self%wf"),
        ("weights.dnorm", "dnorm = dnorm + 2*wf * (self%cn(iref, izp) - cn(iat)) * gw"),
        ("weights.norm_guard", "if (abs(norm) > eps_norm) then"),
        ("weights.fallback", "if (is_exceptional(gwk) .or. norm == 0.0_wp) then"),
        ("weights.fallback_ref", "if (abs(maxcn - self%cn(iref, izp)) < 1e-12_wp) then"),
        ("weights.hardness", "gi = self%eta(izp) * self%gc"),
        (
            "weights.zeta",
            "gwvec(iref, iat, 1) = gwk * zeta(self%ga, gi, self%q(iref, izp)+zi, q(iat)+zi)",
        ),
        ("weights.dgw", "dgwk = norm * (expd - expw * dnorm * norm)"),
        ("model.eeq", "call new_eeq2019_model(mol, d4%mchrg, error)"),
    ],
    ("dftd4", "src/dftd4/model/utils.f90"): [
        ("weights.gaussian", "cngw = exp ( -wf * ( cn - cnref )**2 )"),
        ("zeta.negative", "zeta = exp( a )"),
        ("zeta.value", "zeta = exp( a * ( 1.0_wp - exp( c * ( 1.0_wp - qref/qmod ) ) ) )"),
        (
            "zeta.derivative",
            (
                "dzeta = - a * c * exp( c * ( 1.0_wp - qref/qmod ) ) "
                "* zeta(a,c,qref,qmod) * qref / ( qmod**2 )"
            ),
        ),
        ("c6.trapezoid", "trapzd = sum(pol*weights)"),
    ],
    ("dftd4", "src/dftd4/ncoord.f90"): [
        ("cn.kcn", "real(wp), parameter :: default_kcn = 7.5_wp"),
    ],
    ("dftd4", "src/dftd4/disp.f90"): [
        ("disp.charge_chain", "call d4_gemv(dqdr, dEdq, gradient, beta=1.0_wp)"),
        ("disp.atm_neutral", "q(:) = 0.0_wp"),
        (
            "disp.cn_derivs_lattice",
            (
                "call add_coordination_number_derivs(mol, lattr, cutoff%cn, "
                "disp%rcov, disp%en, dEdcn, gradient, sigma)"
            ),
        ),
    ],
    ("dftd4", "src/dftd4/cutoff.f90"): [
        ("cutoff.cn", "real(wp), parameter :: cn_default = 30.0_wp"),
        ("cutoff.disp2", "real(wp), parameter :: disp2_default = 60.0_wp"),
        ("cutoff.disp3", "real(wp), parameter :: disp3_default = 40.0_wp"),
        ("cutoff.any_periodic", "if (.not.any(periodic)) then"),
    ],
    ("dftd4", "src/dftd4/damping/rational.f90"): [
        ("pair.kappa", "rrij = 3*r4r2(izp)*r4r2(jzp)"),
        ("pair.r0", "r0ij = self%a1 * sqrt(rrij) + self%a2"),
        ("pair.t6", "t6 = 1.0_wp/(r2**3 + r0ij**6)"),
        ("pair.t8", "t8 = 1.0_wp/(r2**4 + r0ij**8)"),
        ("pair.exclusion", "if (r2 > cutoff2 .or. r2 < epsilon(1.0_wp)) cycle"),
        (
            "pair.skip",
            "if (abs(self%s6) < epsilon(1.0_wp) .and. abs(self%s8) < epsilon(1.0_wp)) return",
        ),
        ("atm.enabled", "if (abs(self%s9) < epsilon(1.0_wp)) return"),
    ],
    ("dftd4", "src/dftd4/damping/atm.f90"): [
        ("atm.alp3", "alp3 = alp / 3.0_wp"),
        ("atm.r0", "r0ij = a1 * sqrt(3*r4r2(jzp)*r4r2(izp)) + a2"),
        ("atm.c9", "c9 = -s9 * sqrt(abs(c6ij*c6ik*c6jk))"),
        ("atm.fdmp", "fdmp = 1.0_wp / (1.0_wp + 6.0_wp * (r0 / r1)**alp3)"),
        ("atm.third_edge", "vjk(:) = vik(:) - vij(:)"),
    ],
    ("dftd4", "src/dftd4/param.f90"): [
        ("methods.prefix", "is = index(functional, '/')"),
        ("methods.lowercase", "fname = lowercase(functional(:is-1))"),
        ("methods.mbd", "if (present(s9)) mbd = abs(s9) > epsilon(s9)"),
        ("methods.alp", "alp_ = 16.0_wp"),
    ],
    ("dftd4", "src/dftd4/api.f90"): [
        ("api.wrap", "call wrap_to_central_cell(mol%ptr%xyz, mol%ptr%lattice, mol%ptr%periodic)"),
        ("api.atm_on", "s9 = 1.0_wp"),
        ("api.atm_off", "s9 = 0.0_wp"),
    ],
    ("multicharge", "src/multicharge/param.f90"): [
        ("eeq.cn_cutoff", "real(wp), parameter :: cutoff = 25.0_wp"),
        ("eeq.cn_kcn", "real(wp), parameter :: cn_exp = 7.5_wp"),
        ("eeq.cn_max", "real(wp), parameter :: cn_max = 8.0_wp"),
        ("eeq.rcov", "rcov = get_covalent_rad(mol%num)"),
    ],
    ("multicharge", "src/multicharge/model/eeq.f90"): [
        ("eeq.reg", "real(wp), parameter :: reg = 1.0e-14_wp"),
        ("eeq.xvec", "xvec(iat) = -self%chi(izp) + tmp * ptr%cn(iat)"),
        ("eeq.xvec_tmp", "tmp = self%kcnchi(izp) / sqrt(ptr%cn(iat) + reg)"),
        (
            "eeq.xvec_derivative",
            "dxdr(:, :, iat) = 0.5_wp * tmp * ptr%dcndr(:, :, iat) + dxdr(:, :, iat)",
        ),
        ("eeq.gam_0d", "gam = 1.0_wp / (self%rad(izp)**2 + self%rad(jzp)**2)"),
        ("eeq.amat_0d", "tmp = erf(sqrt(r2 * gam)) / sqrt(r2)"),
        ("eeq.diag_0d", "tmp = self%eta(izp) + sqrt2pi / self%rad(izp)"),
        ("eeq.gam_3d", "gam = 1.0_wp / sqrt(self%rad(izp)**2 + self%rad(jzp)**2)"),
        ("eeq.gam_self", "gam = 1.0_wp / sqrt(2.0_wp * self%rad(izp)**2)"),
        ("eeq.diag_3d", "dtmp = self%eta(izp) + sqrt2pi / self%rad(izp) - 2 * alpha / sqrtpi"),
        ("eeq.dir_eps", "real(wp), parameter :: eps = sqrt(epsilon(0.0_wp))"),
        ("eeq.dir", "tmp = erf(gam * r1) / r1 - erf(alp * r1) / r1"),
        (
            "eeq.rec",
            "tmp = cos(dot_product(rij, vec)) * fac * exp(-0.25_wp * g2 / (alp * alp)) / g2",
        ),
        (
            "eeq.rec_strain",
            (
                "ds(:, :) = ds + etmp * cos(gv) * ((2.0_wp / g2 + 0.5_wp / alp2) "
                "* spread(vec, 1, 3) * spread(vec, 2, 3) - unity)"
            ),
        ),
    ],
    ("multicharge", "src/multicharge/model/type.F90"): [
        ("ewald.rep", "integer, parameter :: rep(3) = [2, 2, 2]"),
        ("ewald.dir", "call get_lattice_points(lattice, rep, .true., trans)"),
        ("ewald.rec_lattice", "rec_lat = twopi * transpose(matinv_3x3(lattice))"),
        ("ewald.rec", "call get_lattice_points(rec_lat, rep, .false., trans)"),
        ("eeq.solver", "call sytrf(ainv, ipiv, info=info, uplo='l')"),
    ],
    ("multicharge", "src/multicharge/ewald.f90"): [
        ("ewald.alpha0", "real(wp), parameter :: alpha0 = 1.0e-8_wp"),
        ("ewald.niter", "integer, parameter :: niter = 30"),
        ("ewald.fallback", "alpha = 0.25_wp"),
        ("ewald.dir_term", "dval = erfc(alpha*rr)/rr"),
        ("ewald.rec_term", "rval = 4.0_wp*pi*(exp(-0.25_wp*gg*gg/(alpha**2))/(vol*gg*gg))"),
    ],
    ("multicharge", "src/multicharge/wignerseitz.f90"): [
        ("ws.thr", "real(wp), parameter :: thr = sqrt(epsilon(0.0_wp))"),
        ("ws.tol", "real(wp), parameter :: tol = 0.01_wp"),
        ("ws.candidates", "call get_lattice_points(mol%periodic, mol%lattice, thr, trans)"),
        ("ws.skip", "if (r2 < thr) cycle"),
        ("ws.compacted_index", "list(iws) = pos"),
        ("ws.tie", "if (abs(dist(pos) - r2) > tol) exit"),
    ],
    ("mctc-lib", "src/mctc/ncoord/type.f90"): [
        ("cn.exclusion", "if (r2 > cutoff2 .or. r2 < 1.0e-12_wp) cycle"),
        ("cn.cap", "cnp = log(1.0_wp + exp(cnmax)) - log(1.0_wp + exp(cnmax - cn))"),
        ("cn.dcap", "dcnpdcn = exp(cnmax)/(exp(cnmax) + exp(cn))"),
    ],
    ("mctc-lib", "src/mctc/ncoord/erf.f90"): [
        ("cn.erf", "count = 0.5_wp * (1.0_wp + erf(-self%kcn*(r-rc)/rc**self%norm_exp))"),
        ("cn.derf", "count = -(self%kcn*expterm)/(sqrtpi*rc**self%norm_exp)"),
    ],
    ("mctc-lib", "src/mctc/ncoord/erf/dftd4.f90"): [
        ("cn.k4", "real(wp), parameter :: k4 = 4.10451_wp"),
        ("cn.k5", "real(wp), parameter :: k5 = 19.08857_wp"),
        ("cn.k6", "real(wp), parameter :: k6 = 2*11.28174_wp**2.0_wp"),
        ("cn.en_factor", "en_factor = k4*exp(-(abs(self%en(izp)-self%en(jzp)) + k5)**2.0_wp/k6)"),
    ],
    ("mctc-lib", "src/mctc/cutoff.f90"): [
        (
            "lattice.translation",
            "trans(:, itr) = lat(:, 1)*ix*jx + lat(:, 2)*iy*jy + lat(:, 3)*iz*jz",
        ),
        ("lattice.rep", "rep(1) = ceiling(abs(rthr/cos10))"),
        ("wrap.eps", "real(wp),parameter :: p_pbc_eps = 1.0e-14_wp"),
        ("wrap.low", "if(in < (0.0_wp - p_pbc_eps)) out = in + real(ceiling(-in),wp)"),
        ("wrap.high", "if(in > (1.0_wp + p_pbc_eps)) out = in - real(floor ( in),wp)"),
        ("wrap.one", "if (abs(in - 1.0_wp) < p_pbc_eps) out = in - 1.0_wp"),
    ],
}


# --------------------------------------------------------------------------
# reference.inc: Fortran data statements
# --------------------------------------------------------------------------

_DATA_STATEMENT = re.compile(r"data\s*(\w+)\s*\(([^)]*)\)\s*/(.*)/")
_DEFAULT_REAL = re.compile(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?")


def round_to_binary32(text: str) -> float:
    """Correctly round a decimal literal to IEEE binary32 (round half to even)."""
    exact = Fraction(text)
    if exact == 0:
        return 0.0
    sign = -1 if exact < 0 else 1
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    exponent = max(exponent, -126)  # subnormals share the minimum exponent
    scale = Fraction(2) ** (exponent - 23)
    mantissa = round(magnitude / scale)  # Fraction.__round__ rounds half to even
    return sign * float(mantissa * scale)


def parse_inc_reals(body: str, single: list[str], where: str) -> list[float]:
    """Parse ``_wp`` literals exactly and default-kind literals as gfortran stores them.

    A real literal without a kind suffix is default (single-precision) real; in a
    DATA statement for a ``real(wp)`` variable gfortran converts that binary32
    value, so the stored double is the decimal rounded to binary32.
    """
    values: list[float] = []
    for token in (part.strip() for part in body.split(",")):
        if REAL_LITERAL.fullmatch(token):
            values.extend(parse_real_list(token))
        elif _DEFAULT_REAL.fullmatch(token):
            values.append(round_to_binary32(token))
            single.append(where)
        else:
            msg = f"unexpected token in {where}: {token!r}"
            raise GenerationError(msg)
    return values


@dataclass
class RawReference:
    """reference.inc arrays in Fortran layout; ``None`` marks uninitialised storage."""

    refn: list[int | None]  # index Z-1, length 118
    per_ref: dict[str, list[list[float | None]]]  # name -> [Z-1][ref-1]
    refsys: list[list[int | None]]
    alphaiw: list[list[list[float] | None]]  # [Z-1][ref-1] -> 23 values
    secondary: dict[str, list[float | None]]  # name -> [s-1]
    secaiw: list[list[float] | None]  # [s-1] -> 23 values
    single_precision: list[str] = field(default_factory=list)  # targets of kind-less literals


def _indices(text: str) -> list[str]:
    return [part.strip() for part in text.split(",")]


def _assign(target: list[Any], index: int, value: Any, what: str) -> None:
    if target[index] is not None:
        msg = f"duplicate data statement for {what}"
        raise GenerationError(msg)
    target[index] = value


def parse_reference_inc(reader: SourceReader) -> RawReference:
    """Parse every data statement of ``reference.inc`` into Fortran-shaped arrays."""
    declarations = {
        squash(s) for s in fortran_statements(reader.text("dftd4", "src/dftd4/reference.f90"))
    }
    required = [
        "integer,parameter::max_elem=118",
        "integer,dimension(max_elem)::refn",
        "real(wp),dimension(7,max_elem)::dftq,pbcq,gffq,clsq,eeqbcq",
        "real(wp),dimension(7,max_elem)::dfth,pbch,gffh,clsh,eeqbch",
        "real(wp),dimension(7,max_elem)::refcovcn",
        "integer,dimension(7,max_elem)::refsys",
        "real(wp),dimension(23,7,max_elem)::alphaiw",
        "real(wp),dimension(17)::sscale",
        "real(wp),dimension(23,17)::secaiw",
        "include'reference.inc'",
    ]
    for line in required:
        if line not in declarations:
            msg = f"reference.f90 no longer declares {line!r}"
            raise GenerationError(msg)

    n = UPSTREAM_MAX_ELEM
    raw = RawReference(
        refn=[None] * n,
        per_ref={name: [[None] * MAX_REF for _ in range(n)] for name in PER_REF_REAL},
        refsys=[[None] * MAX_REF for _ in range(n)],
        alphaiw=[[None] * MAX_REF for _ in range(n)],
        secondary={name: [None] * N_SECONDARY for name in SECONDARY_REAL},
        secaiw=[None] * N_SECONDARY,
    )
    for statement in fortran_statements(reader.text("dftd4", "src/dftd4/reference.inc")):
        match = _DATA_STATEMENT.fullmatch(statement)
        if not match:
            msg = f"unexpected statement in reference.inc: {statement!r}"
            raise GenerationError(msg)
        name = match.group(1).lower()
        index = _indices(match.group(2))
        body = match.group(3).strip()
        _store(raw, name, index, body)
    return raw


def _store(raw: RawReference, name: str, index: list[str], body: str) -> None:
    if name == "refn" and len(index) == 1:
        z = int(index[0])
        (value,) = parse_int_list(body)
        _assign(raw.refn, z - 1, value, f"refn({z})")
    elif name in PER_REF_REAL and len(index) == 2:  # noqa: PLR2004
        ref, z = int(index[0]), int(index[1])
        where = f"{name}({ref},{z})"
        (real,) = parse_inc_reals(body, raw.single_precision, where)
        _assign(raw.per_ref[name][z - 1], ref - 1, real, where)
    elif name in PER_REF_INT and len(index) == 2:  # noqa: PLR2004
        ref, z = int(index[0]), int(index[1])
        (value,) = parse_int_list(body)
        _assign(raw.refsys[z - 1], ref - 1, value, f"refsys({ref},{z})")
    elif name == "alphaiw" and len(index) == 3 and index[0] == ":":  # noqa: PLR2004
        ref, z = int(index[1]), int(index[2])
        values = parse_inc_reals(body, raw.single_precision, f"alphaiw(:,{ref},{z})")
        if len(values) != N_FREQ:
            msg = f"alphaiw(:,{ref},{z}) has {len(values)} values"
            raise GenerationError(msg)
        _assign(raw.alphaiw[z - 1], ref - 1, values, f"alphaiw(:,{ref},{z})")
    elif name in SECONDARY_REAL and len(index) == 1:
        s = int(index[0])
        (real,) = parse_inc_reals(body, raw.single_precision, f"{name}({s})")
        _assign(raw.secondary[name], s - 1, real, f"{name}({s})")
    elif name == "secaiw" and len(index) == 2 and index[0] == ":":  # noqa: PLR2004
        s = int(index[1])
        values = parse_inc_reals(body, raw.single_precision, f"secaiw(:,{s})")
        if len(values) != N_FREQ:
            msg = f"secaiw(:,{s}) has {len(values)} values"
            raise GenerationError(msg)
        _assign(raw.secaiw, s - 1, values, f"secaiw(:,{s})")
    else:
        msg = f"unexpected data statement target {name}({','.join(index)})"
        raise GenerationError(msg)


def _ref_initialised(raw: RawReference, z: int, ref: int) -> list[bool]:
    status = [raw.per_ref[name][z - 1][ref - 1] is not None for name in REF_SCALAR_COLUMNS]
    status.append(raw.refsys[z - 1][ref - 1] is not None)
    status.append(raw.alphaiw[z - 1][ref - 1] is not None)
    return status


def _validate_references(raw: RawReference) -> list[tuple[int, int]]:
    """Require complete or wholly absent references and empty padding; return phantoms."""
    phantoms: list[tuple[int, int]] = []
    for z in range(1, UPSTREAM_MAX_ELEM + 1):
        nref = raw.refn[z - 1]
        supported = z <= Z_MAX or z >= 112  # noqa: PLR2004
        if nref is None and supported:
            msg = f"refn({z}) is not initialised"
            raise GenerationError(msg)
        if nref is not None and not 1 <= nref <= MAX_REF:
            msg = f"refn({z}) = {nref} out of range"
            raise GenerationError(msg)
        for ref in range(1, MAX_REF + 1):
            status = _ref_initialised(raw, z, ref)
            if ref > (nref or 0) and any(status):
                msg = f"padding reference {ref} of Z={z} carries data"
                raise GenerationError(msg)
            if ref <= (nref or 0) and not all(status):
                if any(status):
                    msg = f"reference {ref} of Z={z} is only partly initialised"
                    raise GenerationError(msg)
                phantoms.append((z, ref))
    return [(z, ref) for z, ref in phantoms if z <= Z_MAX]


def _validate_secondary(raw: RawReference) -> list[int]:
    """Require complete or wholly absent secondary systems that every reference can use."""
    empty: list[int] = []
    for s in range(1, N_SECONDARY + 1):
        defined = raw.secondary["sscale"][s - 1] is not None
        if defined != (raw.secaiw[s - 1] is not None):
            msg = f"secondary system {s} is only partly initialised"
            raise GenerationError(msg)
        if not defined:
            empty.append(s)
    for z in range(1, Z_MAX + 1):
        for ref in range(1, (raw.refn[z - 1] or 0) + 1):
            system = raw.refsys[z - 1][ref - 1] or 0
            if not 0 <= system <= N_SECONDARY or system in empty:
                msg = f"reference {ref} of Z={z} uses secondary system {system}"
                raise GenerationError(msg)
    return empty


def validate_reference(raw: RawReference) -> None:
    """Check counts, padding, and secondary systems against the documented anomalies."""
    phantoms = _validate_references(raw)
    if phantoms != EXPECTED_PHANTOMS:
        msg = f"unexpected uninitialised references: {phantoms}"
        raise GenerationError(msg)
    empty = _validate_secondary(raw)
    if empty != EXPECTED_UNINITIALISED_SECONDARY:
        msg = f"unexpected uninitialised secondary systems: {empty}"
        raise GenerationError(msg)


# --------------------------------------------------------------------------
# Element data
# --------------------------------------------------------------------------


def _element_array(
    reader: SourceReader, source: str, relpath: str, name: str, *, integer: bool = False
) -> list[float]:
    statements = fortran_statements(reader.text(source, relpath))
    max_elem = int_parameter(statements, "max_elem")
    match = find_one(statements, rf"real\(wp\), parameter :: {name}\(max_elem\) = \[(.*)\]")
    body = match.group(1)
    values = [float(v) for v in parse_int_list(body)] if integer else parse_real_list(body)
    if len(values) != max_elem:
        msg = f"{relpath}: {name} has {len(values)} values, expected {max_elem}"
        raise GenerationError(msg)
    return values


def parse_covalent_radii(
    reader: SourceReader, source: str, relpath: str, constants: Constants
) -> list[float]:
    """Return ``covalent_rad_d3 = (4/3) * aatoau * covalent_rad_2009`` in upstream order."""
    statements = fortran_statements(reader.text(source, relpath))
    max_elem = int_parameter(statements, "max_elem")
    match = find_one(
        statements,
        r"real\(wp\), parameter :: covalent_rad_2009\(max_elem\) = aatoau \* \[(.*)\]",
    )
    radii = parse_real_list(match.group(1))
    if len(radii) != max_elem:
        msg = f"{relpath}: covalent_rad_2009 length mismatch"
        raise GenerationError(msg)
    scaled = find_one(statements, r"real\(wp\), parameter :: covalent_rad_d3\(max_elem\) = (.*)")
    if squash(scaled.group(1)) != "4.0_wp/3.0_wp*covalent_rad_2009":
        msg = f"{relpath}: covalent_rad_d3 expression changed: {scaled.group(1)!r}"
        raise GenerationError(msg)
    if "rad=covalent_rad_d3(num)" not in {squash(s) for s in statements}:
        msg = f"{relpath}: get_covalent_rad no longer returns covalent_rad_d3"
        raise GenerationError(msg)
    return [(4.0 / 3.0) * (constants.aatoau * value) for value in radii]


def parse_r4r2(reader: SourceReader) -> list[float]:
    """Return dftd4's ``sqrt_z_r4_over_r2`` evaluated in upstream operation order."""
    relpath = "src/dftd4/data/r4r2.f90"
    r4_over_r2 = _element_array(reader, "dftd4", relpath, "r4_over_r2")
    statements = fortran_statements(reader.text("dftd4", relpath))
    expr = find_one(statements, r"real\(wp\), parameter :: sqrt_z_r4_over_r2\(max_elem\) = (.*)")
    if squash(expr.group(1)) != SQRT_Z_EXPR:
        msg = f"sqrt_z_r4_over_r2 expression changed: {expr.group(1)!r}"
        raise GenerationError(msg)
    return [
        math.sqrt(0.5 * (value * math.sqrt(float(z))))
        for z, value in enumerate(r4_over_r2, start=1)
    ]


@dataclass(frozen=True)
class Elements:
    """Per-element arrays (index Z-1) as upstream evaluates them."""

    zeff: list[float]
    hardness: list[float]
    en: list[float]
    rcov_d4: list[float]
    r4r2: list[float]
    eeq_chi: list[float]
    eeq_eta: list[float]
    eeq_kcnchi: list[float]
    eeq_rad: list[float]
    eeq_rcov: list[float]

    def column(self, name: str) -> list[float]:
        """Return one column by its ``ELEMENT_COLUMNS`` name."""
        values: list[float] = getattr(self, name)
        return values


def parse_elements(reader: SourceReader, constants: Constants) -> Elements:
    """Parse the dftd4 element data, the EEQ 2019 parameters, and the EEQ radii."""
    eeq = "src/multicharge/param/eeq2019.f90"
    return Elements(
        zeff=_element_array(
            reader, "dftd4", "src/dftd4/data/zeff.f90", "effective_nuclear_charge", integer=True
        ),
        hardness=_element_array(
            reader, "dftd4", "src/dftd4/data/hardness.f90", "chemical_hardness"
        ),
        en=_element_array(reader, "dftd4", "src/dftd4/data/en.f90", "pauling_en"),
        rcov_d4=parse_covalent_radii(reader, "dftd4", "src/dftd4/data/covrad.f90", constants),
        r4r2=parse_r4r2(reader),
        eeq_chi=_element_array(reader, "multicharge", eeq, "eeq_chi"),
        eeq_eta=_element_array(reader, "multicharge", eeq, "eeq_eta"),
        eeq_kcnchi=_element_array(reader, "multicharge", eeq, "eeq_kcnchi"),
        eeq_rad=_element_array(reader, "multicharge", eeq, "eeq_rad"),
        eeq_rcov=parse_covalent_radii(reader, "mctc-lib", "src/mctc/data/covrad.f90", constants),
    )


def parse_frequencies(reader: SourceReader) -> tuple[list[float], list[float]]:
    """Return the 23 imaginary frequencies and trapezoid weights of ``trapzd``."""
    statements = fortran_statements(reader.text("dftd4", "src/dftd4/model/utils.f90"))
    match = find_one(statements, r"real\(wp\), parameter :: freq\(23\) = \[(.*)\]")
    freq = parse_real_list(match.group(1))
    if len(freq) != N_FREQ:
        msg = "trapzd frequency grid changed length"
        raise GenerationError(msg)
    if TRAPZD_WEIGHTS_EXPR not in {squash(s) for s in statements}:
        msg = "trapzd weights expression changed"
        raise GenerationError(msg)
    weights = [0.5 * (freq[1] - freq[0])]
    weights += [0.5 * ((freq[k] - freq[k - 1]) + (freq[k + 1] - freq[k])) for k in range(1, 22)]
    weights.append(0.5 * (freq[22] - freq[21]))
    return freq, weights


# --------------------------------------------------------------------------
# Derived quantities (for validation and the manifest)
# --------------------------------------------------------------------------


def fortran_nint(value: float) -> int:
    """Fortran ``nint``: round half away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def gaussian_counts(refcn: Sequence[float]) -> list[int]:
    """Upstream ``set_refgw``: Gaussians per reference from the reference-CN histogram."""
    cnc = [1] + [0] * MAX_CN_BIN
    for value in refcn:
        cnc[min(fortran_nint(value), MAX_CN_BIN)] += 1
    counts = []
    for value in refcn:
        icn = cnc[min(fortran_nint(value), MAX_CN_BIN)]
        counts.append(icn * (icn + 1) // 2)
    return counts


def zeta(a: float, c: float, qref: float, qmod: float) -> float:
    """Upstream charge scaling ``zeta``."""
    if qmod < 0.0:
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref / qmod))))


@dataclass(frozen=True)
class Packed:
    """Packed section payloads (valid references of Z = 1..103 in Z, ref order)."""

    nref: list[int]  # slot 0 = 0
    ngw: list[int]
    refsys: list[int]
    ref_scalars: list[float]  # row-major n_refs x 6
    alphaiw: list[float]  # row-major n_refs x 23
    secondary: list[float]  # row-major 18 x 24, row 0 NaN
    element: list[float]  # row-major 104 x 10, row 0 NaN
    freq: list[float]  # 2 x 23: frequencies, then weights


def pack_tables(
    raw: RawReference, elements: Elements, freq: list[float], weights: list[float]
) -> Packed:
    """Pack valid references, zero-filling uninitialised storage as gfortran does."""
    nref = [0] + [raw.refn[z - 1] or 0 for z in range(1, Z_MAX + 1)]
    ngw: list[int] = []
    refsys: list[int] = []
    scalars: list[float] = []
    alphaiw: list[float] = []
    for z in range(1, Z_MAX + 1):
        refcn = [raw.per_ref["refcn"][z - 1][r] or 0.0 for r in range(nref[z])]
        if any(value < -0.5 for value in refcn):  # noqa: PLR2004
            msg = f"negative reference CN for Z={z} would index below the histogram"
            raise GenerationError(msg)
        ngw.extend(gaussian_counts(refcn))
        for r in range(nref[z]):
            refsys.append(raw.refsys[z - 1][r] or 0)
            scalars.extend(raw.per_ref[name][z - 1][r] or 0.0 for name in REF_SCALAR_COLUMNS)
            alphaiw.extend(raw.alphaiw[z - 1][r] or [0.0] * N_FREQ)
    if max(ngw) > 255 or max(refsys) > 255:  # noqa: PLR2004
        msg = "u8 overflow in NGW or REFSYS"
        raise GenerationError(msg)
    secondary = [math.nan] * (N_FREQ + 1)
    for s in range(1, N_SECONDARY + 1):
        secondary.append(raw.secondary["sscale"][s - 1] or 0.0)
        secondary.extend(raw.secaiw[s - 1] or [0.0] * N_FREQ)
    element = [math.nan] * len(ELEMENT_COLUMNS)
    for z in range(1, Z_MAX + 1):
        element.extend(elements.column(name)[z - 1] for name in ELEMENT_COLUMNS)
    return Packed(
        nref=nref,
        ngw=ngw,
        refsys=refsys,
        ref_scalars=scalars,
        alphaiw=alphaiw,
        secondary=secondary,
        element=element,
        freq=freq + weights,
    )


def _ref_slices(packed: Packed) -> list[tuple[int, int]]:
    """``(Z, first packed reference index)`` for every element."""
    offsets = []
    offset = 0
    for z in range(1, Z_MAX + 1):
        offsets.append((z, offset))
        offset += packed.nref[z]
    return offsets


def reference_alpha(packed: Packed, ga: float, gc: float) -> list[list[float]]:
    """Upstream ``set_refalpha_eeq`` for every packed reference (23 values each)."""
    ncol = len(ELEMENT_COLUMNS)
    zeff = ELEMENT_COLUMNS.index("zeff")
    hardness = ELEMENT_COLUMNS.index("hardness")
    alphas: list[list[float]] = []
    for index, system in enumerate(packed.refsys):
        if system == 0:
            alphas.append([0.0] * N_FREQ)
            continue
        clsh, hcount, ascale = packed.ref_scalars[6 * index + 3 : 6 * index + 6]
        row = packed.secondary[(N_FREQ + 1) * system : (N_FREQ + 1) * (system + 1)]
        sscale, secaiw = row[0], row[1:]
        iz = packed.element[ncol * system + zeff]
        scale = zeta(ga, packed.element[ncol * system + hardness] * gc, iz, clsh + iz)
        alpha = packed.alphaiw[N_FREQ * index : N_FREQ * (index + 1)]
        alphas.append(
            [
                max(ascale * (alpha[k] - hcount * ((sscale * secaiw[k]) * scale)), 0.0)
                for k in range(N_FREQ)
            ]
        )
    return alphas


def reference_c6(packed: Packed, alphas: list[list[float]], pi: float) -> list[float]:
    """C6 reference blocks for all pairs ``Zi >= Zj``, rows for ``Zi``, in pair order."""
    weights = packed.freq[N_FREQ:]
    thopi = 3.0 / pi
    slices = _ref_slices(packed)
    c6: list[float] = []
    for zi, oi in slices:
        for zj, oj in slices[:zi]:
            for a in range(packed.nref[zi]):
                ai = alphas[oi + a]
                for b in range(packed.nref[zj]):
                    bj = alphas[oj + b]
                    total = 0.0
                    for k in range(N_FREQ):
                        total += (ai[k] * bj[k]) * weights[k]
                    c6.append(thopi * total)
    return c6


def tied_maxima(packed: Packed) -> list[int]:
    """Atomic numbers whose maximum weighting CN is shared by several references."""
    tied = []
    for z, offset in _ref_slices(packed):
        cns = [packed.ref_scalars[6 * (offset + r)] for r in range(packed.nref[z])]
        top = max(cns)
        if sum(abs(top - cn) < 1e-12 for cn in cns) > 1:  # noqa: PLR2004
            tied.append(z)
    return tied


def zero_alpha_references(packed: Packed, alphas: list[list[float]]) -> list[dict[str, int]]:
    """References whose polarizability vanishes at every frequency (default ga, gc)."""
    found: list[dict[str, int]] = []
    for z, offset in _ref_slices(packed):
        found.extend(
            {"Z": z, "reference": r + 1}
            for r in range(packed.nref[z])
            if not any(alphas[offset + r])
        )
    return found


# --------------------------------------------------------------------------
# Method parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodTable:
    """Parsed D4 rational-damping tables and the functional-name map."""

    aliases: dict[str, str]  # alias -> functional id
    first_alias: dict[str, str]  # functional id -> canonical alias
    tables: dict[str, dict[str, dict[str, Any]]]  # "bj" | "bjatm" -> id -> parameters


_PARAM_KEYS = ("s6", "s8", "a1", "a2", "alp")
_CITATION = re.compile(r"!\s*\(([^)]+)\)")


def _parse_functional_ids(statements: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    start = statements.index("pure function get_functional_id(df) result(num)")
    end = statements.index("end function get_functional_id", start)
    body = statements[start + 1 : end]
    expected = ["integer(df_enum) :: num", "character(len=*), intent(in) :: df", "select case(df)"]
    if body[:3] != expected or body[3:5] != ["case default", "num = p_invalid"]:
        msg = "unexpected preamble in get_functional_id"
        raise GenerationError(msg)
    case_re = re.compile(r"case\(('[^']*'(?:, ?'[^']*')*)\)")
    id_re = re.compile(r"num = (p_\w+)")
    aliases: dict[str, str] = {}
    first_alias: dict[str, str] = {}
    pending: list[str] = []
    for statement in body[5:]:
        if statement == "end select":
            break
        if match := case_re.fullmatch(statement):
            pending = re.findall(r"'([^']*)'", match.group(1))
        elif (match := id_re.fullmatch(statement)) and pending:
            functional = match.group(1)
            for alias in pending:
                if alias in aliases:
                    msg = f"duplicate functional alias {alias!r}"
                    raise GenerationError(msg)
                aliases[alias] = functional
            first_alias.setdefault(functional, pending[0])
            pending = []
        else:
            msg = f"unexpected statement in get_functional_id: {statement!r}"
            raise GenerationError(msg)
    return aliases, first_alias


def _parse_keyword_args(body: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in body.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key in result or key not in _PARAM_KEYS:
            msg = f"unexpected keyword {key} in {body!r}"
            raise GenerationError(msg)
        result[key] = parse_real_list(value.strip())[0]
    return result


def _parse_parameter_block(
    source: str, statements: list[str], subroutine: str, s9_default: str
) -> dict[str, dict[str, Any]]:
    start = statements.index(f"subroutine {subroutine}(dfnum, param, s9)")
    end = statements.index(f"end subroutine {subroutine}", start)
    select = statements.index("select case(dfnum)", start)
    end_select = statements.index("end select", select)
    ids_re = re.compile(r"case\((p_\w+(?:, ?p_\w+)*)\)")
    param_re = re.compile(r"param = dftd_param ?\((.*)\)")
    entries: list[tuple[list[str], dict[str, float]]] = []
    for statement in statements[select + 1 : end_select]:
        if match := ids_re.fullmatch(statement):
            entries.append(([p.strip() for p in match.group(1).split(",")], {}))
        elif (match := param_re.fullmatch(statement)) and entries and not entries[-1][1]:
            entries[-1][1].update(_parse_keyword_args(match.group(1)))
        else:
            msg = f"unexpected statement in {subroutine}: {statement!r}"
            raise GenerationError(msg)
    tail = {squash(s) for s in statements[end_select + 1 : end]}
    for line in ["s6_=1.0_wp", s9_default, "if(present(s9))s9_=s9", "alp_=16.0_wp"]:
        if squash(line) not in tail:
            msg = f"{subroutine} defaults no longer contain {line!r}"
            raise GenerationError(msg)

    citations = _block_citations(source, subroutine)
    if len(citations) != len(entries):
        msg = f"{subroutine}: {len(citations)} citation slots for {len(entries)} entries"
        raise GenerationError(msg)
    table: dict[str, dict[str, Any]] = {}
    for (ids, values), citation in zip(entries, citations, strict=True):
        if set(values) - {"s6"} != {"s8", "a1", "a2"}:
            msg = f"{subroutine} entry for {ids} sets {sorted(values)}"
            raise GenerationError(msg)
        for functional in ids:
            if functional in table:
                msg = f"duplicate {subroutine} entry for {functional}"
                raise GenerationError(msg)
            table[functional] = {
                "s6": values.get("s6", 1.0),
                "s8": values["s8"],
                "a1": values["a1"],
                "a2": values["a2"],
                "alp": values.get("alp", 16.0),
                "citation": citation,
            }
    return table


def _block_citations(source: str, subroutine: str) -> list[str | None]:
    """First ``! (…)`` comment of every case block, read from the raw source lines."""
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if f"subroutine {subroutine}(" in line)
    citations: list[str | None] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("end select"):
            break
        if re.match(r"case\(p_", stripped):
            citations.append(None)
        if citations and citations[-1] is None and (match := _CITATION.search(line)):
            citations[-1] = match.group(1).strip()
    return citations


def parse_methods(reader: SourceReader) -> MethodTable:
    """Parse ``get_functional_id`` and both rational-damping tables from ``param.f90``."""
    source = reader.text("dftd4", "src/dftd4/param.f90")
    statements = fortran_statements(source)
    aliases, first_alias = _parse_functional_ids(statements)
    tables = {
        "bj": _parse_parameter_block(source, statements, "get_d4eeq_bj_parameter", "s9_ = 0.0_wp"),
        "bjatm": _parse_parameter_block(
            source, statements, "get_d4eeq_bjatm_parameter", "s9_ = 1.0_wp"
        ),
    }
    for name, table in tables.items():
        unknown = set(table) - set(aliases.values())
        if unknown:
            msg = f"{name} entries without a functional name: {sorted(unknown)}"
            raise GenerationError(msg)
    return MethodTable(aliases=aliases, first_alias=first_alias, tables=tables)


def build_methods(methods: MethodTable, pin: SourcePin) -> dict[str, Any]:
    """Build the D4 method table document."""
    canonical = {f: methods.first_alias[f] for f in sorted(methods.first_alias)}
    tables = {
        name: {
            canonical[functional]: {**params, "upstream_id": functional}
            for functional, params in sorted(table.items(), key=lambda item: canonical[item[0]])
        }
        for name, table in methods.tables.items()
    }
    with_parameters = set(methods.tables["bj"]) | set(methods.tables["bjatm"])
    known_without = sorted(canonical[f] for f in canonical if f not in with_parameters)
    return {
        "schema": "vanderfast.d4-methods/1",
        "source": f"dftd4 {pin.meta['tag']} ({pin.commit}) src/dftd4/param.f90",
        "defaults": {"s6": 1.0, "alp": 16.0, "s9_bj": 0.0, "s9_bjatm": 1.0},
        "atm_rule": (
            "atm=True sets s9=1.0 and reads the bjatm table, falling back to the bj table; "
            "atm=False sets s9=0.0 and reads the bj table, falling back to the bjatm table; "
            "the resulting s9 is always the requested one"
        ),
        "normalisation": (
            "UTF-8 bytes; keep the text before the first '/'; lowercase ASCII A-Z outside "
            "quotes; trim trailing blanks; hyphens are significant; exact match"
        ),
        "tables": tables,
        "aliases": {alias: canonical[f] for alias, f in sorted(methods.aliases.items())},
        "known_without_parameters": known_without,
    }


# --------------------------------------------------------------------------
# Packing and manifest
# --------------------------------------------------------------------------


def build_binary(packed: Packed, source_id: str) -> tuple[bytes, dict[str, str]]:
    """Serialise the packed tables into format v1 and return per-section hashes."""
    n_refs = len(packed.refsys)
    sections = [
        Section(
            SECTION_NREF, DTYPE_U8, bytes(packed.nref), len(packed.nref), "number_of_references"
        ),
        Section(SECTION_NGW, DTYPE_U8, bytes(packed.ngw), n_refs, "gaussian_counts"),
        Section(SECTION_REFSYS, DTYPE_U8, bytes(packed.refsys), n_refs, "secondary_system"),
        Section(
            SECTION_REF_SCALARS,
            DTYPE_F64,
            pack_f64(packed.ref_scalars),
            len(packed.ref_scalars),
            "reference_scalars",
        ),
        Section(
            SECTION_ALPHAIW, DTYPE_F64, pack_f64(packed.alphaiw), len(packed.alphaiw), "alphaiw"
        ),
        Section(
            SECTION_SECONDARY,
            DTYPE_F64,
            pack_f64(packed.secondary),
            len(packed.secondary),
            "secondary_systems",
        ),
        Section(
            SECTION_ELEMENT, DTYPE_F64, pack_f64(packed.element), len(packed.element), "elements"
        ),
        Section(SECTION_FREQ, DTYPE_F64, pack_f64(packed.freq), len(packed.freq), "frequencies"),
    ]
    prefix = struct.pack(
        "<8sIIIIIIIIQ",
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        BYTE_ORDER_MARK,
        FLAG_RCOV_SCALED | FLAG_ATOMIC_UNITS | FLAG_RAW_POLARIZABILITIES | FLAG_EEQ2019,
        Z_MAX,
        MAX_REF,
        n_refs,
        N_FREQ,
        N_SECONDARY,
    )
    return build_container(prefix, source_id, sections)


@dataclass(frozen=True)
class Derived:
    """Quantities derived from the raw tables at the default ga, gc."""

    refalpha_sha256: str
    refc6_sha256: str
    n_c6: int
    zero_alpha: list[dict[str, int]]
    tied_maxima: list[int]


def derive(packed: Packed, constants: Constants) -> Derived:
    """Derive reference polarizabilities and C6 at the default ga, gc and hash them."""
    alphas = reference_alpha(packed, GA_DEFAULT, GC_DEFAULT)
    c6 = reference_c6(packed, alphas, constants.pi)
    flat = [value for row in alphas for value in row]
    return Derived(
        refalpha_sha256=hashlib.sha256(pack_f64(flat)).hexdigest(),
        refc6_sha256=hashlib.sha256(pack_f64(c6)).hexdigest(),
        n_c6=len(c6),
        zero_alpha=zero_alpha_references(packed, alphas),
        tied_maxima=tied_maxima(packed),
    )


def build_manifest(
    pins: dict[str, SourcePin],
    constants: Constants,
    packed: Packed,
    derived: Derived,
    single_precision: list[str],
    *,
    table_hashes: dict[str, str],
    outputs: dict[str, bytes],
    conventions: dict[str, str],
    methods_doc: dict[str, Any],
) -> dict[str, Any]:
    """Build the provenance manifest."""
    return {
        "schema": "vanderfast.reference-manifest/1",
        "generator": {"script": "tools/generate_d4_data.py", "version": GENERATOR_VERSION},
        "format": {
            "name": "d4_reference.bin",
            "version": FORMAT_VERSION,
            "header_bytes": HEADER_BYTES,
            "section_alignment": SECTION_ALIGN,
            "layout": (
                "references packed for Z = 1..103 in Z order, each element's references in "
                "upstream order; per-reference rows are reference_scalars columns and 23 "
                "alphaiw values; secondary rows (index 0 a NaN sentinel) are sscale then 23 "
                "secaiw values; element rows indexed by Z with row 0 a NaN sentinel; "
                "frequencies then trapezoid weights"
            ),
            "reference_scalar_columns": list(REF_SCALAR_COLUMNS),
            "element_columns": list(ELEMENT_COLUMNS),
        },
        "sources": manifest_sources(pins),
        "outputs": manifest_outputs(outputs),
        "tables": {name: {"sha256": digest} for name, digest in sorted(table_hashes.items())}
        | {"d4_parameters": {"sha256": hashlib.sha256(dumps(methods_doc)).hexdigest()}},
        "supported_atomic_numbers": {"min": 1, "max": Z_MAX},
        "counts": {
            "n_refs_total": len(packed.refsys),
            "n_frequencies": N_FREQ,
            "n_secondary": N_SECONDARY,
            "bj_methods": len(methods_doc["tables"]["bj"]),
            "bjatm_methods": len(methods_doc["tables"]["bjatm"]),
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
                "source": "ase.units (CODATA 2014), identical to dftd4.ase",
                "recipes": {
                    "positions": "x / Bohr",
                    "cutoffs": "x / Bohr",
                    "energy": "e * Hartree",
                    "forces": "-g * Hartree / Bohr",
                    "stress": "virial * Hartree / V -> flat[[0, 4, 8, 5, 2, 1]]",
                },
            },
        },
        "derived": {
            "ga": GA_DEFAULT,
            "gc": GC_DEFAULT,
            "algorithm": (
                "refalpha = max(ascale*(alphaiw - hcount*((sscale*secaiw)*zeta(ga, "
                "hardness[refsys]*gc, zeff[refsys], clsh + zeff[refsys]))), 0), zero for "
                "refsys = 0; c6ref = (3/pi) * sequential sum_k (alpha_a[k]*alpha_b[k])*w[k]"
            ),
            "refalpha_sha256": derived.refalpha_sha256,
            "refc6_sha256": derived.refc6_sha256,
            "refc6_layout": "pairs Zi >= Zj in pair order, blocks row-major nref(Zi) x nref(Zj)",
            "n_c6": derived.n_c6,
        },
        "conventions": {
            "charge_model": "EEQ 2019 (multicharge v0.5.0), Z = 1..103",
            "partial_pbc": (
                "any periodic flag makes the system fully 3D periodic over the supplied cell, "
                "for dispersion and EEQ (dftd4 ignores per-direction flags)"
            ),
            "cn_d4": {
                "count": "en_factor * 0.5*(1 + erf(-7.5*(r - rc)/rc)), rc = rcov_i + rcov_j",
                "en_factor": "4.10451*exp(-(|en_i - en_j| + 19.08857)^2 / (2*11.28174^2))",
                "exclude_r2_below": 1e-12,
                "default_cutoff_bohr": 30.0,
            },
            "cn_eeq": {
                "count": "0.5*(1 + erf(-7.5*(r - rc)/rc)), rc from mctc covalent_rad_d3",
                "cap": "log(1 + e^8) - log(1 + e^(8 - cn)), derivative e^8/(e^8 + e^cn)",
                "cutoff_bohr": 25.0,
            },
            "eeq": {
                "rhs": "-chi + kcnchi/sqrt(cn + 1e-14)*cn; derivative 0.5*kcnchi/sqrt(cn + 1e-14)",
                "molecular": (
                    "A_ij = erf(sqrt(r^2/(rad_i^2 + rad_j^2)))/r, A_ii = eta + sqrt(2/pi)/rad"
                ),
                "constraint": "bordered system with a row and column of ones, total charge",
            },
            "ewald_upstream": {
                "images": (
                    "Wigner-Seitz: 27 lattice points, minimum r^2 plus ties within 0.01 bohr^2, "
                    "weight 1/nimg; self pairs index the compacted distance list (off by one)"
                ),
                "real": (
                    "125 translations (rep 2): erf(gam r)/r - erf(alpha r)/r, skip r < sqrt(eps)"
                ),
                "reciprocal": "124 vectors: 4 pi/V cos(G.r) exp(-G^2/(4 alpha^2))/G^2",
                "diagonal": "eta + sqrt(2/pi)/rad - 2 alpha/sqrt(pi) plus self images",
                "alpha": "bisection balancing truncation terms, fallback 0.25",
                "wrap": "positions wrapped to the central cell (eps 1e-14) before evaluation",
            },
            "weights": {
                "wf": WF_DEFAULT,
                "gaussians": "sum_{k=1..ngw} exp(-(k*wf)*(cn - cnref)^2)",
                "norm_guard": "norm <= sqrt(tiny) -> fallback",
                "fallback": "weight 1 for every reference with |max cnref - cnref| < 1e-12, else 0",
                "tied_maxima": derived.tied_maxima,
            },
            "zeta": {
                "ga": GA_DEFAULT,
                "gc": GC_DEFAULT,
                "value": "exp(ga*(1 - exp(hardness*gc*(1 - qref/qmod)))), exp(ga) if qmod < 0",
                "qref": "clsq + zeff",
                "qmod": "q + zeff",
            },
            "pair": {
                "kappa": "3*r4r2_i*r4r2_j",
                "r0": "a1*sqrt(kappa) + a2",
                "exclude_r2_below": "DBL_EPSILON",
                "default_cutoff_bohr": 60.0,
            },
            "atm": {
                "c6": "evaluated with every charge set to zero; its charge derivative is unused",
                "r0": "product of a1*sqrt(3*r4r2_i*r4r2_j) + a2 over the three pairs",
                "exponent": "alp/3 (alp = 16)",
                "c9": "-s9*sqrt(|c6ij*c6ik*c6jk|)",
                "default_cutoff_bohr": 40.0,
            },
            "source_lines": dict(sorted(conventions.items())),
        },
        "anomalies": {
            "uninitialised_references": [{"Z": z, "reference": r} for z, r in EXPECTED_PHANTOMS],
            "uninitialised_secondary_systems": EXPECTED_UNINITIALISED_SECONDARY,
            "zero_polarizability_references": derived.zero_alpha,
            "single_precision_literals": {
                "note": (
                    "kind-less literals are default (binary32) reals; gfortran stores them "
                    "widened to double, and so do these tables"
                ),
                "targets": single_precision,
            },
            "unsupported_with_references": list(range(112, UPSTREAM_MAX_ELEM + 1)),
        },
        "notices": "THIRD_PARTY_NOTICES.md",
    }


# --------------------------------------------------------------------------
# Binary verification against compiled upstream libraries
# --------------------------------------------------------------------------


def _fortran_ref_array(raw: RawReference, name: str) -> list[float]:
    """Column-major (7, 118) array with gfortran's zero fill."""
    return [
        raw.per_ref[name][z][r] or 0.0 for z in range(UPSTREAM_MAX_ELEM) for r in range(MAX_REF)
    ]


def verify_binary(
    libdir: Path, raw: RawReference, elements_all: dict[str, list[float]], constants: Constants
) -> list[str]:
    """Search compiled upstream libraries for the parsed arrays; return failures."""

    def library(pattern: str) -> bytes:
        return b"".join(p.read_bytes() for p in sorted(libdir.glob(pattern)) if p.is_file())

    dftd4 = library("libdftd4.so*")
    multicharge = library("libmulticharge.so*")
    mctc = library("libmctc-lib.so*")
    if not dftd4 or not multicharge or not mctc:
        return [f"libdftd4/libmulticharge/libmctc-lib not found in {libdir}"]
    refn = [value or 0 for value in raw.refn]
    refsys = [raw.refsys[z][r] or 0 for z in range(UPSTREAM_MAX_ELEM) for r in range(MAX_REF)]
    checks: list[tuple[str, bytes, bytes]] = [
        ("refn", struct.pack(f"<{len(refn)}i", *refn), dftd4),
        ("refsys", struct.pack(f"<{len(refsys)}i", *refsys), dftd4),
        *((name, pack_f64(_fortran_ref_array(raw, name)), dftd4) for name in REF_SCALAR_COLUMNS),
        ("sscale", pack_f64([v or 0.0 for v in raw.secondary["sscale"]]), dftd4),
        ("secaiw", pack_f64([v for s in raw.secaiw for v in (s or [0.0] * N_FREQ)]), dftd4),
        ("covalent_rad_d3 (dftd4)", pack_f64(elements_all["rcov_d4"]), dftd4),
        ("pauling_en", pack_f64(elements_all["en"]), dftd4),
        ("chemical_hardness", pack_f64(elements_all["hardness"]), dftd4),
        ("effective_nuclear_charge", pack_f64(elements_all["zeff"]), dftd4),
        ("sqrt_z_r4_over_r2", pack_f64(elements_all["r4r2"]), dftd4),
        ("eeq_chi", pack_f64(elements_all["eeq_chi"]), multicharge),
        ("eeq_eta", pack_f64(elements_all["eeq_eta"]), multicharge),
        ("eeq_kcnchi", pack_f64(elements_all["eeq_kcnchi"]), multicharge),
        ("eeq_rad", pack_f64(elements_all["eeq_rad"]), multicharge),
        ("covalent_rad_d3 (mctc)", pack_f64(elements_all["eeq_rcov"]), mctc),
        ("aatoau", pack_f64([constants.aatoau]), mctc),
    ]
    alphaiw = [
        value
        for z in range(UPSTREAM_MAX_ELEM)
        for r in range(MAX_REF)
        for value in (raw.alphaiw[z][r] or [0.0] * N_FREQ)
    ]
    chunk = 2000
    for start in range(0, len(alphaiw), chunk):
        values = alphaiw[start : start + chunk]
        checks.append((f"alphaiw[{start + 1}:{start + len(values)}]", pack_f64(values), dftd4))
    return [name for name, needle, haystack in checks if needle not in haystack]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def check_wraps(reader: SourceReader, pins: dict[str, SourcePin]) -> None:
    """Require the upstream wraps to name the pinned multicharge and mctc-lib releases."""
    upstream_mctc = pins["mctc-lib"].meta["upstream_wrap"]["tag"]
    for source in ("dftd4", "multicharge"):
        wrap = reader.text(source, "subprojects/mctc-lib.wrap")
        if f"revision = {upstream_mctc}" not in wrap:
            msg = f"{source} mctc-lib.wrap does not name {upstream_mctc}"
            raise GenerationError(msg)
    wrap = reader.text("dftd4", "subprojects/multicharge.wrap")
    if f"revision = {pins['multicharge'].meta['tag']}" not in wrap:
        msg = "dftd4 multicharge.wrap does not name the pinned multicharge release"
        raise GenerationError(msg)


def generate(reader: SourceReader, pins: dict[str, SourcePin]) -> Generated:
    """Run the full generation and return output bytes keyed by file name."""
    constants = parse_constants(reader)
    check_wraps(reader, pins)
    check_licence_copies(reader, pins)
    raw = parse_reference_inc(reader)
    validate_reference(raw)
    elements = parse_elements(reader, constants)
    if elements.rcov_d4[:Z_MAX] != elements.eeq_rcov[:Z_MAX]:
        msg = "dftd4 and mctc covalent radii differ; the fused CN kernels assume they agree"
        raise GenerationError(msg)
    freq, weights = parse_frequencies(reader)
    conventions = check_convention_lines(reader, CONVENTION_LINES)
    methods = parse_methods(reader)

    packed = pack_tables(raw, elements, freq, weights)
    derived = derive(packed, constants)
    pin = pins["dftd4"]
    source_id = (
        f"dftd4 {pin.meta['tag']} {pin.commit[:12]} + multicharge {pins['multicharge'].meta['tag']}"
    )
    binary, table_hashes = build_binary(packed, source_id)
    methods_doc = build_methods(methods, pin)
    outputs = {"d4_reference.bin": binary, "d4_methods.json": dumps(methods_doc)}
    manifest = build_manifest(
        pins,
        constants,
        packed,
        derived,
        sorted(raw.single_precision),
        table_hashes=table_hashes,
        outputs=outputs,
        conventions=conventions,
        methods_doc=methods_doc,
    )
    outputs["d4_manifest.json"] = dumps(manifest)
    elements_all = {name: elements.column(name) for name in ELEMENT_COLUMNS}
    return Generated(outputs, lambda libdir: verify_binary(libdir, raw, elements_all, constants))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    return run_cli(
        argv,
        description=__doc__.splitlines()[0],
        script="tools/generate_d4_data.py",
        sources_toml=SOURCES_TOML,
        generate=generate,
        verify_help="search libdftd4/libmulticharge/libmctc-lib in LIBDIR for the parsed arrays",
    )


if __name__ == "__main__":
    raise SystemExit(main())
