"""Generate reference fixtures with the pinned dftd4 4.2.0 library.

Each fixture family is written to ``tests/fixtures/dftd4-4.2.0/<family>.jsonl``:
the first line is a header, every following line one case. A case stores the
exact input (atomic units, positions as passed to dftd4, which wraps periodic
positions itself), the total charge, the fully resolved parameters and cutoffs,
and the reference energy, gradient, virial, EEQ charges, coordination
numbers, static polarizabilities, and (for small systems) C6 coefficients.
``--check`` regenerates every case and compares numerically.

Usage::

    python tools/generate_d4_fixtures.py [--check] [--family NAME ...]
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from vanderfast import _oracle_d4
from vanderfast.data.d4 import load_d4_reference_tables
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters, resolve_d4_parameters
from vanderfast.reference_d4 import D4ReferenceEvaluator, reference_version
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "dftd4-4.2.0"
GENERATOR_VERSION = 1
CHECK_RTOL = 1e-12
CHECK_ATOL = 1e-14
MAX_C6_ATOMS = 24

_PBE = resolve_d4_parameters(method="pbe")
PARAMETER_SETS: dict[str, D4Parameters] = {
    "Q1": resolve_d4_parameters(method="pbe", atm=False),
    "Q2": _PBE,
    "Q3": resolve_d4_parameters(method="b3lyp", atm=True),
    "Q4": D4Parameters(s6=0.95, s8=1.3, a1=0.38, a2=4.9, s9=1.0, source="custom"),
    "Q5": D4Parameters(s6=1.0, s8=0.0, a1=0.0, a2=0.0, s9=0.0, source="c6 probe"),
    "Q6": D4Parameters(s6=0.0, s8=0.0, a1=0.4, a2=5.0, s9=1.0, source="atm only"),
    "Q7": D4Parameters(s8=0.0, a1=0.42, a2=5.65, s9=2.0, ga=2.0, gc=1.0, source="r2scan-3c-like"),
    "Q8": resolve_d4_parameters(method="dftb3", atm=False),
    "Q9": D4Parameters(
        s8=_PBE.s8, a1=_PBE.a1, a2=_PBE.a2, s9=1.0, ga=6.0, gc=4.0, wf=10.0, source="model"
    ),
}

SHORT = D4Cutoffs(cn=20.0, disp2=25.0, disp3=15.0)


def _case(
    name: str,
    atoms: Atoms | None,
    sets: Sequence[str],
    *,
    charge: float = 0.0,
    cutoffs: D4Cutoffs | None = None,
    gradient: bool = True,
    bohr: tuple[np.ndarray, np.ndarray] | None = None,
    pbc_override: Sequence[bool] | None = None,
    known_issue: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield fixture inputs for ``atoms`` (Å) or explicit ``(numbers, positions_bohr)``."""
    if bohr is not None:
        numbers, positions = bohr
        cell = np.zeros((3, 3))
        pbc = np.zeros(3, dtype=bool)
    elif atoms is None:
        msg = "either atoms or explicit bohr coordinates are required"
        raise ValueError(msg)
    else:
        numbers = atoms.numbers.astype(np.int64)
        positions = atoms.positions * INV_BOHR
        pbc = np.asarray(atoms.pbc if pbc_override is None else pbc_override, dtype=bool)
        cell = atoms.cell[:] * INV_BOHR if pbc.any() else np.zeros((3, 3))
    for key in sets:
        params = PARAMETER_SETS[key]
        yield {
            "name": name,
            "parameter_set": key,
            "numbers": [int(z) for z in numbers],
            "positions_bohr": positions.tolist(),
            "cell_bohr": cell.tolist() if pbc.any() else None,
            "pbc": [bool(p) for p in pbc],
            "charge": float(charge),
            "parameters": {k: v for k, v in params.as_dict().items() if k != "source"},
            "cutoffs_bohr": (cutoffs or D4Cutoffs()).as_bohr(),
            "gradient_requested": gradient,
            "known_issue": known_issue,
        }


def _tuned_shell(centre: int, shell: int, count: int, target_cn: float) -> Atoms:
    """Return a shell cluster whose centre D4 CN equals ``target_cn`` (bisection)."""
    tables = load_d4_reference_tables()
    rc = tables.element_column("rcov_d4")[[centre]] + tables.element_column("rcov_d4")[[shell]]
    en = tables.element_column("en")
    den = float(_oracle_d4.en_factor(en[[centre]], en[[shell]])[0])
    lo, hi = 0.3, 12.0  # bohr
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cn = count * den * float(_oracle_d4.erf_count(rc, np.array([mid]))[0])
        if cn > target_cn:
            lo = mid
        else:
            hi = mid
    return structures.shell_cluster(centre, shell, count, 0.5 * (lo + hi) / INV_BOHR)


def family_molecules() -> Iterator[dict[str, Any]]:
    """Small molecules with every functional-style parameter set."""
    for name, build in structures.MOLECULES.items():
        yield from _case(name, build(), ["Q1", "Q2", "Q3", "Q4", "Q7", "Q8", "Q9"])
        yield from _case(name, build(), ["Q5"])


def family_ions() -> Iterator[dict[str, Any]]:
    """Charged molecules (total charge passed to the EEQ model)."""
    for name, (build, charge) in structures.IONS.items():
        yield from _case(name, build(), ["Q2", "Q4"], charge=charge)
    yield from _case("water_dimer_anion", structures.water_dimer(), ["Q2"], charge=-1.0)


def family_coverage() -> Iterator[dict[str, Any]]:
    """Clusters covering Z = 1..103, neutral and charged."""
    for index, atoms in enumerate(structures.coverage_clusters()):
        yield from _case(f"coverage_{index}", atoms, ["Q2", "Q5"])
        yield from _case(f"coverage_{index}_cation", atoms, ["Q2"], charge=1.0)
        yield from _case(f"coverage_{index}_anion", atoms, ["Q2"], charge=-1.0)


def family_actinides() -> Iterator[dict[str, Any]]:
    """Elements whose reference lists contain uninitialised (phantom) references."""
    for name, atoms in (
        ("uranyl", structures.uranyl()),
        ("americium_trichloride", structures.americium_trichloride()),
        ("cf_isolated", structures.dimer(98, 1, 12.0)),
        ("u_isolated", structures.dimer(92, 1, 12.0)),
        ("am_dimer", structures.dimer(95, 95, 3.2)),
    ):
        yield from _case(name, atoms, ["Q2", "Q5"])


def family_dimers() -> Iterator[dict[str, Any]]:
    """Dimer distance scans across the periodic table and exact cutoff edges."""
    pairs = [(1, 1), (6, 6), (8, 1), (18, 18), (26, 8), (79, 79), (53, 53), (58, 1), (103, 103)]
    distances = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 31.7]
    for zi, zj in pairs:
        for r in distances:
            yield from _case(f"dimer_{zi}_{zj}_{r:g}", structures.dimer(zi, zj, r), ["Q1"])
        yield from _case(f"dimer_{zi}_{zj}_c6", structures.dimer(zi, zj, 3.0), ["Q5"])
    edges = {"disp2": 60.0, "cn": 30.0, "eeq_cn": 25.0}
    for label, rc in edges.items():
        for tag, factor in (("below", 1.0 - 1e-9), ("exact", 1.0), ("above", 1.0 + 1e-9)):
            numbers = np.array([11, 17], dtype=np.int64)
            positions = np.array([[0.0, 0.0, 0.0], [rc * factor, 0.0, 0.0]])
            yield from _case(f"edge_{label}_{tag}", None, ["Q1"], bohr=(numbers, positions))


def family_trimers() -> Iterator[dict[str, Any]]:
    """Three-body geometries."""
    cases: list[tuple[str, Atoms]] = [
        ("ar3_equilateral_3.5", structures.equilateral_trimer(18, 3.5)),
        ("ar3_linear", structures.linear_trimer(18, 3.8)),
        ("cno_scalene", structures.scalene_trimer()),
    ]
    cases += [
        (f"ar3_apex_{angle}", structures.isosceles_trimer(18, 3.8, angle))
        for angle in (30, 90, 150, 179)
    ]
    for name, atoms in cases:
        yield from _case(name, atoms, ["Q2", "Q6"])
    numbers = np.array([18, 18, 18], dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [39.0, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    yield from _case("edge_atm_outside", None, ["Q2", "Q6"], bohr=(numbers, positions))


def family_periodic() -> Iterator[dict[str, Any]]:
    """Fully periodic cells, including ionic crystals and unwrapped input."""
    for name, build in structures.PERIODIC.items():
        yield from _case(name, build(), ["Q1"])
        yield from _case(name, build(), ["Q2"], cutoffs=SHORT)
    for name, build in structures.IONIC.items():
        yield from _case(name, build(), ["Q1"])
        yield from _case(name, build(), ["Q2"], cutoffs=SHORT)
    yield from _case("nacl_primitive_default_atm", structures.rocksalt_primitive(), ["Q2"])
    yield from _case("skewed_triclinic_cation", structures.skewed_triclinic(), ["Q1"], charge=1.0)
    shifted = structures.skewed_triclinic()
    shifted.positions += np.array([[2, -1, 0], [0, 3, -2], [-1, 0, 1], [0, 0, 0]]) @ shifted.cell[:]
    yield from _case("skewed_triclinic_unwrapped", shifted, ["Q1"])
    yield from _case("c_tiny_cubic", structures.simple_cubic(6, 1.8), ["Q1"])
    custom = D4Cutoffs(cn=25.0, disp2=35.0, disp3=25.0)
    yield from _case("co2_crystal_custom_cutoffs", structures.co2_crystal(), ["Q2"], cutoffs=custom)


def family_cell_dependence() -> Iterator[dict[str, Any]]:
    """One crystal described by different cells (dftd4's EEQ is cell-dependent)."""
    for name, base in (
        ("mgo", structures.magnesia_primitive()),
        ("nacl", structures.rocksalt_primitive()),
    ):
        for reps in ((1, 1, 1), (2, 2, 2)):
            label = f"{name}_replica_{'x'.join(map(str, reps))}"
            yield from _case(label, structures.replicate(base, reps), ["Q1"], cutoffs=SHORT)
    yield from _case("mgo_conventional", structures.magnesia_conventional(), ["Q1"], cutoffs=SHORT)


def family_partial_pbc() -> Iterator[dict[str, Any]]:
    """Slabs and wires; dftd4 treats any periodic flag as fully 3D periodic."""
    for name, build in structures.PARTIAL.items():
        yield from _case(name, build(), ["Q1", "Q2"])
        yield from _case(f"{name}_as_3d", build(), ["Q1"], pbc_override=(True, True, True))


def family_extreme_cn() -> Iterator[dict[str, Any]]:
    """Coordination numbers that exercise the weight fallback and the EEQ CN cap."""
    for target in (10.0, 14.0, 16.5):
        yield from _case(f"h_in_c18_cn{target:g}", _tuned_shell(1, 6, 18, target), ["Q2"])
    yield from _case("c_in_c24_cap", _tuned_shell(6, 6, 24, 11.0), ["Q2"])
    yield from _case("isolated_pair", structures.dimer(18, 36, 25.0), ["Q1"])


ZETA_GRADIENT_ISSUE = "dftd4 4.2.0 evaluates dzeta as 0/0 at q + zeff == 0; no gradient"


def family_zeta() -> Iterator[dict[str, Any]]:
    """Effective charges at and below zero (zeta = exp(ga), dzeta = 0)."""
    hydrogen = structures.simple_cubic(1, 3.0)
    yield from _case("periodic_h_dianion", hydrogen, ["Q2"], charge=-2.0, cutoffs=SHORT)
    yield from _case(
        "periodic_hydride_exact_zero",
        hydrogen,
        ["Q2"],
        charge=-1.0,
        cutoffs=SHORT,
        gradient=False,
        known_issue=ZETA_GRADIENT_ISSUE,
    )


def family_eeq_stress() -> Iterator[dict[str, Any]]:
    """Dense configurations with an ill-conditioned or indefinite EEQ matrix."""
    yield from _case(
        "hydrogen_cluster_indefinite", structures.hydrogen_cluster_indefinite(), ["Q1"]
    )
    yield from _case("hydrogen_solid", structures.hydrogen_solid(), ["Q1"], cutoffs=SHORT)
    compressed = structures.lithium_hydride()
    compressed.set_cell(compressed.cell[:] * 0.7, scale_atoms=True)
    yield from _case("lih_compressed", compressed, ["Q1"], cutoffs=SHORT)


FAMILIES: dict[str, Callable[[], Iterator[dict[str, Any]]]] = {
    "molecules": family_molecules,
    "ions": family_ions,
    "coverage": family_coverage,
    "actinides": family_actinides,
    "dimers": family_dimers,
    "trimers": family_trimers,
    "periodic": family_periodic,
    "cell_dependence": family_cell_dependence,
    "partial_pbc": family_partial_pbc,
    "extreme_cn": family_extreme_cn,
    "zeta": family_zeta,
    "eeq_stress": family_eeq_stress,
}


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Attach dftd4 reference results to a fixture input."""
    params = D4Parameters(**case["parameters"])
    cutoffs = D4Cutoffs(**case["cutoffs_bohr"])
    evaluator = D4ReferenceEvaluator(case["numbers"], params, cutoffs, charge=case["charge"])
    result = evaluator.evaluate(
        np.array(case["positions_bohr"]),
        None if case["cell_bohr"] is None else np.array(case["cell_bohr"]),
        case["pbc"],
        gradient=case["gradient_requested"],
        properties=True,
    )
    out = dict(case)
    out["energy"] = result.energy
    out["gradient"] = None if result.gradient is None else result.gradient.tolist()
    out["virial"] = None if result.virial is None else result.virial.tolist()
    if result.charges is None or result.cn is None or result.polarizabilities is None:
        msg = "the reference did not return properties"
        raise RuntimeError(msg)
    out["charges"] = result.charges.tolist()
    out["cn"] = result.cn.tolist()
    out["polarizabilities"] = result.polarizabilities.tolist()
    small = len(case["numbers"]) <= MAX_C6_ATOMS
    out["c6"] = result.c6.tolist() if small and result.c6 is not None else None
    return out


def header() -> dict[str, Any]:
    """Return the fixture file header."""
    import dftd4.library  # noqa: PLC0415  (imported after the thread counts are set)

    return {
        "schema": "vanderfast.fixtures-d4/1",
        "generator_version": GENERATOR_VERSION,
        "reference": {
            "package": "dftd4",
            "version": reference_version(),
            "api_version": str(dftd4.library.get_api_version()),
            "omp_num_threads": 1,
            "openblas_num_threads": 1,
        },
        "units": "bohr, hartree, e",
    }


def encode(document: dict[str, Any]) -> str:
    """Serialise one JSON line deterministically."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=CHECK_RTOL, abs_tol=CHECK_ATOL)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return bool(a == b)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES))
    parser.add_argument("--output-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)

    if reference_version() != "4.2.0":
        print(f"dftd4 {reference_version()} is installed; fixtures need 4.2.0", file=sys.stderr)
        return 1
    status = 0
    head = header()
    for family in args.family or sorted(FAMILIES):
        cases = [evaluate_case(case) for case in FAMILIES[family]()]
        names = [(c["name"], c["parameter_set"]) for c in cases]
        if len(set(names)) != len(names):
            print(f"{family}: duplicate case names", file=sys.stderr)
            return 1
        path = args.output_dir / f"{family}.jsonl"
        if args.check:
            lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
            stored = [json.loads(line) for line in lines]
            if not stored or not _close(stored[0], head) or len(stored) - 1 != len(cases):
                print(f"{path}: header or case count differs", file=sys.stderr)
                status = 1
                continue
            for old, new in zip(stored[1:], cases, strict=True):
                if not _close(old, new):
                    print(
                        f"{path}: case {new['name']}/{new['parameter_set']} differs",
                        file=sys.stderr,
                    )
                    status = 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join([encode(head), *(encode(c) for c in cases)]) + "\n"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({len(cases)} cases, {len(text.encode()) // 1024} KiB)")
    if args.check and status == 0:
        print("fixtures reproduce")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
