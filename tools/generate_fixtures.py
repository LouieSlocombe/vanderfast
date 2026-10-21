"""Generate reference fixtures with the pinned dftd3 1.6.0 library.

Each fixture family is written to ``tests/fixtures/dftd3-1.6.0/<family>.jsonl``:
the first line is a header, every following line one case. A case stores the
exact input (atomic units, wrapped where upstream requires it), the fully
resolved parameters and cutoffs, and the reference energy, gradient, and
virial. ``--check`` regenerates every case and compares numerically.

Usage::

    OMP_NUM_THREADS=1 python tools/generate_fixtures.py [--check] [--family NAME ...]
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from vanderfast import _oracle
from vanderfast.data import load_reference_tables
from vanderfast.parameters import Cutoffs, D3BJParameters, resolve_parameters
from vanderfast.reference import ReferenceEvaluator, reference_version
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "dftd3-1.6.0"
GENERATOR_VERSION = 1
CHECK_RTOL = 1e-13
CHECK_ATOL = 1e-15

PARAMETER_SETS: dict[str, D3BJParameters] = {
    "P1": resolve_parameters(method="pbe", damping="bj", atm=False),
    "P2": resolve_parameters(method="pbe", damping="bj", atm=True),
    "P3": resolve_parameters(method="b3lyp", damping="bj", atm=True),
    "P4": D3BJParameters(s6=0.95, s8=1.3, a1=0.35, a2=4.6, s9=1.0, alp=16.0, source="custom"),
    "P5": D3BJParameters(s6=1.0, s8=0.0, a1=0.0, a2=0.0, s9=0.0, alp=14.0, source="c6 probe"),
    "P6": D3BJParameters(s6=0.0, s8=0.0, a1=0.4289, a2=4.4407, s9=1.0, alp=14.0, source="atm only"),
}


def _wrap_active(
    numbers: np.ndarray, pos: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Wrap positions into [0, 1) along active directions of the search basis."""
    geometry = _oracle.make_geometry(numbers, pos, cell, pbc)
    basis = _oracle.search_basis(geometry)
    frac = pos @ np.linalg.inv(basis)
    active = np.flatnonzero(pbc)
    frac[:, active] -= np.floor(frac[:, active])
    return frac @ basis


def _case(
    name: str,
    atoms: Atoms | None,
    sets: Sequence[str],
    *,
    cutoffs: Cutoffs | None = None,
    pairwise: bool = False,
    bohr: tuple[np.ndarray, np.ndarray] | None = None,
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
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = atoms.cell[:] * INV_BOHR if pbc.any() else np.zeros((3, 3))
        if pbc.any():
            positions = _wrap_active(numbers, positions, cell, pbc)
    for key in sets:
        yield {
            "name": name,
            "parameter_set": key,
            "numbers": [int(z) for z in numbers],
            "positions_bohr": positions.tolist(),
            "cell_bohr": cell.tolist() if pbc.any() else None,
            "pbc": [bool(p) for p in pbc],
            "parameters": {k: v for k, v in PARAMETER_SETS[key].as_dict().items() if k != "source"},
            "cutoffs_bohr": (cutoffs or Cutoffs()).as_bohr(),
            "pairwise": bool(pairwise and not pbc.any()),
            "known_issue": known_issue,
        }


def _tuned_shell(centre: int, shell: int, count: int, target_cn: float) -> Atoms:
    """Return a shell cluster whose centre CN equals ``target_cn`` (bisection)."""
    tables = load_reference_tables()
    rc = tables.rcov[centre] + tables.rcov[shell]
    lo, hi = 0.3, 8.0  # bohr
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cn = count * float(_oracle.cn_count(np.array([rc]), np.array([mid]))[0])
        if cn > target_cn:
            lo = mid
        else:
            hi = mid
    return structures.shell_cluster(centre, shell, count, 0.5 * (lo + hi) / INV_BOHR)


def family_molecules() -> Iterator[dict[str, Any]]:
    """Small molecules with every parameter set."""
    for name, build in structures.MOLECULES.items():
        yield from _case(name, build(), ["P1", "P2", "P3", "P4"])
        yield from _case(name, build(), ["P5"], pairwise=True)


def family_coverage() -> Iterator[dict[str, Any]]:
    """Clusters covering Z = 1..103."""
    for index, atoms in enumerate(structures.coverage_clusters()):
        yield from _case(f"coverage_{index}", atoms, ["P2"])
        yield from _case(f"coverage_{index}", atoms, ["P5"], pairwise=True)


def family_extreme_cn() -> Iterator[dict[str, Any]]:
    """Coordination numbers that exercise the exceptional weight branches."""
    for target in (10.0, 14.0, 16.5):
        atoms = _tuned_shell(1, 6, 18, target)
        yield from _case(f"h_in_c18_cn{target:g}", atoms, ["P2"])
    yield from _case("cr_near_reference", _tuned_shell(24, 8, 12, 10.6191), ["P2"])
    yield from _case("isolated_pair", structures.dimer(18, 36, 25.0), ["P1"])


def family_dimers() -> Iterator[dict[str, Any]]:
    """Dimer distance scans across the periodic table and exact cutoff edges."""
    pairs = [(1, 1), (6, 6), (8, 1), (18, 18), (26, 8), (79, 79), (53, 53), (58, 1), (96, 1)]
    pairs += [(103, 103), (1, 103)]
    distances = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 31.7]
    for zi, zj in pairs:
        for r in distances:
            yield from _case(f"dimer_{zi}_{zj}_{r:g}", structures.dimer(zi, zj, r), ["P1"])
        yield from _case(
            f"dimer_{zi}_{zj}_c6", structures.dimer(zi, zj, 3.0), ["P5"], pairwise=True
        )
    edges = {"disp2": 60.0, "cn": 40.0}
    for label, rc in edges.items():
        for tag, factor in (("below", 1.0 - 1e-9), ("exact", 1.0), ("above", 1.0 + 1e-9)):
            numbers = np.array([18, 18], dtype=np.int64)
            positions = np.array([[0.0, 0.0, 0.0], [rc * factor, 0.0, 0.0]])
            yield from _case(f"edge_{label}_{tag}", None, ["P1"], bohr=(numbers, positions))


def family_trimers() -> Iterator[dict[str, Any]]:
    """Three-body geometries."""
    cases: list[tuple[str, Atoms]] = [
        ("ar3_equilateral_3.5", structures.equilateral_trimer(18, 3.5)),
        ("ar3_equilateral_4.0", structures.equilateral_trimer(18, 4.0)),
        ("ar3_linear", structures.linear_trimer(18, 3.8)),
        ("cno_scalene", structures.scalene_trimer()),
    ]
    cases += [
        (f"ar3_apex_{angle}", structures.isosceles_trimer(18, 3.8, angle))
        for angle in (30, 90, 120, 150, 179)
    ]
    for name, atoms in cases:
        yield from _case(name, atoms, ["P2", "P6"])
    numbers = np.array([18, 18, 18], dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [39.0, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    yield from _case("edge_atm_outside", None, ["P2", "P6"], bohr=(numbers, positions))


def family_periodic() -> Iterator[dict[str, Any]]:
    """Fully periodic cells."""
    for name, build in structures.PERIODIC.items():
        yield from _case(name, build(), ["P1", "P2"])
    yield from _case("nacl_primitive", structures.rocksalt_primitive(), ["P6"])
    yield from _case("c_tiny_cubic", structures.simple_cubic(6, 1.8), ["P1"])
    base = structures.rocksalt_primitive()
    for reps in ((1, 1, 1), (2, 1, 1), (2, 2, 2)):
        name = "nacl_replica_" + "x".join(map(str, reps))
        yield from _case(name, structures.replicate(base, reps), ["P2"])
    custom = Cutoffs(cn=25.0, disp2=35.0, disp3=20.0)
    yield from _case("co2_crystal_custom_cutoffs", structures.co2_crystal(), ["P2"], cutoffs=custom)


# simple-dftd3 v1.6.0 get_lattice_points (2D branch) masks Cartesian rows with
# spread(periodic, 2, 3) instead of lattice columns, so its image range is
# wrong when an inactive lattice vector has in-plane components. vanderfast
# enumerates the complete image set; such cases are compared by invariance.
UPSTREAM_PARTIAL_PBC_ISSUE = "simple-dftd3 v1.6.0 2D image range ignores tilted inactive vectors"


def family_partial_pbc() -> Iterator[dict[str, Any]]:
    """Slabs and wires."""
    for name, build in structures.PARTIAL.items():
        issue = UPSTREAM_PARTIAL_PBC_ISSUE if name.endswith("tilted_vacuum") else None
        yield from _case(name, build(), ["P1", "P2"], known_issue=issue)


FAMILIES: dict[str, Callable[[], Iterator[dict[str, Any]]]] = {
    "molecules": family_molecules,
    "coverage": family_coverage,
    "extreme_cn": family_extreme_cn,
    "dimers": family_dimers,
    "trimers": family_trimers,
    "periodic": family_periodic,
    "partial_pbc": family_partial_pbc,
}


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Attach dftd3 reference results to a fixture input."""
    params = D3BJParameters(**case["parameters"])
    cutoffs = Cutoffs(**case["cutoffs_bohr"])
    evaluator = ReferenceEvaluator(case["numbers"], params, cutoffs)
    result = evaluator.evaluate(
        np.array(case["positions_bohr"]),
        None if case["cell_bohr"] is None else np.array(case["cell_bohr"]),
        case["pbc"],
        gradient=True,
        pairwise=case["pairwise"],
    )
    out = dict(case)
    out["energy"] = result.energy
    if result.gradient is None or result.virial is None:
        msg = "the reference did not return derivatives"
        raise RuntimeError(msg)
    out["gradient"] = result.gradient.tolist()
    out["virial"] = result.virial.tolist()
    if case["pairwise"]:
        if result.pairwise_additive is None or result.pairwise_nonadditive is None:
            msg = "the reference did not return pairwise energies"
            raise RuntimeError(msg)
        out["pairwise_additive"] = result.pairwise_additive.tolist()
        out["pairwise_nonadditive"] = result.pairwise_nonadditive.tolist()
    return out


def header() -> dict[str, Any]:
    """Return the fixture file header."""
    import dftd3.library  # noqa: PLC0415  (imported after OMP_NUM_THREADS is set)

    return {
        "schema": "vanderfast.fixtures/1",
        "generator_version": GENERATOR_VERSION,
        "reference": {
            "package": "dftd3",
            "version": reference_version(),
            "api_version": str(dftd3.library.get_api_version()),
            "omp_num_threads": 1,
        },
        "units": "bohr, hartree",
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

    if reference_version() != "1.6.0":
        print(f"dftd3 {reference_version()} is installed; fixtures need 1.6.0", file=sys.stderr)
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
