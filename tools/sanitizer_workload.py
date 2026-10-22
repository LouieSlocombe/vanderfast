r"""Representative GPU workload for compute-sanitizer runs.

Usage::

    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \\
        python tools/sanitizer_workload.py
"""

import sys
from typing import Any

import numpy as np
from ase import Atoms

from vanderfast import _native, resolve_parameters
from vanderfast.parameters import Cutoffs
from vanderfast.parameters_d4 import D4Cutoffs, resolve_d4_parameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR


def _context(numbers: np.ndarray, atm: bool, cutoffs: Cutoffs) -> _native.Context:
    p = resolve_parameters(method="pbe", damping="bj", atm=atm)
    return _native.Context(
        numbers.astype(np.int32),
        s6=p.s6,
        s8=p.s8,
        a1=p.a1,
        a2=p.a2,
        s9=p.s9,
        alp=p.alp,
        cutoff_cn=cutoffs.cn,
        cutoff_disp2=cutoffs.disp2,
        cutoff_disp3=cutoffs.disp3,
    )


def _d4_context(numbers: np.ndarray, atm: bool, **options: Any) -> _native.D4Context:
    p = resolve_d4_parameters(method="pbe", atm=atm)
    cutoffs = D4Cutoffs(cn=15.0, disp2=20.0, disp3=12.0)
    return _native.D4Context(
        numbers.astype(np.int32),
        s6=p.s6,
        s8=p.s8,
        a1=p.a1,
        a2=p.a2,
        s9=p.s9,
        alp=p.alp,
        ga=p.ga,
        gc=p.gc,
        wf=p.wf,
        cutoff_cn=cutoffs.cn,
        cutoff_disp2=cutoffs.disp2,
        cutoff_disp3=cutoffs.disp3,
        **options,
    )


def d4_workload(atm: bool) -> None:
    """Run D4 over every charge solver, storage, Ewald mode, and kernel variant."""
    ion, charge = structures.IONS["acetate"]
    cluster = structures.water_dimer()
    for offset in range(1, 6):
        extra = structures.water_dimer()
        extra.translate((8.0 * offset, 0.0, 0.0))
        cluster += extra
    crystal = structures.magnesia_conventional().repeat((2, 1, 1))
    dense_bytes = (len(crystal) + 1) ** 2 * 8
    cases: list[tuple[Atoms, float, dict[str, Any], dict[str, int]]] = [
        (ion(), charge, {}, {}),  # shared-memory LU
        (ion(), charge, {"kernels": "literal", "solver": "direct"}, {}),
        (cluster, 0.0, {"solver": "iterative"}, {}),
        (cluster, 0.0, {"solver": "iterative"}, {"packed": 1}),
        (cluster, 0.0, {"solver": "iterative", "warm_start": True}, {}),
        (crystal, 0.0, {"solver": "iterative"}, {}),  # stored lattice derivatives
        (crystal, 0.0, {"solver": "iterative", "ewald": "converged"}, {"packed": 1}),
        (crystal, 0.0, {"solver": "iterative"}, {"limit": dense_bytes}),  # recomputed
        (crystal, 0.0, {"kernels": "literal", "solver": "direct"}, {}),
    ]
    for atoms, total, options, hooks in cases:
        ctx = _d4_context(atoms.numbers, atm, charge=total, **options)
        if "packed" in hooks:
            ctx._testing_set_packed_min_atoms(hooks["packed"])  # noqa: SLF001
        if "limit" in hooks:
            ctx._testing_set_matrix_limit(hooks["limit"])  # noqa: SLF001
        positions = atoms.positions * INV_BOHR
        periodic = bool(atoms.pbc.any())
        cell = atoms.cell[:] * INV_BOHR if periodic else np.zeros((3, 3))
        pbc = (periodic, periodic, periodic)
        ctx.evaluate(positions, cell, pbc, gradient=True, virial=True, atom_energies=True)
        ctx.evaluate(positions, cell, pbc)
        ctx.debug(positions, cell, pbc, derivatives=True)
        del ctx


def main() -> int:
    """Run molecular, periodic, energy-only, and debug evaluations (D3 and D4)."""
    short = Cutoffs(cn=15.0, disp2=20.0, disp3=15.0)
    atm = "--atm" in sys.argv
    cases = [
        (structures.benzene_dimer(), Cutoffs()),
        (structures.mixed_fragment(), Cutoffs()),
        (structures.skewed_triclinic(), short),
        (structures.graphene_slab(), short),
        (structures.simple_cubic(18, 3.4), short),
    ]
    for atoms, cutoffs in cases:
        ctx = _context(atoms.numbers, atm, cutoffs)
        positions = atoms.positions * INV_BOHR
        cell = atoms.cell[:] * INV_BOHR if atoms.pbc.any() else np.zeros((3, 3))
        pbc = (bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2]))
        ctx.evaluate(positions, cell, pbc, gradient=True, virial=True, atom_energies=True)
        ctx.evaluate(positions, cell, pbc)
        ctx.debug(positions, cell, pbc, derivatives=True)
        del ctx
    d4_workload(atm)
    print("sanitizer workload completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
