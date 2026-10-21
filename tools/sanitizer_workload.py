r"""Representative GPU workload for compute-sanitizer runs.

Usage::

    compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \\
        python tools/sanitizer_workload.py
"""

import sys

import numpy as np

from vanderfast import _native, resolve_parameters
from vanderfast.parameters import Cutoffs
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


def main() -> int:
    """Run molecular, periodic, energy-only, and debug evaluations."""
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
    print("sanitizer workload completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
