"""Shared test inputs: reduced cutoffs and ASE structures in atomic units."""

from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms

from vanderfast.parameters import Cutoffs
from vanderfast.parameters_d4 import D4Cutoffs
from vanderfast.units import INV_BOHR

SHORT_D3 = Cutoffs(cn=15.0, disp2=20.0, disp3=15.0)
"""Reduced D3 cutoffs (bohr) that keep brute-force periodic evaluations fast."""

SHORT_D4 = D4Cutoffs(cn=20.0, disp2=25.0, disp3=15.0)
"""Reduced D4 cutoffs (bohr) that keep brute-force periodic evaluations fast."""


def bohr(
    atoms: Atoms,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None, Any]:
    """Return numbers, positions, cell (``None`` without periodicity), and pbc in bohr."""
    cell = atoms.cell[:] * INV_BOHR if atoms.pbc.any() else None
    return atoms.numbers, atoms.positions * INV_BOHR, cell, atoms.pbc
