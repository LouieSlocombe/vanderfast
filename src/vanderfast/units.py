"""Unit conversion at the Python boundary.

The numerical core works in atomic units (bohr, hartree). Public Python and
ASE interfaces use Ångström and eV. This module is the single place where the
two meet. The constants and conversion recipes are those of ``ase.units``
(CODATA 2014) exactly as used by the upstream ``dftd3.ase`` calculator, so
results agree with the reference to the last bit of the conversion.

The bundled reference tables were converted with mctc-lib's CODATA 2018
``aatoau`` instead; that constant is internal to the tables and recorded in
the data manifest.
"""

from typing import Final

import numpy as np
import numpy.typing as npt

BOHR: Final = 0.5291772105638411
"""Bohr radius in Ångström (``ase.units.Bohr``)."""

HARTREE: Final = 27.211386024367243
"""Hartree energy in eV (``ase.units.Hartree``)."""

INV_BOHR: Final = 1.0 / BOHR
HARTREE_PER_BOHR: Final = HARTREE / BOHR

STRESS_VOIGT_ORDER: Final = (0, 4, 8, 5, 2, 1)
"""Flat 3x3 indices for ASE stress order xx, yy, zz, yz, xz, xy."""

FloatArray = npt.NDArray[np.float64]


def length_to_bohr(values: npt.ArrayLike) -> FloatArray:
    """Convert positions or cell vectors from Ångström to bohr."""
    return np.asarray(values, dtype=np.float64) * INV_BOHR


def cutoff_to_bohr(value: float) -> float:
    """Convert a cutoff radius from Ångström to bohr."""
    return float(value) / BOHR


def cutoff_to_angstrom(value: float) -> float:
    """Convert a cutoff radius from bohr to Ångström."""
    return float(value) * BOHR


def energy_to_ev(value: float) -> float:
    """Convert an energy from hartree to eV."""
    return float(value) * HARTREE


def gradient_to_forces(gradient: npt.ArrayLike) -> FloatArray:
    """Convert an energy gradient in hartree/bohr to forces in eV/Å."""
    return np.asarray(gradient, dtype=np.float64) * -HARTREE_PER_BOHR


def strain_derivative_to_stress(strain_derivative: npt.ArrayLike, volume: float) -> FloatArray:
    """Convert dE/dstrain (hartree) to ASE Voigt stress in eV/Å³.

    Args:
        strain_derivative: The 3x3 derivative of the energy with respect to a
            homogeneous strain of positions and cell, in hartree.
        volume: Cell volume in Å³.

    Returns:
        Six stress components in the order xx, yy, zz, yz, xz, xy.
    """
    matrix = np.asarray(strain_derivative, dtype=np.float64).reshape(3, 3)
    scaled = matrix * (HARTREE / float(volume))
    return scaled.flat[list(STRESS_VOIGT_ORDER)].copy()
