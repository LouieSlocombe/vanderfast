"""Unit conversion constants and recipes."""

import ase.units
import numpy as np

from vanderfast import units


def test_constants_match_ase_units_bitwise() -> None:
    assert ase.units.Bohr == units.BOHR
    assert ase.units.Hartree == units.HARTREE


def test_conversion_recipes_match_upstream_ase_calculator() -> None:
    positions = np.array([[1.0, -2.5, 3.25]])
    np.testing.assert_array_equal(
        units.length_to_bohr(positions), positions * (1.0 / ase.units.Bohr)
    )
    assert units.cutoff_to_bohr(10.0) == 10.0 / ase.units.Bohr
    assert (
        units.cutoff_to_angstrom(units.cutoff_to_bohr(7.0)) == 7.0 * ase.units.Bohr / ase.units.Bohr
    )
    assert units.energy_to_ev(-0.5) == -0.5 * ase.units.Hartree
    gradient = np.array([[0.1, -0.2, 0.3]])
    np.testing.assert_array_equal(
        units.gradient_to_forces(gradient), gradient * -(ase.units.Hartree / ase.units.Bohr)
    )


def test_stress_uses_ase_voigt_order() -> None:
    matrix = np.arange(9.0).reshape(3, 3)
    stress = units.strain_derivative_to_stress(matrix, volume=2.0)
    expected = (matrix * (ase.units.Hartree / 2.0)).flat[[0, 4, 8, 5, 2, 1]]
    np.testing.assert_array_equal(stress, expected)
    assert stress.flags.owndata
