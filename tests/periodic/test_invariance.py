"""Symmetry, wrapping, replication, and partial-PBC invariances."""

from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms
from support.backends import (
    SHORT_CUTOFFS,
    Evaluation,
    Evaluator,
    cuda_evaluator,
    oracle_evaluator,
)

from vanderfast import resolve_parameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
PBE = resolve_parameters(method="pbe", damping="bj", atm=False)

BACKENDS = [
    pytest.param((oracle_evaluator, PBE_ATM), id="oracle"),
    pytest.param((cuda_evaluator, PBE_ATM), id="cuda", marks=pytest.mark.cuda),
]

Run = Callable[[Atoms], Evaluation]


@pytest.fixture(params=BACKENDS)
def evaluate(request: pytest.FixtureRequest) -> Run:
    evaluator: Evaluator
    evaluator, parameters = request.param

    def run(atoms: Atoms) -> Evaluation:
        cell = atoms.cell[:] * INV_BOHR if atoms.pbc.any() else None
        return evaluator(
            atoms.numbers, atoms.positions * INV_BOHR, cell, atoms.pbc, parameters, SHORT_CUTOFFS
        )

    return run


def _rotation() -> np.ndarray:
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(0.3), -np.sin(0.3)], [0.0, np.sin(0.3), np.cos(0.3)]]
    )
    rotation: np.ndarray = rz @ rx
    return rotation


@pytest.mark.parametrize(
    "build", [structures.skewed_triclinic, structures.graphene_slab, structures.carbon_wire]
)
def test_translation_and_rotation_invariance(build: Callable[[], Atoms], evaluate: Run) -> None:
    atoms = build()
    reference = evaluate(atoms)
    moved = atoms.copy()
    moved.translate([0.37, -1.2, 2.9])
    assert evaluate(moved).energy == pytest.approx(reference.energy, rel=1e-12)
    rotation = _rotation()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.cell = atoms.cell[:] @ rotation.T
    result = evaluate(rotated)
    assert result.energy == pytest.approx(reference.energy, rel=1e-12)
    np.testing.assert_allclose(result.gradient, reference.gradient @ rotation.T, atol=1e-14)
    np.testing.assert_allclose(
        result.strain_derivative, rotation @ reference.strain_derivative @ rotation.T, atol=1e-14
    )


def test_permutation_invariance(evaluate: Run) -> None:
    atoms = structures.co2_crystal()
    order = np.random.default_rng(3).permutation(len(atoms))
    reference = evaluate(atoms)
    result = evaluate(atoms[order])
    assert result.energy == pytest.approx(reference.energy, rel=1e-12)
    np.testing.assert_allclose(result.gradient, reference.gradient[order], atol=1e-14)


def test_wrapping_invariance(evaluate: Run) -> None:
    atoms = structures.skewed_triclinic()
    reference = evaluate(atoms)
    unwrapped = atoms.copy()
    unwrapped.positions += np.array([[2, -1, 0], [0, 0, 3], [-2, 1, 1], [1, 1, -1]]) @ atoms.cell[:]
    result = evaluate(unwrapped)
    assert result.energy == pytest.approx(reference.energy, rel=1e-12)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-14)
    np.testing.assert_allclose(result.strain_derivative, reference.strain_derivative, atol=1e-13)


def test_replication_is_extensive(evaluate: Run) -> None:
    base = structures.rocksalt_primitive()
    single = evaluate(base)
    for repeats in ((2, 1, 1), (1, 2, 2)):
        supercell = structures.replicate(base, repeats)
        factor = int(np.prod(repeats))
        result = evaluate(supercell)
        assert result.energy == pytest.approx(factor * single.energy, rel=1e-11)
        np.testing.assert_allclose(
            result.gradient, np.tile(single.gradient, (factor, 1)), atol=1e-14
        )
        np.testing.assert_allclose(
            result.strain_derivative, factor * single.strain_derivative, rtol=1e-10, atol=1e-14
        )


def test_inactive_lattice_vector_is_irrelevant(evaluate: Run) -> None:
    reference = evaluate(structures.graphene_slab())
    for vacuum in ((1.0, 0.5, 12.0), (3.0, 0.0, 12.0), (0.0, 0.0, 2.0)):
        tilted = evaluate(structures.graphene_slab(vacuum))
        assert tilted.energy == pytest.approx(reference.energy, rel=1e-13)
        np.testing.assert_allclose(tilted.gradient, reference.gradient, atol=1e-15)


def test_one_atom_cell_has_energy_and_stress_but_no_force(evaluate: Run) -> None:
    result = evaluate(structures.one_atom_triclinic())
    assert result.energy < 0.0
    np.testing.assert_allclose(result.gradient, 0.0, atol=1e-15)
    assert np.abs(result.strain_derivative).max() > 1e-6
