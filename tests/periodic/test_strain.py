"""Strain derivatives, forces, and stress in periodic cells."""

from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.fd import calculate_numerical_stress
from support.backends import Evaluation, Evaluator, cuda_evaluator, oracle_evaluator
from support.fd import agrees_over_sweep, position_gradient, strain_derivative

from vanderfast import resolve_parameters
from vanderfast.parameters import Cutoffs, D3BJParameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
PBE = resolve_parameters(method="pbe", damping="bj", atm=False)
FD_CUTOFFS = Cutoffs(cn=12.0, disp2=18.0, disp3=12.0)

BACKENDS = [
    pytest.param((oracle_evaluator, PBE_ATM), id="oracle"),
    pytest.param((cuda_evaluator, PBE_ATM), id="cuda", marks=pytest.mark.cuda),
]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> tuple[Evaluator, D3BJParameters]:
    evaluator: Evaluator
    evaluator, parameters = request.param
    return evaluator, parameters


def _bohr(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return atoms.numbers, atoms.positions * INV_BOHR, atoms.cell[:] * INV_BOHR, atoms.pbc


@pytest.mark.parametrize(
    "build",
    [structures.skewed_triclinic, structures.one_atom_triclinic, structures.rocksalt_primitive],
)
def test_strain_derivative_matches_finite_strain(
    build: Callable[[], Atoms], backend: tuple[Evaluator, D3BJParameters]
) -> None:
    evaluator, parameters = backend
    numbers, positions, cell, pbc = _bohr(build())
    analytic = evaluator(numbers, positions, cell, pbc, parameters, FD_CUTOFFS).strain_derivative

    def energy(x: np.ndarray, h: np.ndarray) -> float:
        return evaluator(numbers, x, h, pbc, parameters, FD_CUTOFFS).energy

    ok, errors = agrees_over_sweep(
        analytic, lambda step: strain_derivative(energy, positions, cell, step), 1e-9
    )
    assert ok, errors
    assert np.abs(analytic - analytic.T).max() < 1e-12


@pytest.mark.parametrize("build", [structures.skewed_triclinic, structures.graphene_slab])
def test_periodic_forces_match_finite_differences(
    build: Callable[[], Atoms], backend: tuple[Evaluator, D3BJParameters]
) -> None:
    evaluator, parameters = backend
    numbers, positions, cell, pbc = _bohr(build())
    result: Evaluation = evaluator(numbers, positions, cell, pbc, parameters, FD_CUTOFFS)

    def energy(x: np.ndarray) -> float:
        return evaluator(numbers, x, cell, pbc, parameters, FD_CUTOFFS).energy

    ok, errors = agrees_over_sweep(
        result.gradient, lambda step: position_gradient(energy, positions, step), 1e-10
    )
    assert ok, errors


@pytest.mark.cuda
def test_ase_stress_matches_numerical_stress_and_reference() -> None:
    from vanderfast.ase import D3Calculator  # noqa: PLC0415

    cutoff = {"cn": 6.0, "disp2": 9.0, "disp3": 6.0}
    atoms = structures.skewed_triclinic()
    atoms.calc = D3Calculator(parameters=PBE, cutoff=cutoff)
    stress = atoms.get_stress()
    numerical = calculate_numerical_stress(atoms, eps=1e-5)
    np.testing.assert_allclose(stress, numerical, atol=1e-9)
    reference = atoms.copy()
    reference.calc = D3Calculator(parameters=PBE, cutoff=cutoff, backend="reference")
    np.testing.assert_allclose(stress, reference.get_stress(), atol=1e-12)
    np.testing.assert_allclose(atoms.get_forces(), reference.get_forces(), atol=1e-12)
