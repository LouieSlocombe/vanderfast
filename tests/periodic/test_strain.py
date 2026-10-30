"""Strain derivatives, forces, and stress in periodic cells."""

from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.fd import calculate_numerical_stress
from support.backends import Evaluation, Evaluator, cuda_evaluator, oracle_evaluator
from support.fd import agrees_over_sweep, position_gradient, strain_derivative
from support.inputs import bohr

from vanderfast import resolve_parameters
from vanderfast.ase import D3Calculator
from vanderfast.parameters import Cutoffs
from vanderfast.testing import structures

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
PBE = resolve_parameters(method="pbe", damping="bj", atm=False)
FD_CUTOFFS = Cutoffs(cn=12.0, disp2=18.0, disp3=12.0)

BACKENDS = [
    pytest.param(oracle_evaluator, id="oracle"),
    pytest.param(cuda_evaluator, id="cuda", marks=pytest.mark.cuda),
]


@pytest.fixture(params=BACKENDS)
def evaluator(request: pytest.FixtureRequest) -> Evaluator:
    backend: Evaluator = request.param
    return backend


@pytest.mark.parametrize(
    "build",
    [structures.skewed_triclinic, structures.one_atom_triclinic, structures.rocksalt_primitive],
)
def test_strain_derivative_matches_finite_strain(
    build: Callable[[], Atoms], evaluator: Evaluator
) -> None:
    numbers, positions, cell, pbc = bohr(build())
    assert cell is not None
    analytic = evaluator(numbers, positions, cell, pbc, PBE_ATM, FD_CUTOFFS).strain_derivative

    def energy(x: np.ndarray, h: np.ndarray) -> float:
        return evaluator(numbers, x, h, pbc, PBE_ATM, FD_CUTOFFS).energy

    ok, errors = agrees_over_sweep(
        analytic, lambda step: strain_derivative(energy, positions, cell, step), 1e-9
    )
    assert ok, errors
    assert np.abs(analytic - analytic.T).max() < 1e-12


@pytest.mark.parametrize("build", [structures.skewed_triclinic, structures.graphene_slab])
def test_periodic_forces_match_finite_differences(
    build: Callable[[], Atoms], evaluator: Evaluator
) -> None:
    numbers, positions, cell, pbc = bohr(build())
    result: Evaluation = evaluator(numbers, positions, cell, pbc, PBE_ATM, FD_CUTOFFS)

    def energy(x: np.ndarray) -> float:
        return evaluator(numbers, x, cell, pbc, PBE_ATM, FD_CUTOFFS).energy

    ok, errors = agrees_over_sweep(
        result.gradient, lambda step: position_gradient(energy, positions, step), 1e-10
    )
    assert ok, errors


@pytest.mark.cuda
def test_ase_stress_matches_numerical_stress_and_reference() -> None:
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
