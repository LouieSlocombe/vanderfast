"""The NumPy D4 oracle reproduces every committed dftd4 4.2.0 fixture."""

import numpy as np
import pytest
from support.fixtures_d4 import D4FixtureCase, all_cases, find_case, load_header
from support.tolerances import (
    D4_ORACLE_CHARGE_ABS,
    D4_ORACLE_GRADIENT_ABS,
    D4_ORACLE_VIRIAL_ABS,
    d4_oracle_energy_tolerance,
)

from vanderfast import _oracle_d4

CASES = all_cases()
STRICT = [case for case in CASES if case.known_issue is None]


def _oracle_result(case: D4FixtureCase, *, gradient: bool = True) -> _oracle_d4.D4OracleResult:
    return _oracle_d4.evaluate(
        case.numbers,
        case.positions,
        parameters=case.parameters,
        charge=case.charge,
        cell=case.cell,
        pbc=case.pbc,
        cutoffs=case.cutoffs,
        gradient=gradient,
    )


def test_fixtures_were_generated_by_the_pinned_reference() -> None:
    for family in {case.family for case in CASES}:
        header = load_header(family)
        assert header["reference"]["version"] == "4.2.0"
        assert header["reference"]["omp_num_threads"] == 1
        assert header["reference"]["openblas_num_threads"] == 1


@pytest.mark.parametrize("case", STRICT, ids=[case.id for case in STRICT])
def test_oracle_matches_reference(case: D4FixtureCase) -> None:
    result = _oracle_result(case, gradient=case.gradient is not None)
    assert abs(result.energy - case.energy) <= d4_oracle_energy_tolerance(case.energy)
    np.testing.assert_allclose(result.charges, case.charges, rtol=0, atol=D4_ORACLE_CHARGE_ABS)
    np.testing.assert_allclose(result.cn, case.cn, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.polarizabilities, case.polarizabilities, rtol=1e-12)
    if case.c6 is not None:
        np.testing.assert_allclose(result.c6, case.c6, rtol=1e-12, atol=1e-14)
    if case.gradient is not None:
        assert case.virial is not None
        assert result.gradient is not None
        assert result.strain_derivative is not None
        np.testing.assert_allclose(
            result.gradient, case.gradient, rtol=0, atol=D4_ORACLE_GRADIENT_ABS
        )
        np.testing.assert_allclose(
            result.strain_derivative, case.virial, rtol=0, atol=D4_ORACLE_VIRIAL_ABS
        )


def test_partial_periodicity_matches_the_fully_periodic_reference() -> None:
    """dftd4 ignores per-direction flags; so does vanderfast's D4."""
    for name in ("graphene_slab", "carbon_wire", "graphene_slab_tilted_vacuum"):
        partial = find_case("partial_pbc", name, "Q1")
        full = find_case("partial_pbc", f"{name}_as_3d", "Q1")
        assert partial.energy == full.energy
        np.testing.assert_array_equal(partial.charges, full.charges)


def test_reference_charges_depend_on_the_cell_choice() -> None:
    """Documents dftd4's unconverged periodic EEQ (reproduced by ewald='upstream')."""
    primitive = find_case("cell_dependence", "mgo_replica_1x1x1", "Q1")
    supercell = find_case("cell_dependence", "mgo_replica_2x2x2", "Q1")
    spread = np.abs(supercell.charges.reshape(8, -1) - primitive.charges).max()
    assert 1e-4 < spread < 1e-2
    converged = [
        _oracle_d4.evaluate(
            case.numbers,
            case.positions,
            parameters=case.parameters,
            cell=case.cell,
            pbc=case.pbc,
            cutoffs=case.cutoffs,
            ewald="converged",
            gradient=False,
        )
        for case in (primitive, supercell)
    ]
    np.testing.assert_allclose(
        converged[1].charges.reshape(8, -1), converged[0].charges[None, :].repeat(8, 0), atol=1e-13
    )


def test_exact_zero_effective_charge_has_a_finite_gradient() -> None:
    """Upstream's dzeta is 0/0 at q + zeff == 0; vanderfast uses the limit 0."""
    case = find_case("zeta", "periodic_hydride_exact_zero", "Q2")
    assert case.known_issue is not None
    result = _oracle_result(case)
    assert abs(result.energy - case.energy) <= d4_oracle_energy_tolerance(case.energy)
    assert result.gradient is not None
    assert np.all(np.isfinite(result.gradient))
    assert result.charges[0] == -1.0


def test_stress_fixture_has_an_indefinite_eeq_matrix() -> None:
    """The dense hydrogen cluster gives A negative eigenvalues; dftd4 and the oracle agree."""
    case = find_case("eeq_stress", "hydrogen_cluster_indefinite", "Q1")
    result = _oracle_result(case, gradient=False)
    eigenvalues = np.linalg.eigvalsh(result.eeq_matrix[:-1, :-1])
    assert eigenvalues.min() < -0.1
    np.testing.assert_allclose(result.charges, case.charges, rtol=0, atol=D4_ORACLE_CHARGE_ABS)
