"""The NumPy oracle reproduces every committed dftd3 1.6.0 fixture."""

import numpy as np
import pytest
from support.fixtures import FixtureCase, all_cases, find_case, load_header
from support.tolerances import (
    ORACLE_GRADIENT_ABS,
    ORACLE_VIRIAL_ABS,
    oracle_energy_tolerance,
)

from vanderfast import _oracle

CASES = all_cases()
STRICT = [case for case in CASES if case.known_issue is None]
KNOWN_ISSUES = [case for case in CASES if case.known_issue is not None]


def _oracle_result(case: FixtureCase) -> _oracle.OracleResult:
    return _oracle.evaluate(
        case.numbers,
        case.positions,
        parameters=case.parameters,
        cell=case.cell,
        pbc=case.pbc,
        cutoffs=case.cutoffs,
    )


def test_fixtures_were_generated_by_the_pinned_reference() -> None:
    for family in {case.family for case in CASES}:
        header = load_header(family)
        assert header["reference"]["version"] == "1.6.0"
        assert header["reference"]["omp_num_threads"] == 1


@pytest.mark.parametrize("case", STRICT, ids=[case.id for case in STRICT])
def test_oracle_matches_reference(case: FixtureCase) -> None:
    result = _oracle_result(case)
    assert abs(result.energy - case.energy) <= oracle_energy_tolerance(case.energy)
    assert result.gradient is not None
    assert result.strain_derivative is not None
    np.testing.assert_allclose(result.gradient, case.gradient, rtol=0, atol=ORACLE_GRADIENT_ABS)
    np.testing.assert_allclose(
        result.strain_derivative, case.virial, rtol=0, atol=ORACLE_VIRIAL_ABS
    )


@pytest.mark.parametrize("case", KNOWN_ISSUES, ids=[case.id for case in KNOWN_ISSUES])
def test_known_upstream_partial_pbc_issue(case: FixtureCase) -> None:
    """Upstream misses images for tilted inactive vectors; vanderfast does not.

    The tilted slab is physically identical to the untilted one, so the oracle
    must reproduce the untilted reference while upstream does not.
    """
    untilted = find_case(case.family, case.name.removesuffix("_tilted_vacuum"), case.parameter_set)
    result = _oracle_result(case)
    assert abs(result.energy - untilted.energy) <= oracle_energy_tolerance(untilted.energy)
    assert abs(case.energy - untilted.energy) > 1e-9


C6_CASES = [case for case in CASES if case.parameter_set == "P5"]


@pytest.mark.parametrize("case", C6_CASES, ids=[case.id for case in C6_CASES])
def test_oracle_c6_matches_reference_pairwise_extraction(case: FixtureCase) -> None:
    """With s8 = a1 = a2 = 0, the upstream pair energy is exactly -C6 / r^6."""
    assert case.pairwise_additive is not None
    result = _oracle.evaluate(
        case.numbers, case.positions, parameters=case.parameters, gradient=False
    )
    d = case.positions[None, :, :] - case.positions[:, None, :]
    r6 = np.einsum("ijc,ijc->ij", d, d) ** 3
    off = ~np.eye(case.numbers.size, dtype=bool)
    c6_reference = -(case.pairwise_additive + case.pairwise_additive.T)[off] * r6[off]
    np.testing.assert_allclose(result.c6[off], c6_reference, rtol=1e-12)
