"""GPU ATM three-body term: fixtures, counts, chunking, policy, and derivatives."""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms
from support import native_helpers as nh
from support.fd import agrees_over_sweep, position_gradient
from support.fixtures import FixtureCase, all_cases
from support.inputs import SHORT_D3, bohr

from vanderfast import _oracle, resolve_parameters
from vanderfast.data import SECTION_C6, load_reference_tables, pair_index, parse_reference_tables
from vanderfast.testing import structures

pytestmark = pytest.mark.cuda

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)

ATM_CASES = [c for c in all_cases() if c.parameters.atm and c.known_issue is None]


@pytest.mark.parametrize("case", ATM_CASES, ids=[c.id for c in ATM_CASES])
def test_atm_cases_match_dftd3(case: FixtureCase) -> None:
    ctx = nh.context(case.numbers, case.parameters, case.cutoffs)
    result = nh.run(ctx, case.positions, case.cell, case.pbc)
    assert result["energy"] == pytest.approx(case.energy, rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(result["gradient"], case.gradient, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["virial"], case.virial, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "build",
    [
        lambda: structures.equilateral_trimer(18, 3.5),
        structures.water_dimer,
        structures.rocksalt_primitive,
        structures.graphene_slab,
    ],
    ids=["trimer", "water_dimer", "rocksalt", "slab"],
)
def test_triangle_counts_and_energy_split_match_the_oracle(build: Callable[[], Atoms]) -> None:
    numbers, positions, cell, pbc = bohr(build())
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3, kernels="literal")
    result = nh.run(ctx, positions, cell, pbc)
    oracle = _oracle.evaluate(
        numbers, positions, parameters=PBE_ATM, cell=cell, pbc=pbc, cutoffs=SHORT_D3
    )
    for key in ("atm_candidates", "atm_accepted", "atm_c6_skipped"):
        assert result["info"][key] == oracle.counts[key], key
    # The canonical anchor meets every triangle once instead of three times.
    canonical = nh.run(nh.context(numbers, PBE_ATM, SHORT_D3), positions, cell, pbc)
    assert 3 * canonical["info"]["atm_accepted"] == result["info"]["atm_accepted"]
    assert result["energy_atm"] == pytest.approx(oracle.energy_atm, rel=1e-12, abs=1e-18)
    assert result["energy_pair"] == pytest.approx(oracle.energy_pair, rel=1e-12, abs=1e-18)


def test_chunked_launches_give_the_same_result() -> None:
    numbers, positions, cell, pbc = bohr(structures.co2_crystal())
    whole = nh.run(nh.context(numbers, PBE_ATM, SHORT_D3), positions, cell, pbc)
    chunked = nh.run(
        nh.context(numbers, PBE_ATM, SHORT_D3, atm_items_per_launch=7), positions, cell, pbc
    )
    assert chunked["energy"] == pytest.approx(whole["energy"], rel=1e-14)
    np.testing.assert_allclose(chunked["gradient"], whole["gradient"], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(chunked["virial"], whole["virial"], rtol=1e-12, atol=1e-18)
    assert chunked["info"]["atm_accepted"] == whole["info"]["atm_accepted"]


def test_energy_is_reproducible_and_energy_only_path_agrees() -> None:
    numbers, positions, cell, pbc = bohr(structures.skewed_triclinic())
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3)
    first = nh.run(ctx, positions, cell, pbc)
    for _ in range(3):
        assert nh.run(ctx, positions, cell, pbc)["energy"] == first["energy"]
    energy_only = nh.run(ctx, positions, cell, pbc, gradient=False, virial=False)
    assert energy_only["energy"] == first["energy"]


def test_tiny_c6_triangles_are_skipped_in_every_path() -> None:
    start = int(load_reference_tables().c6_offset[pair_index(18, 18)])
    blob = nh.patched_reference_blob(SECTION_C6, start, [1e-20])
    numbers, positions, _, _ = bohr(structures.equilateral_trimer(18, 3.5))
    ctx = nh.context(
        numbers, PBE_ATM, reference_data=blob, reference_identity="tiny", kernels="literal"
    )
    full = nh.run(ctx, positions)
    energy_only = nh.run(ctx, positions, gradient=False, virial=False)
    tiny = parse_reference_tables(blob)
    oracle = _oracle.evaluate(numbers, positions, parameters=PBE_ATM, tables=tiny)
    assert full["info"]["atm_c6_skipped"] == oracle.counts["atm_c6_skipped"] == 3
    assert full["info"]["atm_accepted"] == 0
    assert full["energy"] == energy_only["energy"]
    assert full["energy"] == pytest.approx(oracle.energy, rel=1e-12)
    assert np.all(np.isfinite(full["gradient"]))


def test_atm_forces_match_finite_differences() -> None:
    numbers, positions, _, _ = bohr(structures.scalene_trimer())
    ctx = nh.context(numbers, PBE_ATM)
    analytic = nh.run(ctx, positions)["gradient"]

    def energy(x: np.ndarray) -> float:
        return float(nh.run(ctx, x, gradient=False, virial=False)["energy"])

    ok, errors = agrees_over_sweep(
        analytic, lambda h: position_gradient(energy, positions, h), 1e-10
    )
    assert ok, errors


def test_atm_only_parameters() -> None:
    atm_only = dataclasses.replace(PBE_ATM, s6=0.0, s8=0.0)
    numbers, positions, cell, pbc = bohr(structures.rocksalt_primitive())
    result = nh.run(nh.context(numbers, atm_only, SHORT_D3), positions, cell, pbc)
    assert result["energy_pair"] == 0.0
    assert result["energy_atm"] > 0.0
    assert result["info"]["pair_edges"] == 0
