"""GPU ATM three-body term: fixtures, counts, chunking, policy, and derivatives."""

import dataclasses

import numpy as np
import pytest
from support import native_helpers as nh
from support.fd import agrees_over_sweep, position_gradient
from support.fixtures import FixtureCase, all_cases

from vanderfast import _oracle, resolve_parameters
from vanderfast.data import load_reference_tables, pair_index, read_reference_blob
from vanderfast.parameters import Cutoffs
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
SHORT = Cutoffs(cn=15.0, disp2=20.0, disp3=15.0)

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
def test_triangle_counts_and_energy_split_match_the_oracle(build) -> None:  # type: ignore[no-untyped-def]
    atoms = build()
    positions = atoms.positions * INV_BOHR
    cell = atoms.cell[:] * INV_BOHR if atoms.pbc.any() else None
    ctx = nh.context(atoms.numbers, PBE_ATM, SHORT)
    result = nh.run(ctx, positions, cell, atoms.pbc)
    oracle = _oracle.evaluate(
        atoms.numbers, positions, parameters=PBE_ATM, cell=cell, pbc=atoms.pbc, cutoffs=SHORT
    )
    for key in ("atm_candidates", "atm_accepted", "atm_c6_skipped"):
        assert result["info"][key] == oracle.counts[key], key
    assert result["energy_atm"] == pytest.approx(oracle.energy_atm, rel=1e-12, abs=1e-18)
    assert result["energy_pair"] == pytest.approx(oracle.energy_pair, rel=1e-12, abs=1e-18)


def test_chunked_launches_give_the_same_result() -> None:
    atoms = structures.co2_crystal()
    positions = atoms.positions * INV_BOHR
    cell = atoms.cell[:] * INV_BOHR
    whole = nh.run(nh.context(atoms.numbers, PBE_ATM, SHORT), positions, cell, atoms.pbc)
    chunked = nh.run(
        nh.context(atoms.numbers, PBE_ATM, SHORT, atm_items_per_launch=7),
        positions,
        cell,
        atoms.pbc,
    )
    assert chunked["energy"] == pytest.approx(whole["energy"], rel=1e-14)
    np.testing.assert_allclose(chunked["gradient"], whole["gradient"], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(chunked["virial"], whole["virial"], rtol=1e-12, atol=1e-18)
    assert chunked["info"]["atm_accepted"] == whole["info"]["atm_accepted"]


def test_energy_is_reproducible_and_energy_only_path_agrees() -> None:
    atoms = structures.skewed_triclinic()
    positions = atoms.positions * INV_BOHR
    cell = atoms.cell[:] * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE_ATM, SHORT)
    first = nh.run(ctx, positions, cell, atoms.pbc)
    for _ in range(3):
        assert nh.run(ctx, positions, cell, atoms.pbc)["energy"] == first["energy"]
    energy_only = nh.run(ctx, positions, cell, atoms.pbc, gradient=False, virial=False)
    assert energy_only["energy"] == first["energy"]


def test_tiny_c6_triangles_are_skipped_in_every_path() -> None:
    import struct  # noqa: PLC0415

    tables = load_reference_tables()
    blob = bytearray(read_reference_blob())
    c6_offset = struct.unpack_from("<Q", blob, 120 + 4 * 24 + 8)[0]
    index = int(tables.c6_offset[pair_index(18, 18)])
    struct.pack_into("<d", blob, c6_offset + 8 * index, 1e-20)
    tiny = dataclasses.replace(
        tables, c6_flat=np.frombuffer(bytes(blob), "<f8", tables.c6_flat.size, c6_offset)
    )
    atoms = structures.equilateral_trimer(18, 3.5)
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE_ATM, reference_data=bytes(blob), reference_identity="tiny")
    full = nh.run(ctx, positions)
    energy_only = nh.run(ctx, positions, gradient=False, virial=False)
    oracle = _oracle.evaluate(atoms.numbers, positions, parameters=PBE_ATM, tables=tiny)
    assert full["info"]["atm_c6_skipped"] == oracle.counts["atm_c6_skipped"] == 3
    assert full["info"]["atm_accepted"] == 0
    assert full["energy"] == energy_only["energy"]
    assert full["energy"] == pytest.approx(oracle.energy, rel=1e-12)
    assert np.all(np.isfinite(full["gradient"]))


def test_atm_forces_match_finite_differences() -> None:
    atoms = structures.scalene_trimer()
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE_ATM)
    analytic = nh.run(ctx, positions)["gradient"]

    def energy(x: np.ndarray) -> float:
        return float(nh.run(ctx, x, gradient=False, virial=False)["energy"])

    ok, errors = agrees_over_sweep(
        analytic, lambda h: position_gradient(energy, positions, h), 1e-10
    )
    assert ok, errors


def test_atm_only_parameters() -> None:
    atm_only = dataclasses.replace(PBE_ATM, s6=0.0, s8=0.0)
    atoms = structures.rocksalt_primitive()
    positions = atoms.positions * INV_BOHR
    cell = atoms.cell[:] * INV_BOHR
    result = nh.run(nh.context(atoms.numbers, atm_only, SHORT), positions, cell, atoms.pbc)
    assert result["energy_pair"] == 0.0
    assert result["energy_atm"] > 0.0
    assert result["info"]["pair_edges"] == 0
