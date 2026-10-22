"""GPU D4: dftd4 fixtures, oracle intermediates, derivatives, and context state."""

import dataclasses
from typing import Any

import numpy as np
import pytest
from support import native_helpers as nh
from support.fd import agrees_over_sweep, position_gradient
from support.fixtures_d4 import D4FixtureCase, all_cases
from support.tolerances import (
    D4_GPU_CHARGE_ABS,
    D4_GPU_GRADIENT_ABS,
    D4_GPU_VIRIAL_ABS,
    d4_gpu_energy_tolerance,
)

from vanderfast import _oracle_d4
from vanderfast.parameters_d4 import D4Cutoffs, resolve_d4_parameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE = resolve_d4_parameters(method="pbe")
PBE_PAIR = resolve_d4_parameters(method="pbe", atm=False)
CASES = all_cases()
MOLECULAR = {"none": (False, False, False)}


def _run(case: D4FixtureCase, **options: Any) -> dict[str, Any]:
    ctx = nh.d4_context(case.numbers, case.parameters, case.cutoffs, charge=case.charge, **options)
    gradient = case.gradient is not None
    return nh.run(ctx, case.positions, case.cell, case.pbc, gradient=gradient, virial=gradient)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_cases_match_dftd4(case: D4FixtureCase) -> None:
    result = _run(case, solver="direct")
    assert abs(result["energy"] - case.energy) <= d4_gpu_energy_tolerance(case.energy)
    np.testing.assert_allclose(result["charges"], case.charges, rtol=0, atol=D4_GPU_CHARGE_ABS)
    if case.gradient is not None:
        assert case.virial is not None
        np.testing.assert_allclose(result["gradient"], case.gradient, rtol=0, atol=D4_GPU_GRADIENT_ABS)
        np.testing.assert_allclose(result["virial"], case.virial, rtol=0, atol=D4_GPU_VIRIAL_ABS)
    assert result["info"]["eeq_solver"] == "direct"
    assert result["info"]["ewald_mode"] == ("upstream" if case.periodic else "none")


def _ion(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    build, charge = structures.IONS[name]
    atoms = build()
    return atoms.numbers, atoms.positions * INV_BOHR, charge


@pytest.mark.parametrize(
    ("numbers", "positions", "charge", "cell", "pbc"),
    [
        pytest.param(*_ion("acetate"), None, (False,) * 3, id="acetate"),
        pytest.param(*_ion("hexaaqua_iron"), None, (False,) * 3, id="hexaaqua_iron"),
        pytest.param(
            structures.magnesia_primitive().numbers,
            structures.magnesia_primitive().positions * INV_BOHR,
            0.0,
            structures.magnesia_primitive().cell[:] * INV_BOHR,
            (True,) * 3,
            id="mgo_primitive",
        ),
    ],
)
def test_intermediates_match_the_oracle(
    numbers: np.ndarray, positions: np.ndarray, charge: float, cell: Any, pbc: tuple[bool, ...]
) -> None:
    ctx = nh.d4_context(numbers, PBE, charge=charge)
    h = np.zeros((3, 3)) if cell is None else cell
    debug = ctx.debug(positions, h, pbc, derivatives=True)
    oracle = _oracle_d4.evaluate(
        numbers, positions, parameters=PBE, charge=charge, cell=cell, pbc=pbc, gradient=True
    )
    nref = oracle.weights.shape[1]
    np.testing.assert_allclose(debug["cn"], oracle.cn, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(debug["cn_eeq"], oracle.cn_eeq, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(debug["charges"], oracle.charges, rtol=0, atol=1e-13)
    np.testing.assert_allclose(debug["weights"][:, :nref], oracle.weights, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(debug["weights_q0"][:, :nref], oracle.weights_q0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        debug["dweights_dcn"][:, :nref], oracle.dweights_dcn, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_allclose(debug["dweights_dq"][:, :nref], oracle.dweights_dq, rtol=1e-11, atol=1e-14)
    assert oracle.dedcn is not None
    assert oracle.dedq is not None
    np.testing.assert_allclose(debug["dEdcn"], oracle.dedcn, rtol=1e-11, atol=1e-16)
    np.testing.assert_allclose(debug["dEdq"], oracle.dedq, rtol=1e-11, atol=1e-16)
    assert not debug["weights"][:, nref:].any()


def test_energy_only_path_matches_the_gradient_path() -> None:
    numbers, positions, charge = _ion("hexaaqua_iron")
    ctx = nh.d4_context(numbers, PBE, charge=charge)
    full = nh.run(ctx, positions)
    energy = nh.run(ctx, positions, gradient=False, virial=False)
    assert energy["energy"] == pytest.approx(full["energy"], rel=1e-14)
    np.testing.assert_array_equal(energy["charges"], full["charges"])
    assert energy["gradient"] is None


def test_results_are_reproducible() -> None:
    numbers, positions, charge = _ion("acetate")
    ctx = nh.d4_context(numbers, PBE_PAIR, charge=charge)
    first = nh.run(ctx, positions)
    second = nh.run(ctx, positions)
    assert first["energy"] == second["energy"]
    np.testing.assert_array_equal(first["charges"], second["charges"])
    np.testing.assert_array_equal(first["gradient"], second["gradient"])
    np.testing.assert_array_equal(first["virial"], second["virial"])


@pytest.mark.parametrize("name", ["ammonium", "sulfate", "hexaaqua_iron"])
def test_gradient_matches_finite_differences_of_gpu_energies(name: str) -> None:
    numbers, positions, charge = _ion(name)
    ctx = nh.d4_context(numbers, PBE, charge=charge)
    analytic = nh.run(ctx, positions)["gradient"]

    def energy(x: np.ndarray) -> float:
        return float(nh.run(ctx, x, gradient=False, virial=False)["energy"])

    ok, errors = agrees_over_sweep(analytic, lambda h: position_gradient(energy, positions, h), 1e-10)
    assert ok, errors


def test_charges_sum_to_the_total_charge() -> None:
    numbers, positions, charge = _ion("sulfate")
    result = nh.run(nh.d4_context(numbers, PBE, charge=charge), positions)
    assert result["charges"].sum() == pytest.approx(charge, abs=1e-13)


def test_atom_energies_sum_to_the_energy() -> None:
    numbers, positions, charge = _ion("acetate")
    ctx = nh.d4_context(numbers, PBE, charge=charge)
    result = ctx.evaluate(positions, np.zeros((3, 3)), (False,) * 3, atom_energies=True)
    assert result["atom_energies"].sum() == pytest.approx(result["energy"], rel=1e-13)


def test_molecular_invariances() -> None:
    numbers, positions, charge = _ion("hexaaqua_iron")
    ctx = nh.d4_context(numbers, PBE, charge=charge)
    reference = nh.run(ctx, positions)
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = positions @ q.T + np.array([3.0, -1.0, 12.0])
    rotated = nh.run(ctx, moved)
    assert rotated["energy"] == pytest.approx(reference["energy"], rel=1e-13)
    np.testing.assert_allclose(rotated["charges"], reference["charges"], atol=1e-13)
    np.testing.assert_allclose(rotated["gradient"], reference["gradient"] @ q.T, atol=1e-14)
    order = rng.permutation(numbers.size)
    permuted = nh.run(nh.d4_context(numbers[order], PBE, charge=charge), positions[order])
    assert permuted["energy"] == pytest.approx(reference["energy"], rel=1e-13)
    np.testing.assert_allclose(permuted["charges"], reference["charges"][order], atol=1e-13)
    np.testing.assert_allclose(permuted["gradient"], reference["gradient"][order], atol=1e-14)


def test_set_charge_matches_a_fresh_context() -> None:
    numbers, positions, _ = _ion("ammonium")
    ctx = nh.d4_context(numbers, PBE, charge=0.0, skin=1.0)  # skin 0 rebuilds every call
    nh.run(ctx, positions)
    rebuilds = ctx.stats()["rebuilds"]
    ctx.set_charge(1.0)
    updated = nh.run(ctx, positions)
    fresh = nh.run(nh.d4_context(numbers, PBE, charge=1.0), positions)
    assert updated["energy"] == pytest.approx(fresh["energy"], rel=1e-15)
    np.testing.assert_array_equal(updated["charges"], fresh["charges"])
    assert ctx.stats()["rebuilds"] == rebuilds


def test_set_parameters_matches_a_fresh_context() -> None:
    numbers, positions, charge = _ion("acetate")
    other = dataclasses.replace(PBE, ga=2.0, gc=1.0, wf=4.0, s9=2.0)
    ctx = nh.d4_context(numbers, PBE, charge=charge)
    nh.run(ctx, positions)
    p = other
    cutoffs = D4Cutoffs()
    ctx.set_parameters(
        s6=p.s6, s8=p.s8, a1=p.a1, a2=p.a2, s9=p.s9, alp=p.alp, ga=p.ga, gc=p.gc, wf=p.wf,
        cutoff_cn=cutoffs.cn, cutoff_disp2=cutoffs.disp2, cutoff_disp3=cutoffs.disp3,
    )  # fmt: skip
    updated = nh.run(ctx, positions)
    fresh = nh.run(nh.d4_context(numbers, other, charge=charge), positions)
    assert updated["energy"] == pytest.approx(fresh["energy"], rel=1e-14)
    np.testing.assert_allclose(updated["gradient"], fresh["gradient"], rtol=0, atol=1e-16)


def test_inactive_terms_give_zero() -> None:
    inactive = dataclasses.replace(PBE, s6=0.0, s8=0.0, s9=0.0)
    ctx = nh.d4_context([1, 8, 1], inactive)
    result = nh.run(ctx, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.7, 0.0, 2.3]]))
    assert result["energy"] == 0.0
    np.testing.assert_array_equal(result["gradient"], 0.0)
    assert result["charges"].sum() == pytest.approx(0.0, abs=1e-15)
