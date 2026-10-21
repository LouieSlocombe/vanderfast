"""GPU pair D3(BJ): fixtures, intermediates, neighbour sets, and derivatives."""

import dataclasses

import numpy as np
import pytest
from support import native_helpers as nh
from support.fd import agrees_over_sweep, position_gradient
from support.fixtures import FixtureCase, all_cases

from vanderfast import _oracle, resolve_parameters
from vanderfast.data import parse_reference_tables
from vanderfast.errors import InputError, UnsupportedFeatureError
from vanderfast.parameters import Cutoffs
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE = resolve_parameters(method="pbe", damping="bj", atm=False)
PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)

PAIR_CASES = [c for c in all_cases() if not c.parameters.atm and c.known_issue is None]


@pytest.mark.parametrize("case", PAIR_CASES, ids=[c.id for c in PAIR_CASES])
def test_pair_cases_match_dftd3(case: FixtureCase) -> None:
    ctx = nh.context(case.numbers, case.parameters, case.cutoffs)
    result = nh.run(ctx, case.positions, case.cell, case.pbc)
    assert result["energy"] == pytest.approx(case.energy, rel=1e-12, abs=1e-14)
    np.testing.assert_allclose(result["gradient"], case.gradient, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["virial"], case.virial, rtol=0, atol=1e-12)
    assert result["energy_atm"] == 0.0


@pytest.mark.parametrize(
    "build",
    [structures.water_dimer, structures.benzene_dimer, structures.mixed_fragment],
)
def test_intermediates_match_the_oracle(build) -> None:  # type: ignore[no-untyped-def]
    atoms = build()
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE)
    debug = ctx.debug(positions, np.zeros((3, 3)), (False, False, False), derivatives=True)
    oracle = _oracle.evaluate(atoms.numbers, positions, parameters=PBE)
    np.testing.assert_allclose(debug["cn"], oracle.cn, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(debug["weights"][:, :7], oracle.weights, rtol=1e-12, atol=1e-15)
    assert oracle.dweights is not None
    np.testing.assert_allclose(debug["dweights"][:, :7], oracle.dweights, rtol=1e-11, atol=1e-14)
    assert oracle.g is not None
    np.testing.assert_allclose(debug["dEdcn"], oracle.g, rtol=1e-11, atol=1e-16)
    assert np.all(debug["weights"][:, 7] == 0.0)


@pytest.mark.parametrize("target", [10.0, 14.0, 16.5, 40.0])
def test_exceptional_weight_branches_match_the_oracle(target: float) -> None:
    from vanderfast.data import load_reference_tables  # noqa: PLC0415

    tables = load_reference_tables()
    rc = tables.rcov[1] + tables.rcov[6]
    if target < 18:  # CN = 18 c(R) for an H atom inside a shell of 18 carbon atoms
        radius = rc / (1.0 - np.log(18.0 / target - 1.0) / 16.0) / INV_BOHR
        atoms = structures.shell_cluster(1, 6, 18, radius)
    else:
        atoms = structures.shell_cluster(1, 6, 60, 0.6)
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE)
    debug = ctx.debug(positions, np.zeros((3, 3)), (False, False, False), derivatives=True)
    oracle = _oracle.evaluate(atoms.numbers, positions, parameters=PBE)
    assert oracle.dweights is not None
    np.testing.assert_array_equal(np.isfinite(debug["dweights"][:, :7]), True)
    np.testing.assert_allclose(debug["weights"][:, :7], oracle.weights, rtol=1e-12, atol=1e-14)
    # dw is a difference of O(wf * dCN) terms; compare on that absolute scale.
    np.testing.assert_allclose(debug["dweights"][:, :7], oracle.dweights, rtol=0, atol=1e-11)
    if 4.0 * (oracle.cn[0] - 0.9118) ** 2 > 745.0:  # all Gaussians underflow: fallback branch
        np.testing.assert_array_equal(debug["weights"][0, :2], [1.0, 0.0])
        np.testing.assert_array_equal(debug["dweights"][0, :2], [0.0, 0.0])


def test_tied_maximum_references_on_the_device() -> None:
    from vanderfast.data import load_reference_tables  # noqa: PLC0415

    carbon = list(load_reference_tables().reference_cn(6))
    carbon[3] = carbon[4]
    blob = nh.tables_with_reference_cn(6, carbon)
    tied = parse_reference_tables(blob)
    atoms = structures.shell_cluster(6, 6, 60, 0.6)
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE, reference_data=blob, reference_identity="tied")
    debug = ctx.debug(positions, np.zeros((3, 3)), (False, False, False), derivatives=True)
    expected, _ = _oracle.reference_weights(atoms.numbers, debug["cn"], tied)
    np.testing.assert_array_equal(debug["weights"][:, :7], expected)
    np.testing.assert_array_equal(debug["weights"][0, :5], [0.0, 0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("build", "cutoff"),
    [
        (structures.benzene_dimer, 40.0),
        (structures.rocksalt_primitive, 25.0),
        (structures.skewed_triclinic, 18.0),
        (structures.graphene_slab, 25.0),
        (structures.carbon_wire, 30.0),
        (lambda: structures.simple_cubic(6, 1.8), 20.0),
    ],
)
def test_neighbour_graph_equals_the_cpu_enumeration(build, cutoff: float) -> None:  # type: ignore[no-untyped-def]
    atoms = build()
    positions = atoms.positions * INV_BOHR
    cell = atoms.cell[:] * INV_BOHR if atoms.pbc.any() else np.zeros((3, 3))
    pbc = tuple(bool(p) for p in atoms.pbc)
    cutoffs = Cutoffs(cn=cutoff, disp2=cutoff, disp3=cutoff)
    ctx = nh.context(atoms.numbers, PBE, cutoffs)
    debug = ctx.debug(positions, cell, pbc, derivatives=False)
    cpu = nh.native().enumerate_edges(positions, cell, pbc, cutoff)
    gpu_edges = nh.edge_keys(debug["edge_i"], debug["edge_j"], debug["edge_shift"])
    cpu_edges = nh.edge_keys(cpu["i"], cpu["j"], cpu["shift"])
    assert len(gpu_edges) == len(debug["edge_i"])
    assert gpu_edges == cpu_edges
    for i, j, *n in gpu_edges:
        assert (j, i, *(-x for x in n)) in gpu_edges
    oracle = _oracle.directed_edges(
        _oracle.make_geometry(atoms.numbers, positions, cell, atoms.pbc), cutoff
    )
    assert len(oracle) == len(cpu_edges)


def test_energy_only_path_reproduces_the_energy_exactly() -> None:
    atoms = structures.mixed_fragment()
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE)
    full = nh.run(ctx, positions)
    energy_only = nh.run(ctx, positions, gradient=False, virial=False)
    assert energy_only["energy"] == full["energy"]
    assert energy_only["gradient"] is None
    assert energy_only["virial"] is None


def test_results_are_reproducible() -> None:
    atoms = structures.benzene_dimer()
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE)
    first = nh.run(ctx, positions)
    for _ in range(5):
        again = nh.run(ctx, positions)
        assert again["energy"] == first["energy"]
        np.testing.assert_allclose(again["gradient"], first["gradient"], rtol=1e-13, atol=1e-18)


def test_gradient_matches_finite_differences_of_gpu_energies() -> None:
    atoms = structures.water_dimer()
    positions = atoms.positions * INV_BOHR
    ctx = nh.context(atoms.numbers, PBE)
    analytic = nh.run(ctx, positions)["gradient"]

    def energy(x: np.ndarray) -> float:
        return float(nh.run(ctx, x, gradient=False, virial=False)["energy"])

    ok, errors = agrees_over_sweep(
        analytic, lambda h: position_gradient(energy, positions, h), 1e-10
    )
    assert ok, errors


def test_atm_is_rejected_until_implemented() -> None:
    ctx = nh.context([18, 18, 18], PBE_ATM)
    with pytest.raises(UnsupportedFeatureError, match="ATM"):
        nh.run(ctx, np.eye(3) * 6.0)


def test_inactive_terms_give_zero() -> None:
    inactive = dataclasses.replace(PBE, s6=0.0, s8=0.0)
    ctx = nh.context([1, 1], inactive)
    result = nh.run(ctx, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    assert result["energy"] == 0.0
    np.testing.assert_array_equal(result["gradient"], 0.0)


def test_coincident_sites_are_rejected() -> None:
    ctx = nh.context([1, 1], PBE)
    with pytest.raises(InputError, match="closer than"):
        nh.run(ctx, np.zeros((2, 3)))
