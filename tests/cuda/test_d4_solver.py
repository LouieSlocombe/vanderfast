"""D4 charge solvers and optimised EEQ kernels: agreement, determinism, and diagnostics."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from support import native_helpers as nh
from support.fixtures_d4 import D4FixtureCase, load_family
from support.tolerances import D4_GPU_CHARGE_ABS, d4_gpu_energy_tolerance

from vanderfast.errors import ChargeSolverError, DeviceOutOfMemoryError
from vanderfast.parameters_d4 import D4Cutoffs, resolve_d4_parameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE = resolve_d4_parameters(method="pbe")
PBE_PAIR = resolve_d4_parameters(method="pbe", atm=False)
SHORT = D4Cutoffs(cn=20.0, disp2=25.0, disp3=15.0)
REFINE_TARGET = 2.0**-48


def _cluster(build: Callable[[], Atoms], copies: int, spacing: float = 9.0) -> Atoms:
    """Randomly oriented copies of a molecule on a cubic grid (Å)."""
    rng = np.random.default_rng(1)
    atoms = Atoms()
    side = int(np.ceil(copies ** (1 / 3)))
    for placed, index in enumerate(np.ndindex(side, side, side)):
        if placed == copies:
            break
        molecule = build()
        molecule.rotate(float(rng.uniform(0, 360)), rng.normal(size=3))
        molecule.translate(np.array(index) * spacing)
        atoms += molecule
    return atoms


def _system(name: str) -> tuple[Atoms, float, bool]:
    if name == "water_cluster_48":
        atoms, charge, periodic = _cluster(structures.water_dimer, 8), 0.0, False
    elif name == "acetate_cluster_189":
        atoms, charge, periodic = _cluster(structures.acetate, 27), -27.0, False
    elif name == "acetate_cluster_819":
        atoms, charge, periodic = _cluster(structures.acetate, 117), -117.0, False
    elif name == "mgo_216":
        atoms, charge, periodic = structures.magnesia_conventional().repeat((3, 3, 3)), 0.0, True
    elif name == "mgo_1000":
        atoms, charge, periodic = structures.magnesia_conventional().repeat((5, 5, 5)), 0.0, True
    else:
        atoms, charge, periodic = structures.fluorite().repeat((4, 4, 4)), 0.0, True
    atoms.rattle(stdev=0.05, seed=2)
    return atoms, charge, periodic


def _evaluate(
    atoms: Atoms,
    charge: float,
    periodic: bool,
    parameters: Any = PBE,
    *,
    gradient: bool = True,
    **options: Any,
) -> dict[str, Any]:
    ctx = nh.d4_context(atoms.numbers, parameters, SHORT, charge=charge, **options)
    cell = atoms.cell[:] * INV_BOHR if periodic else None
    return nh.run(
        ctx, atoms.positions * INV_BOHR, cell, (periodic,) * 3, gradient=gradient, virial=gradient
    )


CASES = [
    ("water_cluster_48", "upstream"),
    ("acetate_cluster_189", "upstream"),
    ("mgo_216", "upstream"),
    ("mgo_216", "converged"),
    ("fluorite_192", "upstream"),
    ("fluorite_192", "converged"),
]


@pytest.mark.parametrize(("name", "ewald"), CASES)
def test_optimised_kernels_match_the_literal_path(name: str, ewald: str) -> None:
    atoms, charge, periodic = _system(name)
    literal = _evaluate(atoms, charge, periodic, kernels="literal", solver="direct", ewald=ewald)
    for solver in ("auto", "direct"):
        result = _evaluate(atoms, charge, periodic, solver=solver, ewald=ewald)
        assert result["energy"] == pytest.approx(literal["energy"], rel=1e-13)
        np.testing.assert_allclose(result["charges"], literal["charges"], rtol=0, atol=2e-12)
        np.testing.assert_allclose(result["gradient"], literal["gradient"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["virial"], literal["virial"], rtol=0, atol=1e-13)


def test_solvers_agree_beyond_the_shared_memory_limit() -> None:
    atoms, charge, periodic = _system("acetate_cluster_189")
    results = {
        s: _evaluate(atoms, charge, periodic, solver=s) for s in ("auto", "iterative", "direct")
    }
    assert results["auto"]["info"]["eeq_solver"] == "minres"
    assert results["direct"]["info"]["eeq_solver"] == "direct"
    for solver in ("iterative", "direct"):
        np.testing.assert_allclose(
            results[solver]["charges"], results["auto"]["charges"], atol=1e-12
        )
        assert results[solver]["energy"] == pytest.approx(results["auto"]["energy"], rel=1e-13)
        np.testing.assert_allclose(
            results[solver]["gradient"], results["auto"]["gradient"], atol=1e-14
        )


@pytest.mark.parametrize("name", ["acetate_cluster_819", "mgo_1000"])
def test_packed_minres_matches_the_literal_path(name: str) -> None:
    """Large systems store A as packed lower-triangle 32 x 32 tiles."""
    atoms, charge, periodic = _system(name)
    literal = _evaluate(atoms, charge, periodic, PBE_PAIR, kernels="literal", solver="direct")
    packed = _evaluate(atoms, charge, periodic, PBE_PAIR)
    rows = (len(atoms) + 31) // 32
    assert packed["info"]["eeq_solver"] == "minres"
    assert packed["info"]["eeq_matrix_bytes"] == rows * (rows + 1) // 2 * 32 * 32 * 8
    assert literal["info"]["eeq_matrix_bytes"] == (len(atoms) + 1) ** 2 * 8
    assert packed["energy"] == pytest.approx(literal["energy"], rel=1e-13)
    np.testing.assert_allclose(packed["charges"], literal["charges"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(packed["gradient"], literal["gradient"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(packed["virial"], literal["virial"], rtol=0, atol=1e-13)


def _context(atoms: Atoms, charge: float, **options: Any) -> Any:
    return nh.d4_context(atoms.numbers, PBE, SHORT, charge=charge, **options)


def _run(ctx: Any, atoms: Atoms, periodic: bool) -> dict[str, Any]:
    cell = atoms.cell[:] * INV_BOHR if periodic else None
    return nh.run(ctx, atoms.positions * INV_BOHR, cell, (periodic,) * 3)


@pytest.mark.parametrize(
    ("name", "ewald"), [("acetate_cluster_189", "upstream"), ("mgo_216", "converged")]
)
def test_packed_storage_matches_dense_storage(name: str, ewald: str) -> None:
    atoms, charge, periodic = _system(name)
    dense = _run(_context(atoms, charge, ewald=ewald), atoms, periodic)
    ctx = _context(atoms, charge, ewald=ewald)
    ctx._testing_set_packed_min_atoms(1)  # noqa: SLF001
    packed = _run(ctx, atoms, periodic)
    rows = (len(atoms) + 31) // 32
    assert packed["info"]["eeq_matrix_bytes"] == rows * (rows + 1) // 2 * 32 * 32 * 8
    assert dense["info"]["eeq_matrix_bytes"] == (len(atoms) + 1) ** 2 * 8
    assert packed["energy"] == pytest.approx(dense["energy"], rel=1e-13)
    np.testing.assert_allclose(packed["charges"], dense["charges"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(packed["gradient"], dense["gradient"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(packed["virial"], dense["virial"], rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    ("name", "ewald"), [("mgo_216", "upstream"), ("fluorite_192", "converged")]
)
def test_recomputed_lattice_derivatives_match_stored_ones(name: str, ewald: str) -> None:
    """Without room for the pair derivatives the reverse pass repeats the lattice sums."""
    atoms, charge, periodic = _system(name)
    stored = _run(_context(atoms, charge, ewald=ewald), atoms, periodic)
    ctx = _context(atoms, charge, ewald=ewald)
    ctx._testing_set_matrix_limit((len(atoms) + 1) ** 2 * 8)  # noqa: SLF001
    recomputed = _run(ctx, atoms, periodic)
    # The two kernel variants may contract the lattice sums into FMAs differently.
    assert recomputed["energy"] == pytest.approx(stored["energy"], rel=1e-14)
    np.testing.assert_allclose(recomputed["charges"], stored["charges"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(recomputed["gradient"], stored["gradient"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(recomputed["virial"], stored["virial"], rtol=0, atol=1e-14)


def test_matrix_limit_raises_and_leaves_the_context_usable() -> None:
    atoms, charge, periodic = _system("acetate_cluster_189")
    ctx = _context(atoms, charge)
    ctx._testing_set_matrix_limit(1024)  # noqa: SLF001
    with pytest.raises(DeviceOutOfMemoryError, match="EEQ matrix of 189 atoms"):
        _run(ctx, atoms, periodic)
    ctx._testing_set_matrix_limit(0)  # noqa: SLF001
    assert np.isfinite(_run(ctx, atoms, periodic)["energy"])


@pytest.mark.parametrize("name", ["acetate_cluster_189", "mgo_216", "acetate_cluster_819"])
def test_pair_only_results_are_bitwise_reproducible(name: str) -> None:
    """MINRES and the tiled reverse use fixed-order reductions (ATM still uses atomics)."""
    atoms, charge, periodic = _system(name)
    ctx = nh.d4_context(atoms.numbers, PBE_PAIR, SHORT, charge=charge)
    cell = atoms.cell[:] * INV_BOHR if periodic else None
    x = atoms.positions * INV_BOHR
    first = nh.run(ctx, x, cell, (periodic,) * 3)
    again = nh.run(ctx, x, cell, (periodic,) * 3)
    fresh = _evaluate(atoms, charge, periodic, PBE_PAIR)
    for other in (again, fresh):
        assert other["energy"] == first["energy"]
        for key in ("charges", "gradient", "virial"):
            np.testing.assert_array_equal(other[key], first[key])


def test_minres_reports_its_diagnostics() -> None:
    atoms, charge, periodic = _system("mgo_216")
    full = _evaluate(atoms, charge, periodic)
    info = full["info"]
    assert info["eeq_solver"] == "minres"
    assert 0 < info["eeq_iterations"] < 200
    assert 0 < info["eeq_adjoint_iterations"] < 200
    assert info["eeq_restarts"] >= 0
    assert 0.0 <= info["eeq_backward_error"] <= REFINE_TARGET
    assert info["eeq_warm_started"] is False
    assert info["eeq_matrix_bytes"] == 217 * 217 * 8
    energy_only = _evaluate(atoms, charge, periodic, gradient=False)
    assert energy_only["info"]["eeq_adjoint_iterations"] == 0
    small = _evaluate(*_system("water_cluster_48"))
    assert small["info"]["eeq_solver"] == "direct"
    assert small["info"]["eeq_iterations"] == 0


def test_warm_start_reuses_the_previous_solution() -> None:
    atoms, charge, _ = _system("acetate_cluster_189")
    ctx = nh.d4_context(atoms.numbers, PBE, SHORT, charge=charge, warm_start=True)
    x = atoms.positions * INV_BOHR
    cold = nh.run(ctx, x)
    moved = x + np.random.default_rng(4).normal(scale=1e-3, size=x.shape)
    warm = nh.run(ctx, moved)
    reference = _evaluate_positions(atoms, charge, moved)
    assert cold["info"]["eeq_warm_started"] is False
    assert warm["info"]["eeq_warm_started"] is True
    assert warm["info"]["eeq_iterations"] < cold["info"]["eeq_iterations"]
    np.testing.assert_allclose(warm["charges"], reference["charges"], atol=1e-12)
    assert warm["energy"] == pytest.approx(reference["energy"], rel=1e-13)


def _evaluate_positions(atoms: Atoms, charge: float, positions: np.ndarray) -> dict[str, Any]:
    return nh.run(nh.d4_context(atoms.numbers, PBE, SHORT, charge=charge), positions)


def test_iteration_cap_raises_a_charge_solver_error() -> None:
    atoms, charge, periodic = _system("acetate_cluster_189")
    with pytest.raises(ChargeSolverError, match="did not converge"):
        _evaluate(atoms, charge, periodic, solver="iterative", solver_max_iterations=3)


def _stress_cases() -> list[D4FixtureCase]:
    return [*load_family("eeq_stress"), *load_family("ions"), *load_family("periodic")[:8]]


@pytest.mark.parametrize("case", _stress_cases(), ids=lambda case: case.id)
def test_minres_matches_dftd4_fixtures(case: D4FixtureCase) -> None:
    """Including the dense hydrogen cluster whose EEQ matrix is indefinite."""
    ctx = nh.d4_context(
        case.numbers, case.parameters, case.cutoffs, charge=case.charge, solver="iterative"
    )
    result = nh.run(ctx, case.positions, case.cell, case.pbc, gradient=case.gradient is not None)
    assert result["info"]["eeq_solver"] == "minres"
    assert abs(result["energy"] - case.energy) <= d4_gpu_energy_tolerance(case.energy)
    np.testing.assert_allclose(result["charges"], case.charges, rtol=0, atol=D4_GPU_CHARGE_ABS)
