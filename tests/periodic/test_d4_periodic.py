"""Periodic GPU D4: both Ewald modes, derivatives, partial periodicity, and the ASE path."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from support import native_helpers as nh
from support.fd import agrees_over_sweep, position_gradient, strain_derivative
from support.inputs import SHORT_D4

from vanderfast import _oracle_d4
from vanderfast.ase import D4Calculator
from vanderfast.context_d4 import EwaldMode
from vanderfast.parameters_d4 import resolve_d4_parameters
from vanderfast.testing import structures
from vanderfast.units import BOHR, INV_BOHR

pytestmark = pytest.mark.cuda

PBE = resolve_d4_parameters(method="pbe")
PBE_PAIR = resolve_d4_parameters(method="pbe", atm=False)
SHORT_D4_ANGSTROM = {key: getattr(SHORT_D4, key) * BOHR for key in SHORT_D4.KEYS}
FULL = (True, True, True)
EWALD: list[EwaldMode] = ["upstream", "converged"]


def _rattled(
    build: Callable[[], Atoms], seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms = build()
    rng = np.random.default_rng(seed)
    positions = atoms.positions * INV_BOHR + rng.normal(scale=0.05, size=atoms.positions.shape)
    return atoms.numbers, positions, atoms.cell[:] * INV_BOHR


@pytest.mark.parametrize("ewald", EWALD)
@pytest.mark.parametrize("name", sorted(structures.IONIC))
def test_gpu_matches_the_oracle(name: str, ewald: EwaldMode) -> None:
    numbers, positions, cell = _rattled(structures.IONIC[name])
    result = nh.run(nh.d4_context(numbers, PBE, SHORT_D4, ewald=ewald), positions, cell, FULL)
    oracle = _oracle_d4.evaluate(
        numbers,
        positions,
        parameters=PBE,
        cell=cell,
        pbc=FULL,
        cutoffs=SHORT_D4,
        ewald=ewald,
        gradient=True,
    )
    assert oracle.gradient is not None
    assert oracle.strain_derivative is not None
    assert result["energy"] == pytest.approx(oracle.energy, rel=1e-13)
    np.testing.assert_allclose(result["charges"], oracle.charges, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["gradient"], oracle.gradient, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["virial"], oracle.strain_derivative, rtol=0, atol=1e-14)
    assert result["info"]["ewald_alpha"] == pytest.approx(oracle.ewald_alpha, rel=1e-15)
    assert result["info"]["ewald_mode"] == ewald
    if ewald == "upstream":
        assert (result["info"]["ewald_translations"], result["info"]["ewald_reciprocal"]) == (
            125,
            124,
        )


@pytest.mark.parametrize("ewald", EWALD)
@pytest.mark.parametrize("name", ["mgo_primitive", "caf2_primitive", "elongated_cell"])
def test_forces_and_strain_match_finite_differences(name: str, ewald: EwaldMode) -> None:
    numbers, positions, cell = _rattled(structures.IONIC[name])
    ctx = nh.d4_context(numbers, PBE, SHORT_D4, ewald=ewald)
    result = nh.run(ctx, positions, cell, FULL)

    def energy(x: np.ndarray, h: np.ndarray = cell) -> float:
        return float(nh.run(ctx, x, h, FULL, gradient=False, virial=False)["energy"])

    ok, errors = agrees_over_sweep(
        result["gradient"], lambda step: position_gradient(energy, positions, step), 1e-10
    )
    assert ok, errors
    ok, errors = agrees_over_sweep(
        result["virial"], lambda step: strain_derivative(energy, positions, cell, step), 1e-9
    )
    assert ok, errors


@pytest.mark.parametrize("name", sorted(structures.PARTIAL))
def test_partial_periodicity_is_fully_periodic(name: str) -> None:
    atoms = structures.PARTIAL[name]()
    positions, cell = atoms.positions * INV_BOHR, atoms.cell[:] * INV_BOHR
    ctx = nh.d4_context(atoms.numbers, PBE, SHORT_D4)
    partial = nh.run(ctx, positions, cell, tuple(bool(p) for p in atoms.pbc))
    full = nh.run(ctx, positions, cell, FULL)
    assert partial["energy"] == full["energy"]
    np.testing.assert_array_equal(partial["charges"], full["charges"])
    np.testing.assert_array_equal(partial["gradient"], full["gradient"])
    np.testing.assert_array_equal(partial["virial"], full["virial"])


@pytest.mark.parametrize("ewald", EWALD)
def test_lattice_translations_of_atoms_change_nothing(ewald: EwaldMode) -> None:
    numbers, positions, cell = _rattled(structures.magnesia_conventional)
    ctx = nh.d4_context(numbers, PBE, SHORT_D4, ewald=ewald)
    reference = nh.run(ctx, positions, cell, FULL)
    shifted = positions.copy()
    shifted[0] += cell[0] - 2.0 * cell[2]
    shifted[3] -= cell[1]
    moved = nh.run(ctx, shifted, cell, FULL)
    assert moved["energy"] == pytest.approx(reference["energy"], rel=1e-13)
    np.testing.assert_allclose(moved["charges"], reference["charges"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(moved["gradient"], reference["gradient"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(moved["virial"], reference["virial"], rtol=0, atol=1e-14)


def test_converged_ewald_is_supercell_consistent() -> None:
    crystal = structures.magnesia_primitive()
    big = crystal.repeat((2, 2, 2))
    results: list[dict[str, Any]] = []
    for atoms in (crystal, big):
        ctx = nh.d4_context(atoms.numbers, PBE_PAIR, SHORT_D4, ewald="converged")
        results.append(nh.run(ctx, atoms.positions * INV_BOHR, atoms.cell[:] * INV_BOHR, FULL))
    primitive, supercell = results
    assert supercell["energy"] / 8 == pytest.approx(primitive["energy"], rel=1e-12)
    np.testing.assert_allclose(
        supercell["charges"].reshape(8, -1), np.tile(primitive["charges"], (8, 1)), atol=1e-13
    )
    np.testing.assert_allclose(supercell["virial"] / 8, primitive["virial"], atol=1e-14)


def test_upstream_ewald_reproduces_the_reference_cell_dependence() -> None:
    crystal = structures.magnesia_primitive()
    big = crystal.repeat((2, 2, 2))
    charges = [
        nh.run(
            nh.d4_context(atoms.numbers, PBE_PAIR, SHORT_D4),
            atoms.positions * INV_BOHR,
            atoms.cell[:] * INV_BOHR,
            FULL,
            gradient=False,
            virial=False,
        )["charges"]
        for atoms in (crystal, big)
    ]
    spread = np.abs(charges[1].reshape(8, -1) - charges[0]).max()
    assert 1e-4 < spread < 1e-2


@pytest.mark.reference_d4
def test_calculator_matches_the_reference_backend_with_stress() -> None:
    atoms = structures.fluorite()
    atoms.rattle(stdev=0.03, seed=5)
    atoms.set_initial_charges(np.zeros(len(atoms)))
    values = {}
    for backend in ("cuda", "reference"):
        probe = atoms.copy()
        probe.calc = D4Calculator(method="pbe", backend=backend, cutoff=SHORT_D4_ANGSTROM)
        values[backend] = (
            probe.get_potential_energy(),
            probe.get_forces(),
            probe.get_stress(),
            probe.calc.get_property("charges", probe),
        )
    (e, f, s, q), (e_ref, f_ref, s_ref, q_ref) = values["cuda"], values["reference"]
    assert e == pytest.approx(e_ref, rel=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(s, s_ref, rtol=0, atol=1e-11)
    np.testing.assert_allclose(q, q_ref, rtol=0, atol=1e-12)
