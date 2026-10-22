"""The Python D4Context wrapper (on the oracle-backed fake native module)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from support import oracle_native

import vanderfast.context_d4 as context_module
from vanderfast import _oracle_d4
from vanderfast.context_d4 import D4Context, EEQSolverOptions, evaluate_d4, promote_pbc
from vanderfast.data.d4 import read_d4_reference_blob
from vanderfast.errors import InputError, ParameterError, UnsupportedFeatureError
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters, resolve_d4_parameters
from vanderfast.testing import structures
from vanderfast.units import BOHR, HARTREE

PBE = resolve_d4_parameters(method="pbe")
SHORT = D4Cutoffs(cn=15.0, disp2=20.0, disp3=12.0)


@pytest.fixture(autouse=True)
def fake_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_module, "load_native", lambda: oracle_native)


def _native(context: D4Context) -> oracle_native.D4Context:
    native = context.native
    assert isinstance(native, oracle_native.D4Context)
    return native


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"parameters": {"s8": 1.0}}, ParameterError),
        ({"precision": "float32"}, UnsupportedFeatureError),
        ({"skin": -1.0}, InputError),
        ({"numbers": [[1]]}, InputError),
        ({"numbers": [1.5]}, InputError),
        ({"numbers": [104]}, InputError),
        ({"charge": float("nan")}, InputError),
        ({"charge": True}, InputError),
        ({"ewald": "fast"}, InputError),
        ({"cutoff": {"width2": 1.0}}, UnsupportedFeatureError),
    ],
)
def test_construction_validates_input(kwargs: dict[str, object], error: type[Exception]) -> None:
    arguments: dict[str, object] = {"numbers": [1, 8], "parameters": PBE, **kwargs}
    numbers = arguments.pop("numbers")
    parameters = arguments.pop("parameters")
    with pytest.raises(error):
        D4Context(numbers, parameters, **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "cg"}, "solver method"),
        ({"tolerance": 2.0}, "tolerance"),
        ({"tolerance": "1e-12"}, "tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": True}, "max_iterations"),
    ],
)
def test_solver_options_are_validated(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(InputError, match=match):
        EEQSolverOptions(**kwargs)  # type: ignore[arg-type]


def test_solver_options_map_to_native_defaults() -> None:
    assert EEQSolverOptions().native_kwargs() == {
        "solver": "auto",
        "solver_tolerance": 0.0,
        "solver_max_iterations": 0,
        "warm_start": False,
    }
    custom = EEQSolverOptions(method="direct", tolerance=1e-12, max_iterations=50, warm_start=True)
    assert custom.native_kwargs()["solver_max_iterations"] == 50
    context = D4Context([1, 8], PBE, solver=custom)
    assert _native(context).solver == ("direct", 1e-12, 50, True)


def test_periodic_results_match_the_oracle_in_dftd4_units() -> None:
    atoms = structures.skewed_triclinic()
    context = D4Context(atoms.numbers, PBE, cutoff=SHORT, charge=0.5)
    result = context.evaluate(
        atoms.positions,
        cell=atoms.cell[:],
        pbc=True,
        properties=("energy", "forces", "strain_derivative", "stress"),
    )
    reference = _oracle_d4.evaluate(
        atoms.numbers,
        atoms.positions / BOHR,
        parameters=PBE,
        charge=0.5,
        cell=atoms.cell[:] / BOHR,
        pbc=atoms.pbc,
        cutoffs=SHORT,
    )
    assert reference.gradient is not None
    assert reference.strain_derivative is not None
    assert result.energy == reference.energy * HARTREE
    assert result.energy_pair + result.energy_atm == pytest.approx(result.energy, rel=1e-14)
    assert result.forces is not None
    np.testing.assert_array_equal(result.forces, -reference.gradient * HARTREE / BOHR)
    assert result.strain_derivative is not None
    np.testing.assert_array_equal(result.strain_derivative, reference.strain_derivative * HARTREE)
    volume = abs(np.linalg.det(atoms.cell[:]))
    assert result.stress is not None
    expected = (reference.strain_derivative * HARTREE / volume).flat[[0, 4, 8, 5, 2, 1]]
    np.testing.assert_array_equal(result.stress, expected)
    np.testing.assert_array_equal(result.charges, reference.charges)
    assert result.charges.sum() == pytest.approx(0.5, abs=1e-13)
    assert result.info["ewald_mode"] == "upstream"
    assert result.info["pbc_input"] == (True, True, True)


def test_partial_periodicity_is_promoted_and_gives_stress() -> None:
    slab = structures.graphene_slab()
    context = D4Context(slab.numbers, PBE, cutoff=SHORT)
    partial = context.evaluate(
        slab.positions, cell=slab.cell[:], pbc=slab.pbc, properties=("stress",)
    )
    full = context.evaluate(slab.positions, cell=slab.cell[:], pbc=True, properties=("stress",))
    assert partial.energy == full.energy
    assert partial.stress is not None
    assert full.stress is not None
    np.testing.assert_array_equal(partial.stress, full.stress)
    assert partial.info["pbc_input"] == (True, True, False)


def test_degenerate_cells_and_missing_cells_are_rejected() -> None:
    flat = np.diag([3.0, 3.0, 0.0])
    with pytest.raises(InputError, match="vacuum lattice vector"):
        promote_pbc((True, True, False), flat)
    with pytest.raises(InputError, match="cell is required"):
        promote_pbc(True, None)
    flags, cell = promote_pbc(False, None)
    assert flags == (False, False, False)
    assert not cell.any()


def test_molecules_report_no_ewald_and_refuse_stress() -> None:
    water = structures.water_dimer()
    context = D4Context(water.numbers, PBE)
    result = context.evaluate(water.positions, properties=("energy", "forces"))
    assert result.info["ewald_mode"] == "none"
    assert result.stress is None
    assert result.strain_derivative is None
    with pytest.raises(InputError, match="periodic"):
        context.evaluate(water.positions, properties=("stress",))
    with pytest.raises(InputError, match="unknown properties"):
        context.evaluate(water.positions, properties=("hessian",))


def test_converged_ewald_reaches_the_native_layer() -> None:
    crystal = structures.rocksalt_primitive()
    context = D4Context(crystal.numbers, PBE, cutoff=SHORT, ewald="converged")
    result = context.evaluate(crystal.positions, cell=crystal.cell[:], pbc=True)
    assert result.info["ewald_mode"] == "converged"
    assert _native(context).ewald == "converged"


def test_state_updates_reach_the_native_layer() -> None:
    water = structures.water_dimer()
    context = D4Context(water.numbers, PBE)
    native = _native(context)
    before = context.evaluate(water.positions).energy
    context.set_charge(-1.0)
    assert native.charge == -1.0
    assert context.evaluate(water.positions).energy != before
    custom = D4Parameters(s8=1.0, a1=0.4, a2=5.0, ga=2.0, gc=1.0)
    context.set_parameters(custom)
    assert native.parameters == custom
    with pytest.raises(ParameterError):
        context.set_parameters({"s8": 1.0})  # type: ignore[arg-type]
    with pytest.raises(InputError):
        context.set_charge(float("inf"))
    context.set_cutoffs({"disp2": 10.0})
    assert native.cutoffs.disp2 == 10.0 / BOHR
    assert native.neighbor_invalidations == 1
    context.invalidate_neighbors()
    assert native.neighbor_invalidations == 2


def test_atomic_interface_and_debug_intermediates() -> None:
    water = structures.water_dimer()
    context = D4Context(water.numbers, PBE)
    positions = water.positions / BOHR
    raw = context.evaluate_atomic(positions, gradient=True, virial=True, atom_energies=True)
    assert raw.gradient is not None
    assert raw.atom_energies is not None
    assert raw.atom_energies.sum() == pytest.approx(raw.energy, rel=1e-14)
    debug = context.debug_atomic(positions)
    assert debug["weights"].shape == (6, 8)
    assert debug["charges"] == pytest.approx(raw.charges)
    assert debug["dEdq"].shape == (6,)
    with pytest.raises(InputError, match="shape"):
        context.evaluate_atomic(positions[:2])


def test_diagnostics_lifecycle_and_convenience_function() -> None:
    water = structures.water_dimer()
    with D4Context(water.numbers, PBE, charge=1.0, device="cuda:1") as context:
        report = context.diagnostics()
        assert report["charge"] == 1.0
        assert report["device"] == "cuda:1"
        assert report["ewald"] == "upstream"
        assert report["stats"]["evaluations"] == 0
        assert context.n_atoms == 6
    with pytest.raises(InputError, match="closed"):
        context.native  # noqa: B018
    result = evaluate_d4(water.numbers, water.positions, PBE, charge=-1.0, properties=("forces",))
    assert result.forces is not None
    assert result.charges.sum() == pytest.approx(-1.0, abs=1e-13)


def test_replacement_reference_data_is_validated_and_identified(tmp_path: Path) -> None:
    blob = read_d4_reference_blob()
    path = tmp_path / "tables.bin"
    path.write_bytes(blob)
    context = D4Context([1, 8], PBE, reference_data=path)
    assert _native(context).reference_identity == hashlib.sha256(blob).hexdigest()
    path.write_bytes(b"junk")
    with pytest.raises(InputError):
        D4Context([1, 8], PBE, reference_data=path)
