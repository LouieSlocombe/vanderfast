"""The Python D3Context wrapper (on the oracle-backed fake native module)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from support import oracle_native
from support.inputs import SHORT_D3

import vanderfast.context as context_module
from vanderfast import _oracle, resolve_parameters
from vanderfast.context import D3Context, evaluate, parse_device
from vanderfast.data import read_reference_blob
from vanderfast.errors import InputError, ParameterError, UnsupportedFeatureError
from vanderfast.testing import structures
from vanderfast.units import BOHR, HARTREE, INV_BOHR

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)


@pytest.fixture(autouse=True)
def fake_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_module, "load_native", lambda: oracle_native)


@pytest.mark.parametrize(("device", "index"), [("cuda", 0), ("cuda:3", 3), (2, 2)])
def test_parse_device(device: str | int, index: int) -> None:
    assert parse_device(device) == index


@pytest.mark.parametrize("device", ["cpu", "cuda:x", "cuda:", -1, True])
def test_parse_device_rejects_invalid(device: object) -> None:
    with pytest.raises(InputError):
        parse_device(device)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"parameters": {"s6": 1.0}}, ParameterError),
        ({"precision": "float32"}, UnsupportedFeatureError),
        ({"skin": -1.0}, InputError),
        ({"skin": float("nan")}, InputError),
        ({"numbers": [[1]]}, InputError),
        ({"numbers": [1.5]}, InputError),
        ({"numbers": [0]}, InputError),
        ({"cutoff": {"width3": 1.0}}, UnsupportedFeatureError),
    ],
)
def test_construction_validates_input(kwargs: dict[str, object], error: type[Exception]) -> None:
    arguments: dict[str, object] = {"numbers": [1, 8], "parameters": PBE_ATM, **kwargs}
    numbers = arguments.pop("numbers")
    parameters = arguments.pop("parameters")
    with pytest.raises(error):
        D3Context(numbers, parameters, **arguments)  # type: ignore[arg-type]


def test_energy_forces_and_strain_match_the_oracle_in_public_units() -> None:
    atoms = structures.skewed_triclinic()
    context = D3Context(atoms.numbers, PBE_ATM, cutoff=SHORT_D3)
    result = context.evaluate(
        atoms.positions,
        cell=atoms.cell[:],
        pbc=True,
        properties=("energy", "forces", "strain_derivative", "stress"),
    )
    reference = _oracle.evaluate(
        atoms.numbers,
        atoms.positions * INV_BOHR,
        parameters=PBE_ATM,
        cell=atoms.cell[:] * INV_BOHR,
        pbc=atoms.pbc,
        cutoffs=SHORT_D3,
    )
    assert reference.gradient is not None
    assert reference.strain_derivative is not None
    assert result.energy == reference.energy * HARTREE
    assert result.energy_pair + result.energy_atm == pytest.approx(result.energy, rel=1e-14)
    assert result.forces is not None
    np.testing.assert_array_equal(result.forces, reference.gradient * -(HARTREE / BOHR))
    assert result.strain_derivative is not None
    np.testing.assert_array_equal(result.strain_derivative, reference.strain_derivative * HARTREE)
    expected_stress = (reference.strain_derivative * (HARTREE / atoms.get_volume())).flat[
        [0, 4, 8, 5, 2, 1]
    ]
    assert result.stress is not None
    np.testing.assert_allclose(result.stress, expected_stress, rtol=1e-14)
    assert result.info["atm_accepted"] > 0


def test_energy_only_requests_return_no_derivatives() -> None:
    atoms = structures.water_dimer()
    result = D3Context(atoms.numbers, PBE_ATM).evaluate(atoms.positions)
    assert result.forces is None
    assert result.stress is None
    assert result.strain_derivative is None
    assert result.energy < 0.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"properties": ("energy", "hessian")}, InputError),
        ({"properties": ("stress",)}, UnsupportedFeatureError),
        ({"pbc": True}, InputError),
        ({"pbc": (True, False)}, InputError),
        ({"cell": np.eye(2)}, InputError),
        ({"positions": np.zeros((5, 3))}, InputError),
    ],
)
def test_evaluate_validates_requests(kwargs: dict[str, object], error: type[Exception]) -> None:
    atoms = structures.water_dimer()
    context = D3Context(atoms.numbers, PBE_ATM)
    arguments: dict[str, object] = {"positions": atoms.positions, **kwargs}
    positions = arguments.pop("positions")
    with pytest.raises(error):
        context.evaluate(positions, **arguments)  # type: ignore[arg-type]


def test_stress_requires_a_non_degenerate_cell() -> None:
    context = D3Context([18], PBE_ATM, cutoff=SHORT_D3)
    with pytest.raises(UnsupportedFeatureError):
        context.evaluate(
            [[0.0, 0.0, 0.0]], cell=np.diag([3.0, 3.0, 0.0]), pbc=True, properties=["stress"]
        )


def test_atomic_interface_uses_bohr_and_hartree() -> None:
    atoms = structures.water_dimer()
    context = D3Context(atoms.numbers, PBE_ATM)
    raw = context.evaluate_atomic(
        atoms.positions * INV_BOHR, gradient=True, virial=True, atom_energies=True
    )
    public = context.evaluate(atoms.positions)
    assert raw.energy * HARTREE == public.energy
    assert raw.gradient is not None
    assert raw.virial is not None
    assert raw.atom_energies is not None
    assert raw.atom_energies.sum() == pytest.approx(raw.energy, rel=1e-13)


def test_parameter_and_cutoff_updates_reach_the_native_context() -> None:
    atoms = structures.water_dimer()
    context = D3Context(atoms.numbers, PBE_ATM)
    before = context.evaluate(atoms.positions).energy
    context.set_parameters(resolve_parameters(method="pbe", damping="bj", atm=False))
    without_atm = context.evaluate(atoms.positions)
    assert without_atm.energy_atm == 0.0
    assert without_atm.energy != before
    context.set_cutoffs({"disp2": 5.0})
    native = context.native
    assert isinstance(native, oracle_native.Context)
    assert native.neighbor_invalidations == 1
    assert native.cutoffs.disp2 == 5.0 / BOHR
    context.invalidate_neighbors()
    assert native.neighbor_invalidations == 2
    with pytest.raises(ParameterError):
        context.set_parameters({"s6": 1.0})  # type: ignore[arg-type]


def test_diagnostics_close_and_context_manager() -> None:
    with D3Context([1, 1], PBE_ATM, device="cuda:0", skin=0.5) as context:
        report = context.diagnostics()
        assert report["device"] == "cuda:0"
        assert report["skin_angstrom"] == 0.5
        assert report["cutoffs_bohr"]["disp2"] == 60.0
        assert report["stats"] == {"evaluations": 0}
        assert context.n_atoms == 2
    with pytest.raises(InputError, match="closed"):
        context.evaluate([[0, 0, 0], [0, 0, 1]])


def test_replacement_reference_data_is_validated_and_identified(tmp_path: Path) -> None:
    blob = read_reference_blob()
    path = tmp_path / "tables.bin"
    path.write_bytes(blob)
    context = D3Context([1, 8], PBE_ATM, reference_data=path)
    native = context.native
    assert isinstance(native, oracle_native.Context)
    assert native.reference_identity == hashlib.sha256(blob).hexdigest()
    path.write_bytes(blob[:100])
    with pytest.raises(InputError):
        D3Context([1, 8], PBE_ATM, reference_data=path)


def test_debug_intermediates() -> None:
    atoms = structures.water_dimer()
    context = D3Context(atoms.numbers, PBE_ATM, skin=0.0)
    debug = context.debug_atomic(atoms.positions * INV_BOHR)
    assert debug["weights"].shape == (6, 8)
    assert debug["dEdcn"].shape == (6,)
    assert len(debug["edge_i"]) == 30
    energy_only = context.debug_atomic(atoms.positions * INV_BOHR, derivatives=False)
    assert energy_only["dweights"] is None


def test_convenience_evaluate_and_empty_systems() -> None:
    result = evaluate(np.zeros(0, dtype=int), np.zeros((0, 3)), PBE_ATM, properties=("forces",))
    assert result.energy == 0.0
    assert result.forces is not None
    assert result.forces.shape == (0, 3)
