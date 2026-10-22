"""The D4 ASE calculator (on the oracle-backed fake native module)."""

import json

import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from support import oracle_native

import vanderfast.context_d4 as context_module
from vanderfast.ase import D4Calculator
from vanderfast.context_d4 import D4Context, EEQSolverOptions
from vanderfast.errors import InputError, ParameterError, UnsupportedFeatureError
from vanderfast.parameters_d4 import D4Parameters
from vanderfast.testing import structures

SHORT = {"cn": 8.0, "disp2": 10.0, "disp3": 6.0}


@pytest.fixture(autouse=True)
def fake_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_module, "load_native", lambda: oracle_native)


def _context(calc: D4Calculator) -> D4Context:
    context = calc._context  # noqa: SLF001
    assert context is not None
    return context


def _native(calc: D4Calculator) -> oracle_native.D4Context:
    native = _context(calc).native
    assert isinstance(native, oracle_native.D4Context)
    return native


def test_energy_forces_stress_and_charges_match_the_context() -> None:
    atoms = structures.rocksalt_primitive()
    calc = D4Calculator(method="pbe", cutoff=SHORT)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    charges = atoms.get_charges()
    direct = D4Context(atoms.numbers, calc.resolved_parameters, cutoff=SHORT).evaluate(
        atoms.positions, cell=atoms.cell[:], pbc=True, properties=("forces", "stress")
    )
    assert energy == direct.energy
    np.testing.assert_array_equal(forces, direct.forces)
    np.testing.assert_array_equal(stress, direct.stress)
    np.testing.assert_array_equal(charges, direct.charges)
    assert calc.resolved_parameters.s9 == 1.0
    assert calc.evaluations == 2  # forces and stress came from one evaluation


def test_total_charge_follows_initial_charges_without_rebuilding() -> None:
    atoms = structures.water_dimer()
    calc = D4Calculator(method="pbe")
    atoms.calc = calc
    neutral = atoms.get_potential_energy()
    context = _context(calc)
    atoms.set_initial_charges([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    anion = atoms.get_potential_energy()
    assert anion != neutral
    assert _context(calc) is context
    assert _native(calc).charge == -1.0
    assert atoms.get_charges().sum() == pytest.approx(-1.0, abs=1e-13)
    atoms.set_initial_magnetic_moments([1.0] * 6)
    assert calc.check_state(atoms) == []


def test_explicit_charge_overrides_initial_charges() -> None:
    atoms = structures.water_dimer()
    atoms.set_initial_charges([1.0] * 6)
    calc = D4Calculator(method="pbe", charge=0.0)
    atoms.calc = calc
    atoms.get_potential_energy()
    assert _native(calc).charge == 0.0
    atoms.set_initial_charges([2.0] * 6)
    assert calc.check_state(atoms) == []
    assert calc.total_charge(atoms) == 0.0


def test_partial_periodicity_gives_stress_like_dftd4() -> None:
    slab = structures.graphene_slab()
    slab.calc = D4Calculator(method="pbe", cutoff=SHORT)
    assert slab.get_stress().shape == (6,)
    molecule = structures.water_dimer()
    molecule.calc = D4Calculator(method="pbe")
    with pytest.raises(PropertyNotImplementedError, match="periodic"):
        molecule.get_stress()


def test_parameter_changes_update_the_context_in_place() -> None:
    atoms = structures.water_dimer()
    calc = D4Calculator(method="pbe")
    atoms.calc = calc
    atoms.get_potential_energy()
    context = _context(calc)
    calc.set(atm=False)
    assert _context(calc) is context
    assert _native(calc).parameters.s9 == 0.0
    calc.set(cutoff={"disp2": 12.0})
    assert _native(calc).neighbor_invalidations == 1
    custom = D4Parameters(s8=1.0, a1=0.4, a2=5.0)
    calc.set(method=None, atm=None, parameters=custom)
    assert _native(calc).parameters == custom
    calc.set(ewald="converged")
    assert calc._context is None  # noqa: SLF001
    assert calc.set() == {}


def test_new_numbers_create_a_new_context() -> None:
    calc = D4Calculator(method="pbe")
    water = structures.water_dimer()
    water.calc = calc
    water.get_potential_energy()
    first = _context(calc)
    benzene = structures.benzene_dimer()
    benzene.calc = calc
    benzene.get_potential_energy()
    assert _context(calc) is not first


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"method": "pbe", "backend": "cpu"}, InputError),
        ({"method": "pbe", "ewald": "pme"}, InputError),
        ({"method": "pbe", "backend": "reference", "ewald": "converged"}, UnsupportedFeatureError),
        ({"method": "pbe", "device": "gpu"}, InputError),
        ({}, ParameterError),
        ({"parameters": D4Parameters(s8=1.0, a1=0.4, a2=5.0), "atm": True}, ParameterError),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        D4Calculator(**kwargs)  # type: ignore[arg-type]


def test_todict_is_json_serialisable() -> None:
    calc = D4Calculator(
        parameters=D4Parameters(s8=1.0, a1=0.4, a2=5.0),
        solver=EEQSolverOptions(method="direct"),
        charge=1.0,
    )
    document = calc.todict()
    json.dumps(document)
    assert document["d4_parameters"]["s8"] == 1.0
    assert document["solver"]["solver"] == "direct"
    assert document["charge"] == 1.0
    assert "d4_parameters" not in D4Calculator(method="pbe").todict()


@pytest.mark.reference_d4
def test_reference_backend_matches_dftd4_ase() -> None:
    dftd4_ase = pytest.importorskip("dftd4.ase")
    atoms = structures.skewed_triclinic()
    atoms.set_initial_charges([0.5, 0.0, 0.0, 0.0])
    ours = D4Calculator(method="pbe", backend="reference")
    upstream = dftd4_ase.DFTD4(method="pbe")
    atoms.calc = ours
    energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
    charges = atoms.get_charges()
    atoms.calc = upstream
    assert energy == atoms.get_potential_energy()
    np.testing.assert_array_equal(forces, atoms.get_forces())
    np.testing.assert_array_equal(stress, atoms.get_stress())
    assert charges.sum() == pytest.approx(0.5, abs=1e-13)
    molecule = structures.water_dimer()
    molecule.calc = D4Calculator(method="pbe", backend="reference")
    assert molecule.get_potential_energy() < 0.0
