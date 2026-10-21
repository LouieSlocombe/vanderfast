"""The ASE calculator (on the oracle-backed fake native module)."""

import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from support import oracle_native

import vanderfast.context as context_module
from vanderfast import D3BJParameters, resolve_parameters
from vanderfast.ase import D3Calculator
from vanderfast.context import D3Context
from vanderfast.errors import InputError, ParameterError
from vanderfast.testing import structures

SHORT = {"cn": 8.0, "disp2": 10.0, "disp3": 8.0}


@pytest.fixture(autouse=True)
def fake_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_module, "load_native", lambda: oracle_native)


def _native(calc: D3Calculator) -> oracle_native.Context:
    context = calc._context  # noqa: SLF001
    assert context is not None
    native = context.native
    assert isinstance(native, oracle_native.Context)
    return native


def test_energy_forces_and_stress_match_the_context() -> None:
    atoms = structures.rocksalt_primitive()
    calc = D3Calculator(method="pbe", damping="bj", atm=True, cutoff=SHORT)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    direct = D3Context(atoms.numbers, calc.resolved_parameters, cutoff=SHORT).evaluate(
        atoms.positions, cell=atoms.cell[:], pbc=True, properties=("forces", "stress")
    )
    assert energy == direct.energy
    assert atoms.get_potential_energy(force_consistent=True) == energy
    assert direct.forces is not None
    assert direct.stress is not None
    np.testing.assert_array_equal(forces, direct.forces)
    np.testing.assert_array_equal(stress, direct.stress)


def test_results_are_cached_and_charges_do_not_invalidate() -> None:
    atoms = structures.water_dimer()
    atoms.calc = D3Calculator(method="pbe", damping="bj", atm=False)
    atoms.get_forces()
    native = _native(atoms.calc)
    count = native.evaluations
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.set_initial_charges(np.ones(len(atoms)))
    atoms.get_forces()
    assert native.evaluations == count
    atoms.positions[0, 0] += 0.01
    atoms.get_forces()
    assert native.evaluations == count + 1


def test_stress_is_not_implemented_without_three_dimensional_pbc() -> None:
    atoms = structures.water_dimer()
    atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
    slab = structures.graphene_slab()
    slab.calc = D3Calculator(method="pbe", damping="bj", atm=True, cutoff=SHORT)
    with pytest.raises(PropertyNotImplementedError):
        slab.get_stress()


def test_parameter_changes_update_the_context_in_place() -> None:
    atoms = structures.water_dimer()
    calc = D3Calculator(method="pbe", damping="bj", atm=True)
    atoms.calc = calc
    with_atm = atoms.get_potential_energy()
    context = calc._context  # noqa: SLF001
    calc.set(atm=False)
    without_atm = atoms.get_potential_energy()
    assert calc._context is context  # noqa: SLF001
    assert without_atm != with_atm
    assert calc.set(atm=False) == {}
    calc.set(skin=0.5)
    assert calc._context is None  # noqa: SLF001


def test_new_numbers_create_a_new_context() -> None:
    calc = D3Calculator(method="pbe", damping="bj", atm=False)
    water = structures.water_dimer()
    water.calc = calc
    water.get_potential_energy()
    first = calc._context  # noqa: SLF001
    benzene = structures.benzene_dimer()
    benzene.calc = calc
    benzene.get_potential_energy()
    assert calc._context is not first  # noqa: SLF001


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"method": "pbe", "damping": "bj"}, ParameterError),
        ({"method": "pbe", "damping": "bj", "atm": True, "backend": "cpu"}, InputError),
        ({"method": "pbe", "damping": "bj", "atm": True, "device": "gpu"}, InputError),
    ],
)
def test_invalid_configuration_is_rejected_at_construction(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        D3Calculator(**kwargs)  # type: ignore[arg-type]


def test_todict_is_json_serialisable() -> None:
    params = D3BJParameters(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0, alp=14.0)
    calc = D3Calculator(parameters=params)
    assert calc.todict()["d3_parameters"]["s8"] == 0.7875
    calc.set(parameters=params)
    assert calc.resolved_parameters is params
    assert calc.resolved_parameters is params
    named = D3Calculator(method="b3lyp", damping="bj", atm=True)
    assert named.todict()["method"] == "b3lyp"


@pytest.mark.reference
def test_reference_backend_matches_the_cuda_path_definition() -> None:
    atoms = structures.rocksalt_primitive()
    params = resolve_parameters(method="pbe", damping="bj", atm=True)
    reference = D3Calculator(parameters=params, backend="reference", cutoff=SHORT)
    atoms.calc = reference
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    ours = D3Calculator(parameters=params, cutoff=SHORT)
    atoms.calc = ours
    assert atoms.get_potential_energy() == pytest.approx(energy, rel=1e-12)
    np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-12)
    np.testing.assert_allclose(atoms.get_stress(), stress, atol=1e-12)
    molecule = structures.water_dimer()
    molecule.calc = D3Calculator(parameters=params, backend="reference")
    assert molecule.get_potential_energy() < 0.0
