"""The ASE invalidation matrix (spec section 8.3) on the oracle-backed fake native."""

import numpy as np
import pytest
from support import oracle_native

import vanderfast.context as context_module
from vanderfast.ase import D3Calculator
from vanderfast.testing import structures

CUTOFF = {"cn": 8.0, "disp2": 10.0, "disp3": 8.0}


@pytest.fixture(autouse=True)
def fake_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_module, "load_native", lambda: oracle_native)


def _setup() -> tuple[object, D3Calculator]:
    atoms = structures.rocksalt_primitive()
    calc = D3Calculator(method="pbe", damping="bj", atm=True, cutoff=CUTOFF)
    atoms.calc = calc
    atoms.get_forces()
    return atoms, calc


def test_no_change_returns_cached_results() -> None:
    atoms, calc = _setup()
    count = calc.evaluations
    atoms.get_forces()  # type: ignore[attr-defined]
    atoms.get_potential_energy()  # type: ignore[attr-defined]
    assert calc.evaluations == count


def test_more_properties_run_only_the_missing_path() -> None:
    atoms = structures.rocksalt_primitive()
    calc = D3Calculator(method="pbe", damping="bj", atm=True, cutoff=CUTOFF)
    atoms.calc = calc
    atoms.get_potential_energy()
    assert "forces" not in calc.results
    atoms.get_forces()
    assert calc.evaluations == 2
    atoms.get_stress()
    assert calc.evaluations == 3
    atoms.get_potential_energy()
    atoms.get_forces()
    assert calc.evaluations == 3


def test_positions_reuse_the_context() -> None:
    atoms, calc = _setup()
    context = calc._context  # noqa: SLF001
    atoms.positions[0, 0] += 0.01  # type: ignore[attr-defined]
    atoms.get_forces()  # type: ignore[attr-defined]
    assert calc._context is context  # noqa: SLF001


def test_numbers_cell_and_pbc_changes() -> None:
    atoms, calc = _setup()
    context = calc._context  # noqa: SLF001
    atoms.set_cell(atoms.cell[:] * 1.01, scale_atoms=True)  # type: ignore[attr-defined]
    count = calc.evaluations
    atoms.get_potential_energy()  # type: ignore[attr-defined]
    assert calc.evaluations == count + 1
    assert calc._context is context  # noqa: SLF001
    atoms.numbers = np.array([11, 11])  # type: ignore[attr-defined]
    atoms.get_potential_energy()  # type: ignore[attr-defined]
    assert calc._context is not context  # noqa: SLF001


def test_parameter_and_cutoff_changes_keep_the_context() -> None:
    atoms, calc = _setup()
    context = calc._context  # noqa: SLF001
    assert context is not None
    with_atm = atoms.get_potential_energy()  # type: ignore[attr-defined]
    calc.set(atm=False)
    assert not calc.results
    assert atoms.get_potential_energy() != with_atm  # type: ignore[attr-defined]
    calc.set(cutoff={"cn": 8.0, "disp2": 12.0, "disp3": 8.0})
    assert calc._context is context  # noqa: SLF001
    native = context.native
    assert isinstance(native, oracle_native.Context)
    assert native.neighbor_invalidations == 1


@pytest.mark.parametrize(
    ("key", "value"), [("skin", 0.25), ("device", "cuda:1"), ("backend", "cuda")]
)
def test_context_level_changes_create_a_new_context(key: str, value: object) -> None:
    atoms, calc = _setup()
    context = calc._context  # noqa: SLF001
    changed = calc.set(**{key: value})
    if changed:
        assert calc._context is None  # noqa: SLF001
        atoms.get_potential_energy()  # type: ignore[attr-defined]
        assert calc._context is not context  # noqa: SLF001
    else:
        assert calc._context is context  # noqa: SLF001


def test_charges_and_magnetic_moments_are_ignored() -> None:
    atoms, calc = _setup()
    count = calc.evaluations
    atoms.set_initial_charges([1.0, -1.0])  # type: ignore[attr-defined]
    atoms.set_initial_magnetic_moments([0.5, 0.5])  # type: ignore[attr-defined]
    atoms.get_forces()  # type: ignore[attr-defined]
    assert calc.evaluations == count
