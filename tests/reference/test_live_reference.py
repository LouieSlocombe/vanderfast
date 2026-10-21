"""Live comparisons with the installed dftd3 1.6.0 library."""

import importlib

import numpy as np
import pytest

from vanderfast import _oracle, resolve_parameters
from vanderfast.errors import ReferenceBackendUnavailableError
from vanderfast.parameters import available_methods
from vanderfast.reference import PINNED_VERSION, ReferenceEvaluator, reference_version
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.reference


def test_installed_reference_is_the_pinned_version() -> None:
    assert reference_version() == PINNED_VERSION


@pytest.mark.parametrize("atm", [False, True])
def test_every_method_matches_dftd3_method_lookup(atm: bool) -> None:
    """dftd3's own method lookup equals our explicit coefficients for every method."""
    interface = importlib.import_module("dftd3.interface")
    water = structures.water_dimer()
    positions = water.positions * INV_BOHR
    for method in available_methods():
        params = resolve_parameters(method=method, damping="bj", atm=atm)
        model = interface.DispersionModel(water.numbers, positions)
        upstream = model.get_dispersion(
            param=interface.RationalDampingParam(method=method, atm=atm), grad=False
        )
        ours = ReferenceEvaluator(water.numbers, params).evaluate(positions, gradient=False)
        assert ours.energy == pytest.approx(float(upstream["energy"]), rel=1e-14), method


def test_random_molecules_match_the_oracle() -> None:
    params = resolve_parameters(method="tpss", damping="bj", atm=True)
    rng = np.random.default_rng(7)
    for _ in range(5):
        numbers = rng.choice([1, 6, 7, 8, 9, 15, 16, 17, 35], size=9)
        atoms = structures._random_cluster(list(numbers), seed=int(rng.integers(1 << 30)))  # noqa: SLF001
        positions = atoms.positions * INV_BOHR
        reference = ReferenceEvaluator(numbers, params).evaluate(positions)
        oracle = _oracle.evaluate(numbers, positions, parameters=params)
        assert oracle.energy == pytest.approx(reference.energy, rel=1e-13)
        assert reference.gradient is not None
        assert oracle.gradient is not None
        np.testing.assert_allclose(oracle.gradient, reference.gradient, rtol=0, atol=1e-13)


def test_reference_adapter_reports_pairwise_energies_and_periodic_virial() -> None:
    params = resolve_parameters(method="pbe", damping="bj", atm=True)
    water = structures.water_dimer()
    result = ReferenceEvaluator(water.numbers, params).evaluate(
        water.positions * INV_BOHR, pairwise=True, gradient=False
    )
    assert result.pairwise_additive is not None
    assert result.pairwise_nonadditive is not None
    assert result.gradient is None
    total = result.pairwise_additive.sum() + result.pairwise_nonadditive.sum()
    assert total == pytest.approx(result.energy, rel=1e-12)
    crystal = structures.rocksalt_primitive()
    periodic = ReferenceEvaluator(crystal.numbers, params).evaluate(
        crystal.positions * INV_BOHR, crystal.cell[:] * INV_BOHR, crystal.pbc
    )
    assert periodic.virial is not None
    assert periodic.virial.shape == (3, 3)


def test_missing_reference_library_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("vanderfast.reference.importlib.import_module", missing)
    with pytest.raises(ReferenceBackendUnavailableError, match="pip install"):
        reference_version()
