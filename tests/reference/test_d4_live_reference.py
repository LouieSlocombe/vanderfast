"""Live comparisons with the installed dftd4 4.2.0 library."""

import importlib

import numpy as np
import pytest

from vanderfast.data.d4 import load_d4_methods
from vanderfast.errors import ReferenceBackendUnavailableError
from vanderfast.parameters_d4 import D4Parameters, resolve_d4_parameters
from vanderfast.reference_d4 import (
    PINNED_VERSION,
    D4ReferenceEvaluator,
    reference_version,
    threads,
)
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.reference_d4


def test_installed_reference_is_the_pinned_version() -> None:
    assert reference_version() == PINNED_VERSION
    assert set(threads()) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}


@pytest.mark.parametrize("atm", [False, True])
def test_every_alias_matches_dftd4_functional_lookup(atm: bool) -> None:
    """dftd4's own name lookup equals our explicit coefficients for every alias, bitwise."""
    interface = importlib.import_module("dftd4.interface")
    water = structures.water_dimer()
    positions = water.positions * INV_BOHR
    for alias in load_d4_methods()["aliases"]:
        params = resolve_d4_parameters(method=alias, atm=atm)
        model = interface.DispersionModel(water.numbers, positions)
        upstream = model.get_dispersion(interface.DampingParam(method=alias, atm=atm), grad=False)
        ours = D4ReferenceEvaluator(water.numbers, params).evaluate(positions, gradient=False)
        assert ours.energy == float(upstream["energy"]), alias


def test_reference_adapter_reports_properties_pairwise_and_virial() -> None:
    params = resolve_d4_parameters(method="pbe")
    water = structures.water_dimer()
    result = D4ReferenceEvaluator(water.numbers, params).evaluate(
        water.positions * INV_BOHR, gradient=False, properties=True, pairwise=True
    )
    assert result.gradient is None
    assert result.charges is not None
    assert result.charges.sum() == pytest.approx(0.0, abs=1e-13)
    assert result.cn is not None
    assert result.c6 is not None
    assert result.c6.shape == (6, 6)
    assert result.polarizabilities is not None
    assert result.pairwise_additive is not None
    assert result.pairwise_nonadditive is not None
    total = result.pairwise_additive.sum() + result.pairwise_nonadditive.sum()
    assert total == pytest.approx(result.energy, rel=1e-12)
    crystal = structures.rocksalt_primitive()
    periodic = D4ReferenceEvaluator(crystal.numbers, params, charge=0.0).evaluate(
        crystal.positions * INV_BOHR, crystal.cell[:] * INV_BOHR, crystal.pbc
    )
    assert periodic.virial is not None
    assert periodic.virial.shape == (3, 3)


def test_total_charge_and_model_parameters_reach_the_reference() -> None:
    water = structures.water_dimer()
    positions = water.positions * INV_BOHR
    base = D4Parameters(s8=1.0, a1=0.4, a2=5.0, s9=0.0)
    neutral = D4ReferenceEvaluator(water.numbers, base).evaluate(positions, properties=True)
    cation = D4ReferenceEvaluator(water.numbers, base, charge=1.0).evaluate(
        positions, properties=True
    )
    assert cation.charges is not None
    assert cation.charges.sum() == pytest.approx(1.0, abs=1e-12)
    assert cation.energy != neutral.energy
    scaled = D4Parameters(s8=1.0, a1=0.4, a2=5.0, s9=0.0, ga=2.0, gc=1.0, wf=4.0)
    other = D4ReferenceEvaluator(water.numbers, scaled).evaluate(positions, gradient=False)
    assert other.energy != neutral.energy
    assert neutral.gradient is not None
    assert np.isfinite(neutral.gradient).all()


def test_missing_reference_library_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("vanderfast.reference_d4.importlib.import_module", missing)
    with pytest.raises(ReferenceBackendUnavailableError, match=r"dftd4-python=4\.2\.0"):
        reference_version()
