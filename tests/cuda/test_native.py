"""The compiled vanderfast._native extension (M1 infrastructure checks)."""

import importlib
from typing import Any

import numpy as np
import pytest

from vanderfast import resolve_parameters
from vanderfast.context import D3Context
from vanderfast.data import load_manifest, load_reference_tables, read_reference_blob
from vanderfast.diagnostics import backend_info
from vanderfast.errors import CudaUnavailableError, InputError, ParameterError

pytestmark = pytest.mark.native

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)


@pytest.fixture
def native() -> Any:
    return importlib.import_module("vanderfast._native")


def _kwargs(**overrides: float) -> dict[str, float]:
    values = {
        "s6": 1.0,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,
        "s9": 1.0,
        "alp": 14.0,
        "cutoff_cn": 40.0,
        "cutoff_disp2": 60.0,
        "cutoff_disp3": 40.0,
    }
    values.update(overrides)
    return values


def test_build_info_describes_the_policy(native: Any) -> None:
    info = native.build_info()
    assert info["cuda_runtime"] == "static"
    assert info["fast_math"] == "false"
    assert info["precision"] == "float64"
    assert info["cuda_compiler"].startswith("NVIDIA 13.")


def test_embedded_tables_match_the_python_package(native: Any) -> None:
    summary = native.reference_summary()
    assert summary["sha256"] == load_manifest()["outputs"]["d3_reference.bin"]["sha256"]
    assert summary["max_z"] == 103
    tables = load_reference_tables()
    assert native.reference_value("nref", 6) == 5
    assert native.reference_value("reference_cn", 1, a=0) == tables.reference_cn(1)[0]
    assert native.reference_value("c6", 6, 1, 4, 1) == tables.c6_block(6, 1)[4, 1]
    assert native.reference_value("c6", 1, 6, 1, 4) == tables.c6_block(6, 1)[4, 1]
    assert native.reference_value("rcov", 26) == tables.rcov[26]
    assert native.reference_value("r4r2", 53) == tables.r4r2[53]
    assert native.reference_value("rvdw", 8, 1) == tables.vdw_radius(1, 8)
    with pytest.raises(InputError, match="unknown table"):
        native.reference_value("bogus", 1)
    with pytest.raises(InputError, match="not supported"):
        native.reference_value("rcov", 104)


@pytest.mark.cuda
def test_devices_are_reported(native: Any) -> None:
    assert native.device_count() >= 1
    device = native.devices()[0]
    assert {"name", "compute_capability", "total_memory"} <= set(device)
    report = backend_info(initialize_cuda=True)
    assert report.native is not None
    assert report.devices is not None


@pytest.mark.cuda
def test_trivial_systems_evaluate_to_zero(native: Any) -> None:
    one = native.Context(np.array([18], np.int32), **_kwargs())
    result = one.evaluate(
        np.zeros((1, 3)), np.zeros((3, 3)), (False, False, False), gradient=True, virial=True
    )
    assert result["energy"] == 0.0
    np.testing.assert_array_equal(result["gradient"], np.zeros((1, 3)))
    np.testing.assert_array_equal(result["virial"], np.zeros((3, 3)))
    assert result["atom_energies"] is None
    empty = native.Context(np.zeros(0, np.int32), **_kwargs())
    result = empty.evaluate(
        np.zeros((0, 3)), np.zeros((3, 3)), (False, False, False), gradient=True
    )
    assert result["gradient"].shape == (0, 3)
    assert one.stats()["evaluations"] == 1


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("numbers", "kwargs", "error"),
    [
        ([0], {}, InputError),
        ([1], {"device": 99}, InputError),
        ([1], {"alp": -1.0}, ParameterError),
        ([1], {"cutoff_cn": 0.0}, ParameterError),
        ([1], {"skin": -1.0}, InputError),
        ([1], {"builder": "magic"}, InputError),
    ],
)
def test_construction_errors_are_translated(
    native: Any,
    numbers: list[int],
    kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    arguments: dict[str, Any] = _kwargs()
    arguments.update(kwargs)
    with pytest.raises(error):
        native.Context(np.array(numbers, np.int32), **arguments)


@pytest.mark.cuda
def test_evaluation_errors_are_translated(native: Any) -> None:
    context = native.Context(np.array([1, 1], np.int32), **_kwargs())
    zeros = np.zeros((3, 3))
    flags = (False, False, False)
    with pytest.raises(InputError, match="shape"):
        context.evaluate(np.zeros((2, 2)), zeros, flags)
    with pytest.raises(InputError, match="expected 2 positions"):
        context.evaluate(np.zeros((3, 3)), zeros, flags)
    with pytest.raises(InputError, match="finite"):
        context.evaluate(np.full((2, 3), np.nan), zeros, flags)
    with pytest.raises(InputError, match="independent"):
        context.evaluate(np.zeros((2, 3)), zeros, (True, False, False))
    with pytest.raises(ParameterError):
        context.set_parameters(**_kwargs(s6=np.inf))


@pytest.mark.cuda
def test_replacement_tables_are_validated(native: Any) -> None:
    blob = read_reference_blob()
    context = native.Context(
        np.array([1], np.int32), **_kwargs(), reference_data=blob, reference_identity="copy"
    )
    assert context.n_atoms == 1
    with pytest.raises(InputError, match="reference data is invalid"):
        native.Context(np.array([1], np.int32), **_kwargs(), reference_data=blob[:600])


@pytest.mark.cuda
def test_python_context_drives_the_native_backend() -> None:
    context = D3Context([18], PBE_ATM)
    result = context.evaluate([[0.0, 0.0, 0.0]], properties=("forces",))
    assert result.energy == 0.0
    assert context.diagnostics()["stats"]["evaluations"] == 1


def test_missing_devices_are_reported_cleanly(native: Any) -> None:
    if native.device_count() > 0:
        pytest.skip("a CUDA device is present")
    with pytest.raises(CudaUnavailableError):
        native.Context(np.array([1], np.int32), **_kwargs())
