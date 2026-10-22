"""The compiled D4 entry points: embedded tables, host precompute, and option checks."""

import hashlib
import importlib
from typing import Any

import numpy as np
import pytest
from support import native_helpers as nh

from vanderfast import _oracle_d4
from vanderfast.data.d4 import load_d4_manifest, load_d4_reference_tables, read_d4_reference_blob
from vanderfast.errors import InputError, ParameterError
from vanderfast.parameters_d4 import resolve_d4_parameters

pytestmark = pytest.mark.native

PBE = resolve_d4_parameters(method="pbe")


@pytest.fixture
def native() -> Any:
    return importlib.import_module("vanderfast._native")


def test_embedded_tables_are_the_bundled_file(native: Any) -> None:
    summary = native.d4_reference_summary()
    digest = hashlib.sha256(read_d4_reference_blob()).hexdigest()
    assert summary["sha256"] == digest
    assert summary["max_z"] == 103
    assert summary["source"].startswith("dftd4 v4.2.0")
    assert native.build_info()["d4_reference_sha256"] == digest


def test_host_precompute_reproduces_the_manifest_hashes(native: Any) -> None:
    derived = load_d4_manifest()["derived"]
    ga, gc = derived["ga"], derived["gc"]
    alphas = [native.d4_reference_alpha(z, ga, gc) for z in range(1, 104)]
    flat_alpha = np.concatenate([a.ravel() for a in alphas]).astype("<f8")
    assert hashlib.sha256(flat_alpha.tobytes()).hexdigest() == derived["refalpha_sha256"]
    blocks = [
        native.d4_reference_c6(zi, zj, ga, gc).ravel()
        for zi in range(1, 104)
        for zj in range(1, zi + 1)
    ]
    flat_c6 = np.concatenate(blocks).astype("<f8")
    assert flat_c6.size == derived["n_c6"]
    assert hashlib.sha256(flat_c6.tobytes()).hexdigest() == derived["refc6_sha256"]


@pytest.mark.parametrize(("ga", "gc"), [(2.0, 1.0), (4.5, 0.7)])
def test_host_precompute_matches_the_oracle_bitwise(native: Any, ga: float, gc: float) -> None:
    tables = load_d4_reference_tables()
    alpha = _oracle_d4.reference_alpha(tables, ga, gc)
    elements = (1, 3, 6, 8, 26, 38, 55, 56, 79, 92, 95, 98, 103)
    for z in elements:
        np.testing.assert_array_equal(
            native.d4_reference_alpha(z, ga, gc), alpha[tables.references(z)]
        )
    for zi in elements:
        for zj in elements:
            np.testing.assert_array_equal(
                native.d4_reference_c6(zi, zj, ga, gc),
                _oracle_d4.reference_c6(tables, alpha, zi, zj),
            )


def test_reference_accessors_reject_unsupported_elements(native: Any) -> None:
    with pytest.raises(InputError, match="not supported by D4"):
        native.d4_reference_alpha(104)
    with pytest.raises(InputError, match="not supported by D4"):
        native.d4_reference_c6(1, 0)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"ewald": "fast"}, InputError, "ewald must be"),
        ({"solver": "cg"}, InputError, "solver must be"),
        ({"kernels": "fused"}, InputError, "kernels must be"),
        ({"builder": "octree"}, InputError, "builder"),
        ({"atm_items_per_launch": -1}, InputError, "atm_items_per_launch"),
        ({"solver_tolerance": 2.0}, InputError, "tolerance"),
    ],
)
def test_constructor_rejects_invalid_options(
    options: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        nh.d4_context([1, 1], PBE, **options)


RAW = {
    "s6": 1.0,
    "s8": 1.0,
    "a1": 0.4,
    "a2": 5.0,
    "s9": 1.0,
    "alp": 16.0,
    "ga": 3.0,
    "gc": 2.0,
    "wf": 6.0,
    "cutoff_cn": 30.0,
    "cutoff_disp2": 60.0,
    "cutoff_disp3": 40.0,
}


def test_constructor_rejects_invalid_inputs(native: Any) -> None:
    pair = np.array([1, 1], dtype=np.int32)
    with pytest.raises(InputError, match="not supported by D4"):
        nh.d4_context([1, 104], PBE)
    with pytest.raises(ParameterError, match="wf must be positive"):
        native.D4Context(pair, **{**RAW, "wf": 0.0})
    with pytest.raises(ParameterError, match="finite"):
        native.D4Context(pair, **{**RAW, "ga": float("nan")})
    with pytest.raises(InputError, match="one-dimensional"):
        native.D4Context(np.ones((2, 2), dtype=np.int32), **RAW)
