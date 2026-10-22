"""Shared pytest configuration: capability markers and strict-run options."""

import importlib
import importlib.util
import os

import pytest

# The dftd3 and dftd4 references use OpenMP reductions, and dftd4 solves its charge
# model with LAPACK; one thread of each makes them deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register options that turn capability skips into failures."""
    group = parser.getgroup("vanderfast")
    group.addoption(
        "--require-native",
        action="store_true",
        help="fail instead of skipping tests that need vanderfast._native",
    )
    group.addoption(
        "--require-cuda",
        action="store_true",
        help="fail instead of skipping tests that need a CUDA device",
    )
    group.addoption(
        "--require-reference",
        action="store_true",
        help="fail instead of skipping tests that need dftd3 1.6.0 or dftd4 4.2.0",
    )
    group.addoption("--run-slow", action="store_true", help="run tests marked slow")


def _has_native() -> bool:
    return importlib.util.find_spec("vanderfast._native") is not None


def _has_cuda() -> bool:
    if not _has_native():
        return False
    native = importlib.import_module("vanderfast._native")
    try:
        return int(native.device_count()) > 0
    except RuntimeError:
        return False


def _has_reference() -> bool:
    if importlib.util.find_spec("dftd3") is None:
        return False
    return str(importlib.import_module("dftd3").__version__) == "1.6.0"


def _has_reference_d4() -> bool:
    if importlib.util.find_spec("dftd4") is None:
        return False
    return str(importlib.import_module("dftd4").__version__) == "4.2.0"


_CAPABILITIES = {
    "native": ("--require-native", _has_native, "vanderfast._native is not built"),
    "cuda": ("--require-cuda", _has_cuda, "no CUDA device is available"),
    "reference": ("--require-reference", _has_reference, "dftd3 1.6.0 is not installed"),
    "reference_d4": ("--require-reference", _has_reference_d4, "dftd4 4.2.0 is not installed"),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip, or fail when required, tests whose capabilities are missing."""
    cache: dict[str, bool] = {}
    require_all = os.environ.get("VANDERFAST_REQUIRE_CUDA") == "1"
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="slow test; use --run-slow"))
        for marker, (option, probe, reason) in _CAPABILITIES.items():
            if marker not in item.keywords:
                continue
            if marker not in cache:
                cache[marker] = probe()
            if cache[marker]:
                continue
            if config.getoption(option) or (require_all and marker in {"native", "cuda"}):
                msg = f"{option} was given but {reason}"
                raise pytest.UsageError(msg)
            item.add_marker(pytest.mark.skip(reason=reason))
