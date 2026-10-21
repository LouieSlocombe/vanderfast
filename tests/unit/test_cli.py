"""Command-line interface and diagnostics."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vanderfast import __version__
from vanderfast.__main__ import main
from vanderfast.diagnostics import backend_info


def test_methods_lists_canonical_names(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["methods"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 161
    assert any(line.startswith("pbe ") for line in lines)


def test_verify_data(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify-data"]) == 0
    assert "simple-dftd3 v1.6.0" in capsys.readouterr().out


def test_info_reports_provenance(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["package_version"] == __version__
    assert report["reference_data"]["supported_atomic_numbers"] == {"min": 1, "max": 103}
    assert main(["info"]) == 0
    assert "reference_data" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_backend_info_without_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("vanderfast._native_loader.importlib.import_module", missing)
    info = backend_info()
    assert info.native is None
    assert info.native_error is not None
    assert "VANDERFAST_ENABLE_CUDA" in info.native_error


def test_backend_info_with_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        build_info=lambda: {"cuda_compiler": "nvcc 13.1"},
        devices=lambda: [{"name": "fake", "compute_capability": "8.9"}],
    )
    monkeypatch.setattr("vanderfast.diagnostics.load_native", lambda: fake)
    info = backend_info(initialize_cuda=True)
    assert info.native == {"cuda_compiler": "nvcc 13.1"}
    assert info.devices == [{"name": "fake", "compute_capability": "8.9"}]
    assert backend_info().devices is None


def test_import_does_not_load_native_or_frameworks() -> None:
    code = (
        "import sys, vanderfast, vanderfast.reference, vanderfast.parameters;"
        "bad = [m for m in ('vanderfast._native', 'torch', 'cupy', 'dftd3') if m in sys.modules];"
        "print(','.join(bad))"
    )
    command = [sys.executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
    assert result.stdout.strip() == ""
