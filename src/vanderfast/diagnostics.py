"""Backend, build, and provenance reporting."""

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from typing import Any

from vanderfast._native_loader import load_native
from vanderfast.data import load_manifest, load_reference_tables
from vanderfast.data.d4 import load_d4_manifest, load_d4_reference_tables
from vanderfast.errors import NativeExtensionMissingError


@dataclass(frozen=True)
class BackendInfo:
    """A reproducibility report for bug reports and benchmark records."""

    package_version: str
    reference_data: dict[str, Any]
    native: dict[str, Any] | None
    native_error: str | None
    devices: list[dict[str, Any]] | None = field(default=None)
    reference_data_d4: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable mapping."""
        return asdict(self)


def _reference_summary() -> dict[str, Any]:
    tables = load_reference_tables()
    manifest = load_manifest()
    return {
        "source": tables.source_id,
        "sha256": tables.sha256,
        "supported_atomic_numbers": manifest["supported_atomic_numbers"],
        "table_aatoau": manifest["units"]["table_aatoau"]["value"],
    }


def _d4_reference_summary() -> dict[str, Any]:
    tables = load_d4_reference_tables()
    manifest = load_d4_manifest()
    return {
        "source": tables.source_id,
        "sha256": tables.sha256,
        "supported_atomic_numbers": manifest["supported_atomic_numbers"],
        "charge_model": manifest["conventions"]["charge_model"],
        "table_aatoau": manifest["units"]["table_aatoau"]["value"],
    }


def backend_info(*, initialize_cuda: bool = False) -> BackendInfo:
    """Describe the installed package, bundled data, and native backend.

    Args:
        initialize_cuda: Also query the CUDA driver and devices. This creates a
            CUDA context in the calling process.
    """
    try:
        native = load_native()
    except NativeExtensionMissingError as error:
        return BackendInfo(
            package_version=version("vanderfast"),
            reference_data=_reference_summary(),
            native=None,
            native_error=str(error),
            reference_data_d4=_d4_reference_summary(),
        )
    build: dict[str, Any] = dict(native.build_info())
    devices = [dict(d) for d in native.devices()] if initialize_cuda else None
    return BackendInfo(
        package_version=version("vanderfast"),
        reference_data=_reference_summary(),
        native=build,
        native_error=None,
        devices=devices,
        reference_data_d4=_d4_reference_summary(),
    )
