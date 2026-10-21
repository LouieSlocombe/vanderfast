"""GPU DFT-D3(BJ) dispersion energies, forces, and stress."""

from importlib.metadata import version

from vanderfast.context import D3Context, D3Result
from vanderfast.diagnostics import backend_info
from vanderfast.errors import (
    ContextBusyError,
    CudaError,
    CudaUnavailableError,
    DeviceOutOfMemoryError,
    InputError,
    NativeExtensionMissingError,
    ParameterError,
    ReferenceBackendUnavailableError,
    UnsupportedFeatureError,
    VanderfastError,
)
from vanderfast.parameters import Cutoffs, D3BJParameters, available_methods, resolve_parameters

__version__ = version("vanderfast")

__all__ = [
    "ContextBusyError",
    "CudaError",
    "CudaUnavailableError",
    "Cutoffs",
    "D3BJParameters",
    "D3Context",
    "D3Result",
    "DeviceOutOfMemoryError",
    "InputError",
    "NativeExtensionMissingError",
    "ParameterError",
    "ReferenceBackendUnavailableError",
    "UnsupportedFeatureError",
    "VanderfastError",
    "__version__",
    "available_methods",
    "backend_info",
    "resolve_parameters",
]
