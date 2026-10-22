"""GPU DFT-D3(BJ) and DFT-D4 dispersion energies, forces, and stress."""

from importlib.metadata import version

from vanderfast.context import D3Context, D3Result
from vanderfast.context_d4 import D4Context, D4Result, EEQSolverOptions
from vanderfast.diagnostics import backend_info
from vanderfast.errors import (
    ChargeSolverError,
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
from vanderfast.parameters_d4 import (
    D4Cutoffs,
    D4Parameters,
    available_d4_methods,
    resolve_d4_parameters,
)

__version__ = version("vanderfast")

__all__ = [
    "ChargeSolverError",
    "ContextBusyError",
    "CudaError",
    "CudaUnavailableError",
    "Cutoffs",
    "D3BJParameters",
    "D3Context",
    "D3Result",
    "D4Context",
    "D4Cutoffs",
    "D4Parameters",
    "D4Result",
    "DeviceOutOfMemoryError",
    "EEQSolverOptions",
    "InputError",
    "NativeExtensionMissingError",
    "ParameterError",
    "ReferenceBackendUnavailableError",
    "UnsupportedFeatureError",
    "VanderfastError",
    "__version__",
    "available_d4_methods",
    "available_methods",
    "backend_info",
    "resolve_d4_parameters",
    "resolve_parameters",
]
