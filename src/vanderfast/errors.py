"""Exception hierarchy shared by the Python layer and the native extension."""


class VanderfastError(Exception):
    """Base class for all vanderfast errors."""


class ParameterError(VanderfastError, ValueError):
    """Invalid, ambiguous, or unknown dispersion parameters."""


class InputError(VanderfastError, ValueError):
    """Invalid geometry, element, or array input."""


class UnsupportedFeatureError(VanderfastError, NotImplementedError):
    """A requested feature is not implemented by the selected backend."""


class CudaError(VanderfastError, RuntimeError):
    """A CUDA runtime failure reported by the native backend."""


class CudaUnavailableError(CudaError):
    """No usable CUDA device or driver is available."""


class DeviceOutOfMemoryError(CudaError, MemoryError):
    """A device allocation failed."""


class ContextBusyError(VanderfastError, RuntimeError):
    """A context was used concurrently; one context runs one evaluation at a time."""


class NativeExtensionMissingError(VanderfastError, ImportError):
    """The compiled ``vanderfast._native`` extension is not installed."""


class ReferenceBackendUnavailableError(VanderfastError, ImportError):
    """The optional ``dftd3`` reference library is not installed."""
