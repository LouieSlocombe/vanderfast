"""Exception hierarchy."""

import pytest

from vanderfast import errors


@pytest.mark.parametrize(
    ("error", "bases"),
    [
        (errors.ParameterError, (ValueError,)),
        (errors.InputError, (ValueError,)),
        (errors.UnsupportedFeatureError, (NotImplementedError,)),
        (errors.CudaError, (RuntimeError,)),
        (errors.CudaUnavailableError, (errors.CudaError,)),
        (errors.DeviceOutOfMemoryError, (errors.CudaError, MemoryError)),
        (errors.ContextBusyError, (RuntimeError,)),
        (errors.NativeExtensionMissingError, (ImportError,)),
        (errors.ReferenceBackendUnavailableError, (ImportError,)),
    ],
)
def test_errors_derive_from_builtin_categories(
    error: type[Exception], bases: tuple[type, ...]
) -> None:
    assert issubclass(error, errors.VanderfastError)
    for base in bases:
        assert issubclass(error, base)
