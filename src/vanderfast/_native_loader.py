"""Lazy loading of the compiled ``vanderfast._native`` extension.

Importing :mod:`vanderfast` never loads the extension (and therefore never
initialises CUDA); it is loaded on first use by a CUDA context or diagnostics.
"""

import importlib
from types import ModuleType

from vanderfast.errors import NativeExtensionMissingError

_MODULE = "vanderfast._native"


def load_native() -> ModuleType:
    """Import and return the native extension.

    Raises:
        NativeExtensionMissingError: If the extension was not built.
    """
    try:
        return importlib.import_module(_MODULE)
    except ImportError as error:
        msg = (
            "the vanderfast CUDA extension is not available "
            f"({error}); reinstall with VANDERFAST_ENABLE_CUDA=ON and a CUDA >= 13 toolkit"
        )
        raise NativeExtensionMissingError(msg) from error
