"""Optional adapter to the pinned simple-dftd3 reference implementation.

The adapter always passes fully explicit damping coefficients and cutoffs to
``dftd3`` (never a method name), so a comparison with vanderfast uses
identical resolved parameters. Importing this module does not import
``dftd3``; it is loaded on first use.
"""

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from vanderfast.errors import ReferenceBackendUnavailableError
from vanderfast.parameters import Cutoffs, D3BJParameters

PINNED_VERSION: Final = "1.6.0"

FloatArray = npt.NDArray[np.float64]


def _load_dftd3() -> tuple[ModuleType, ModuleType]:
    try:
        package = importlib.import_module("dftd3")
        interface = importlib.import_module("dftd3.interface")
    except ImportError as error:
        msg = (
            "the dftd3 reference library is not installed; install it with "
            "'pip install vanderfast[reference]' or 'conda install dftd3-python=1.6.0'"
        )
        raise ReferenceBackendUnavailableError(msg) from error
    return package, interface


def reference_version() -> str:
    """Return the installed dftd3 version."""
    package, _ = _load_dftd3()
    return str(package.__version__)


@dataclass(frozen=True, eq=False)
class ReferenceResult:
    """Reference results in atomic units (hartree, bohr)."""

    energy: float
    gradient: FloatArray | None
    virial: FloatArray | None
    pairwise_additive: FloatArray | None = None
    pairwise_nonadditive: FloatArray | None = None


class ReferenceEvaluator:
    """Evaluate D3(BJ) with ``dftd3`` using explicit parameters and cutoffs."""

    def __init__(
        self,
        numbers: npt.ArrayLike,
        parameters: D3BJParameters,
        cutoffs: Cutoffs | None = None,
    ) -> None:
        """Bind atomic numbers, resolved parameters, and cutoffs in bohr."""
        _, self._interface = _load_dftd3()
        self.numbers = np.asarray(numbers, dtype=np.int32)
        self.parameters = parameters
        self.cutoffs = cutoffs or Cutoffs()

    def _model(
        self,
        positions: npt.ArrayLike,
        cell: npt.ArrayLike | None,
        pbc: npt.ArrayLike,
    ) -> Any:
        periodic = np.asarray(pbc, dtype=bool)
        lattice = None if cell is None else np.asarray(cell, dtype=np.float64)
        model = self._interface.DispersionModel(
            self.numbers,
            np.asarray(positions, dtype=np.float64),
            lattice,
            periodic if periodic.any() else None,
        )
        model.set_realspace_cutoff(
            disp2=self.cutoffs.disp2, disp3=self.cutoffs.disp3, cn=self.cutoffs.cn
        )
        return model

    def _param(self) -> Any:
        p = self.parameters
        return self._interface.RationalDampingParam(
            s6=p.s6, s8=p.s8, s9=p.s9, a1=p.a1, a2=p.a2, alp=p.alp
        )

    def evaluate(
        self,
        positions: npt.ArrayLike,
        cell: npt.ArrayLike | None = None,
        pbc: npt.ArrayLike = (False, False, False),
        *,
        gradient: bool = True,
        pairwise: bool = False,
    ) -> ReferenceResult:
        """Evaluate at positions and cell rows given in bohr."""
        model = self._model(positions, cell, pbc)
        param = self._param()
        result = model.get_dispersion(param=param, grad=gradient)
        additive = nonadditive = None
        if pairwise:
            pairs = model.get_pairwise_dispersion(param=param)
            additive = np.array(pairs["additive pairwise energy"], dtype=np.float64)
            nonadditive = np.array(pairs["non-additive pairwise energy"], dtype=np.float64)
        return ReferenceResult(
            energy=float(result["energy"]),
            gradient=np.array(result["gradient"], dtype=np.float64) if gradient else None,
            virial=np.array(result["virial"], dtype=np.float64) if gradient else None,
            pairwise_additive=additive,
            pairwise_nonadditive=nonadditive,
        )


__all__ = [
    "PINNED_VERSION",
    "ReferenceEvaluator",
    "ReferenceResult",
    "reference_version",
]
