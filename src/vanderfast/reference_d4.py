"""Optional adapter to the pinned dftd4 reference implementation.

The adapter always passes fully explicit damping coefficients, charge-scaling
parameters, and cutoffs to ``dftd4`` (never a functional name), so a
comparison with vanderfast uses identical resolved parameters. Importing this
module does not import ``dftd4``; it is loaded on first use.

dftd4 sums with OpenMP critical sections and solves the charge model with
LAPACK; for bitwise-stable results set ``OMP_NUM_THREADS=1`` and
``OPENBLAS_NUM_THREADS=1`` before the library is first imported. Its energy-only
path solves the EEQ system with ``sytrs`` and its gradient path with an
explicit inverse, so charges differ between the two in the last bits.
"""

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from vanderfast.errors import ReferenceBackendUnavailableError
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters

PINNED_VERSION: Final = "4.2.0"

FloatArray = npt.NDArray[np.float64]


def _load_dftd4() -> tuple[ModuleType, ModuleType]:
    try:
        package = importlib.import_module("dftd4")
        interface = importlib.import_module("dftd4.interface")
    except ImportError as error:
        msg = (
            "the dftd4 reference library is not installed; install it with "
            "'pip install vanderfast[reference]' or 'conda install dftd4-python=4.2.0'"
        )
        raise ReferenceBackendUnavailableError(msg) from error
    return package, interface


def reference_version() -> str:
    """Return the installed dftd4 version."""
    package, _ = _load_dftd4()
    return str(package.__version__)


def threads() -> dict[str, str | None]:
    """Return the thread settings that determine dftd4's summation order."""
    return {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}


@dataclass(frozen=True, eq=False)
class D4ReferenceResult:
    """Reference results in atomic units (hartree, bohr, e).

    ``virial`` is dE/dstrain. The properties (``charges``, ``cn``, ``c6``,
    ``polarizabilities``) come from dftd4's ``get_properties``, which solves the
    charge model on its energy-only path.
    """

    energy: float
    gradient: FloatArray | None
    virial: FloatArray | None
    charges: FloatArray | None = None
    cn: FloatArray | None = None
    c6: FloatArray | None = None
    polarizabilities: FloatArray | None = None
    pairwise_additive: FloatArray | None = None
    pairwise_nonadditive: FloatArray | None = None


class D4ReferenceEvaluator:
    """Evaluate D4 with ``dftd4`` using explicit parameters, cutoffs, and total charge."""

    def __init__(
        self,
        numbers: npt.ArrayLike,
        parameters: D4Parameters,
        cutoffs: D4Cutoffs | None = None,
        charge: float = 0.0,
    ) -> None:
        """Bind atomic numbers, resolved parameters, cutoffs in bohr, and total charge."""
        _, self._interface = _load_dftd4()
        self.numbers = np.asarray(numbers, dtype=np.int32)
        self.parameters = parameters
        self.cutoffs = cutoffs or D4Cutoffs()
        self.charge = float(charge)

    def _model(
        self,
        positions: npt.ArrayLike,
        cell: npt.ArrayLike | None,
        pbc: npt.ArrayLike,
    ) -> Any:
        periodic = np.asarray(pbc, dtype=bool)
        lattice = None if cell is None else np.asarray(cell, dtype=np.float64)
        p = self.parameters
        model = self._interface.DispersionModel(
            self.numbers,
            np.asarray(positions, dtype=np.float64),
            self.charge,
            lattice if periodic.any() else None,
            periodic if periodic.any() else None,
            model="d4",
            ga=p.ga,
            gc=p.gc,
            wf=p.wf,
        )
        model.set_realspace_cutoff(
            disp2=self.cutoffs.disp2, disp3=self.cutoffs.disp3, cn=self.cutoffs.cn
        )
        return model

    def _param(self) -> Any:
        p = self.parameters
        return self._interface.DampingParam(s6=p.s6, s8=p.s8, s9=p.s9, a1=p.a1, a2=p.a2, alp=p.alp)

    def evaluate(
        self,
        positions: npt.ArrayLike,
        cell: npt.ArrayLike | None = None,
        pbc: npt.ArrayLike = (False, False, False),
        *,
        gradient: bool = True,
        properties: bool = False,
        pairwise: bool = False,
    ) -> D4ReferenceResult:
        """Evaluate at positions and cell rows given in bohr."""
        model = self._model(positions, cell, pbc)
        param = self._param()
        result = model.get_dispersion(param=param, grad=gradient)
        extra: dict[str, FloatArray] = {}
        if properties:
            props = model.get_properties()
            extra["charges"] = np.array(props["partial charges"], dtype=np.float64)
            extra["cn"] = np.array(props["coordination numbers"], dtype=np.float64)
            extra["c6"] = np.array(props["c6 coefficients"], dtype=np.float64)
            extra["polarizabilities"] = np.array(props["polarizabilities"], dtype=np.float64)
        if pairwise:
            pairs = model.get_pairwise_dispersion(param=param)
            extra["pairwise_additive"] = np.array(
                pairs["additive pairwise energy"], dtype=np.float64
            )
            extra["pairwise_nonadditive"] = np.array(
                pairs["non-additive pairwise energy"], dtype=np.float64
            )
        return D4ReferenceResult(
            energy=float(result["energy"]),
            gradient=np.array(result["gradient"], dtype=np.float64) if gradient else None,
            virial=np.array(result["virial"], dtype=np.float64) if gradient else None,
            **extra,
        )


__all__ = [
    "PINNED_VERSION",
    "D4ReferenceEvaluator",
    "D4ReferenceResult",
    "reference_version",
    "threads",
]
