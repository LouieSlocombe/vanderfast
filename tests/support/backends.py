"""Backend-neutral evaluation helpers for periodic and state tests."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from support import native_helpers
from vanderfast import _oracle
from vanderfast.parameters import Cutoffs, D3BJParameters

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Evaluation:
    """Energy, gradient, and strain derivative in atomic units."""

    energy: float
    gradient: FloatArray
    strain_derivative: FloatArray


Evaluator = Callable[
    [npt.ArrayLike, FloatArray, FloatArray | None, npt.ArrayLike, D3BJParameters, Cutoffs],
    Evaluation,
]


def cuda_evaluator(
    numbers: npt.ArrayLike,
    positions: FloatArray,
    cell: FloatArray | None,
    pbc: npt.ArrayLike,
    parameters: D3BJParameters,
    cutoffs: Cutoffs,
) -> Evaluation:
    """Evaluate with the CUDA backend through the native extension."""
    ctx = native_helpers.context(numbers, parameters, cutoffs)
    result = native_helpers.run(ctx, positions, cell, pbc)
    return Evaluation(result["energy"], result["gradient"], result["virial"])


def oracle_evaluator(
    numbers: npt.ArrayLike,
    positions: FloatArray,
    cell: FloatArray | None,
    pbc: npt.ArrayLike,
    parameters: D3BJParameters,
    cutoffs: Cutoffs,
) -> Evaluation:
    """Evaluate with the NumPy oracle."""
    result = _oracle.evaluate(
        numbers, positions, parameters=parameters, cell=cell, pbc=pbc, cutoffs=cutoffs
    )
    assert result.gradient is not None
    assert result.strain_derivative is not None
    return Evaluation(result.energy, result.gradient, result.strain_derivative)
