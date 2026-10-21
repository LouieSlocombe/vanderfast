"""Central finite differences with a step-size sweep."""

import itertools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_STEPS: tuple[float, ...] = (1e-4, 1e-5, 1e-6)


def position_gradient(
    energy: Callable[[FloatArray], float], positions: FloatArray, step: float
) -> FloatArray:
    """Return the central-difference gradient dE/dx."""
    grad = np.zeros_like(positions)
    for index in np.ndindex(positions.shape):
        plus = positions.copy()
        minus = positions.copy()
        plus[index] += step
        minus[index] -= step
        grad[index] = (energy(plus) - energy(minus)) / (2.0 * step)
    return grad


def strain_derivative(
    energy: Callable[[FloatArray, FloatArray], float],
    positions: FloatArray,
    cell: FloatArray,
    step: float,
) -> FloatArray:
    """Return dE/dstrain for symmetric strains of positions and cell together.

    Row-vector convention: ``x -> x (I + eps)`` and ``H -> H (I + eps)``.
    """
    result = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            eps = np.zeros((3, 3))
            eps[a, b] += 0.5 * step
            eps[b, a] += 0.5 * step
            values = []
            for sign in (1.0, -1.0):
                deform = np.eye(3) + sign * eps
                values.append(energy(positions @ deform, cell @ deform))
            result[a, b] = result[b, a] = (values[0] - values[1]) / (2.0 * step)
    return result


def agrees_over_sweep(
    analytic: FloatArray,
    numeric: Callable[[float], FloatArray],
    tolerance: float,
    steps: Sequence[float] = DEFAULT_STEPS,
) -> tuple[bool, list[float]]:
    """Return whether two consecutive step sizes agree with ``analytic``."""
    errors = [float(np.abs(numeric(h) - analytic).max()) for h in steps]
    ok = any(a <= tolerance and b <= tolerance for a, b in itertools.pairwise(errors))
    return ok, errors
