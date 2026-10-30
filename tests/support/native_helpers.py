"""Helpers for tests that drive the compiled extension directly (atomic units)."""

import importlib
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from vanderfast.data import read_reference_blob
from vanderfast.parameters import Cutoffs, D3BJParameters
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters

FloatArray = npt.NDArray[np.float64]


def native() -> Any:
    return importlib.import_module("vanderfast._native")


def context(
    numbers: npt.ArrayLike,
    parameters: D3BJParameters,
    cutoffs: Cutoffs | None = None,
    **options: Any,
) -> Any:
    cutoffs = cutoffs or Cutoffs()
    p = parameters
    return native().Context(
        np.asarray(numbers, dtype=np.int32),
        s6=p.s6,
        s8=p.s8,
        a1=p.a1,
        a2=p.a2,
        s9=p.s9,
        alp=p.alp,
        cutoff_cn=cutoffs.cn,
        cutoff_disp2=cutoffs.disp2,
        cutoff_disp3=cutoffs.disp3,
        **options,
    )


def d4_context(
    numbers: npt.ArrayLike,
    parameters: D4Parameters,
    cutoffs: D4Cutoffs | None = None,
    **options: Any,
) -> Any:
    cutoffs = cutoffs or D4Cutoffs()
    p = parameters
    return native().D4Context(
        np.asarray(numbers, dtype=np.int32),
        s6=p.s6,
        s8=p.s8,
        a1=p.a1,
        a2=p.a2,
        s9=p.s9,
        alp=p.alp,
        ga=p.ga,
        gc=p.gc,
        wf=p.wf,
        cutoff_cn=cutoffs.cn,
        cutoff_disp2=cutoffs.disp2,
        cutoff_disp3=cutoffs.disp3,
        **options,
    )


def run(
    ctx: Any,
    positions: FloatArray,
    cell: FloatArray | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    *,
    gradient: bool = True,
    virial: bool = True,
) -> dict[str, Any]:
    flags = tuple(bool(p) for p in np.asarray(pbc, dtype=bool))
    h = np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=np.float64)
    result: dict[str, Any] = ctx.evaluate(
        np.asarray(positions, dtype=np.float64), h, flags, gradient=gradient, virial=virial
    )
    return result


def patched_reference_blob(section: int, start: int, values: Sequence[float]) -> bytes:
    """Return the bundled D3 blob with ``values`` written into ``section`` from entry ``start``."""
    blob = bytearray(read_reference_blob())
    offset = struct.unpack_from("<Q", blob, 120 + (section - 1) * 24 + 8)[0]
    for index, value in enumerate(values):
        struct.pack_into("<d", blob, offset + 8 * (start + index), value)
    return bytes(blob)
