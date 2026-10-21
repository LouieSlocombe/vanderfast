"""Helpers for tests that drive the compiled extension directly (atomic units)."""

import importlib
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from vanderfast.data import SECTION_REF_CN, read_reference_blob
from vanderfast.parameters import Cutoffs, D3BJParameters

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


def edge_keys(i: npt.ArrayLike, j: npt.ArrayLike, shift: npt.ArrayLike) -> set[tuple[int, ...]]:
    shifts = np.asarray(shift).reshape(-1, 3)
    return {
        (int(a), int(b), *map(int, n))
        for a, b, n in zip(np.asarray(i), np.asarray(j), shifts, strict=True)
    }


def tables_with_reference_cn(z: int, values: list[float]) -> bytes:
    """Return the bundled blob with the reference CNs of element ``z`` replaced."""
    from vanderfast.data import load_reference_tables  # noqa: PLC0415

    blob = bytearray(read_reference_blob())
    offset = struct.unpack_from("<Q", blob, 120 + (SECTION_REF_CN - 1) * 24 + 8)[0]
    start = int(load_reference_tables().ref_offset[z])
    for index, value in enumerate(values):
        struct.pack_into("<d", blob, offset + 8 * (start + index), value)
    return bytes(blob)
