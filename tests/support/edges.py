"""Directed edge lists ``(i -> j, n)`` as sorted integer keys for exact comparisons."""

import numpy as np
import numpy.typing as npt

ATOM_BITS = 16
SHIFT_BITS = 10

IntArray = npt.NDArray[np.int64]


def edge_keys(i: npt.ArrayLike, j: npt.ArrayLike, shift: npt.ArrayLike) -> IntArray:
    """Pack every edge into one int64 and sort: equal edge sets give equal arrays.

    Duplicate edges keep duplicate keys, which ``has_duplicates`` detects. Sorting
    packed keys is much faster than building Python sets of millions of tuples.
    """
    first = np.asarray(i, dtype=np.int64)
    second = np.asarray(j, dtype=np.int64)
    images = np.asarray(shift, dtype=np.int64).reshape(-1, 3) + (1 << (SHIFT_BITS - 1))
    atoms = np.concatenate([first, second])
    if atoms.size and (atoms.min() < 0 or atoms.max() >= 1 << ATOM_BITS):
        msg = f"atom indices must fit in {ATOM_BITS} bits"
        raise ValueError(msg)
    if images.size and (images.min() < 0 or images.max() >= 1 << SHIFT_BITS):
        msg = f"image shifts must fit in {SHIFT_BITS} signed bits"
        raise ValueError(msg)
    keys = (first << ATOM_BITS) | second
    for column in images.T:
        keys = (keys << SHIFT_BITS) | column
    keys.sort()
    return keys


def reversed_edge_keys(i: npt.ArrayLike, j: npt.ArrayLike, shift: npt.ArrayLike) -> IntArray:
    """Keys of the reverse edges ``(j -> i, -n)``."""
    return edge_keys(j, i, -np.asarray(shift))


def has_duplicates(keys: IntArray) -> bool:
    """Whether a sorted key array lists any edge twice."""
    return bool(np.any(keys[1:] == keys[:-1]))
