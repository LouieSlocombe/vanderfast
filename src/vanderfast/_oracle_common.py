"""Model-agnostic pieces of the brute-force NumPy oracles.

Geometry validation, complete lattice-image enumeration, directed edges, the
energy/force/virial/adjoint accumulator, and the Axilrod-Teller-Muto triangle
machinery are shared by the D3 oracle (``_oracle``) and the D4 oracle
(``_oracle_d4``). Everything is in atomic units; lattice vectors are rows of
``cell`` and a site is ``(atom index, integer image n)`` at ``r_j + n @ cell``.
"""

import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from vanderfast.errors import InputError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

DBL_EPSILON: Final = sys.float_info.epsilon

CHUNK: Final = 1 << 18

AtmCounting = Literal["anchored", "upstream"]


# --------------------------------------------------------------------------
# Geometry and lattice images
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Geometry:
    """Validated input geometry in bohr."""

    numbers: IntArray
    positions: FloatArray
    cell: FloatArray
    pbc: BoolArray

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the primary cell."""
        return int(self.numbers.shape[0])


def validate_geometry(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    cell: npt.ArrayLike | None,
    pbc: npt.ArrayLike,
    max_z: int,
) -> Geometry:
    """Validate numbers (1..max_z), positions and cell rows (bohr), and periodicity."""
    z = np.asarray(numbers)
    if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer):
        msg = "numbers must be a one-dimensional integer array"
        raise InputError(msg)
    z = z.astype(np.int64)
    if z.size and (z.min() < 1 or z.max() > max_z):
        msg = f"atomic numbers must be within 1..{max_z}"
        raise InputError(msg)
    x = np.asarray(positions, dtype=np.float64)
    if x.shape != (z.size, 3):
        msg = f"positions must have shape ({z.size}, 3), got {x.shape}"
        raise InputError(msg)
    if not np.all(np.isfinite(x)):
        msg = "positions must be finite"
        raise InputError(msg)
    h = np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        msg = "cell must be a finite 3x3 array with lattice vectors as rows"
        raise InputError(msg)
    p = np.asarray(pbc, dtype=bool)
    if p.shape != (3,):
        msg = "pbc must contain three booleans"
        raise InputError(msg)
    geometry = Geometry(numbers=z, positions=x, cell=h, pbc=p)
    if p.any():
        search_basis(geometry)
    return geometry


def search_basis(geometry: Geometry) -> FloatArray:
    """Return the search basis: active lattice vectors completed orthogonally.

    Inactive directions are replaced by unit vectors orthogonal to the span of
    the active lattice vectors. With no periodic direction the identity is
    returned.

    Raises:
        InputError: If the active lattice vectors are not linearly independent.
    """
    active = np.flatnonzero(geometry.pbc)
    basis = np.eye(3)
    vectors = geometry.cell[active]
    if active.size == 0:
        return basis
    gram = vectors @ vectors.T
    scale = float(np.prod(np.diag(gram)))
    if scale <= 0.0 or np.linalg.det(gram) <= 1e-12 * scale:
        msg = "active lattice vectors must be linearly independent"
        raise InputError(msg)
    if active.size == 3:  # noqa: PLR2004
        return geometry.cell.copy()
    if active.size == 2:  # noqa: PLR2004
        normal = np.cross(vectors[0], vectors[1])
        completion = [normal / np.linalg.norm(normal)]
    else:
        a = vectors[0] / np.linalg.norm(vectors[0])
        trial = np.eye(3)[int(np.argmin(np.abs(a)))]
        u = np.cross(a, trial)
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        completion = [u, v / np.linalg.norm(v)]
    inactive = np.flatnonzero(~geometry.pbc)
    basis[active] = vectors
    for index, vector in zip(inactive, completion, strict=True):
        basis[index] = vector
    return basis


def image_ranges(geometry: Geometry, cutoff: float) -> IntArray:
    """Return per-direction image bounds that are complete for any input wrapping.

    For an active direction ``k`` with plane spacing ``h_k`` and fractional
    spread ``s_k`` of the (unwrapped) positions, every site within ``cutoff``
    satisfies ``|n_k| <= cutoff / h_k + s_k``.
    """
    rep = np.zeros(3, dtype=np.int64)
    if not geometry.pbc.any():
        return rep
    basis = search_basis(geometry)
    volume = abs(float(np.linalg.det(basis)))
    frac = geometry.positions @ np.linalg.inv(basis)
    for k in np.flatnonzero(geometry.pbc):
        area = float(np.linalg.norm(np.cross(basis[(k + 1) % 3], basis[(k + 2) % 3])))
        spacing = volume / area
        spread = float(frac[:, k].max() - frac[:, k].min()) if geometry.n_atoms else 0.0
        rep[k] = math.ceil(cutoff / spacing) + math.ceil(spread)
    return rep


def lattice_shifts(rep: IntArray) -> IntArray:
    """Return all integer image shifts within ``[-rep, rep]`` per direction."""
    axes = [np.arange(-r, r + 1, dtype=np.int64) for r in rep]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1)


@dataclass(frozen=True, eq=False)
class Edges:
    """Directed site pairs ``(i -> j, n)`` with ``d = r_j + n H - r_i``."""

    i: IntArray
    j: IntArray
    shift: IntArray
    d: FloatArray
    r2: FloatArray

    def select(self, mask: BoolArray) -> Edges:
        """Return the subset of edges where ``mask`` is true."""
        return Edges(self.i[mask], self.j[mask], self.shift[mask], self.d[mask], self.r2[mask])

    def __len__(self) -> int:
        """Number of edges."""
        return int(self.i.shape[0])


def directed_edges(geometry: Geometry, cutoff: float) -> Edges:
    """Enumerate every directed site pair with ``r^2 <= cutoff^2``.

    The zero-shift self pair ``(i, i, 0)`` is excluded. Other images of the
    same atom are kept. No lower distance threshold is applied here; each term
    applies its own.
    """
    n = geometry.n_atoms
    shifts = lattice_shifts(image_ranges(geometry, cutoff))
    translations = shifts.astype(np.float64) @ geometry.cell
    cutoff2 = cutoff * cutoff
    parts: list[tuple[IntArray, IntArray, IntArray, FloatArray, FloatArray]] = []
    x = geometry.positions
    zero = np.all(shifts == 0, axis=1)
    step = max(1, CHUNK // max(n, 1))
    for i in range(n):
        for start in range(0, shifts.shape[0], step):
            t = translations[start : start + step]
            d = (x[None, :, :] - x[i]) + t[:, None, :]
            r2 = np.einsum("mnc,mnc->mn", d, d)
            keep = r2 <= cutoff2
            keep[zero[start : start + step], i] = False
            m_idx, j_idx = np.nonzero(keep)
            parts.append(
                (
                    np.full(m_idx.size, i, dtype=np.int64),
                    j_idx.astype(np.int64),
                    shifts[start + m_idx],
                    d[m_idx, j_idx],
                    r2[m_idx, j_idx],
                )
            )
    if not parts:
        empty_i = np.zeros(0, dtype=np.int64)
        return Edges(empty_i, empty_i, np.zeros((0, 3), np.int64), np.zeros((0, 3)), np.zeros(0))
    return Edges(*(np.concatenate(arrays) for arrays in zip(*parts, strict=True)))


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Accumulator:
    """Per-atom energies, gradient, virial (strain derivative), and dE/dCN adjoint."""

    n: int
    gradient: bool
    atom_energy: FloatArray = field(init=False)
    grad: FloatArray = field(init=False)
    sigma: FloatArray = field(init=False)
    g: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Zero every accumulator."""
        self.atom_energy = np.zeros(self.n)
        self.grad = np.zeros((self.n, 3))
        self.sigma = np.zeros((3, 3))
        self.g = np.zeros(self.n)

    def energy(self, index: IntArray, values: FloatArray) -> None:
        """Book energies onto atoms."""
        self.atom_energy += np.bincount(index, weights=values, minlength=self.n)

    def triangle_energy(
        self, i: IntArray, j: IntArray, k: IntArray, values: FloatArray, *, book_all: bool
    ) -> None:
        """Book triangle energies onto all three corners (thirds) or the centre only."""
        if book_all:
            third = values / 3.0
            self.energy(i, third)
            self.energy(j, third)
            self.energy(k, third)
        else:
            self.energy(i, values)

    def force(self, index: IntArray, values: FloatArray) -> None:
        """Add gradient contributions ``values`` (dE/dr) to atoms."""
        for c in range(3):
            self.grad[:, c] += np.bincount(index, weights=values[:, c], minlength=self.n)

    def adjoint(self, index: IntArray, values: FloatArray) -> None:
        """Add dE/dCN contributions."""
        self.g += np.bincount(index, weights=values, minlength=self.n)

    def virial(self, vectors: FloatArray, derivatives: FloatArray) -> None:
        """Add ``sum_e vectors_e (x) derivatives_e`` to the strain derivative."""
        self.sigma += np.einsum("ea,eb->ab", vectors, derivatives)


# --------------------------------------------------------------------------
# Axilrod-Teller-Muto triangles
# --------------------------------------------------------------------------


@dataclass(eq=False)
class AtmCounters:
    """Triangle bookkeeping for one evaluation."""

    candidates: int = 0
    accepted: int = 0
    c6_skipped: int = 0


@dataclass(eq=False)
class AtmModel:
    """Everything the ATM triangle evaluator needs.

    ``r0[i, j]`` is the per-pair damping radius; a triangle uses
    ``r0[i, j] * r0[i, k] * r0[j, k]``. Triangles with any ``|C6| < c6_floor``
    are counted in ``counters.c6_skipped`` and dropped when ``skip_small``.
    ``dc6[i, j]`` is the derivative of ``c6[i, j]`` with respect to ``cn[i]``.
    """

    c6: FloatArray
    dc6: FloatArray | None
    r0: FloatArray
    s9: float
    alp3: float
    cutoff2: float
    c6_floor: float
    skip_small: bool
    acc: Accumulator
    counters: AtmCounters = field(default_factory=AtmCounters)


@dataclass(frozen=True, eq=False)
class Triangles:
    """Candidate triangles: centre i, sites j and k at ``vij`` and ``vik``."""

    i: IntArray
    j: IntArray
    k: IntArray
    vij: FloatArray
    vik: FloatArray
    weight: FloatArray


def _screen_triangles(model: AtmModel, tri: Triangles) -> tuple[Triangles, FloatArray] | None:
    """Apply the third-edge cutoff and the C6 policy; return accepted triangles."""
    counters = model.counters
    counters.candidates += int(tri.i.size)
    vjk = tri.vik - tri.vij
    r2jk = np.einsum("ec,ec->e", vjk, vjk)
    valid = (r2jk <= model.cutoff2) & (r2jk >= DBL_EPSILON)
    c6 = model.c6
    small = (
        (np.abs(c6[tri.i, tri.j]) < model.c6_floor)
        | (np.abs(c6[tri.i, tri.k]) < model.c6_floor)
        | (np.abs(c6[tri.j, tri.k]) < model.c6_floor)
    )
    counters.c6_skipped += int(np.count_nonzero(valid & small))
    if model.skip_small:
        valid &= ~small
    counters.accepted += int(np.count_nonzero(valid))
    if not valid.any():
        return None
    accepted = Triangles(
        tri.i[valid], tri.j[valid], tri.k[valid], tri.vij[valid], tri.vik[valid], tri.weight[valid]
    )
    return accepted, vjk[valid]


def _atm_triangles(model: AtmModel, candidates: Triangles, *, book_all: bool) -> None:
    """Evaluate candidate triangles whose two centre edges already passed screening."""
    screened = _screen_triangles(model, candidates)
    if screened is None:
        return
    tri, vjk = screened
    i, j, k, vij, vik = tri.i, tri.j, tri.k, tri.vij, tri.vik
    c6ij, c6ik, c6jk = model.c6[i, j], model.c6[i, k], model.c6[j, k]
    r2ij = np.einsum("ec,ec->e", vij, vij)
    r2ik = np.einsum("ec,ec->e", vik, vik)
    r2jk = np.einsum("ec,ec->e", vjk, vjk)
    r0 = model.r0[i, j] * model.r0[i, k] * model.r0[j, k]
    alp3 = model.alp3
    r2 = r2ij * r2ik * r2jk
    r1 = np.sqrt(r2)
    r3 = r2 * r1
    r5 = r3 * r2
    q = 6.0 * (r0 / r1) ** alp3
    fdmp = 1.0 / (1.0 + q)
    a = r2ij + r2jk - r2ik
    b = r2ij - r2jk + r2ik
    c = -r2ij + r2jk + r2ik
    ang = 0.375 * a * b * c / r5 + 1.0 / r3
    pre = tri.weight * model.s9 * np.sqrt(np.abs(c6ij * c6ik * c6jk))
    energy = pre * ang * fdmp
    acc = model.acc
    acc.triangle_energy(i, j, k, energy, book_all=book_all)
    if model.dc6 is None or not acc.gradient:
        return

    def partial(x: FloatArray, dp: FloatArray) -> FloatArray:
        """Derivative of the triangle energy with respect to one squared edge."""
        dang = 0.375 * (dp / r5 - 2.5 * a * b * c / (r5 * x)) - 1.5 / (r3 * x)
        dfdmp = fdmp * fdmp * (0.5 * alp3) * q / x
        result: FloatArray = pre * (dang * fdmp + ang * dfdmp)
        return result

    g_ij = (2.0 * partial(r2ij, b * c + a * c - a * b))[:, None] * vij
    g_ik = (2.0 * partial(r2ik, -b * c + a * c + a * b))[:, None] * vik
    g_jk = (2.0 * partial(r2jk, b * c - a * c + a * b))[:, None] * vjk
    acc.force(i, -(g_ij + g_ik))
    acc.force(j, g_ij - g_jk)
    acc.force(k, g_ik + g_jk)
    acc.virial(vij, g_ij)
    acc.virial(vik, g_ik)
    acc.virial(vjk, g_jk)
    half = 0.5 * energy
    dc6 = model.dc6
    acc.adjoint(i, half * (dc6[i, j] / c6ij + dc6[i, k] / c6ik))
    acc.adjoint(j, half * (dc6[j, i] / c6ij + dc6[j, k] / c6jk))
    acc.adjoint(k, half * (dc6[k, i] / c6ik + dc6[k, j] / c6jk))


def triple_scale(i: IntArray, j: IntArray, k: IntArray) -> FloatArray:
    """Upstream ``triple_scale``: 1, 1/2 for exactly two equal, 1/6 for all equal."""
    scale = np.ones(i.size)
    scale[(i == j) | (j == k) | (i == k)] = 0.5
    scale[(i == j) & (j == k)] = 1.0 / 6.0
    return scale


def chunked_pairs(n: int) -> Iterator[tuple[IntArray, IntArray]]:
    """Yield blocks of unordered index pairs ``a < b`` of ``range(n)``."""
    if n < 2:  # noqa: PLR2004
        return
    rows_per_block = max(1, CHUNK // n)
    for start in range(0, n - 1, rows_per_block):
        rows = np.arange(start, min(n - 1, start + rows_per_block), dtype=np.int64)
        counts = n - 1 - rows
        a = np.repeat(rows, counts)
        offsets = np.arange(a.size, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        yield a, a + 1 + offsets


def anchored_triangles(centre: int, site_j: IntArray, site_d: FloatArray) -> Iterator[Triangles]:
    """Unordered pairs of distinct neighbour sites, weight 1/3 per anchor."""
    for a, b in chunked_pairs(site_j.size):
        count = a.size
        yield Triangles(
            np.full(count, centre, dtype=np.int64),
            site_j[a],
            site_j[b],
            site_d[a],
            site_d[b],
            np.full(count, 1.0 / 3.0),
        )


def upstream_triangles(centre: int, site_j: IntArray, site_d: FloatArray) -> Iterator[Triangles]:
    """Upstream ``iat >= jat >= kat`` loops over all translations with ``triple_scale``."""
    outer = np.flatnonzero(site_j <= centre)
    step = max(1, CHUNK // max(site_j.size, 1))
    for start in range(0, outer.size, step):
        grid_u, grid_v = np.meshgrid(
            outer[start : start + step], np.arange(site_j.size), indexing="ij"
        )
        flat_u: IntArray = grid_u.ravel().astype(np.int64)
        flat_v: IntArray = grid_v.ravel().astype(np.int64)
        keep = site_j[flat_v] <= site_j[flat_u]
        uu, vv = flat_u[keep], flat_v[keep]
        ii = np.full(uu.size, centre, dtype=np.int64)
        yield Triangles(
            ii,
            site_j[uu],
            site_j[vv],
            site_d[uu],
            site_d[vv],
            triple_scale(ii, site_j[uu], site_j[vv]),
        )


def atm_term(model: AtmModel, edges: Edges, n_atoms: int, counting: AtmCounting) -> None:
    """Sum the ATM energy (and derivatives) over all triangles anchored on ``edges``."""
    order = np.argsort(edges.i, kind="stable")
    bounds = np.searchsorted(edges.i[order], np.arange(n_atoms + 1))
    enumerate_triangles = anchored_triangles if counting == "anchored" else upstream_triangles
    for centre in range(n_atoms):
        rows = order[bounds[centre] : bounds[centre + 1]]
        for candidates in enumerate_triangles(centre, edges.j[rows], edges.d[rows]):
            _atm_triangles(model, candidates, book_all=counting == "upstream")
