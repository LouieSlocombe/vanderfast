"""Brute-force NumPy reference implementation of D3(BJ) with ATM.

This module is an executable specification of the numerical contract. It
enumerates explicit lattice sites, differentiates the literal weighted sums,
and exposes every intermediate quantity. It is meant for small systems in
tests and fixture checks, never for production work.

All quantities are in atomic units (bohr, hartree). Lattice vectors are rows
of ``cell``; a site is ``(atom index, integer image n)`` at ``r_j + n @ cell``.

Counting conventions (``pair_counting`` / ``atm_counting``):

``"directed"`` / ``"anchored"``
    The vanderfast baseline: every directed edge ``(i -> j, n)`` with weight
    1/2, and every primary-cell centre with unordered pairs of distinct
    neighbour sites with weight 1/3.
``"upstream"``
    The simple-dftd3 loops: ``jat <= iat`` over all translations with weight
    1/2 for self-images, and ``iat >= jat >= kat`` with ``triple_scale``.

Both describe the same scalar sums; tests use them to check each other.
"""

import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from vanderfast.data import ReferenceTables, load_reference_tables
from vanderfast.errors import InputError
from vanderfast.parameters import Cutoffs, D3BJParameters

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

DBL_EPSILON: Final = sys.float_info.epsilon
CN_EXCLUSION_R2: Final = 1.0e-12
KCN: Final = 16.0
WF: Final = 4.0
MAX_REF: Final = 7

_CHUNK: Final = 1 << 18

PairCounting = Literal["directed", "upstream"]
AtmCounting = Literal["anchored", "upstream"]
AtmC6Policy = Literal["skip", "keep"]


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


def make_geometry(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    tables: ReferenceTables | None = None,
) -> Geometry:
    """Validate numbers, positions (bohr), cell rows (bohr), and periodicity."""
    tables = tables or load_reference_tables()
    z = np.asarray(numbers)
    if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer):
        msg = "numbers must be a one-dimensional integer array"
        raise InputError(msg)
    z = z.astype(np.int64)
    if z.size and (z.min() < 1 or z.max() > tables.max_z):
        msg = f"atomic numbers must be within 1..{tables.max_z}"
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
    step = max(1, _CHUNK // max(n, 1))
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
# Coordination numbers, weights, and C6
# --------------------------------------------------------------------------


def cn_count(rc: FloatArray, r: FloatArray) -> FloatArray:
    """D3 counting function ``1 / (1 + exp(-kcn (rc / r - 1)))``."""
    return 1.0 / (1.0 + np.exp(-KCN * (rc / r - 1.0)))


def coordination_numbers(geometry: Geometry, edges: Edges, tables: ReferenceTables) -> FloatArray:
    """Sum the counting function over directed CN edges (already cutoff-screened)."""
    z = geometry.numbers
    rc = tables.rcov[z[edges.i]] + tables.rcov[z[edges.j]]
    counts = cn_count(rc, np.sqrt(edges.r2))
    return np.bincount(edges.i, weights=counts, minlength=geometry.n_atoms)


def _padded_reference_cn(
    numbers: IntArray, tables: ReferenceTables
) -> tuple[FloatArray, BoolArray]:
    refs = np.zeros((numbers.size, MAX_REF))
    mask = np.zeros((numbers.size, MAX_REF), dtype=bool)
    for index, z in enumerate(numbers):
        values = tables.reference_cn(int(z))
        refs[index, : values.size] = values
        mask[index, : values.size] = True
    return refs, mask


def reference_weights(
    numbers: npt.ArrayLike,
    cn: npt.ArrayLike,
    tables: ReferenceTables | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return Gaussian reference weights and their CN derivatives, padded to 7.

    Reproduces simple-dftd3's ``weight_references`` including the exceptional
    branch: a non-finite normalised weight becomes 1 for every reference whose
    CN equals the maximum valid reference CN (ties all receive 1) and 0
    otherwise; a non-finite derivative becomes 0. Padded entries are 0.
    """
    tables = tables or load_reference_tables()
    z = np.asarray(numbers, dtype=np.int64)
    c = np.asarray(cn, dtype=np.float64)
    refs, mask = _padded_reference_cn(z, tables)
    with np.errstate(all="ignore"):
        dcn = refs - c[:, None]
        gauss = np.where(mask, np.exp(-WF * (c[:, None] - refs) ** 2), 0.0)
        norm = 1.0 / gauss.sum(axis=1)
        dgauss = np.where(mask, (2.0 * WF) * dcn * gauss, 0.0)
        dnorm = dgauss.sum(axis=1)
        weights = gauss * norm[:, None]
        dweights = dgauss * norm[:, None] - (gauss * dnorm[:, None]) * (norm * norm)[:, None]
    max_ref = np.where(mask, refs, -np.inf).max(axis=1)
    fallback = np.where(refs == max_ref[:, None], 1.0, 0.0)
    exceptional = mask & ~np.isfinite(weights)
    weights = np.where(exceptional, fallback, weights)
    dweights = np.where(mask & ~np.isfinite(dweights), 0.0, dweights)
    return np.where(mask, weights, 0.0), np.where(mask, dweights, 0.0)


def c6_coefficients(
    numbers: npt.ArrayLike,
    weights: FloatArray,
    dweights: FloatArray | None,
    tables: ReferenceTables | None = None,
) -> tuple[FloatArray, FloatArray | None]:
    """Return ``C6[i, j]`` and ``dC6[i, j] = dC6_ij / dCN_i`` for all atom pairs."""
    tables = tables or load_reference_tables()
    z = np.asarray(numbers, dtype=np.int64)
    n = z.size
    c6 = np.zeros((n, n))
    dc6 = None if dweights is None else np.zeros((n, n))
    species = np.unique(z)
    for zi in species:
        rows = np.flatnonzero(z == zi)
        ni = int(tables.nref[zi])
        for zj in species:
            cols = np.flatnonzero(z == zj)
            nj = int(tables.nref[zj])
            block = tables.c6_block(int(zi), int(zj))
            wi = weights[rows, :ni]
            wj = weights[cols, :nj]
            c6[np.ix_(rows, cols)] = wi @ block @ wj.T
            if dc6 is not None and dweights is not None:
                dc6[np.ix_(rows, cols)] = dweights[rows, :ni] @ block @ wj.T
    return c6, dc6


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------


@dataclass(eq=False)
class _Accumulator:
    n: int
    gradient: bool
    atom_energy: FloatArray = field(init=False)
    grad: FloatArray = field(init=False)
    sigma: FloatArray = field(init=False)
    g: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.atom_energy = np.zeros(self.n)
        self.grad = np.zeros((self.n, 3))
        self.sigma = np.zeros((3, 3))
        self.g = np.zeros(self.n)

    def energy(self, index: IntArray, values: FloatArray) -> None:
        self.atom_energy += np.bincount(index, weights=values, minlength=self.n)

    def triangle_energy(
        self, i: IntArray, j: IntArray, k: IntArray, values: FloatArray, *, book_all: bool
    ) -> None:
        if book_all:
            third = values / 3.0
            self.energy(i, third)
            self.energy(j, third)
            self.energy(k, third)
        else:
            self.energy(i, values)

    def force(self, index: IntArray, values: FloatArray) -> None:
        for c in range(3):
            self.grad[:, c] += np.bincount(index, weights=values[:, c], minlength=self.n)

    def adjoint(self, index: IntArray, values: FloatArray) -> None:
        self.g += np.bincount(index, weights=values, minlength=self.n)

    def virial(self, vectors: FloatArray, derivatives: FloatArray) -> None:
        self.sigma += np.einsum("ea,eb->ab", vectors, derivatives)


@dataclass(eq=False)
class _AtmCounters:
    candidates: int = 0
    accepted: int = 0
    c6_skipped: int = 0


@dataclass(eq=False)
class _Model:
    """Per-evaluation state shared by the term evaluators."""

    geometry: Geometry
    tables: ReferenceTables
    params: D3BJParameters
    c6: FloatArray
    dc6: FloatArray | None
    acc: _Accumulator
    atm_cutoff2: float
    c6_policy: AtmC6Policy
    counters: _AtmCounters = field(default_factory=_AtmCounters)


@dataclass(frozen=True, eq=False)
class _Triangles:
    """Candidate triangles: centre i, sites j and k at ``vij`` and ``vik``."""

    i: IntArray
    j: IntArray
    k: IntArray
    vij: FloatArray
    vik: FloatArray
    weight: FloatArray


def _pair_term(model: _Model, edges: Edges, weight: FloatArray, *, book_both: bool) -> None:
    z = model.geometry.numbers
    tables, params, acc = model.tables, model.params, model.acc
    i, j, d, r2 = edges.i, edges.j, edges.d, edges.r2
    kappa = 3.0 * tables.r4r2[z[i]] * tables.r4r2[z[j]]
    r0 = params.a1 * np.sqrt(kappa) + params.a2
    r0_2 = r0 * r0
    r0_6 = r0_2 * r0_2 * r0_2
    r0_8 = r0_6 * r0_2
    r4 = r2 * r2
    t6 = 1.0 / (r4 * r2 + r0_6)
    t8 = 1.0 / (r4 * r4 + r0_8)
    edisp = params.s6 * t6 + params.s8 * kappa * t8
    c6e = model.c6[i, j]
    energy = -weight * c6e * edisp
    if book_both:
        acc.energy(i, 0.5 * energy)
        acc.energy(j, 0.5 * energy)
    else:
        acc.energy(i, energy)
    if model.dc6 is None or not acc.gradient:
        return
    d6 = -6.0 * r4 * t6 * t6
    d8 = -8.0 * r4 * r2 * t8 * t8
    gdisp = params.s6 * d6 + params.s8 * kappa * d8
    de_dd = (-weight * c6e * gdisp)[:, None] * d
    acc.force(j, de_dd)
    acc.force(i, -de_dd)
    acc.virial(d, de_dd)
    acc.adjoint(i, -weight * model.dc6[i, j] * edisp)
    acc.adjoint(j, -weight * model.dc6[j, i] * edisp)


def _screen_triangles(model: _Model, tri: _Triangles) -> tuple[_Triangles, FloatArray] | None:
    """Apply the third-edge cutoff and the C6 policy; return accepted triangles."""
    counters = model.counters
    counters.candidates += int(tri.i.size)
    vjk = tri.vik - tri.vij
    r2jk = np.einsum("ec,ec->e", vjk, vjk)
    valid = (r2jk <= model.atm_cutoff2) & (r2jk >= DBL_EPSILON)
    c6 = model.c6
    small = (
        (np.abs(c6[tri.i, tri.j]) < DBL_EPSILON)
        | (np.abs(c6[tri.i, tri.k]) < DBL_EPSILON)
        | (np.abs(c6[tri.j, tri.k]) < DBL_EPSILON)
    )
    counters.c6_skipped += int(np.count_nonzero(valid & small))
    if model.c6_policy == "skip":
        valid &= ~small
    counters.accepted += int(np.count_nonzero(valid))
    if not valid.any():
        return None
    accepted = _Triangles(
        tri.i[valid], tri.j[valid], tri.k[valid], tri.vij[valid], tri.vik[valid], tri.weight[valid]
    )
    return accepted, vjk[valid]


def _atm_triangles(model: _Model, candidates: _Triangles, *, book_all: bool) -> None:
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
    r0 = _atm_r0(model, i, j, k)
    alp3 = model.params.atm_exponent / 3.0
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
    pre = tri.weight * model.params.s9 * np.sqrt(np.abs(c6ij * c6ik * c6jk))
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


def _atm_r0(model: _Model, i: IntArray, j: IntArray, k: IntArray) -> FloatArray:
    """Product of the three scaled vdW radii, in upstream operation order."""
    z = model.geometry.numbers
    rvdw = model.tables.rvdw_flat
    rs9 = D3BJParameters.rs9
    r0: FloatArray = (
        (rs9 * rvdw[_pair_indices(z[i], z[j])])
        * (rs9 * rvdw[_pair_indices(z[i], z[k])])
        * (rs9 * rvdw[_pair_indices(z[j], z[k])])
    )
    return r0


def _pair_indices(zi: IntArray, zj: IntArray) -> IntArray:
    big = np.maximum(zi, zj)
    small = np.minimum(zi, zj)
    index: IntArray = big * (big - 1) // 2 + (small - 1)
    return index


def _triple_scale(i: IntArray, j: IntArray, k: IntArray) -> FloatArray:
    """Upstream ``triple_scale``: 1, 1/2 for exactly two equal, 1/6 for all equal."""
    scale = np.ones(i.size)
    scale[(i == j) | (j == k) | (i == k)] = 0.5
    scale[(i == j) & (j == k)] = 1.0 / 6.0
    return scale


def _chunked_pairs(n: int) -> Iterator[tuple[IntArray, IntArray]]:
    """Yield blocks of unordered index pairs ``a < b`` of ``range(n)``."""
    if n < 2:  # noqa: PLR2004
        return
    rows_per_block = max(1, _CHUNK // n)
    for start in range(0, n - 1, rows_per_block):
        rows = np.arange(start, min(n - 1, start + rows_per_block), dtype=np.int64)
        counts = n - 1 - rows
        a = np.repeat(rows, counts)
        offsets = np.arange(a.size, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        yield a, a + 1 + offsets


def _anchored_triangles(centre: int, site_j: IntArray, site_d: FloatArray) -> Iterator[_Triangles]:
    """Unordered pairs of distinct neighbour sites, weight 1/3 per anchor."""
    for a, b in _chunked_pairs(site_j.size):
        count = a.size
        yield _Triangles(
            np.full(count, centre, dtype=np.int64),
            site_j[a],
            site_j[b],
            site_d[a],
            site_d[b],
            np.full(count, 1.0 / 3.0),
        )


def _upstream_triangles(centre: int, site_j: IntArray, site_d: FloatArray) -> Iterator[_Triangles]:
    """Upstream ``iat >= jat >= kat`` loops over all translations with ``triple_scale``."""
    outer = np.flatnonzero(site_j <= centre)
    step = max(1, _CHUNK // max(site_j.size, 1))
    for start in range(0, outer.size, step):
        grid_u, grid_v = np.meshgrid(
            outer[start : start + step], np.arange(site_j.size), indexing="ij"
        )
        flat_u: IntArray = grid_u.ravel().astype(np.int64)
        flat_v: IntArray = grid_v.ravel().astype(np.int64)
        keep = site_j[flat_v] <= site_j[flat_u]
        uu, vv = flat_u[keep], flat_v[keep]
        ii = np.full(uu.size, centre, dtype=np.int64)
        yield _Triangles(
            ii,
            site_j[uu],
            site_j[vv],
            site_d[uu],
            site_d[vv],
            _triple_scale(ii, site_j[uu], site_j[vv]),
        )


def _atm_term(model: _Model, edges: Edges, counting: AtmCounting) -> None:
    order = np.argsort(edges.i, kind="stable")
    bounds = np.searchsorted(edges.i[order], np.arange(model.geometry.n_atoms + 1))
    enumerate_triangles = _anchored_triangles if counting == "anchored" else _upstream_triangles
    for centre in range(model.geometry.n_atoms):
        rows = order[bounds[centre] : bounds[centre + 1]]
        for candidates in enumerate_triangles(centre, edges.j[rows], edges.d[rows]):
            _atm_triangles(model, candidates, book_all=counting == "upstream")


def _cn_reverse(model: _Model, edges: Edges) -> None:
    z = model.geometry.numbers
    rc = model.tables.rcov[z[edges.i]] + model.tables.rcov[z[edges.j]]
    r = np.sqrt(edges.r2)
    cf = cn_count(rc, r)
    dcf = -cf * (1.0 - cf) * KCN * rc / edges.r2
    acc = model.acc
    de_dd = (acc.g[edges.i] * dcf / r)[:, None] * edges.d
    acc.force(edges.j, de_dd)
    acc.force(edges.i, -de_dd)
    acc.virial(edges.d, de_dd)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class OracleResult:
    """Energies, derivatives, and intermediates in atomic units.

    ``gradient`` is dE/dr (hartree/bohr). ``strain_derivative`` is dE/dstrain
    (hartree) for a homogeneous strain of positions and cell. ``dc6[i, j]`` is
    the derivative of ``c6[i, j]`` with respect to ``cn[i]``. ``g`` is dE/dCN.
    """

    energy: float
    energy_pair: float
    energy_atm: float
    atom_energies: FloatArray
    gradient: FloatArray | None
    strain_derivative: FloatArray | None
    cn: FloatArray
    weights: FloatArray
    dweights: FloatArray | None
    c6: FloatArray
    dc6: FloatArray | None
    g: FloatArray | None
    counts: dict[str, int]


def evaluate(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    parameters: D3BJParameters,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    cutoffs: Cutoffs | None = None,
    gradient: bool = True,
    tables: ReferenceTables | None = None,
    pair_counting: PairCounting = "directed",
    atm_counting: AtmCounting = "anchored",
    atm_c6_policy: AtmC6Policy = "skip",
) -> OracleResult:
    """Evaluate D3(BJ) (+ ATM when ``parameters.atm``) by brute force.

    Args:
        numbers: Atomic numbers, shape (N,).
        positions: Cartesian positions in bohr, shape (N, 3).
        parameters: Resolved damping parameters.
        cell: Lattice vectors in bohr as rows; ignored along non-periodic axes.
        pbc: Periodicity flags per lattice vector.
        cutoffs: Real-space cutoffs in bohr (reference defaults if omitted).
        gradient: Also compute gradient, strain derivative, and adjoints.
        tables: Reference tables (the bundled tables if omitted).
        pair_counting: Enumeration of the pair sum; see the module docstring.
        atm_counting: Enumeration of the triple sum; see the module docstring.
        atm_c6_policy: ``"skip"`` drops triangles with any ``|C6| < eps`` (the
            upstream derivative path and vanderfast policy); ``"keep"`` keeps
            them (the upstream energy-only path).

    Returns:
        The full set of results and intermediates.
    """
    tables = tables or load_reference_tables()
    cutoffs = cutoffs or Cutoffs()
    geometry = make_geometry(numbers, positions, cell, pbc, tables)
    n = geometry.n_atoms
    use_pair = parameters.pair
    use_atm = parameters.atm
    rmax = max(
        cutoffs.cn,
        cutoffs.disp2 if use_pair else 0.0,
        cutoffs.disp3 if use_atm else 0.0,
    )
    edges = directed_edges(geometry, rmax)
    cn_edges = edges.select((edges.r2 <= cutoffs.cn**2) & (edges.r2 >= CN_EXCLUSION_R2))
    pair_edges = edges.select((edges.r2 <= cutoffs.disp2**2) & (edges.r2 >= DBL_EPSILON))
    atm_edges = edges.select((edges.r2 <= cutoffs.disp3**2) & (edges.r2 >= DBL_EPSILON))

    cn = coordination_numbers(geometry, cn_edges, tables)
    weights, all_dweights = reference_weights(geometry.numbers, cn, tables)
    dweights = all_dweights if gradient else None
    c6, dc6 = c6_coefficients(geometry.numbers, weights, dweights, tables)

    acc = _Accumulator(n, gradient)
    model = _Model(geometry, tables, parameters, c6, dc6, acc, cutoffs.disp3**2, atm_c6_policy)
    energy_pair = 0.0
    if use_pair:
        if pair_counting == "directed":
            _pair_term(model, pair_edges, np.full(len(pair_edges), 0.5), book_both=False)
        else:
            half = pair_edges.select(pair_edges.j <= pair_edges.i)
            _pair_term(model, half, np.where(half.i == half.j, 0.5, 1.0), book_both=True)
        energy_pair = float(acc.atom_energy.sum())
    if use_atm:
        _atm_term(model, atm_edges, atm_counting)
    energy_total = float(acc.atom_energy.sum())
    if gradient:
        _cn_reverse(model, cn_edges)

    counts = {
        "cn_edges": len(cn_edges),
        "pair_edges": len(pair_edges) if use_pair else 0,
        "atm_edges": len(atm_edges) if use_atm else 0,
        "atm_candidates": model.counters.candidates,
        "atm_accepted": model.counters.accepted,
        "atm_c6_skipped": model.counters.c6_skipped,
    }
    return OracleResult(
        energy=energy_total,
        energy_pair=energy_pair,
        energy_atm=energy_total - energy_pair,
        atom_energies=acc.atom_energy,
        gradient=acc.grad if gradient else None,
        strain_derivative=acc.sigma if gradient else None,
        cn=cn,
        weights=weights,
        dweights=dweights,
        c6=c6,
        dc6=dc6,
        g=acc.g if gradient else None,
        counts=counts,
    )


__all__ = [
    "Edges",
    "Geometry",
    "OracleResult",
    "c6_coefficients",
    "cn_count",
    "coordination_numbers",
    "directed_edges",
    "evaluate",
    "image_ranges",
    "lattice_shifts",
    "make_geometry",
    "reference_weights",
    "search_basis",
]
