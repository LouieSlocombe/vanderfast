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

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from vanderfast._oracle_common import (
    DBL_EPSILON,
    Accumulator,
    AtmCounting,
    AtmModel,
    BoolArray,
    Edges,
    FloatArray,
    Geometry,
    IntArray,
    atm_term,
    directed_edges,
    image_ranges,
    lattice_shifts,
    search_basis,
    validate_geometry,
)
from vanderfast.data import ReferenceTables, load_reference_tables
from vanderfast.parameters import Cutoffs, D3BJParameters

CN_EXCLUSION_R2: Final = 1.0e-12
KCN: Final = 16.0
WF: Final = 4.0
MAX_REF: Final = 7

PairCounting = Literal["directed", "upstream"]
AtmC6Policy = Literal["skip", "keep"]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def make_geometry(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    tables: ReferenceTables | None = None,
) -> Geometry:
    """Validate numbers, positions (bohr), cell rows (bohr), and periodicity."""
    tables = tables or load_reference_tables()
    return validate_geometry(numbers, positions, cell, pbc, tables.max_z)


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
# Terms
# --------------------------------------------------------------------------


@dataclass(eq=False)
class _Model:
    """Per-evaluation state shared by the term evaluators."""

    geometry: Geometry
    tables: ReferenceTables
    params: D3BJParameters
    c6: FloatArray
    dc6: FloatArray | None
    acc: Accumulator


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


def _atm_model(model: _Model, cutoff2: float, c6_policy: AtmC6Policy) -> AtmModel:
    """ATM inputs: pair radii ``rs9 * rvdw_ij`` and the D3 exponent ``(alp + 2) / 3``."""
    z = model.geometry.numbers
    rvdw = model.tables.rvdw_flat
    rs9 = D3BJParameters.rs9
    r0 = rs9 * rvdw[_pair_indices(z[:, None], z[None, :])]
    return AtmModel(
        c6=model.c6,
        dc6=model.dc6,
        r0=r0,
        s9=model.params.s9,
        alp3=model.params.atm_exponent / 3.0,
        cutoff2=cutoff2,
        c6_floor=DBL_EPSILON,
        skip_small=c6_policy == "skip",
        acc=model.acc,
    )


def _pair_indices(zi: IntArray, zj: IntArray) -> IntArray:
    big = np.maximum(zi, zj)
    small = np.minimum(zi, zj)
    index: IntArray = big * (big - 1) // 2 + (small - 1)
    return index


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

    acc = Accumulator(n, gradient)
    model = _Model(geometry, tables, parameters, c6, dc6, acc)
    atm = _atm_model(model, cutoffs.disp3**2, atm_c6_policy)
    energy_pair = 0.0
    if use_pair:
        if pair_counting == "directed":
            _pair_term(model, pair_edges, np.full(len(pair_edges), 0.5), book_both=False)
        else:
            half = pair_edges.select(pair_edges.j <= pair_edges.i)
            _pair_term(model, half, np.where(half.i == half.j, 0.5, 1.0), book_both=True)
        energy_pair = float(acc.atom_energy.sum())
    if use_atm:
        atm_term(atm, atm_edges, n, atm_counting)
    energy_total = float(acc.atom_energy.sum())
    if gradient:
        _cn_reverse(model, cn_edges)

    counts = {
        "cn_edges": len(cn_edges),
        "pair_edges": len(pair_edges) if use_pair else 0,
        "atm_edges": len(atm_edges) if use_atm else 0,
        "atm_candidates": atm.counters.candidates,
        "atm_accepted": atm.counters.accepted,
        "atm_c6_skipped": atm.counters.c6_skipped,
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
