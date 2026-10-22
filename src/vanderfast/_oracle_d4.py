"""Brute-force NumPy reference implementation of DFT-D4 with EEQ charges and ATM.

This module is an executable specification of the D4 numerical contract
(dftd4 v4.2.0 with the multicharge v0.5.0 EEQ model). It enumerates explicit
lattice sites, solves the charge equations densely, differentiates the literal
sums, and exposes every intermediate quantity. It is meant for small systems in
tests and fixture checks, never for production work.

All quantities are in atomic units (bohr, hartree, e). Lattice vectors are rows
of ``cell``. Any periodic flag makes the system fully periodic over the supplied
cell, as in dftd4.

Periodic charges (``ewald``):

``"upstream"``
    The multicharge definition: positions wrapped to the central cell,
    Wigner-Seitz image averaging (including its self-pair indexing quirk), a
    fixed 125-translation real-space and 124-vector reciprocal sum, and the
    Ewald parameter from multicharge's bisection. This is what dftd4 computes;
    it is not converged and depends on the choice of cell.
``"converged"``
    The same model with complete lattice sums: real-space images and
    reciprocal vectors are included until each omitted term is below
    ``1e-20``. The result is independent of the Ewald parameter and of the
    supercell chosen to describe the same crystal.

The force chain through the charges uses the adjoint of the bordered EEQ
system, which is exactly the contraction ``dq/dr . dE/dq`` that dftd4 forms
from an explicit inverse.
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from vanderfast._oracle_common import (
    DBL_EPSILON,
    Accumulator,
    AtmCounting,
    AtmModel,
    Edges,
    FloatArray,
    Geometry,
    IntArray,
    atm_term,
    directed_edges,
    validate_geometry,
)
from vanderfast.data.d4 import D4ReferenceTables, load_d4_reference_tables
from vanderfast.errors import ChargeSolverError
from vanderfast.parameters_d4 import EEQ_CN_CUTOFF, D4Cutoffs, D4Parameters

MAX_REF: Final = 7
CN_EXCLUSION_R2: Final = 1.0e-12
KCN: Final = 7.5
K4: Final = 4.10451
K5: Final = 19.08857
K6: Final = 2 * 11.28174**2.0
CN_MAX: Final = 8.0
EEQ_REG: Final = 1.0e-14
SQRTPI: Final = math.sqrt(math.pi)
SQRT2PI: Final = math.sqrt(2.0 / math.pi)
EWALD_EPS: Final = math.sqrt(DBL_EPSILON)
WS_TOL: Final = 0.01
EPS_NORM: Final = sys.float_info.min**0.5
DBL_MIN: Final = sys.float_info.min
CONVERGED_TAIL: Final = 1e-20
ALPHA_ITERATIONS: Final = 30

EwaldMode = Literal["upstream", "converged"]
PairCounting = Literal["directed", "upstream"]

_erf = np.vectorize(math.erf, otypes=[np.float64])
_erfc = np.vectorize(math.erfc, otypes=[np.float64])


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def make_geometry(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    tables: D4ReferenceTables | None = None,
) -> Geometry:
    """Validate the input and promote any periodicity to all three directions."""
    tables = tables or load_d4_reference_tables()
    flags = np.asarray(pbc, dtype=bool)
    promoted = np.full(3, bool(flags.any())) if flags.shape == (3,) else flags
    return validate_geometry(numbers, positions, cell, promoted, tables.max_z)


def matdet_3x3(a: FloatArray) -> float:
    """Replica of mctc ``matdet_3x3`` in its operation order (``a`` in Fortran index order)."""
    return float(
        a[0, 0] * a[1, 1] * a[2, 2]
        - a[0, 0] * a[1, 2] * a[2, 1]
        - a[0, 1] * a[1, 0] * a[2, 2]
        + a[0, 1] * a[1, 2] * a[2, 0]
        + a[0, 2] * a[1, 0] * a[2, 1]
        - a[0, 2] * a[1, 1] * a[2, 0]
    )


def matinv_3x3(a: FloatArray) -> FloatArray:
    """Replica of mctc ``matinv_3x3`` in its operation order."""
    d = 1.0 / matdet_3x3(a)
    b = np.empty((3, 3))
    b[0, 0] = +d * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
    b[1, 0] = -d * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
    b[2, 0] = +d * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    b[0, 1] = -d * (a[0, 1] * a[2, 2] - a[0, 2] * a[2, 1])
    b[1, 1] = +d * (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0])
    b[2, 1] = -d * (a[0, 0] * a[2, 1] - a[0, 1] * a[2, 0])
    b[0, 2] = +d * (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1])
    b[1, 2] = -d * (a[0, 0] * a[1, 2] - a[0, 2] * a[1, 0])
    b[2, 2] = +d * (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])
    return b


def _matvec(a: FloatArray, x: FloatArray) -> FloatArray:
    """Fortran ``matmul(a, x)`` with sequential summation, ``x`` of shape (n, 3)."""
    return np.stack(
        [(a[r, 0] * x[:, 0] + a[r, 1] * x[:, 1]) + a[r, 2] * x[:, 2] for r in range(3)], axis=1
    )


def shift_back_abc(frac: FloatArray) -> FloatArray:
    """Replica of mctc ``shift_back_abc``: wrap fractional coordinates into ``[0, 1)``."""
    eps = 1.0e-14
    out = frac.copy()
    low = frac < (0.0 - eps)
    out[low] = frac[low] + np.ceil(-frac[low])
    high = frac > (1.0 + eps)
    out[high] = frac[high] - np.floor(frac[high])
    one = np.abs(frac - 1.0) < eps
    out[one] = frac[one] - 1.0
    return out


def wrap_to_central_cell(positions: FloatArray, cell: FloatArray) -> FloatArray:
    """Replica of mctc ``wrap_to_central_cell`` (as dftd4's API applies before every evaluation)."""
    lattice = cell.T
    frac = _matvec(matinv_3x3(lattice), positions)
    return _matvec(lattice, shift_back_abc(frac))


def lattice_points(lattice: FloatArray, rep: tuple[int, int, int], *, origin: bool) -> FloatArray:
    """Replica of mctc ``get_lattice_points`` (rep form); ``lattice`` columns are vectors."""
    points: list[FloatArray] = []
    for ix in range(rep[0] + 1):
        for iy in range(rep[1] + 1):
            for iz in range(rep[2] + 1):
                if not origin and ix == iy == iz == 0:
                    continue
                for jx in (1, -1) if ix > 0 else (1,):
                    for jy in (1, -1) if iy > 0 else (1,):
                        for jz in (1, -1) if iz > 0 else (1,):
                            points.append(  # noqa: PERF401 - keep the upstream loop order
                                ((lattice[:, 0] * ix) * jx + (lattice[:, 1] * iy) * jy)
                                + (lattice[:, 2] * iz) * jz
                            )
    return np.array(points)


# --------------------------------------------------------------------------
# Reference polarizabilities and C6
# --------------------------------------------------------------------------


def zeta(a: float, c: FloatArray, qref: FloatArray, qmod: FloatArray) -> FloatArray:
    """dftd4 charge scaling; ``exp(a)`` for negative effective charge."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = np.exp(a * (1.0 - np.exp(c * (1.0 - qref / qmod))))
    result: FloatArray = np.where(qmod < 0.0, math.exp(a), value)
    return result


def dzeta(a: float, c: FloatArray, qref: FloatArray, qmod: FloatArray) -> FloatArray:
    """Derivative of :func:`zeta` with respect to ``qmod``.

    Zero for negative effective charge. At ``qmod == 0`` upstream evaluates
    ``0/0``; vanderfast uses the analytic limit 0.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = (
            -a * c * np.exp(c * (1.0 - qref / qmod)) * zeta(a, c, qref, qmod) * qref / (qmod * qmod)
        )
    result: FloatArray = np.where(qmod <= 0.0, 0.0, value)
    return result


def reference_alpha(tables: D4ReferenceTables, ga: float, gc: float) -> FloatArray:
    """dftd4 ``set_refalpha_eeq`` for every reference: shape ``(n_refs, n_freq)``."""
    system = tables.refsys.astype(np.int64)
    zeff = tables.element_column("zeff")[system]
    hardness = tables.element_column("hardness")[system]
    clsh = tables.reference_column("clsh")
    scale = zeta(ga, hardness * gc, zeff, clsh + zeff)
    aiw = (tables.sscale[system][:, None] * tables.secaiw[system]) * scale[:, None]
    ascale = tables.reference_column("ascale")[:, None]
    hcount = tables.reference_column("hcount")[:, None]
    alpha = np.maximum(ascale * (tables.alphaiw - hcount * aiw), 0.0)
    result: FloatArray = np.where((system == 0)[:, None], 0.0, alpha)
    return result


def reference_c6(tables: D4ReferenceTables, alpha: FloatArray, zi: int, zj: int) -> FloatArray:
    """Casimir-Polder C6 of every reference pair, ``(nref(zi), nref(zj))``.

    ``(3/pi) * sum_k (alpha_a[k] * alpha_b[k]) * w[k]`` summed sequentially, as
    dftd4's ``trapzd`` does.
    """
    ai = alpha[tables.references(zi)]
    aj = alpha[tables.references(zj)]
    total = np.zeros((ai.shape[0], aj.shape[0]))
    for k, weight in enumerate(tables.weights):
        total = total + (ai[:, None, k] * aj[None, :, k]) * weight
    result: FloatArray = (3.0 / math.pi) * total
    return result


@dataclass(frozen=True, eq=False)
class _Species:
    """Per-atom model data; reference arrays padded to ``MAX_REF``."""

    rcov: FloatArray
    en: FloatArray
    rcov_eeq: FloatArray
    chi: FloatArray
    eta: FloatArray
    kcnchi: FloatArray
    rad: FloatArray
    zeff: FloatArray
    hardness: FloatArray
    r4r2: FloatArray
    nref: IntArray
    ngw: IntArray
    cnref: FloatArray
    qref: FloatArray
    alpha: FloatArray


def _species(numbers: IntArray, tables: D4ReferenceTables, alpha: FloatArray) -> _Species:
    def column(name: str) -> FloatArray:
        values: FloatArray = tables.element_column(name)[numbers]
        return values

    n = numbers.size
    nref = tables.nref[numbers].astype(np.int64)
    ngw = np.zeros((n, MAX_REF), dtype=np.int64)
    cnref = np.zeros((n, MAX_REF))
    qref = np.zeros((n, MAX_REF))
    ref_alpha = np.zeros((n, MAX_REF, tables.freq.size))
    zeff = column("zeff")
    for atom, z in enumerate(numbers):
        rows = tables.references(int(z))
        count = rows.stop - rows.start
        ngw[atom, :count] = tables.ngw[rows]
        cnref[atom, :count] = tables.reference_column("refcovcn")[rows]
        qref[atom, :count] = tables.reference_column("clsq")[rows] + zeff[atom]
        ref_alpha[atom, :count] = alpha[rows]
    return _Species(
        rcov=column("rcov_d4"),
        en=column("en"),
        rcov_eeq=column("eeq_rcov"),
        chi=column("eeq_chi"),
        eta=column("eeq_eta"),
        kcnchi=column("eeq_kcnchi"),
        rad=column("eeq_rad"),
        zeff=zeff,
        hardness=column("hardness"),
        r4r2=column("r4r2"),
        nref=nref,
        ngw=ngw,
        cnref=cnref,
        qref=qref,
        alpha=ref_alpha,
    )


# --------------------------------------------------------------------------
# Coordination numbers
# --------------------------------------------------------------------------


def erf_count(rc: FloatArray, r: FloatArray) -> FloatArray:
    """Replica of mctc erf counting function ``0.5 * (1 + erf(-kcn (r - rc) / rc))``."""
    result: FloatArray = 0.5 * (1.0 + _erf(-KCN * (r - rc) / rc))
    return result


def erf_dcount(rc: FloatArray, r: FloatArray) -> FloatArray:
    """Derivative of :func:`erf_count` with respect to ``r``."""
    exponent = KCN * (r - rc) / rc
    result: FloatArray = -(KCN * np.exp(-(exponent * exponent))) / (SQRTPI * rc)
    return result


def en_factor(en_i: FloatArray, en_j: FloatArray) -> FloatArray:
    """D4 electronegativity factor ``k4 exp(-(|en_i - en_j| + k5)^2 / k6)``."""
    x = np.abs(en_i - en_j) + K5
    result: FloatArray = K4 * np.exp(-(x * x) / K6)
    return result


def coordination_numbers(edges: Edges, species: _Species, n: int) -> FloatArray:
    """D4 coordination numbers over directed edges (already cutoff-screened)."""
    rc = species.rcov[edges.i] + species.rcov[edges.j]
    counts = en_factor(species.en[edges.i], species.en[edges.j]) * erf_count(rc, np.sqrt(edges.r2))
    return np.bincount(edges.i, weights=counts, minlength=n)


def eeq_coordination_numbers(edges: Edges, species: _Species, n: int) -> FloatArray:
    """Uncapped EEQ coordination numbers over directed edges (already screened)."""
    rc = species.rcov_eeq[edges.i] + species.rcov_eeq[edges.j]
    return np.bincount(edges.i, weights=erf_count(rc, np.sqrt(edges.r2)), minlength=n)


def cap_cn(cn: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Replica of mctc ``cut_coordination_number`` at 8: capped value and its derivative."""
    capped = np.log(1.0 + np.exp(CN_MAX)) - np.log(1.0 + np.exp(CN_MAX - cn))
    dcap = np.exp(CN_MAX) / (np.exp(CN_MAX) + np.exp(cn))
    return capped, dcap


# --------------------------------------------------------------------------
# EEQ charges: matrices
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EwaldSetup:
    """Lattice data shared by the periodic Coulomb matrix and its derivatives."""

    mode: EwaldMode
    alpha: float
    volume: float
    positions: FloatArray
    dtrans: FloatArray
    rtrans: FloatArray
    images: list[list[FloatArray]]
    real_cutoff: float = 0.0


def _rec_dir_diff(alpha: float, rlen: float, dlen: float, volume: float) -> float:
    def rec(gg: float) -> float:
        return 4.0 * math.pi * (math.exp(-0.25 * gg * gg / (alpha**2)) / (volume * gg * gg))

    def direct(rr: float) -> float:
        return math.erfc(alpha * rr) / rr

    return (rec(4 * rlen) - rec(5 * rlen)) - (direct(2 * dlen) - direct(3 * dlen))


def search_alpha(lattice: FloatArray) -> float:
    """Replica of multicharge ``get_alpha``: bisection for the Ewald parameter, 0.25 on failure."""
    volume = abs(matdet_3x3(lattice))
    rec_lat = 2.0 * math.pi * matinv_3x3(lattice).T
    rlen = math.sqrt(float(np.min(np.sum(rec_lat**2, axis=0))))
    dlen = math.sqrt(float(np.min(np.sum(lattice**2, axis=0))))
    tol = EWALD_EPS
    alpha = 1.0e-8
    huge = sys.float_info.max

    def diff(value: float) -> float:
        return _rec_dir_diff(value, rlen, dlen, volume)

    current = diff(alpha)
    while current < -tol and alpha <= huge:
        alpha *= 2.0
        current = diff(alpha)
    if alpha > huge or alpha == 1.0e-8:  # noqa: PLR2004
        return 0.25
    left = 0.5 * alpha
    while current < tol and alpha <= huge:
        alpha *= 2.0
        current = diff(alpha)
    if alpha > huge:  # pragma: no cover - unreachable for physical cells
        return 0.25
    right = alpha
    alpha = (left + right) * 0.5
    iterations = 0
    current = diff(alpha)
    while abs(current) > tol and iterations <= ALPHA_ITERATIONS:
        if current < 0:
            left = alpha
        else:
            right = alpha
        alpha = (left + right) * 0.5
        current = diff(alpha)
        iterations += 1
    return 0.25 if iterations > ALPHA_ITERATIONS else alpha


def wigner_seitz_images(positions: FloatArray, lattice: FloatArray) -> list[list[FloatArray]]:
    """Replica of multicharge ``new_wignerseitz_cell``, including its self-pair indexing quirk.

    ``images[i][j]`` holds the translations ``T`` used for the pair as
    ``r_j - r_i + T``. Candidates within ``r^2 < sqrt(eps)`` are dropped from
    the distance list, but the kept positions index the full translation
    list. That only happens for self pairs, where the origin is dropped, so
    their images are shifted by one translation.
    """
    candidates = lattice_points(lattice, (1, 1, 1), origin=True)
    n = positions.shape[0]
    images: list[list[FloatArray]] = [[np.zeros((0, 3))] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rij = positions[i] - positions[j]
            vec = rij[None, :] - candidates
            r2 = vec[:, 0] * vec[:, 0] + vec[:, 1] * vec[:, 1] + vec[:, 2] * vec[:, 2]
            dist = r2[r2 >= EWALD_EPS]
            if dist.size == 0:  # pragma: no cover - a single point cannot coincide with all
                continue
            order = np.argsort(dist, kind="stable")
            best = dist[order[0]]
            chosen = [int(k) for k in order if abs(dist[k] - best) <= WS_TOL]
            images[i][j] = candidates[chosen]
    return images


def _real_space_sites(vec: FloatArray, setup: EwaldSetup) -> tuple[FloatArray, FloatArray]:
    """Sites ``vec + T`` and their distances, excluding coincident sites and the cutoff."""
    sites = vec[None, :] + setup.dtrans
    r1 = np.sqrt(np.einsum("tc,tc->t", sites, sites))
    keep = r1 >= EWALD_EPS
    if setup.mode == "converged":
        keep &= r1 <= setup.real_cutoff
    return sites[keep], r1[keep]


def _real_space_kernel(r1: FloatArray, gam: float, setup: EwaldSetup) -> FloatArray:
    """``erf(gam r)/r - erf(alpha r)/r`` (the erfc form in converged mode)."""
    if setup.mode == "converged":
        result: FloatArray = (_erfc(setup.alpha * r1) - _erfc(gam * r1)) / r1
    else:
        result = _erf(gam * r1) / r1 - _erf(setup.alpha * r1) / r1
    return result


def _real_space_dkernel(r1: FloatArray, gam: float, setup: EwaldSetup) -> FloatArray:
    """``(1/r) d/dr`` of :func:`_real_space_kernel`."""
    alpha = setup.alpha
    r2 = r1 * r1
    if setup.mode == "converged":
        value = _real_space_kernel(r1, gam, setup)
        slope = (
            2 * gam * np.exp(-r2 * gam * gam) - 2 * alpha * np.exp(-r2 * alpha * alpha)
        ) / SQRTPI
        result: FloatArray = (slope - value) / r2
        return result
    gtmp = 2 * gam * np.exp(-r2 * gam * gam) / (SQRTPI * r2) - _erf(r1 * gam) / (r2 * r1)
    atmp = -2 * alpha * np.exp(-r2 * alpha * alpha) / (SQRTPI * r2) + _erf(r1 * alpha) / (r2 * r1)
    total: FloatArray = gtmp + atmp
    return total


def _reciprocal_factors(setup: EwaldSetup) -> tuple[FloatArray, FloatArray]:
    """``|G|^2`` and ``4 pi / V exp(-G^2/(4 alpha^2)) / G^2`` per reciprocal vector."""
    g2 = np.einsum("gc,gc->g", setup.rtrans, setup.rtrans)
    fac = 4 * math.pi / setup.volume
    etmp: FloatArray = fac * np.exp(-0.25 * g2 / (setup.alpha * setup.alpha)) / g2
    return g2, etmp


def _ewald_value(vec: FloatArray, gam: float, setup: EwaldSetup) -> float:
    """Real-space plus reciprocal Ewald sum for one displacement."""
    _, r1 = _real_space_sites(vec, setup)
    _, etmp = _reciprocal_factors(setup)
    return float(np.sum(_real_space_kernel(r1, gam, setup))) + float(
        np.sum(np.cos(setup.rtrans @ vec) * etmp)
    )


def _ewald_derivative(
    vec: FloatArray, gam: float, setup: EwaldSetup
) -> tuple[FloatArray, FloatArray]:
    """Derivative of :func:`_ewald_value` by ``vec`` and by a homogeneous strain."""
    sites, r1 = _real_space_sites(vec, setup)
    dtmp = _real_space_dkernel(r1, gam, setup)
    dg = dtmp @ sites
    ds = np.einsum("t,ta,tb->ab", dtmp, sites, sites)
    g2, etmp = _reciprocal_factors(setup)
    gv = setup.rtrans @ vec
    dg += (-np.sin(gv) * etmp) @ setup.rtrans
    gfac = 2.0 / g2 + 0.5 / (setup.alpha * setup.alpha)
    weight = etmp * np.cos(gv)
    ds += np.einsum("g,ga,gb->ab", weight * gfac, setup.rtrans, setup.rtrans)
    ds -= float(np.sum(weight)) * np.eye(3)
    return dg, ds


def converged_setup(
    geometry: Geometry, species: _Species, alpha: float | None = None
) -> EwaldSetup:
    """Complete lattice sums: every term above ``1e-20`` is kept."""
    cell = geometry.cell
    lattice = cell.T
    volume = abs(matdet_3x3(lattice))
    alpha = alpha if alpha is not None else math.sqrt(math.pi) / volume ** (1.0 / 3.0)
    gam_min = 1.0 / math.sqrt(2.0 * float(np.max(species.rad)) ** 2)
    slowest = min(alpha, gam_min)
    real_cutoff = 1.0
    while math.erfc(slowest * real_cutoff) / real_cutoff > CONVERGED_TAIL:
        real_cutoff *= 1.1
    span = real_cutoff + float(
        np.max(
            np.linalg.norm(geometry.positions[:, None] - geometry.positions[None], axis=-1),
            initial=0.0,
        )
    )
    dtrans = _sphere_points(lattice, span, origin=True)
    rec_lat = 2.0 * math.pi * matinv_3x3(lattice).T
    g_cut = 1.0
    while (4 * math.pi / volume) * math.exp(-0.25 * g_cut * g_cut / (alpha * alpha)) / (
        g_cut * g_cut
    ) > CONVERGED_TAIL:
        g_cut *= 1.1
    rtrans = _sphere_points(rec_lat, g_cut, origin=False)
    return EwaldSetup(
        mode="converged",
        alpha=alpha,
        volume=volume,
        positions=geometry.positions.copy(),
        dtrans=dtrans,
        rtrans=rtrans,
        images=[],
        real_cutoff=real_cutoff,
    )


def _sphere_points(lattice: FloatArray, radius: float, *, origin: bool) -> FloatArray:
    """Every lattice point within ``radius`` of the origin (columns are vectors)."""
    normals = np.cross(lattice[:, [1, 2, 0]].T, lattice[:, [2, 0, 1]].T)
    volume = abs(float(np.linalg.det(lattice)))
    spacing = volume / np.linalg.norm(normals, axis=1)
    rep = np.ceil(radius / spacing).astype(np.int64)
    axes = [np.arange(-r, r + 1) for r in rep]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = grid.astype(np.float64) @ lattice.T
    keep = np.einsum("pc,pc->p", points, points) <= radius * radius
    if not origin:
        keep &= np.any(grid != 0, axis=1)
    result: FloatArray = points[keep]
    return result


def upstream_setup(geometry: Geometry) -> EwaldSetup:
    """Periodic setup of multicharge, on positions wrapped to the central cell."""
    lattice = geometry.cell.T
    positions = wrap_to_central_cell(geometry.positions, geometry.cell)
    rec_lat = 2.0 * math.pi * matinv_3x3(lattice).T
    return EwaldSetup(
        mode="upstream",
        alpha=search_alpha(lattice),
        volume=abs(matdet_3x3(lattice)),
        positions=positions,
        dtrans=lattice_points(lattice, (2, 2, 2), origin=True),
        rtrans=lattice_points(rec_lat, (2, 2, 2), origin=False),
        images=wigner_seitz_images(positions, lattice),
    )


def _pair_value(setup: EwaldSetup, i: int, j: int, gam: float) -> float:
    vec0 = setup.positions[j] - setup.positions[i]
    if setup.mode == "converged":
        return _ewald_value(vec0, gam, setup)
    images = setup.images[i][j]
    wsw = 1.0 / images.shape[0]
    total = 0.0
    for trans in images:
        total += _ewald_value(vec0 + trans, gam, setup) * wsw
    return total


def _self_value(setup: EwaldSetup, i: int, gam: float) -> float:
    if setup.mode == "converged":
        return _ewald_value(np.zeros(3), gam, setup)
    images = setup.images[i][i]
    wsw = 1.0 / images.shape[0]
    total = 0.0
    for trans in images:
        total += _ewald_value(trans, gam, setup) * wsw
    return total


def coulomb_matrix(geometry: Geometry, species: _Species, setup: EwaldSetup | None) -> FloatArray:
    """The EEQ Coulomb matrix ``A`` (without the charge-constraint border)."""
    n = geometry.n_atoms
    amat = np.zeros((n, n))
    rad = species.rad
    if setup is None:
        x = geometry.positions
        for i in range(n):
            for j in range(i):
                vec = x[j] - x[i]
                r2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
                gam = 1.0 / (rad[i] ** 2 + rad[j] ** 2)
                amat[i, j] = amat[j, i] = math.erf(math.sqrt(r2 * gam)) / math.sqrt(r2)
            amat[i, i] = species.eta[i] + SQRT2PI / rad[i]
        return amat
    for i in range(n):
        for j in range(i):
            gam = 1.0 / math.sqrt(rad[i] ** 2 + rad[j] ** 2)
            amat[i, j] = amat[j, i] = _pair_value(setup, i, j, gam)
        gam = 1.0 / math.sqrt(2.0 * rad[i] ** 2)
        amat[i, i] = _self_value(setup, i, gam) + (
            species.eta[i] + SQRT2PI / rad[i] - 2 * setup.alpha / SQRTPI
        )
    return amat


def coulomb_reverse(
    geometry: Geometry,
    species: _Species,
    setup: EwaldSetup | None,
    q: FloatArray,
    nu: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Gradient and strain derivative of ``-nu^T A q`` at fixed ``nu`` and ``q``."""
    n = geometry.n_atoms
    grad = np.zeros((n, 3))
    sigma = np.zeros((3, 3))
    rad = species.rad
    for i in range(n):
        for j in range(i):
            w = nu[i] * q[j] + nu[j] * q[i]
            if setup is None:
                vec = geometry.positions[j] - geometry.positions[i]
                r2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
                gam = 1.0 / math.sqrt(rad[i] ** 2 + rad[j] ** 2)
                arg = gam * gam * r2
                dtmp = 2.0 * gam * math.exp(-arg) / (SQRTPI * r2) - math.erf(math.sqrt(arg)) / (
                    r2 * math.sqrt(r2)
                )
                dg, ds = dtmp * vec, dtmp * np.outer(vec, vec)
            else:
                gam = 1.0 / math.sqrt(rad[i] ** 2 + rad[j] ** 2)
                dg, ds = _pair_derivative(setup, i, j, gam)
            grad[i] += w * dg
            grad[j] -= w * dg
            sigma -= w * ds
        if setup is not None:
            gam = 1.0 / math.sqrt(2.0 * rad[i] ** 2)
            sigma -= nu[i] * q[i] * _self_derivative(setup, i, gam)
    return grad, sigma


def _pair_derivative(
    setup: EwaldSetup, i: int, j: int, gam: float
) -> tuple[FloatArray, FloatArray]:
    vec0 = setup.positions[j] - setup.positions[i]
    if setup.mode == "converged":
        return _ewald_derivative(vec0, gam, setup)
    images = setup.images[i][j]
    wsw = 1.0 / images.shape[0]
    dg = np.zeros(3)
    ds = np.zeros((3, 3))
    for trans in images:
        gd, sd = _ewald_derivative(vec0 + trans, gam, setup)
        dg += gd * wsw
        ds += sd * wsw
    return dg, ds


def _self_derivative(setup: EwaldSetup, i: int, gam: float) -> FloatArray:
    if setup.mode == "converged":
        return _ewald_derivative(np.zeros(3), gam, setup)[1]
    images = setup.images[i][i]
    wsw = 1.0 / images.shape[0]
    ds = np.zeros((3, 3))
    for trans in images:
        ds += _ewald_derivative(trans, gam, setup)[1] * wsw
    return ds


def bordered(amat: FloatArray) -> FloatArray:
    """The EEQ system matrix: ``A`` bordered by a row and column of ones."""
    n = amat.shape[0]
    kmat = np.ones((n + 1, n + 1))
    kmat[:n, :n] = amat
    kmat[n, n] = 0.0
    return kmat


def solve_bordered(kmat: FloatArray, rhs: FloatArray) -> FloatArray:
    """Solve the (indefinite) bordered system with LU; raise on singular input."""
    try:
        with np.errstate(all="raise"):
            solution: FloatArray = np.linalg.solve(kmat, rhs)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        msg = "the EEQ charge equations are singular"
        raise ChargeSolverError(msg) from error
    if not np.all(np.isfinite(solution)):  # pragma: no cover - LAPACK reports singularity first
        msg = "the EEQ charge equations have no finite solution"
        raise ChargeSolverError(msg)
    return solution


# --------------------------------------------------------------------------
# Reference weights and C6
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Weights:
    """Charge-scaled Gaussian weights, padded to ``MAX_REF``."""

    w: FloatArray
    dw_dcn: FloatArray
    dw_dq: FloatArray


def reference_weights(
    species: _Species, cn: FloatArray, q: FloatArray, ga: float, gc: float, wf: float
) -> Weights:
    """dftd4 ``weight_references`` (with derivatives) in its summation order."""
    n = cn.size
    valid = np.arange(MAX_REF)[None, :] < species.nref[:, None]
    max_ngw = int(species.ngw.max(initial=0))
    norm = np.zeros(n)
    dnorm = np.zeros(n)
    expw = np.zeros((n, MAX_REF))
    expd = np.zeros((n, MAX_REF))
    for ref in range(MAX_REF):
        for igw in range(1, max_ngw + 1):
            use = valid[:, ref] & (igw <= species.ngw[:, ref])
            wfk = igw * wf
            gw = np.exp(-wfk * (cn - species.cnref[:, ref]) ** 2)
            dgw = 2 * wfk * (species.cnref[:, ref] - cn) * gw
            norm = norm + np.where(use, gw, 0.0)
            dnorm = dnorm + np.where(use, dgw, 0.0)
            expw[:, ref] = expw[:, ref] + np.where(use, gw, 0.0)
            expd[:, ref] = expd[:, ref] + np.where(use, dgw, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv = np.where(np.abs(norm) > EPS_NORM, 1.0 / norm, 0.0)
        gwk = expw * inv[:, None]
        dgwk = inv[:, None] * (expd - expw * dnorm[:, None] * inv[:, None])
    maxcn = np.where(valid, species.cnref, -np.inf).max(axis=1)
    fallback = np.where(np.abs(maxcn[:, None] - species.cnref) < 1e-12, 1.0, 0.0)  # noqa: PLR2004
    bad = ~np.isfinite(gwk) | (inv == 0.0)[:, None]
    gwk = np.where(bad, fallback, gwk)
    dgwk = np.where(~np.isfinite(dgwk) | (inv == 0.0)[:, None], 0.0, dgwk)
    gi = (species.hardness * gc)[:, None]
    zi = species.zeff[:, None]
    qmod = (q[:, None] + zi) * np.ones((1, MAX_REF))
    scale = zeta(ga, gi, species.qref, qmod)
    dscale = dzeta(ga, gi, species.qref, qmod)
    return Weights(
        w=np.where(valid, gwk * scale, 0.0),
        dw_dcn=np.where(valid, dgwk * scale, 0.0),
        dw_dq=np.where(valid, gwk * dscale, 0.0),
    )


@dataclass(frozen=True, eq=False)
class C6:
    """Atomic C6 and derivatives: ``d*[i, j]`` differentiates ``c6[i, j]`` by atom ``i``."""

    c6: FloatArray
    dcn: FloatArray
    dq: FloatArray


def c6_coefficients(
    numbers: IntArray, weights: Weights, tables: D4ReferenceTables, alpha: FloatArray
) -> C6:
    """``C6_ij = w_i^T C6ref(Z_i, Z_j) w_j`` and its CN and charge derivatives."""
    n = numbers.size
    c6 = np.zeros((n, n))
    dcn = np.zeros((n, n))
    dq = np.zeros((n, n))
    species = np.unique(numbers)
    for zi in species:
        rows = np.flatnonzero(numbers == zi)
        ni = int(tables.nref[zi])
        for zj in species:
            cols = np.flatnonzero(numbers == zj)
            nj = int(tables.nref[zj])
            block = reference_c6(tables, alpha, int(zi), int(zj))
            wj = weights.w[cols, :nj]
            c6[np.ix_(rows, cols)] = weights.w[rows, :ni] @ block @ wj.T
            dcn[np.ix_(rows, cols)] = weights.dw_dcn[rows, :ni] @ block @ wj.T
            dq[np.ix_(rows, cols)] = weights.dw_dq[rows, :ni] @ block @ wj.T
    return C6(c6, dcn, dq)


# --------------------------------------------------------------------------
# Terms
# --------------------------------------------------------------------------


@dataclass(eq=False)
class _State:
    geometry: Geometry
    species: _Species
    params: D4Parameters
    acc: Accumulator
    gq: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.gq = np.zeros(self.geometry.n_atoms)


def _pair_term(state: _State, c6: C6, edges: Edges, weight: FloatArray, *, book_both: bool) -> None:
    sp, params, acc = state.species, state.params, state.acc
    i, j, d, r2 = edges.i, edges.j, edges.d, edges.r2
    kappa = 3.0 * sp.r4r2[i] * sp.r4r2[j]
    r0 = params.a1 * np.sqrt(kappa) + params.a2
    r0_2 = r0 * r0
    r0_6 = r0_2 * r0_2 * r0_2
    r0_8 = r0_6 * r0_2
    r4 = r2 * r2
    t6 = 1.0 / (r4 * r2 + r0_6)
    t8 = 1.0 / (r4 * r4 + r0_8)
    edisp = params.s6 * t6 + params.s8 * kappa * t8
    c6e = c6.c6[i, j]
    energy = -weight * c6e * edisp
    if book_both:
        acc.energy(i, 0.5 * energy)
        acc.energy(j, 0.5 * energy)
    else:
        acc.energy(i, energy)
    if not acc.gradient:
        return
    d6 = -6.0 * r4 * t6 * t6
    d8 = -8.0 * r4 * r2 * t8 * t8
    gdisp = params.s6 * d6 + params.s8 * kappa * d8
    de_dd = (-weight * c6e * gdisp)[:, None] * d
    acc.force(j, de_dd)
    acc.force(i, -de_dd)
    acc.virial(d, de_dd)
    acc.adjoint(i, -weight * c6.dcn[i, j] * edisp)
    acc.adjoint(j, -weight * c6.dcn[j, i] * edisp)
    n = state.geometry.n_atoms
    state.gq += np.bincount(i, weights=-weight * c6.dq[i, j] * edisp, minlength=n)
    state.gq += np.bincount(j, weights=-weight * c6.dq[j, i] * edisp, minlength=n)


def _count_reverse(
    acc: Accumulator,
    edges: Edges,
    adjoint: FloatArray,
    rc: FloatArray,
    factor: FloatArray,
) -> None:
    r = np.sqrt(edges.r2)
    de_dd = (adjoint[edges.i] * factor * erf_dcount(rc, r) / r)[:, None] * edges.d
    acc.force(edges.j, de_dd)
    acc.force(edges.i, -de_dd)
    acc.virial(edges.d, de_dd)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class D4OracleResult:
    """Energies, derivatives, and intermediates in atomic units.

    ``gradient`` is dE/dr (hartree/bohr) and ``strain_derivative`` dE/dstrain
    (hartree). ``dc6dcn[i, j]`` and ``dc6dq[i, j]`` differentiate ``c6[i, j]``
    with respect to atom ``i``. ``dedcn`` and ``dedq`` are dE/dCN and dE/dq.
    ``eeq_matrix`` is the bordered system and ``eeq_adjoint`` the solution of
    the adjoint system (``None`` without gradients).
    """

    energy: float
    energy_pair: float
    energy_atm: float
    atom_energies: FloatArray
    gradient: FloatArray | None
    strain_derivative: FloatArray | None
    charges: FloatArray
    cn: FloatArray
    cn_eeq: FloatArray
    weights: FloatArray
    weights_q0: FloatArray
    dweights_dcn: FloatArray
    dweights_dq: FloatArray
    c6: FloatArray
    c6_q0: FloatArray
    dc6dcn: FloatArray
    dc6dq: FloatArray
    polarizabilities: FloatArray
    dedcn: FloatArray | None
    dedq: FloatArray | None
    eeq_matrix: FloatArray
    eeq_rhs: FloatArray
    eeq_adjoint: FloatArray | None
    ewald_alpha: float | None
    counts: dict[str, int]


def _edge_views(
    geometry: Geometry, cutoffs: D4Cutoffs, params: D4Parameters
) -> tuple[Edges, Edges, Edges, Edges]:
    rmax = max(
        cutoffs.cn_radius(),
        cutoffs.disp2 if params.pair else 0.0,
        cutoffs.disp3 if params.atm else 0.0,
    )
    edges = directed_edges(geometry, rmax)
    usable = edges.r2 >= CN_EXCLUSION_R2
    return (
        edges.select(usable & (edges.r2 <= cutoffs.cn**2)),
        edges.select(usable & (edges.r2 <= EEQ_CN_CUTOFF**2)),
        edges.select((edges.r2 <= cutoffs.disp2**2) & (edges.r2 >= DBL_EPSILON)),
        edges.select((edges.r2 <= cutoffs.disp3**2) & (edges.r2 >= DBL_EPSILON)),
    )


def evaluate(  # noqa: PLR0913
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    parameters: D4Parameters,
    charge: float = 0.0,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    cutoffs: D4Cutoffs | None = None,
    gradient: bool = True,
    tables: D4ReferenceTables | None = None,
    ewald: EwaldMode = "upstream",
    ewald_alpha: float | None = None,
    pair_counting: PairCounting = "directed",
    atm_counting: AtmCounting = "anchored",
) -> D4OracleResult:
    """Evaluate DFT-D4 (+ ATM when ``parameters.atm``) by brute force.

    Args:
        numbers: Atomic numbers, shape (N,).
        positions: Cartesian positions in bohr, shape (N, 3).
        parameters: Resolved damping and model parameters.
        charge: Total charge of the system in e.
        cell: Lattice vectors in bohr as rows.
        pbc: Periodicity flags; any true flag makes all three periodic.
        cutoffs: Real-space cutoffs in bohr (reference defaults if omitted).
        gradient: Also compute gradient, strain derivative, and adjoints.
        tables: D4 reference tables (the bundled tables if omitted).
        ewald: Periodic charge model; see the module docstring.
        ewald_alpha: Ewald parameter for ``ewald="converged"`` (any value
            gives the same result to the truncation level).
        pair_counting: ``"directed"`` (vanderfast) or ``"upstream"`` loops.
        atm_counting: ``"anchored"`` (vanderfast) or ``"upstream"`` loops.

    Returns:
        The full set of results and intermediates.

    Raises:
        ChargeSolverError: If the bordered EEQ system is singular.
    """
    tables = tables or load_d4_reference_tables()
    cutoffs = cutoffs or D4Cutoffs()
    geometry = make_geometry(numbers, positions, cell, pbc, tables)
    n = geometry.n_atoms
    alpha_ref = reference_alpha(tables, parameters.ga, parameters.gc)
    species = _species(geometry.numbers, tables, alpha_ref)
    cn_edges, eeq_edges, pair_edges, atm_edges = _edge_views(geometry, cutoffs, parameters)

    cn = coordination_numbers(cn_edges, species, n)
    cn_eeq, dcap = cap_cn(eeq_coordination_numbers(eeq_edges, species, n))

    setup: EwaldSetup | None = None
    if geometry.pbc.any():
        setup = (
            upstream_setup(geometry)
            if ewald == "upstream"
            else converged_setup(geometry, species, ewald_alpha)
        )
    tmp = species.kcnchi / np.sqrt(cn_eeq + EEQ_REG)
    rhs = np.append(-species.chi + tmp * cn_eeq, float(charge))
    kmat = bordered(coulomb_matrix(geometry, species, setup))
    solution = solve_bordered(kmat, rhs)
    q = solution[:n]

    p = parameters
    weights = reference_weights(species, cn, q, p.ga, p.gc, p.wf)
    c6 = c6_coefficients(geometry.numbers, weights, tables, alpha_ref)

    acc = Accumulator(n, gradient)
    state = _State(geometry, species, p, acc)
    if p.pair:
        if pair_counting == "directed":
            _pair_term(state, c6, pair_edges, np.full(len(pair_edges), 0.5), book_both=False)
        else:
            half = pair_edges.select(pair_edges.j <= pair_edges.i)
            _pair_term(state, c6, half, np.where(half.i == half.j, 0.5, 1.0), book_both=True)
    energy_pair = float(acc.atom_energy.sum())

    weights0 = reference_weights(species, cn, np.zeros(n), p.ga, p.gc, p.wf)
    c6_0 = c6_coefficients(geometry.numbers, weights0, tables, alpha_ref)
    atm = AtmModel(
        c6=c6_0.c6,
        dc6=c6_0.dcn,
        r0=p.a1 * np.sqrt(3.0 * species.r4r2[:, None] * species.r4r2[None, :]) + p.a2,
        s9=p.s9,
        alp3=p.atm_exponent / 3.0,
        cutoff2=cutoffs.disp3**2,
        c6_floor=DBL_MIN,
        skip_small=True,
        acc=acc,
    )
    if p.atm:
        atm_term(atm, atm_edges, n, atm_counting)
    energy_total = float(acc.atom_energy.sum())

    nu: FloatArray | None = None
    if gradient:
        nu = solve_bordered(kmat, np.append(state.gq, 0.0))
        grad_q, sigma_q = coulomb_reverse(geometry, species, setup, q, nu[:n])
        acc.grad += grad_q
        acc.sigma += sigma_q
        rc_d4 = species.rcov[cn_edges.i] + species.rcov[cn_edges.j]
        den = en_factor(species.en[cn_edges.i], species.en[cn_edges.j])
        _count_reverse(acc, cn_edges, acc.g, rc_d4, den)
        h = nu[:n] * (0.5 * tmp) * dcap
        rc_eeq = species.rcov_eeq[eeq_edges.i] + species.rcov_eeq[eeq_edges.j]
        _count_reverse(acc, eeq_edges, h, rc_eeq, np.ones(len(eeq_edges)))

    counts = {
        "cn_edges": len(cn_edges),
        "eeq_cn_edges": len(eeq_edges),
        "pair_edges": len(pair_edges) if p.pair else 0,
        "atm_edges": len(atm_edges) if p.atm else 0,
        "atm_candidates": atm.counters.candidates,
        "atm_accepted": atm.counters.accepted,
        "atm_c6_skipped": atm.counters.c6_skipped,
    }
    return D4OracleResult(
        energy=energy_total,
        energy_pair=energy_pair,
        energy_atm=energy_total - energy_pair,
        atom_energies=acc.atom_energy,
        gradient=acc.grad if gradient else None,
        strain_derivative=acc.sigma if gradient else None,
        charges=q,
        cn=cn,
        cn_eeq=cn_eeq,
        weights=weights.w,
        weights_q0=weights0.w,
        dweights_dcn=weights.dw_dcn,
        dweights_dq=weights.dw_dq,
        c6=c6.c6,
        c6_q0=c6_0.c6,
        dc6dcn=c6.dcn,
        dc6dq=c6.dq,
        polarizabilities=np.einsum("ir,ir->i", weights.w, species.alpha[:, :, 0]),
        dedcn=acc.g if gradient else None,
        dedq=state.gq if gradient else None,
        eeq_matrix=kmat,
        eeq_rhs=rhs,
        eeq_adjoint=nu,
        ewald_alpha=None if setup is None else setup.alpha,
        counts=counts,
    )


__all__ = [
    "C6",
    "D4OracleResult",
    "EwaldSetup",
    "Weights",
    "bordered",
    "c6_coefficients",
    "cap_cn",
    "converged_setup",
    "coordination_numbers",
    "coulomb_matrix",
    "coulomb_reverse",
    "dzeta",
    "eeq_coordination_numbers",
    "en_factor",
    "erf_count",
    "erf_dcount",
    "evaluate",
    "lattice_points",
    "make_geometry",
    "matdet_3x3",
    "matinv_3x3",
    "reference_alpha",
    "reference_c6",
    "reference_weights",
    "search_alpha",
    "shift_back_abc",
    "solve_bordered",
    "upstream_setup",
    "wigner_seitz_images",
    "wrap_to_central_cell",
    "zeta",
]
