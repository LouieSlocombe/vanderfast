# Scientific conventions

vanderfast reproduces D3(BJ) with the ATM three-body term exactly as defined by
simple-dftd3 v1.6.0, and DFT-D4 as defined by dftd4 v4.2.0 with multicharge
v0.5.0 ([DFT-D4](#dft-d4) below). Every convention is checked by the data
generators (`tools/generate_d3_data.py`, `tools/generate_d4_data.py`) against
the pinned upstream source lines and recorded in
`src/vanderfast/data/manifest.json` and `src/vanderfast/data/d4_manifest.json`.

## Units

The numerical core uses atomic units (bohr, hartree). The Python and ASE
interfaces use Å and eV, converted exactly once in Python with the
`ase.units` constants (CODATA 2014) and the same recipes as `dftd3.ase`:
positions `x * (1/Bohr)`, cutoffs `x / Bohr`, energy `e * Hartree`,
forces `g * -(Hartree/Bohr)`, stress `G * (Hartree/V)` in the order
xx, yy, zz, yz, xz, xy. The bundled tables were converted with mctc-lib's
CODATA 2018 Å-to-bohr factor (1.8897261246204404), as upstream does.

## Geometry

Lattice vectors are rows of the cell `H`. A site is `(atom j, image n)` at
`r_j + n H`; the displacement from centre `i` is `d = r_j + n H - r_i`. The
zero-shift self site is excluded; other images of the same atom are kept.
Only active periodic directions generate images, and every site within each
cutoff is included regardless of how the input is wrapped.

## Neighbour lists

Each atom's row of the directed edge list `(i -> j, n)` is ordered by radius
band, so the CN, pair, and ATM views are prefixes of one list sized for the
largest cutoff plus the skin. Every kernel still applies its exact cutoff.
Two builders produce the same edge set (checked against a CPU enumeration):

- **Brute force** over all atoms and lattice images, for small systems.
- **Cell lists** in the search basis (active lattice vectors completed with
  orthonormal vectors). Bins hold wrapped fractional coordinates, and an
  unbounded periodic stencil maps each bin offset to one (bin, image) pair, so
  cutoffs larger than the cell yield every image exactly once. Image shifts are
  stored relative to the raw input coordinates.

The builder is chosen automatically by system size, or explicitly with
`builder=`. Distinct sites closer than 1e-6 bohr are rejected as unphysical
input.

## Terms

| Quantity | Definition | Exclusion |
| --- | --- | --- |
| CN | `1/(1+exp(-16 (rc/r - 1)))`, `rc` = sum of D3-scaled covalent radii | `r² > rc_cn²` or `r² < 1e-12` |
| Weights | `exp(-4 (CN - CNref)²)` normalised over valid references | non-finite weight → 1 for every max-CNref reference, else 0; non-finite derivative → 0 |
| Pair | `-C6 (s6/(r⁶+R⁶) + s8 κ/(r⁸+R⁸))`, `κ = 3 r4r2_i r4r2_j`, `R = a1 √κ + a2` | `r² > rc_disp2²` or `r² < DBL_EPSILON` |
| ATM | `+s9 √|C6ij C6ik C6jk| (1 + 3 cos θi cos θj cos θk)/(rij rik rjk)³ / (1 + 6 (R0/(rij rik rjk))^((alp+2)/3))`, `R0 = Π (4/3) rvdw` | each edge as for the pair term; any `|C6| < DBL_EPSILON` |

Counting: the pair sum runs over directed site pairs with weight ½; the ATM sum
runs over primary-cell centres and unordered pairs of distinct neighbour sites
with weight ⅓. Both equal the upstream loop structures (checked in
`tests/unit/test_oracle.py`).

## Parameters

A method name requires `damping="bj"` and an explicit `atm=True/False`. ATM on
sets `s9 = 1`, ATM off sets `s9 = 0`; the pair coefficients are identical
(simple-dftd3 v1.6.0 has no pair-only refits). Explicit parameters are used
unchanged. Note that the upstream `dftd3.ase.DFTD3(method=...)` calculator
loads parameters with ATM *off* by default.

## Known differences from simple-dftd3 v1.6.0

- **Partial PBC with tilted inactive vectors.** The 2D branch of upstream
  `get_lattice_points` masks Cartesian rows (`spread(periodic, 2, 3)`) instead
  of lattice columns, so its image range is too small when an inactive lattice
  vector has components along the periodic Cartesian axes (for example a slab
  whose vacuum vector is not along z). Upstream then misses interactions;
  vanderfast includes every site within the cutoff, so the result does not
  depend on the inactive vector. See the `graphene_slab_tilted_vacuum`
  fixtures.
- **ATM triangles with `|C6| < DBL_EPSILON`.** Upstream skips them only in its
  derivative path. vanderfast skips them in every path, so energy-only and
  energy+force calls agree. With the bundled tables this case cannot occur for
  physical inputs.
- **Upstream ASE stress** is reported for any periodic system; vanderfast
  reports stress only for 3D periodic cells until partial-PBC stress is
  qualified.

# DFT-D4

D4 (model `d4`, EEQ charges, rational damping with ATM) follows dftd4 v4.2.0,
multicharge v0.5.0 and mctc-lib v0.5.2. Elements 1-103 are supported.

## Units

As for D3, with the recipes of `dftd4.ase` (which match `dftd3.ase`):
positions and cutoffs `x / Bohr`, energy `e * Hartree`, forces
`-g * Hartree/Bohr`, stress `virial * Hartree/V` in the order xx, yy, zz, yz,
xz, xy. The tables use mctc-lib's CODATA 2018 Å-to-bohr factor.

## Periodicity

Any periodic flag makes the system fully three-dimensional periodic over the
supplied cell, for both the dispersion terms and the charges, because dftd4
ignores the per-direction flags. A slab or wire therefore needs a cell whose
non-periodic vectors are long enough to separate the images, and the cell must
not be degenerate. Positions are wrapped into the central cell (mctc
`shift_back_abc`, tolerance 1e-14) before the charge model, as dftd4's API does.
Stress is available for every periodic input.

## Terms

| Quantity | Definition |
| --- | --- |
| D4 CN | `Σ_j f_EN · ½(1 + erf(-7.5 (r - rc)/rc))`, `rc = rcov_i + rcov_j`, `f_EN = 4.10451 exp(-(|EN_i - EN_j| + 19.08857)² / (2·11.28174²))`; cutoff 30 bohr; `r² < 1e-12` excluded |
| EEQ CN | the same count without `f_EN`, cutoff 25 bohr, capped as `log(1 + e⁸) - log(1 + e^(8 - CN))` |
| Charges | bordered system `[[A, 1], [1ᵀ, 0]] [q; μ] = [X; Q]` with `X_i = -χ_i + κ_i CN_i / √(CN_i + 1e-14)` and total charge `Q` |
| A (molecules) | `A_ij = erf(√(r²/(rad_i² + rad_j²)))/r`, `A_ii = η_i + √(2/π)/rad_i` |
| Weights | `ζ(ga, gc η_i, qref + zeff, q_i + zeff) · Σ_{k=1..ngw} exp(-k wf (CN_i - CNref)²) / norm`; `ngw` from the reference-CN histogram |
| ζ | `exp(ga (1 - exp(c (1 - qref/qmod))))`, `exp(ga)` for `qmod < 0` |
| C6 | `Σ w_i w_j C6ref`, `C6ref = (3/π) Σ_k w_k α_a(iω_k) α_b(iω_k)` over 23 frequencies |
| Pair | as D3(BJ) with `κ = 3 r4r2_i r4r2_j`, `R0 = a1 √κ + a2`; cutoff 60 bohr |
| ATM | `-s9 √|C6ij C6ik C6jk|` with C6 at zero charge, damping `1/(1 + 6 (R0/r)^(alp/3))`, `R0 = Π (a1 √κ + a2)`, `alp = 16`; cutoff 40 bohr |

Weights fall back to 1 for every reference at the maximum reference CN (within
1e-12) when the Gaussian norm is below `√tiny` or not finite. Reference
polarisabilities are `max(ascale (α_ref - hcount ζ(...) α_secondary), 0)` and
are recomputed whenever `ga` or `gc` change, so every D4 parameter is honoured.

## Periodic charges

`ewald="upstream"` (the default) reproduces multicharge, whose periodic EEQ
matrix is a fixed truncation:

- the Ewald parameter from multicharge's bisection (0.25 if it fails);
- 125 real-space translations, `erf(γr)/r - erf(αr)/r` with `γ = 1/√(rad_i² + rad_j²)`;
- 124 reciprocal vectors, `(4π/V) cos(G·r) exp(-G²/4α²)/G²`;
- Wigner-Seitz averaging over the nearest of 27 images (ties within
  0.01 bohr², weight 1/n), including upstream's self-pair indexing quirk;
- the diagonal `η + √(2/π)/rad - 2α/√π` plus the self images.

Because the sums are truncated, the charges depend on the choice of cell: an
MgO primitive cell and its 2×2×2 supercell differ by up to 4e-3 e.
`ewald="converged"` instead keeps every real-space and reciprocal term above
1e-20, with the constant G = 0 term absorbed by the charge constraint, so the
charges do not depend on the Ewald parameter or the cell (supercell energies
agree to about 1e-14). It does not reproduce dftd4.

## Solving the charge equations

The bordered matrix is symmetric and indefinite (and `A` itself can be
indefinite, for example in dense hydrogen clusters), so there is no Cholesky
or conjugate-gradient solve. vanderfast uses:

- `solver="auto"` (default): LU with partial pivoting in shared memory when the
  matrix fits (up to 111 atoms on Ada), otherwise preconditioned MINRES;
- `solver="direct"`: always LU;
- `solver="iterative"`: always MINRES.

MINRES uses a Jacobi preconditioner (`|A_ii|`, with `Σ 1/|A_ii|` for the
border) and stops when the normwise backward error
`|b - Kx|∞ / (|K|∞ |x|∞ + |b|∞)` is below 2⁻⁴⁸, refining from the current
solution up to three times; it accepts at most `tolerance` (1e-13) and raises
`ChargeSolverError` otherwise. From 768 atoms the matrix is stored as packed
lower-triangle tiles. Forces come from the adjoint system
`K [ν; λ] = [∂E/∂q; 0]` and the derivatives of `X` and `A`, so `dq/dr` is
never formed.

## Parameters

`resolve_d4_parameters(method=...)` looks a functional up as dftd4 does: the
name before any `/` (so `"pbe/def2-tzvp"` gives `pbe`), lower-cased, through the
dftd4 alias table. With `atm=True` (the default, as in `dftd4.ase`) the rational
damping parameters with ATM are used, falling back to the pair-only table;
`atm=False` prefers the pair-only table and sets `s9 = 0`. `ga = 3`, `gc = 2`,
`wf = 6` and `alp = 16` unless given. The D4 calculator takes the total charge
from `atoms.get_initial_charges().sum()`, as `dftd4.ase` does.

## Known differences from dftd4 v4.2.0

- **ζ derivative at zero effective charge.** Upstream evaluates `0/0` when
  `q + zeff = 0` exactly; vanderfast uses the analytic limit 0.
- **ATM triangles with `C6 = 0`.** Skipped (upstream's charge derivative would
  divide by zero). dftd4 has no small-C6 rule otherwise, and neither does
  vanderfast.
- **CN derivatives with a short ATM cutoff.** Upstream evaluates the CN
  derivative over the ATM lattice. vanderfast uses the CN lattice, which only
  differs if the ATM cutoff is shorter than the CN cutoff; no fixture shows a
  difference.
- **Solver round-off.** dftd4 factorises with `sytrf` and, for gradients,
  inverts with `sytri`, so its own energy-only and gradient calls differ at the
  level of the condition number times machine epsilon. vanderfast agrees with
  both to that level.
