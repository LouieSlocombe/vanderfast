# Scientific conventions

vanderfast reproduces D3(BJ) with the ATM three-body term exactly as defined by
simple-dftd3 v1.6.0. Every convention below is checked by
`tools/generate_d3_data.py` against the pinned upstream source lines and
recorded in `src/vanderfast/data/manifest.json`.

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
