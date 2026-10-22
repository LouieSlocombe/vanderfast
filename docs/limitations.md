# Limitations

- **Scope.** D3 with rational (Becke-Johnson) damping and the ATM three-body
  term, and DFT-D4 (model `d4` with EEQ charges, rational damping, ATM). No
  zero damping or other D3 families, no D4S, EEQ-BC or other charge models, no
  self-consistent or externally supplied charges, no Hessians, smooth cutoffs,
  ghost atoms, or batching of several structures.
- **Precision.** FP64 throughout. Consumer GPUs have low FP64 throughput; the
  ATM term at the default 40 bohr cutoff is O(N k²) and dominates the cost
  (see `docs/performance.md`).
- **Stress.** D3 reports it only for three-dimensional periodic cells (the
  strain derivative is available for any periodicity through `D3Context`).
  D4 treats every periodic input as three-dimensional, as dftd4 does, and
  reports stress for all of them.
- **D4 periodicity and charges.** Any periodic flag makes a D4 system fully
  periodic over the supplied cell (a slab needs a vacuum vector), and the
  default `ewald="upstream"` reproduces dftd4's truncated Ewald sums, whose
  charges depend slightly on the choice of cell. `ewald="converged"` removes
  that dependence but no longer matches dftd4.
- **Hardware.** Linux x86_64 with an NVIDIA GPU of compute capability 7.5 or
  newer. Tested only on the devices listed in `docs/build.md`.
- **Determinism.** Energies are bitwise reproducible. Pair-only forces and
  virials are reproducible with the default kernels. With ATM, forces, dE/dCN
  (and therefore the CN part of every force), and per-atom ATM energies use
  atomic additions, so they may differ in the last bits between runs. D4
  charges, including MINRES solves, are reproducible unless warm starts are
  enabled, which makes them depend in the last bits on the previous call.
- **Upstream differences.** See `docs/conventions.md` (D3: partial-PBC image
  ranges with tilted inactive vectors; the ATM small-C6 rule in energy-only
  calls; stress for partially periodic systems. D4: the ζ derivative at zero
  effective charge; ATM triangles with zero C6).
- **Memory.** The neighbour list stores 16 bytes per directed edge within the
  largest cutoff plus skin. At the default cutoffs and a 1 Å skin, total
  device memory measured 236 kB per atom for liquid water and 322 kB per atom
  for MgO. That is 1.3 GB for 4096 MgO atoms, so a 16 GB GPU holds roughly
  50 000 dense-solid atoms. An oversized request raises
  `DeviceOutOfMemoryError` and leaves the context usable. D4 adds the dense
  charge matrix, (N+1)² doubles, or half that as packed tiles from 768 atoms,
  capped at half of the device memory (about 45 000 atoms on 16 GB); periodic
  gradients use up to 4.5 times more for stored lattice derivatives when it
  fits, and recompute them otherwise. The charge solve is O(N²) per iteration,
  so very large D4 systems are bounded by memory bandwidth.
- **Single structure per context.** There is no batching of several small
  structures. Below about 100 atoms an evaluation is bound by kernel-launch
  latency (about 0.1-0.3 ms per call; see `docs/performance.md`).
