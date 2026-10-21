# Limitations

- **Scope.** D3 with rational (Becke-Johnson) damping and the ATM three-body
  term only. No zero damping or other families, D4, Hessians, smooth cutoffs,
  reciprocal-space summation, ghost atoms, or batching of several structures.
- **Precision.** FP64 throughout. Consumer GPUs have low FP64 throughput; the
  ATM term at the default 40 bohr cutoff is O(N k²) and dominates the cost
  (see `docs/performance.md`).
- **Stress.** Reported only for three-dimensional periodic cells. The strain
  derivative is available for any periodicity through `D3Context`.
- **Hardware.** Linux x86_64 with an NVIDIA GPU of compute capability 7.5 or
  newer. Tested only on the devices listed in `docs/build.md`.
- **Determinism.** Energies are bitwise reproducible. Pair-only forces and
  virials are reproducible with the default kernels; with ATM, forces and
  dE/dCN use atomic additions and may differ in the last bits between runs.
- **Upstream differences.** See `docs/conventions.md` (partial-PBC image
  ranges with tilted inactive vectors; the ATM small-C6 rule in energy-only
  calls; stress for partially periodic systems).
- **Memory.** The neighbour list stores 16 bytes per directed edge within the
  largest cutoff plus skin: about 235 kB per atom in a dense solid at the
  default 60 bohr pair cutoff.
