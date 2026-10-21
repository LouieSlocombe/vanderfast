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
  virials are reproducible with the default kernels. With ATM, forces, dE/dCN
  (and therefore the CN part of every force), and per-atom ATM energies use
  atomic additions, so they may differ in the last bits between runs.
- **Upstream differences.** See `docs/conventions.md` (partial-PBC image
  ranges with tilted inactive vectors; the ATM small-C6 rule in energy-only
  calls; stress for partially periodic systems).
- **Memory.** The neighbour list stores 16 bytes per directed edge within the
  largest cutoff plus skin. At the default cutoffs and a 1 Å skin, total
  device memory measured 236 kB per atom for liquid water and 322 kB per atom
  for MgO. That is 1.3 GB for 4096 MgO atoms, so a 16 GB GPU holds roughly
  50 000 dense-solid atoms. An oversized request raises
  `DeviceOutOfMemoryError` and leaves the context usable.
- **Single structure per context.** There is no batching of several small
  structures. Below about 100 atoms an evaluation is bound by kernel-launch
  latency (about 0.1-0.3 ms per call; see `docs/performance.md`).
