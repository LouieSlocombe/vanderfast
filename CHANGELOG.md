# Changelog

All notable changes to vanderfast are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): within
a major version, the public Python API (`vanderfast`, `vanderfast.ase`), the
command line, the bundled reference tables, and the numerical results they
produce are stable.

## [Unreleased]

### Changed

- vanderfast supports CUDA 12.6 and newer, CUDA 13 included. The release
  wheel is built with CUDA 12.9 instead of 13.1, so it runs on drivers
  that support CUDA 12.6 or newer (previously 13.1). It keeps SASS for
  sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, and sm_120 and PTX for
  compute_120.
- Source builds accept any CUDA toolkit from 12.6 (previously 13.0). The
  `release` architecture list needs CUDA 12.8 or newer, for Blackwell;
  with CUDA 12.6, pass an explicit list such as
  `75-real;80-real;86-real;89-real;90-real;90-virtual`.
- The conda development environment and its lock move to CUDA 12.9 and
  GCC 14; recreate `vanderfast-dev` from `build_tools/conda-linux-64.lock`.
  CI also builds the extension with CUDA 12.6 and GCC 13, the oldest
  supported toolchain, in the oldest-dependencies job, and with the newest
  CUDA 13 release in a new `cuda13` job.

## [1.1.2] - 2026-10-23

The first release published to PyPI. The v1.0.0, v1.1.0, and v1.1.1 tags
exist but their release workflows failed before publishing, so 1.1.2
carries the 1.0.0 content plus the release-tooling fixes below and in
1.1.0; the code and the bundled tables are unchanged.

### Fixed

- The release workflow's wheel job failed in
  `test_derived_reference_tables_match_the_manifest`, which checks the
  NumPy oracle's reference polarizabilities against the manifest hash bit
  for bit. glibc 2.28, in the manylinux_2_28 build image, has an `exp` that
  is not correctly rounded for 3 of the test's 365 inputs, so 61
  polarizabilities differ by about 1e-15 relative and the hash cannot
  match there. The wheel tests now deselect that test; CI and the release
  workflow's GPU job still run it on newer glibc.
- The wheel tests no longer set the CPU code-path pins added in 1.1.1;
  they had no effect on this failure.

## [1.1.1] - 2026-10-23

Tagged on GitHub; not published to PyPI, because the wheel job of its
release workflow still failed (fixed in 1.1.2). It set CPU code-path pins
(`OPENBLAS_CORETYPE`, `NPY_DISABLE_CPU_FEATURES`) for the wheel tests on the
mistaken diagnosis that NumPy's AVX-512 `exp` caused the 1.1.0 failure.

## [1.1.0] - 2026-10-23

Tagged on GitHub; not published to PyPI, because the wheel job of its
release workflow failed (fixed in 1.1.2). It carries the 1.0.0 content plus
the release-tooling fixes below; the code and the bundled tables are
unchanged.

### Fixed

- The release workflow's sdist job needs `twine`, which the locked conda
  environment did not contain. It is now part of the environment, the
  lock, and the `build` dependency group; `make sdist` runs the same build
  and checks as the release job, and CI runs `twine check` on every build.
- A source distribution built from a working tree could contain Python
  bytecode caches, because the `src/vanderfast/data/*` include pattern
  overrode `.gitignore`. The pattern now names the data file types and
  `tools/check_sdist.py` rejects any bytecode.
- `benchmarks/schema.py` no longer fails when `nvidia-smi` or `git` is not
  installed (first shipped in 1.0.0; recorded here for the PyPI release).

## [1.0.0] - 2026-10-23

First stable release (tagged on GitHub; not published to PyPI).

### Added

- **DFT-D3(BJ) on the GPU.** Energies, forces, strain derivatives, and
  stress for molecules and periodic systems (3D cells, slabs, and wires)
  with the Axilrod-Teller-Muto three-body term, in double precision. `D3Context`
  keeps the device tables and the neighbour list between evaluations;
  `resolve_parameters`, `D3BJParameters`, and `Cutoffs` expose the 161
  methods and 201 aliases of simple-dftd3 1.6.0.
- **DFT-D4 on the GPU.** The D4 model with EEQ partial charges (multicharge
  0.5.0), rational damping, and the three-body term: energies, forces,
  stress, and charges for molecules, ions, and periodic systems, with
  dftd4's truncated Ewald sums (`ewald="upstream"`) or complete sums
  (`ewald="converged"`). The charge equations are solved by a packed MINRES
  solver or a shared-memory dense LU. `D4Context`, `resolve_d4_parameters`,
  `D4Parameters`, `D4Cutoffs`, and `EEQSolverOptions` cover the 118
  functionals with ATM parameters, 5 without, and 319 aliases of dftd4 4.2.0.
- **ASE calculators.** `vanderfast.ase.D3Calculator` and `D4Calculator`
  with energy, forces, stress, and (D4) charges, result caching that follows
  ASE's rules, in-place parameter updates, and a `backend="reference"`
  option that delegates to the dftd3 or dftd4 Python packages.
- **Two neighbour-list builders** (brute force and cell lists) that give
  identical edge sets for any cell shape, periodicity, wrapping, and cutoff,
  including cutoffs larger than the cell.
- **Bundled reference data** generated byte for byte from pinned sources:
  simple-dftd3 v1.6.0 and mctc-lib for D3; dftd4 v4.2.0, multicharge
  v0.5.0, and mctc-lib v0.5.2 for D4. The generators verify every parsed
  array inside the compiled upstream libraries, and the manifests record
  the hashes, units, conventions, and upstream anomalies.
- **NumPy oracles** (`vanderfast._oracle`, `vanderfast._oracle_d4`) as
  executable specifications of both models, used by the tests and usable
  without a GPU.
- **Command line:** `vanderfast info [--json] [--cuda]`, `vanderfast
  methods [--model d3|d4]`, `vanderfast verify-data`, and `--version`.
- **Diagnostics:** `backend_info()` reports the package version, build
  (compilers, architectures, math flags), reference-data hashes, and CUDA
  devices; the contexts report resolved parameters and runtime counters.
- **Release wheel** for CPython 3.14 on manylinux_2_28 x86_64 built with
  CUDA 13.1: SASS for sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, and
  sm_120, PTX for compute_120, a statically linked CUDA runtime, and no
  register spills in any kernel. `VANDERFAST_ENABLE_CUDA=OFF` installs the
  pure-Python parts without the CUDA backend.
- **Benchmarks** (`benchmarks/`) with a versioned JSON schema, a comparison
  tool, and a Markdown report renderer; measured results in
  `docs/performance.md`.

### Validation

- D3 agrees with simple-dftd3 1.6.0 to about 1e-15 hartree on 233 fixtures;
  D4, including the EEQ charges, agrees with dftd4 4.2.0 to about 1e-13 on
  291 fixtures. Forces and strain derivatives match finite differences, and
  the full suite, the Catch2 host and device tests, and compute-sanitizer
  (memcheck, racecheck, synccheck, initcheck) pass on an RTX 4080 SUPER.
  See `docs/validation.md`.

### Requirements

- Linux x86_64, Python 3.14, NumPy 2.3.3 or newer, ASE 3.26 or newer.
- An NVIDIA GPU of compute capability 7.5 or newer and a driver that
  supports CUDA 13.1. No CUDA toolkit is needed at run time.

### Known limitations

- Only D3 with Becke-Johnson damping and D4 with the EEQ model; no zero
  damping, D4S, EEQ-BC, external charges, Hessians, smooth cutoffs, ghost
  atoms, or batching. D3 stress is reported for three-dimensional cells
  only; D4 treats any periodic input as fully periodic, as dftd4 does.
- With ATM, forces and the CN-mediated derivatives use atomic additions
  and may differ in the last bits between runs; energies are bitwise
  reproducible.
- Tested on one GPU model (sm_89). The other release architectures are
  compiled and checked for spills but have not run on hardware.
  See `docs/limitations.md` and the support matrix in `docs/build.md`.

[1.1.2]: https://github.com/LouieSlocombe/vanderfast/releases/tag/v1.1.2
[1.1.1]: https://github.com/LouieSlocombe/vanderfast/releases/tag/v1.1.1
[1.1.0]: https://github.com/LouieSlocombe/vanderfast/releases/tag/v1.1.0
[1.0.0]: https://github.com/LouieSlocombe/vanderfast/releases/tag/v1.0.0
