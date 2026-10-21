# Validation

vanderfast is checked against simple-dftd3 v1.6.0 at three levels.

1. **Reference data.** `tools/generate_d3_data.py --check --verify-binary`
   regenerates the tables byte for byte from the pinned sources and finds every
   parsed array inside the compiled `libs-dftd3` and `libmctc-lib`.
2. **NumPy oracle.** `vanderfast._oracle` is a brute-force executable
   specification with explicit lattice sites. It matches every dftd3 fixture
   (energies, gradients, virials) to about 1e-16 hartree, both with the
   vanderfast counting (directed pairs with weight 1/2, anchored triangles with
   weight 1/3) and with the upstream loop structure.
3. **CUDA backend.** Every fixture (233 cases: molecules, all elements 1-103,
   extreme coordination numbers, dimer scans and exact cutoff edges, trimers,
   one- and two-atom cells, triclinic and tiny cells with many images, slabs,
   wires, replicas, custom cutoffs) agrees to about 1e-15 hartree in energy and
   1e-15 hartree/bohr in gradient, far inside the agreement targets
   (1e-8 eV + N 1e-10 eV, 1e-7 eV/Å, 1e-8 eV/Å³).

Further checks: GPU neighbour lists equal a CPU enumeration exactly for both
builders (fuzzed over cell shapes, periodicities, unwrapped coordinates,
cutoffs much larger than the cell, and vacuum); CN, weights (including the
exceptional fallback and synthetic ties), C6 and dE/dCN equal the oracle;
forces and strain derivatives (normal and shear) match finite differences with
a step-size sweep; ASE stress matches `ase.calculators.fd`; translation,
rotation, permutation, wrapping, replication, and inactive-vector invariances
hold; neighbour-list reuse equals fresh evaluations; and compute-sanitizer
(memcheck with leak check, racecheck, synccheck, initcheck) is clean.

Run everything on a GPU machine with `make test-gpu cpp-test sanitize`.

## Release artifacts

The tests above run against an editable development build. A release wheel is
tested as shipped with `make wheel gpu-verify`, which the release workflow
also runs:

- `tools/check_wheel.py` confirms the wheel has SASS for every release
  architecture, compute_120 PTX, no register spills in any vanderfast kernel,
  and no CUDA library among its dynamic dependencies.
- The wheel is installed into a fresh virtual environment with PyPI
  dependencies only (numpy, ase, `dftd3==1.6.0`).
- The whole suite then runs with `--require-native --require-cuda
  --require-reference --run-slow` in a scrubbed environment.
- `tools/gpu_report.py` records the wheel's sha256 with the result.

For commit 7ee4352 this gave 826 passed and 1 skipped (the test that needs
*no* CUDA device) on an RTX 4080 SUPER.
