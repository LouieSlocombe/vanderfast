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

## DFT-D4

D4 is checked against dftd4 v4.2.0 (with multicharge v0.5.0) the same way.

1. **Reference data.** `tools/generate_d4_data.py --check --verify-binary`
   regenerates `d4_reference.bin` byte for byte and finds every parsed array
   inside the compiled `libdftd4`, `libmulticharge`, and `libmctc-lib`. The
   reference polarisabilities and C6 derived from it are hashed in the
   manifest; the native host code reproduces both hashes bitwise, and agrees
   bitwise with the oracle for other `ga`/`gc`. Every method alias resolves to
   parameters for which dftd4's own lookup gives a bitwise identical energy,
   with and without ATM.
2. **NumPy oracle.** `vanderfast._oracle_d4` evaluates both coordination
   numbers, the bordered EEQ system (dense solve), the weights, C6, the pair
   and ATM terms, and the adjoint gradient and virial, including multicharge's
   periodic Ewald sums with their quirks. It matches every fixture: energies
   to 3e-13 relative, charges to 2e-14 e, gradients and virials to about
   1e-15. Its `ewald="converged"` mode is independent of the Ewald parameter
   and agrees between a cell and its supercells.
3. **CUDA backend.** All 291 fixtures (`tests/fixtures/dftd4-4.2.0`:
   molecules and ions, every element 1-103 neutral and charged, actinides
   with upstream's zero-filled references, dimer scans across the 25, 30, 40,
   and 60 bohr cutoffs, trimers, ionic and skewed crystals, cell dependence,
   slabs and wires, extreme coordination numbers, zero and negative effective
   charges, and a dense hydrogen cluster whose EEQ matrix is indefinite) agree
   to 1.1e-13 relative in energy, 2e-14 e in charges, 2e-15 hartree/bohr in
   gradients, and 3e-14 hartree in virials. The tests allow 1e-12. MINRES
   and the shared-memory LU give the same agreement.

Further D4 checks: coordination numbers, weights and their derivatives, dE/dCN
and dE/dq equal the oracle; forces and strain derivatives match finite
differences in both Ewald modes; slabs and wires equal the same cell with every
flag set; translations, rotations, permutations, and lattice shifts leave
results unchanged; the converged mode gives the same energy per cell for a
crystal and its supercell; the optimised kernels (tiled assembly, structure
factors, packed MINRES, stored lattice derivatives) agree with the literal
kernels and dense LU to 2e-13 e and 1e-14 hartree/bohr; pair-only results
are bitwise reproducible; `set_charge` and `set_parameters` equal fresh
contexts; and the ASE calculator's reference backend reproduces `dftd4.ase`
bitwise, with the CUDA backend matching it in energy, forces, stress, and
charges.

Run everything on a GPU machine with `make test-gpu cpp-test sanitize`.

## Release artifacts

The tests above run against an editable development build. A release wheel is
tested as shipped with `make wheel gpu-verify`, which the release workflow
also runs:

- `tools/check_wheel.py` confirms the wheel has SASS for every release
  architecture, compute_120 PTX, no register spills in any vanderfast kernel,
  and no CUDA library among its dynamic dependencies.
- The wheel is installed into a fresh virtual environment with PyPI
  dependencies only (numpy, ase, `dftd3==1.6.0`, `dftd4==4.2.0`).
- The whole suite then runs with `--require-native --require-cuda
  --require-reference --run-slow` in a scrubbed environment.
- `tools/gpu_report.py` records the wheel's sha256 with the result.

For the 1.0.0 wheel this gave 1653 passed and 1 skipped (the test that needs
*no* CUDA device) on an RTX 4080 SUPER, with compute-sanitizer memcheck,
racecheck, synccheck, and initcheck reporting no errors.
