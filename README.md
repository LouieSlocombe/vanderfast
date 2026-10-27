# vanderfast

GPU DFT-D3(BJ) and DFT-D4 dispersion energies, forces, and stress, with the
Axilrod-Teller-Muto three-body term, for molecules and periodic systems. The
numerical core is CUDA C++ in double precision; the Python layer provides a
NumPy API and ASE calculators. D3 results agree with simple-dftd3 v1.6.0 to
about 1e-15 in atomic units on every reference fixture; D4 results, including
the EEQ partial charges, agree with dftd4 v4.2.0 to about 1e-13.

> **Status: 1.0.** D3(BJ) + ATM and D4 (EEQ charges, rational damping, ATM)
> on molecules, 3D cells, slabs, and wires; the public API is stable and
> follows [semantic versioning](https://semver.org/). Tested on one GPU
> (RTX 4080 SUPER, sm_89); see the support matrix in
> [docs/build.md](docs/build.md) and the release notes in
> [CHANGELOG.md](CHANGELOG.md).

## Requirements

- Linux x86_64, Python 3.14
- An NVIDIA GPU with compute capability 7.5 or newer and a driver that supports
  CUDA 12.6 or newer, CUDA 13 included (the CUDA runtime is linked statically;
  no CUDA toolkit is needed at run time)

## Installation

From PyPI (a CUDA wheel for Linux x86_64 and CPython 3.14, plus the sdist):

```bash
pip install vanderfast             # add [reference] for the dftd3 1.6.0 and dftd4 4.2.0 backends
python -m vanderfast info --json   # build, data, and device report
```

From source, use the conda-forge toolchain in `build_tools/` (any CUDA toolkit
from 12.6 on works; see [docs/build.md](docs/build.md)):

```bash
conda env create -f build_tools/environment.yml
conda activate vanderfast-dev
make dev        # editable CUDA build for this machine's GPU
make test-gpu   # full suite; fails rather than skips GPU and reference tests
```

`VANDERFAST_ENABLE_CUDA=OFF pip install .` installs the pure-Python parts
(parameters, data, the NumPy oracles, and the reference backends) without the
CUDA backend.

## Usage

```python
from ase.build import bulk
from vanderfast.ase import D3Calculator

atoms = bulk("MgO", "rocksalt", a=4.21, cubic=True).repeat(2)
atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()  # eV/Å
stress = atoms.get_stress()  # eV/Å³, 3D periodic cells only
```

ATM must be chosen explicitly: `atm=True` sets `s9 = 1` and leaves the pair
coefficients unchanged. For repeated evaluations without ASE objects, a
`D3Context` keeps the device tables and the neighbour list between calls:

```python
from vanderfast import D3Context, resolve_parameters

params = resolve_parameters(method="pbe", damping="bj", atm=False)
with D3Context(atoms.numbers, params, skin=1.0) as context:
    result = context.evaluate(
        atoms.positions, cell=atoms.cell[:], pbc=True, properties=("energy", "forces", "stress")
    )
```

### DFT-D4

```python
from vanderfast.ase import D4Calculator

atoms.calc = D4Calculator(method="r2scan")  # ATM on by default, as in dftd4
energy = atoms.get_potential_energy()
charges = atoms.calc.get_property("charges", atoms)  # EEQ partial charges, e
```

The total charge is the sum of `atoms.get_initial_charges()` unless `charge=`
is given. As in dftd4, any periodic direction makes the system fully
periodic (slabs and wires need vacuum vectors), and stress is available for
every periodic system. The periodic charges reproduce dftd4's truncated Ewald
sums by default; `ewald="converged"` uses complete sums, which do not depend
on the choice of cell. `D4Context` is the array interface.

More in [examples/](examples/): a molecule, crystal stress, comparisons with
`dftd3.ase` and `dftd4.ase`, D4 charges of an anion, both periodic charge
models, repeated MD evaluations, and the array interface.

## Performance

Warm energy-and-force evaluations of the release wheel on an RTX 4080 SUPER,
against simple-dftd3 1.6.0 (conda-forge, OpenMP) on 28 threads of an
i7-14700KF. Default cutoffs: 40 bohr for CN and ATM, 60 bohr for the pair term.
Times include host-device transfers and Python. dftd3 was not timed above 1024
atoms.

| System | Atoms | Term | vanderfast (ms) | dftd3, 28 threads (ms) | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| NaCl primitive cell | 2 | pair | 0.152 | 0.411 | 3x |
| NaCl primitive cell | 2 | pair + ATM | 5.65 | 79.9 | 14x |
| benzene cluster | 1020 | pair | 0.78 | 9.82 | 13x |
| benzene cluster | 1020 | pair + ATM | 39 | 147 | 4x |
| liquid water | 1023 | pair | 6.31 | 43.6 | 7x |
| liquid water | 1023 | pair + ATM | 1,505 | 6,756 | 4x |
| MgO crystal | 512 | pair | 3.19 | 14 | 4x |
| MgO crystal | 512 | pair + ATM | 817 | 5,651 | 7x |
| liquid water | 4095 | pair | 19.6 | – | – |
| liquid water | 4095 | pair + ATM | 3,729 | – | – |
| MgO crystal | 4096 | pair | 25.5 | – | – |
| MgO crystal | 4096 | pair + ATM | 6,682 | – | – |

Against a 28-thread CPU run of the reference, the speedup is 3–14×. With
ATM on, the three-body term takes over 98% of GPU time once a system passes
about 1000 atoms (1.5 million triangles per atom in MgO). It is FP64-bound, and
this consumer GPU runs FP64 at 1/64 of its FP32 rate. A data-centre GPU with
full-rate FP64 should do much better, but none has been measured. Full
tables, stage timings, the method, and the remaining bottlenecks are in
[docs/performance.md](docs/performance.md).

DFT-D4 on the same machine against dftd4 4.2.0 (conda-forge, OpenMP, 28
threads, single-threaded OpenBLAS, its fastest setting), energy and forces
with the EEQ charges, default cutoffs, development build:

| System | Atoms | Term | vanderfast (ms) | dftd4, 28 threads (ms) | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| benzene cluster | 252 | pair | 1.55 | 13 | 8x |
| benzene cluster | 1020 | pair | 5.17 | 337 | 65x |
| benzene cluster | 1020 | pair + ATM | 42 | 622 | 15x |
| liquid water | 255 | pair | 4.32 | 104 | 24x |
| liquid water | 1023 | pair | 19.4 | 1,196 | 61x |
| liquid water | 1023 | pair + ATM | 1,474 | 12,781 | 9x |
| MgO crystal | 512 | pair | 13.2 | 394 | 30x |
| MgO crystal | 512 | pair + ATM | 899 | 9,239 | 10x |
| liquid water | 4095 | pair | 211 | – | – |
| MgO crystal | 4096 | pair | 287 | – | – |

Periodic D4 is dominated by dftd4's real-space Ewald sums for the charges,
large molecules by the iterative charge solve, and anything with ATM by the
three-body term.

## Documentation

- [docs/api.md](docs/api.md): parameters, `D3Context`, `D4Context`, results,
  errors, diagnostics
- [docs/ase.md](docs/ase.md): the ASE calculators and their caching rules
- [docs/conventions.md](docs/conventions.md): units, geometry, counting,
  parameters, the D4 charge model, and known differences from the references
- [docs/validation.md](docs/validation.md): how correctness is established
- [docs/performance.md](docs/performance.md): measured results and method
- [docs/limitations.md](docs/limitations.md): what is out of scope
- [docs/build.md](docs/build.md): toolchain, build options, support matrix,
  releasing
- [docs/data-provenance.md](docs/data-provenance.md): how the bundled tables
  are generated and verified

## Development

```bash
make lint            # pre-commit hooks (prek) and mypy
make test            # CPU: oracle, fixtures, fake-native API tests, 100% coverage
make cpp-test        # Catch2 host and device tests
make sanitize        # compute-sanitizer memcheck, racecheck, synccheck, initcheck
make data-check      # regenerate the D3 and D4 reference tables byte for byte
make fixtures-check  # regenerate the dftd3 and dftd4 fixtures
make bench-smoke     # benchmark harness sanity run
```

## Citing

If vanderfast contributes to published work, cite it with the metadata in
[CITATION.cff](CITATION.cff) alongside the D3 and D4 papers listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

vanderfast's code is released under the [MIT License](LICENSE). The bundled
reference tables derive from simple-dftd3 and dftd4 (LGPL-3.0-or-later),
multicharge and mctc-lib (Apache-2.0); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A replacement table can be
loaded at run time with `reference_data=`.
