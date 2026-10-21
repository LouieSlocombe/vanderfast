# vanderfast

GPU DFT-D3(BJ) dispersion energies, forces, and stress, with the
Axilrod-Teller-Muto three-body term, for molecules and periodic systems. The
numerical core is CUDA C++; the Python layer provides a NumPy API and an ASE
calculator. Results are validated against simple-dftd3 v1.6.0.

> **Status: under development.** The scientific specification, reference data,
> NumPy oracle, and dftd3 fixtures are in place. The CUDA backend is being
> built milestone by milestone (see `docs/`). No performance claims are made yet.

## Requirements

- Linux x86_64
- Python 3.14
- For the CUDA backend: an NVIDIA GPU with compute capability 7.5 or newer and a
  CUDA 13 capable driver
- Development: the conda-forge environment in `build_tools/`

## Development setup

```bash
conda env create -f build_tools/environment.yml
conda activate vanderfast-dev
make dev-cpu         # editable install of the pure-Python parts
make test            # oracle, fixture, and reference tests
make lint            # pre-commit hooks (prek) and mypy
make data-check      # regenerate the reference tables byte for byte
make fixtures-check  # regenerate the dftd3 fixtures
```

## Parameters

```python
from vanderfast import D3BJParameters, resolve_parameters

params = resolve_parameters(method="pbe", damping="bj", atm=True)
custom = D3BJParameters(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0, alp=14.0)
```

ATM must be chosen explicitly. `atm=True` sets `s9 = 1`; the pair coefficients
do not change.

## Documentation

- `docs/conventions.md`: units, geometry, terms, counting, parameters, and
  known differences from simple-dftd3
- `docs/data-provenance.md`: how the bundled tables are generated and verified
- `build_tools/README.md`: environment and toolchain

## License

vanderfast's code is released under the [MIT License](LICENSE). The bundled
reference tables derive from simple-dftd3 (LGPL-3.0-or-later) and mctc-lib
(Apache-2.0); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
