# Third-party notices

vanderfast's source code is released under the MIT License (see `LICENSE`).
The package also bundles scientific reference data derived from the third-party
projects listed below. Their licence texts are reproduced in `data/licenses/`.

## simple-dftd3

- Project: <https://github.com/dftd3/simple-dftd3>
- Version: v1.6.0 (commit `6004cfd62ad2a248ad32167392bd08c674da6161`)
- Licence: GNU Lesser General Public License v3.0 or later
  (`data/licenses/simple-dftd3-COPYING`, `data/licenses/simple-dftd3-COPYING.LESSER`)
- Used for: reference coordination numbers and C6 coefficients
  (`src/dftd3/reference.f90`), r4/r2 expectation values (`src/dftd3/data/r4r2.f90`),
  pairwise van der Waals radii (`src/dftd3/data/vdwrad.f90`), and D3(BJ) damping
  parameters (`src/dftd3/param.f90`).

The derived tables are `src/vanderfast/data/d3_reference.bin` and
`src/vanderfast/data/d3bj_methods.json`. `tools/generate_d3_data.py` reproduces them
byte for byte from the pinned sources listed in `data/sources.toml`. A different
table in the same documented format can be supplied at run time without recompiling
vanderfast (`reference_data=` in `vanderfast.D3Context`).

## mctc-lib

- Project: <https://github.com/grimme-lab/mctc-lib>
- Version: commit `06db4d899b8146ee96720d97ee405982b8b6f6f6`, pinned by simple-dftd3
  v1.6.0. The files used are byte-identical to release v0.5.2.
- Licence: Apache License 2.0 (`data/licenses/mctc-lib-LICENSE`)
- Used for: D3 covalent radii (`src/mctc/data/covrad.f90`) and the Ångström-to-bohr
  conversion factor (`src/mctc/io/convert.f90`, `constants.f90`, `codata2018.f90`).

## Scientific references

Publications using vanderfast should cite the original D3 and damping papers:

- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104 (2010),
  doi:10.1063/1.3382344.
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011),
  doi:10.1002/jcc.21759.
- The method-specific parameter references listed by simple-dftd3.
