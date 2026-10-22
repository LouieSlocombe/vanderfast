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

## dftd4

- Project: <https://github.com/dftd4/dftd4>
- Version: v4.2.0 (commit `6e1f59c3f39d919a2dbef0601d2576727c8b30e8`)
- Licence: GNU Lesser General Public License v3.0 or later
  (`data/licenses/dftd4-COPYING`, `data/licenses/dftd4-COPYING.LESSER`)
- Used for: D4 reference polarisabilities, reference coordination numbers and
  charges (`src/dftd4/reference.inc`, `src/dftd4/reference.f90`), effective
  nuclear charges, chemical hardnesses, electronegativities, covalent radii,
  and r4/r2 values (`src/dftd4/data/`), the Casimir-Polder frequency grid
  (`src/dftd4/model/d4.f90`), and the D4 rational-damping parameters
  (`src/dftd4/param.f90`).

## multicharge

- Project: <https://github.com/grimme-lab/multicharge>
- Version: v0.5.0 (commit `6a5d63f9e9e29dcf13cc47cc27f33bf9015681bf`), pinned by
  dftd4 v4.2.0
- Licence: Apache License 2.0 (`data/licenses/multicharge-LICENSE`)
- Used for: the EEQ 2019 charge-model parameters
  (`src/multicharge/param/eeq2019.f90`).

The derived tables are `src/vanderfast/data/d4_reference.bin` and
`src/vanderfast/data/d4_methods.json`. `tools/generate_d4_data.py` reproduces
them byte for byte from the pinned sources listed in `data/sources-d4.toml`
(dftd4, multicharge, and mctc-lib v0.5.2, commit
`e9de066d89f250d1cfb6de3a33f0c27c0e2f855d`, for the covalent radii and unit
conversion). A replacement table in the same format can be loaded with
`reference_data=` in `vanderfast.D4Context`.

## Scientific references

Publications using vanderfast should cite the original D3 and damping papers:

- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104 (2010),
  doi:10.1063/1.3382344.
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011),
  doi:10.1002/jcc.21759.
- The method-specific parameter references listed by simple-dftd3.

Publications using the D4 model should cite:

- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **147**, 034112 (2017),
  doi:10.1063/1.4993215.
- E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth,
  S. Grimme, *J. Chem. Phys.* **150**, 154122 (2019), doi:10.1063/1.5090222.
- E. Caldeweyher, J.-M. Mewes, S. Ehlert, S. Grimme, *Phys. Chem. Chem. Phys.*
  **22**, 8499 (2020), doi:10.1039/D0CP00502A (periodic systems).
- The method-specific parameter references listed by dftd4.
