# Reference data provenance

`src/vanderfast/data/` holds three generated files:

| File | Content |
| --- | --- |
| `d3_reference.bin` | Packed D3 tables, format v1 (below) |
| `d3bj_methods.json` | D3(BJ) parameters for 161 methods and 201 aliases |
| `manifest.json` | Source pins, file and table hashes, units, conventions, anomalies |

They are produced by `tools/generate_d3_data.py` from the files pinned in
`data/sources.toml` (simple-dftd3 v1.6.0 `6004cfd6`, mctc-lib `06db4d89`). The
generator verifies each file's sha256, parses the Fortran data statements with a
literal-only tokenizer, evaluates the unit conversions in upstream operation
order, and asserts the convention-bearing source lines. `--check` requires a
byte-identical regeneration; `--verify-binary LIBDIR` finds every parsed array
byte for byte inside the compiled `libs-dftd3` and `libmctc-lib`.

## Format v1

Little-endian. A 512-byte header (`VDFD3TBL`, version 1, byte-order mark
`0x01020304`, flags: radii already D3-scaled | atomic units, `max_z = 103`,
`max_ref = 7`, 338 references, 5356 element pairs, 57,810 C6 values, a source id,
and a table of six sections) is followed by 64-byte aligned sections:

| Section | Type | Count | Content |
| --- | --- | --- | --- |
| NREF | u8 | 104 | reference counts, indexed by Z (slot 0 = 0) |
| REF_CN | f64 | 338 | reference CNs of valid references, element-major |
| RCOV_D3 | f64 | 104 | D3-scaled covalent radii (bohr), slot 0 = NaN |
| R4R2 | f64 | 104 | `sqrt(0.5 r4/r2 sqrt(Z))`, slot 0 = NaN |
| C6 | f64 | 57,810 | packed valid blocks per pair `ic = Zi(Zi-1)/2 + (Zj-1)`, `Zi >= Zj`, row-major `nref(Zi) x nref(Zj)` |
| RVDW | f64 | 5356 | pairwise van der Waals radii (bohr), same pair order |

## Recorded anomalies

- Cm (Z = 96) has six references, but upstream also stores a seventh reference
  CN and 344 nonzero C6 entries for it. Upstream never reads them; they are
  dropped.
- Md (Z = 101) reference 6 is valid, with reference CN −1 and zero C6 against
  every partner. It takes part in the weight normalisation as upstream.
- No element has a tied maximum reference CN.

## DFT-D4 tables

Three more files hold the D4 data:

| File | Content |
| --- | --- |
| `d4_reference.bin` | Packed D4 and EEQ tables, format `VDFD4TBL` v1 (below) |
| `d4_methods.json` | rational-damping parameters: 118 sets with ATM, 5 without, 319 aliases |
| `d4_manifest.json` | source pins, file and table hashes, units, conventions, anomalies |

`tools/generate_d4_data.py` produces them from the files pinned in
`data/sources-d4.toml`: dftd4 v4.2.0 (`6e1f59c3`), multicharge v0.5.0
(`6a5d63f9`, pinned by dftd4) and mctc-lib v0.5.2 (`e9de066d`; dftd4's wraps
name v0.5.1, whose data files are identical, while conda-forge links 0.5.2).
It checks every file's sha256, parses `reference.inc` and the element data
with the same literal-only tokenizer as the D3 generator, asserts 108
convention-bearing source lines, and records the sha256 of the reference
polarisabilities and C6 that the tables imply at the default `ga`, `gc`.
`--check` requires a byte-identical regeneration and `--verify-binary LIBDIR`
finds every array inside `libdftd4`, `libmulticharge`, and `libmctc-lib`.

The tables are stored raw rather than as precomputed C6, so `ga`, `gc`, and
`wf` stay free: the C6 of the species in a context are computed on the host
when it is created or its parameters change.

### Format VDFD4TBL v1

Little-endian, with the D3 container layout: a 512-byte header (`VDFD4TBL`,
version 1, byte-order mark `0x01020304`, flags: scaled covalent radii | atomic
units | raw polarisabilities | EEQ 2019 parameters, `max_z = 103`,
`max_ref = 7`, 365 references, 23 frequencies, 17 secondary systems, a source
id, and a table of eight sections) followed by 64-byte aligned sections:

| Section | Type | Count | Content |
| --- | --- | --- | --- |
| NREF | u8 | 104 | reference counts by Z (slot 0 = 0) |
| NGW | u8 | 365 | Gaussians per reference (from the reference-CN histogram; checked on load) |
| REFSYS | u8 | 365 | secondary system of each reference (0 = none) |
| REF_SCALARS | f64 | 365 × 6 | refcovcn, refcn, clsq, clsh, hcount, ascale |
| ALPHAIW | f64 | 365 × 23 | reference polarisabilities at imaginary frequencies |
| SECONDARY | f64 | 18 × 24 | sscale and 23 secondary polarisabilities (row 0 a NaN sentinel) |
| ELEMENT | f64 | 104 × 10 | zeff, hardness, EN, D4 covalent radius, r4r2, EEQ χ, η, κ, radius, covalent radius |
| FREQ | f64 | 2 × 23 | frequencies, then trapezoid weights |

### Recorded anomalies

- Uranium reference 3, americium reference 5, and californium reference 6
  are counted by upstream (in the weights and the Gaussian counts) but never
  initialised; they are stored as zeros and have zero polarisability, as
  upstream evaluates them.
- Secondary systems 3, 4, 5, and 12-16 are never initialised upstream; no
  reference uses them.
- 28 values of strontium and barium reference 4 are written without a kind
  suffix in `reference.inc`, so gfortran stores them as single-precision
  literals widened to double; the tables keep those widened values.
- Elements 112-118 have references upstream but no EEQ parameters, so D4
  supports Z = 1-103 like dftd4.
