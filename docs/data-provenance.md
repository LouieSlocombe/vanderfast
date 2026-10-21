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
