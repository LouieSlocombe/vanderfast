"""Bundled D4 reference tables: provenance, layout, and spot values."""

import hashlib
import math
import struct
from importlib import resources

import numpy as np
import pytest

from vanderfast.data import ReferenceDataError, d4
from vanderfast.data.d4 import (
    fortran_nint,
    gaussian_counts,
    load_d4_manifest,
    load_d4_methods,
    load_d4_reference_tables,
    parse_d4_reference_tables,
    read_d4_reference_blob,
)
from vanderfast.errors import InputError

AATOAU = 1.8897261246204404


def test_manifest_hashes_match_bundled_files() -> None:
    manifest = load_d4_manifest()
    assert set(manifest["outputs"]) == {"d4_reference.bin", "d4_methods.json"}
    for name, entry in manifest["outputs"].items():
        blob = resources.files("vanderfast.data").joinpath(name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"]
        assert len(blob) == entry["bytes"]


def test_manifest_records_pins_and_anomalies() -> None:
    manifest = load_d4_manifest()
    sources = {source["name"]: source for source in manifest["sources"]}
    assert sources["dftd4"]["commit"] == "6e1f59c3f39d919a2dbef0601d2576727c8b30e8"
    assert sources["multicharge"]["commit"] == "6a5d63f9e9e29dcf13cc47cc27f33bf9015681bf"
    assert sources["mctc-lib"]["commit"] == "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d"
    assert manifest["supported_atomic_numbers"] == {"min": 1, "max": 103}
    assert manifest["units"]["table_aatoau"]["value"] == AATOAU
    phantoms = manifest["anomalies"]["uninitialised_references"]
    assert phantoms == [
        {"Z": 92, "reference": 3},
        {"Z": 95, "reference": 5},
        {"Z": 98, "reference": 6},
    ]
    assert manifest["anomalies"]["zero_polarizability_references"] == phantoms
    singles = manifest["anomalies"]["single_precision_literals"]["targets"]
    assert "refcovcn(4,38)" in singles
    assert len(singles) == 28


def test_table_dimensions_and_reference_counts() -> None:
    tables = load_d4_reference_tables()
    assert tables.max_z == 103
    assert tables.max_ref == 7
    assert tables.n_refs == 365
    assert int(tables.nref.sum()) == 365
    assert tables.alphaiw.shape == (365, 23)
    assert tables.secaiw.shape == (18, 23)
    assert tables.element.shape == (104, 10)
    assert tables.source_id.startswith("dftd4 v4.2.0")
    assert tables.sha256 == load_d4_manifest()["outputs"]["d4_reference.bin"]["sha256"]


def test_spot_values_match_upstream_tables() -> None:
    tables = load_d4_reference_tables()
    assert tables.nref[1] == 2
    rows = tables.references(1)
    np.testing.assert_array_equal(tables.reference_column("refcovcn")[rows], [0.0, 0.8942243654604])
    np.testing.assert_array_equal(tables.ngw[rows], [3, 1])
    np.testing.assert_array_equal(tables.ngw[tables.references(8)], [3, 3, 1, 3])
    assert tables.element_column("rcov_d4")[1] == (4.0 / 3.0) * (AATOAU * 0.32)
    assert tables.element_column("eeq_rcov")[6] == tables.element_column("rcov_d4")[6]
    assert tables.element_column("r4r2")[1] == math.sqrt(0.5 * (8.0589 * math.sqrt(1.0)))
    assert tables.element_column("eeq_eta")[1] == -0.35015861
    assert tables.element_column("zeff")[37] == 9.0
    assert tables.element_column("en")[9] == 3.98
    assert math.isnan(tables.element_column("zeff")[0])
    assert tables.freq[0] == 1e-6
    assert tables.weights[0] == 0.5 * (0.05 - 1e-6)
    assert math.isnan(tables.sscale[0])


def test_kindless_literals_are_stored_as_binary32() -> None:
    tables = load_d4_reference_tables()
    strontium = tables.references(38)
    assert tables.reference_column("refcovcn")[strontium][3] == float(np.float32(15.121))


def test_phantom_references_are_zero_filled_but_counted() -> None:
    tables = load_d4_reference_tables()
    uranium = tables.references(92)
    assert tables.nref[92] == 7
    assert tables.refsys[uranium][2] == 0
    assert not np.any(tables.alphaiw[uranium][2])
    assert tables.reference_column("refcn")[uranium][2] == 0.0


def test_gaussian_counts_follow_the_histogram() -> None:
    assert gaussian_counts([0.0, 0.9]) == [3, 1]
    assert gaussian_counts([0.0, 0.0, 2.0]) == [6, 6, 1]
    assert gaussian_counts([25.0]) == [1]
    assert fortran_nint(2.5) == 3
    assert fortran_nint(-2.5) == -3
    assert fortran_nint(2.4999) == 2


def test_unsupported_elements_are_rejected() -> None:
    tables = load_d4_reference_tables()
    assert not tables.supports(0)
    assert not tables.supports(104)
    with pytest.raises(InputError, match="104"):
        tables.references(104)


def test_methods_table_is_hash_checked() -> None:
    methods = load_d4_methods()
    assert methods["tables"]["bjatm"]["pbe"]["s8"] == 0.95948085
    assert methods["aliases"]["gga_x_pbe:gga_c_pbe"] == "pbe"
    assert len(methods["tables"]["bjatm"]) == 118
    assert len(methods["tables"]["bj"]) == 5


def _patched(offset: int, fmt: str, value: object) -> bytes:
    blob = bytearray(read_d4_reference_blob())
    struct.pack_into(fmt, blob, offset, value)
    return bytes(blob)


@pytest.mark.parametrize(
    ("blob", "match"),
    [
        (b"short", "shorter"),
        (_patched(0, "<8s", b"VDFD3TBL"), "magic"),
        (_patched(8, "<I", 2), "format"),
        (_patched(20, "<I", 1), "flags"),
        (_patched(28, "<I", 9), "dimensions"),
        (_patched(112, "<I", 7), "sections"),
        (_patched(120, "<I", 9), "section 0"),
        (_patched(124, "<I", 7), "dtype"),
        (_patched(128, "<Q", 513), "misaligned"),
        (_patched(128, "<Q", 1 << 40), "misaligned or out of bounds"),
    ],
    ids=["short", "magic", "version", "flags", "dims", "nsections", "id", "dtype", "align", "oob"],
)
def test_structural_validation_rejects_corruption(blob: bytes, match: str) -> None:
    with pytest.raises(ReferenceDataError, match=match):
        parse_d4_reference_tables(blob)


def _section_offset(section: int) -> int:
    blob = read_d4_reference_blob()
    return int(struct.unpack_from("<Q", blob, 120 + (section - 1) * 24 + 8)[0])


def test_validation_rejects_bad_counts_and_values() -> None:
    nref = _section_offset(d4.SECTION_NREF)
    with pytest.raises(ReferenceDataError, match="out of range"):
        parse_d4_reference_tables(_patched(nref + 1, "<B", 9))
    with pytest.raises(ReferenceDataError, match="sum"):
        parse_d4_reference_tables(_patched(nref + 2, "<B", 2))
    alphaiw = _section_offset(d4.SECTION_ALPHAIW)
    with pytest.raises(ReferenceDataError, match="non-finite"):
        parse_d4_reference_tables(_patched(alphaiw, "<d", math.nan))
    refsys = _section_offset(d4.SECTION_REFSYS)
    with pytest.raises(ReferenceDataError, match="secondary"):
        parse_d4_reference_tables(_patched(refsys, "<B", 18))
    freq = _section_offset(d4.SECTION_FREQ)
    with pytest.raises(ReferenceDataError, match="increasing"):
        parse_d4_reference_tables(_patched(freq + 8, "<d", 0.0))
    ngw = _section_offset(d4.SECTION_NGW)
    with pytest.raises(ReferenceDataError, match="Gaussian counts of Z=1"):
        parse_d4_reference_tables(_patched(ngw, "<B", 4))


def test_loader_detects_manifest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    original = d4._read_resource  # noqa: SLF001

    def tampered(name: str) -> bytes:
        payload = original(name)
        if name == "d4_methods.json":
            return payload + b" "
        if name == "d4_reference.bin":
            return payload + b"\0"
        return payload

    load_d4_reference_tables.cache_clear()
    load_d4_methods.cache_clear()
    monkeypatch.setattr(d4, "_read_resource", tampered)
    try:
        with pytest.raises(ReferenceDataError, match="manifest"):
            load_d4_reference_tables()
        with pytest.raises(ReferenceDataError, match="manifest"):
            load_d4_methods()
    finally:
        monkeypatch.undo()
        load_d4_reference_tables.cache_clear()
        load_d4_methods.cache_clear()
