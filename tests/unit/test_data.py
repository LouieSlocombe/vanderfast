"""Bundled reference tables: provenance, layout, and spot values."""

import hashlib
import math
import struct
from importlib import resources

import numpy as np
import pytest

from vanderfast import data
from vanderfast.data import (
    ReferenceDataError,
    load_manifest,
    load_methods,
    load_reference_tables,
    pair_index,
    parse_reference_tables,
    read_reference_blob,
)
from vanderfast.errors import InputError

AATOAU = 1.8897261246204404


def test_manifest_hashes_match_bundled_files() -> None:
    manifest = load_manifest()
    for name, entry in manifest["outputs"].items():
        blob = resources.files("vanderfast.data").joinpath(name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"]
        assert len(blob) == entry["bytes"]


def test_manifest_records_pins_units_and_anomalies() -> None:
    manifest = load_manifest()
    sources = {source["name"]: source for source in manifest["sources"]}
    assert sources["simple-dftd3"]["commit"] == "6004cfd62ad2a248ad32167392bd08c674da6161"
    assert sources["mctc-lib"]["commit"] == "06db4d899b8146ee96720d97ee405982b8b6f6f6"
    assert manifest["supported_atomic_numbers"] == {"min": 1, "max": 103}
    assert manifest["units"]["table_aatoau"]["value"] == AATOAU
    assert manifest["conventions"]["weights"]["tied_maxima"] == []
    kinds = {(a["Z"], a["kind"]) for a in manifest["anomalies"]}
    assert kinds == {(96, "populated_padding_reference_cn"), (96, "populated_padding_c6")}
    assert manifest["zero_c6_valid_references"] == [{"Z": 101, "reference": 6}]


def test_table_dimensions_and_reference_counts() -> None:
    tables = load_reference_tables()
    assert tables.max_z == 103
    assert tables.max_ref == 7
    assert int(tables.nref.sum()) == 338
    assert tables.c6_flat.size == 57810
    assert tables.nref[1] == 2
    assert tables.nref[6] == 5
    assert tables.nref[96] == 6
    assert tables.source_id.startswith("simple-dftd3 v1.6.0")
    assert tables.sha256 == load_manifest()["outputs"]["d3_reference.bin"]["sha256"]


def test_spot_values_match_upstream_tables() -> None:
    tables = load_reference_tables()
    np.testing.assert_array_equal(tables.reference_cn(1), [0.9118, 0.0])
    np.testing.assert_array_equal(tables.c6_block(1, 1), [[3.0267, 4.7379], [4.7379, 7.5916]])
    np.testing.assert_array_equal(tables.reference_cn(101)[-1], -1.0)
    assert tables.rcov[1] == (4.0 / 3.0) * (AATOAU * 0.32)
    assert tables.r4r2[1] == math.sqrt(0.5 * (8.0589 * math.sqrt(1.0)))
    assert math.isnan(tables.rcov[0])


def test_c6_block_transposes_for_reversed_pairs() -> None:
    tables = load_reference_tables()
    forward = tables.c6_block(6, 1)
    assert forward.shape == (5, 2)
    np.testing.assert_array_equal(tables.c6_block(1, 6), forward.T)
    assert tables.vdw_radius(1, 6) == tables.vdw_radius(6, 1)


def test_pair_index_is_symmetric() -> None:
    assert pair_index(1, 1) == 0
    assert pair_index(2, 1) == pair_index(1, 2) == 1
    assert pair_index(103, 103) == 103 * 104 // 2 - 1


def test_unsupported_elements_are_rejected() -> None:
    tables = load_reference_tables()
    assert not tables.supports(0)
    assert not tables.supports(104)
    with pytest.raises(InputError, match="104"):
        tables.reference_cn(104)
    with pytest.raises(InputError):
        tables.c6_block(1, 0)
    with pytest.raises(InputError):
        tables.vdw_radius(200, 1)


def test_methods_table_is_hash_checked() -> None:
    methods = load_methods()
    assert methods["methods"]["pbe"]["s8"] == 0.7875
    assert methods["aliases"]["b3lyp5"] == "b3lyp"


def _patched(offset: int, fmt: str, value: object) -> bytes:
    blob = bytearray(read_reference_blob())
    struct.pack_into(fmt, blob, offset, value)
    return bytes(blob)


@pytest.mark.parametrize(
    ("blob", "match"),
    [
        (b"short", "shorter"),
        (_patched(0, "<8s", b"NOTMAGIC"), "magic"),
        (_patched(8, "<I", 2), "format"),
        (_patched(20, "<I", 0), "flags"),
        (_patched(36, "<I", 7), "dimensions"),
        (_patched(112, "<I", 5), "sections"),
        (_patched(120, "<I", 9), "section 0"),
        (_patched(124, "<I", 7), "dtype"),
        (_patched(128, "<Q", 513), "misaligned"),
        (_patched(152, "<Q", 1 << 40), "misaligned or out of bounds"),
    ],
    ids=[
        "short",
        "magic",
        "version",
        "flags",
        "dims",
        "nsections",
        "id",
        "dtype",
        "align",
        "bounds",
    ],
)
def test_structural_validation_rejects_corruption(blob: bytes, match: str) -> None:
    with pytest.raises(ReferenceDataError, match=match):
        parse_reference_tables(blob)


def _section_offset(section: int) -> int:
    blob = read_reference_blob()
    return int(struct.unpack_from("<Q", blob, 120 + (section - 1) * 24 + 8)[0])


def test_validation_rejects_bad_counts_and_values() -> None:
    nref = _section_offset(data.SECTION_NREF)
    with pytest.raises(ReferenceDataError, match="out of range"):
        parse_reference_tables(_patched(nref + 1, "<B", 9))
    with pytest.raises(ReferenceDataError, match="sum"):
        parse_reference_tables(_patched(nref + 2, "<B", 2))
    c6 = _section_offset(data.SECTION_C6)
    with pytest.raises(ReferenceDataError, match="non-finite"):
        parse_reference_tables(_patched(c6, "<d", math.inf))
    header = bytearray(read_reference_blob())
    struct.pack_into("<I", header, 32, 339)  # n_refs no longer matches the counts
    struct.pack_into("<Q", header, 120 + 24 + 16, 339)
    with pytest.raises(ReferenceDataError, match="sum"):
        parse_reference_tables(bytes(header))


def test_validation_rejects_c6_length_mismatch() -> None:
    blob = bytearray(read_reference_blob())
    struct.pack_into("<Q", blob, 40, 57809)
    struct.pack_into("<Q", blob, 120 + 4 * 24 + 16, 57809)
    with pytest.raises(ReferenceDataError, match="C6 table length"):
        parse_reference_tables(bytes(blob))


def test_loader_detects_manifest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    original = data._read_resource  # noqa: SLF001

    def tampered(name: str) -> bytes:
        payload = original(name)
        if name == "d3bj_methods.json":
            return payload + b" "
        if name == "d3_reference.bin":
            return payload + b"\0"
        return payload

    load_reference_tables.cache_clear()
    load_methods.cache_clear()
    monkeypatch.setattr(data, "_read_resource", tampered)
    try:
        with pytest.raises(ReferenceDataError, match="manifest"):
            load_reference_tables()
        with pytest.raises(ReferenceDataError, match="manifest"):
            load_methods()
    finally:
        monkeypatch.undo()
        load_reference_tables.cache_clear()
        load_methods.cache_clear()
