"""The D3 path is frozen: its data, provenance, and fixtures must not change.

Adding DFT-D4 touches the shared generator, oracle, and test machinery. These
hashes pin every D3 artefact so that an accidental change (a manifest key
reordered, a fixture regenerated with different numerics) fails loudly. A
deliberate scientific change to D3 updates this table in the same commit.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FROZEN = {
    "src/vanderfast/data/d3_reference.bin": (
        "1361e7e437052c4f21759518cb53303c399be2be78caea0afaf9bb0450ea3507"
    ),
    "src/vanderfast/data/d3bj_methods.json": (
        "cfbaf25b26934b5a9aa21da7b7de34c53ca34b322fb632d5e82a2c953f2b8a6c"
    ),
    "src/vanderfast/data/manifest.json": (
        "d873084f4a8551fac32774c9f88d890c70286ef954dedc84d336b507a19df319"
    ),
    "data/sources.toml": "7c7b85217356ab4b32d49cd58827724932b9a141aa945bd94c51d70f99aa0091",
    "tests/fixtures/dftd3-1.6.0/coverage.jsonl": (
        "8b381643ffb2a571c5a46dfcb0302a7c13585ef22d112fac881e4ac724287829"
    ),
    "tests/fixtures/dftd3-1.6.0/dimers.jsonl": (
        "47c1590fa62007825246afaddbfbece86b0ed2d6df9f840a09fe5001bcffceac"
    ),
    "tests/fixtures/dftd3-1.6.0/extreme_cn.jsonl": (
        "7052201e1d15eabb42eea19eba20606adba60b95dec3c887fffdad8848d414d9"
    ),
    "tests/fixtures/dftd3-1.6.0/molecules.jsonl": (
        "f88b92e997b6cf3ea81737f94c8c240c40ff3646193672c9df9eef4aea95373a"
    ),
    "tests/fixtures/dftd3-1.6.0/partial_pbc.jsonl": (
        "72a63c1e8a19ee22e528376b8c286941a106100521f9f2135fc29910926ccddf"
    ),
    "tests/fixtures/dftd3-1.6.0/periodic.jsonl": (
        "c0ae2d6fce84231373688ebe132b438f30fca7a1e4829fc8ae0eec544659bb25"
    ),
    "tests/fixtures/dftd3-1.6.0/trimers.jsonl": (
        "83dcfa62545dff78270d032eda4eb9787ce75a821728ac6d658a61a1ebab52dc"
    ),
}


@pytest.mark.parametrize("relpath", sorted(FROZEN))
def test_d3_artefact_is_unchanged(relpath: str) -> None:
    path = ROOT / relpath
    if not path.is_file():  # an installed wheel ships the data but not the repository files
        pytest.skip(f"{relpath} is not part of this installation")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN[relpath]
