"""Check that a source distribution contains everything needed to build and test."""

import sys
import tarfile
from pathlib import Path

REQUIRED = [
    "pyproject.toml",
    "CMakeLists.txt",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "data/sources.toml",
    "data/licenses/simple-dftd3-COPYING.LESSER",
    "data/licenses/mctc-lib-LICENSE",
    "data/sources-d4.toml",
    "data/licenses/dftd4-COPYING.LESSER",
    "data/licenses/multicharge-LICENSE",
    "src/vanderfast/__init__.py",
    "src/vanderfast/data/d3_reference.bin",
    "src/vanderfast/data/d3bj_methods.json",
    "src/vanderfast/data/manifest.json",
    "src/vanderfast/data/d4_reference.bin",
    "src/vanderfast/data/d4_methods.json",
    "src/vanderfast/data/d4_manifest.json",
    "tests/fixtures/dftd3-1.6.0/molecules.jsonl",
]
FORBIDDEN_PREFIXES = ["build/", "reports/", ".github/", "benchmarks/results/"]


def main(argv: list[str]) -> int:
    """Return 0 when the sdist given as the only argument is complete."""
    (path,) = argv
    with tarfile.open(path) as archive:
        names = archive.getnames()
    root = Path(names[0]).parts[0]
    members = {name.removeprefix(root + "/") for name in names}
    status = 0
    for required in REQUIRED:
        if required not in members:
            print(f"missing from sdist: {required}", file=sys.stderr)
            status = 1
    for member in members:
        if any(member.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            print(f"unexpected in sdist: {member}", file=sys.stderr)
            status = 1
    if status == 0:
        print(f"{path}: {len(members)} files, all required files present")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
