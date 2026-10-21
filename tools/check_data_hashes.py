"""Check that the bundled reference data matches its manifest (stdlib only)."""

import hashlib
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "vanderfast" / "data"


def main() -> int:
    """Return 0 when every output listed in the manifest has the recorded hash."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    status = 0
    for name, entry in manifest["outputs"].items():
        path = DATA_DIR / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if digest != entry["sha256"]:
            print(f"{path}: sha256 {digest} does not match the manifest", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
