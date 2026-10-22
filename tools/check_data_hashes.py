"""Check that the bundled reference data matches its manifests (stdlib only)."""

import hashlib
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "vanderfast" / "data"


def main() -> int:
    """Return 0 when every output listed in every manifest has the recorded hash."""
    status = 0
    for manifest_path in sorted(DATA_DIR.glob("*manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, entry in manifest["outputs"].items():
            path = DATA_DIR / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
            if digest != entry["sha256"]:
                print(
                    f"{path}: sha256 {digest} does not match {manifest_path.name}", file=sys.stderr
                )
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
