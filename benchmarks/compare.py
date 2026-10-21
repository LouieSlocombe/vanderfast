"""Compare warm-native medians between two benchmark result files."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _medians(path: Path) -> dict[str, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for case in document["cases"]:
        wall = case.get("modes", {}).get("warm_native", {}).get("wall_ms")
        if wall:
            result[case["case_id"]] = float(wall["median"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Print a Markdown table of baseline and candidate medians."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    base, cand = _medians(args.baseline), _medians(args.candidate)
    print("| case | baseline ms | candidate ms | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for case_id in sorted(base.keys() & cand.keys()):
        b, c = base[case_id], cand[case_id]
        print(f"| {case_id} | {b:.3f} | {c:.3f} | {b / c:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
