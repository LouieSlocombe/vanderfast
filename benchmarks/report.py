"""Render Markdown tables from a benchmark result file (for docs/performance.md)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STAGES = ("upload", "neighbors", "cn_weights", "pair", "atm", "cn_reverse", "reduce")


def _median(case: dict[str, Any], mode: str) -> float | None:
    wall = case.get("modes", {}).get(mode, {}).get("wall_ms")
    return None if not wall else float(wall["median"])


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}" if value < 100 else f"{value:.0f}"  # noqa: PLR2004


def _speed(cases: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| case | edges (pair / ATM) | ATM triangles | warm native ms (p10-p90) "
            "| ASE E+F ms | ASE E+F+S ms | rebuild ms | dftd3 ms | speedup | max ΔF eV/Å |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        counts = case.get("counts", {})
        wall = case.get("modes", {}).get("warm_native", {}).get("wall_ms")
        native = None if not wall else float(wall["median"])
        spread = "" if not wall else f" ({_ms(wall['p10'])}-{_ms(wall['p90'])})"
        errors = case.get("errors") or {}
        reference = errors.get("reference_wall_ms")
        speedup = "n/a" if reference is None or not native else f"{reference / native:.0f}x"
        triangles = counts.get("atm_accepted", 0) if "atm" in case["case_id"] else 0
        force = errors.get("force_max_ev_per_ang")
        edges = f"{counts.get('pair_edges', 0):,} / {counts.get('atm_edges', 0):,}"
        lines.append(
            f"| {case['case_id']} | {edges} | {triangles:,} | {_ms(native)}{spread}"
            f" | {_ms(_median(case, 'warm_ase_ef'))} | {_ms(_median(case, 'warm_ase_efs'))}"
            f" | {_ms(_median(case, 'rebuild'))} | {_ms(reference)} | {speedup}"
            f" | {'n/a' if force is None else f'{force:.1e}'} |"
        )
    return lines


def _stages(cases: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| case | " + " | ".join(STAGES) + " | GPU total | wall |",
        "| --- |" + " ---: |" * (len(STAGES) + 2),
    ]
    for case in cases:
        native = case.get("modes", {}).get("warm_native", {})
        stages = native.get("gpu_stage_ms")
        if not stages:
            continue
        cells = [_ms(stages.get(s)) for s in STAGES]
        total = sum(float(v) for v in stages.values())
        lines.append(
            f"| {case['case_id']} | " + " | ".join(cells) + f" | {_ms(total)}"
            f" | {_ms(_median(case, 'warm_native'))} |"
        )
    return lines


def _cold(cases: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| case | import ms | context ms | first evaluation ms | device MB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        modes = case.get("modes", {})
        cold = modes.get("cold")
        if not cold or "status" in cold:
            continue
        device = modes.get("memory", {}).get("device_bytes")
        lines.append(
            f"| {case['case_id']} | {_ms(cold['import_ms'])} | {_ms(cold['context_ms'])}"
            f" | {_ms(cold['first_eval_ms'])}"
            f" | {'n/a' if device is None else f'{device / 2**20:.1f}'} |"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Print the speed, stage, and cold-start tables of one result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.result.read_text(encoding="utf-8"))
    cases = [c for c in document["cases"] if c["status"] == "ok"]
    failed = [f"{c['case_id']}: {c['status']}" for c in document["cases"] if c["status"] != "ok"]
    for title, table in (
        ("Warm evaluation", _speed(cases)),
        ("GPU stages (warm native, ms)", _stages(cases)),
        ("Cold start", _cold(cases)),
    ):
        print(f"### {title}\n")
        print("\n".join(table))
        print()
    if failed:
        print("### Not completed\n")
        print("\n".join(f"- {line}" for line in failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
