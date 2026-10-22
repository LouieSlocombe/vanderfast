"""Render Markdown tables from a benchmark result file (for docs/performance.md)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STAGES = ("upload", "neighbors", "cn_weights", "pair", "atm", "cn_reverse", "reduce")
D4_STAGES = (
    "upload",
    "neighbors",
    "cn",
    "eeq_setup",
    "eeq_solve",
    "weights",
    "pair",
    "eeq_adjoint",
    "atm",
    "eeq_reverse",
    "cn_reverse",
    "reduce",
)


def _is_d4(case: dict[str, Any]) -> bool:
    return "/d4" in case["case_id"]


def _median(case: dict[str, Any], mode: str) -> float | None:
    wall = case.get("modes", {}).get(mode, {}).get("wall_ms")
    return None if not wall else float(wall["median"])


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}" if value < 100 else f"{value:.0f}"  # noqa: PLR2004


def _speed(cases: list[dict[str, Any]]) -> list[str]:
    reference = "dftd4" if cases and all(_is_d4(c) for c in cases) else "dftd3"
    lines = [
        (
            "| case | edges (pair / ATM) | ATM triangles | warm native ms (p10-p90) "
            f"| ASE E+F ms | ASE E+F+S ms | rebuild ms | {reference} ms | speedup | max ΔF eV/Å |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        counts = case.get("counts", {})
        wall = case.get("modes", {}).get("warm_native", {}).get("wall_ms")
        native = None if not wall else float(wall["median"])
        spread = "" if not wall else f" ({_ms(wall['p10'])}-{_ms(wall['p90'])})"
        errors = case.get("errors") or {}
        reference_ms = errors.get("reference_wall_ms")
        speedup = "n/a" if reference_ms is None or not native else f"{reference_ms / native:.0f}x"
        triangles = counts.get("atm_accepted", 0) if "atm" in case["case_id"] else 0
        force = errors.get("force_max_ev_per_ang")
        edges = f"{counts.get('pair_edges', 0):,} / {counts.get('atm_edges', 0):,}"
        lines.append(
            f"| {case['case_id']} | {edges} | {triangles:,} | {_ms(native)}{spread}"
            f" | {_ms(_median(case, 'warm_ase_ef'))} | {_ms(_median(case, 'warm_ase_efs'))}"
            f" | {_ms(_median(case, 'rebuild'))} | {_ms(reference_ms)} | {speedup}"
            f" | {'n/a' if force is None else f'{force:.1e}'} |"
        )
    return lines


def _stages(cases: list[dict[str, Any]], stage_names: tuple[str, ...] = STAGES) -> list[str]:
    lines = [
        "| case | " + " | ".join(stage_names) + " | GPU total | wall |",
        "| --- |" + " ---: |" * (len(stage_names) + 2),
    ]
    for case in cases:
        native = case.get("modes", {}).get("warm_native", {})
        stages = native.get("gpu_stage_ms")
        if not stages:
            continue
        cells = [_ms(stages.get(s)) for s in stage_names]
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


def _override_reference(document: dict[str, Any], other: dict[str, Any]) -> None:
    timings = {
        c["case_id"]: c["errors"]
        for c in other["cases"]
        if (c.get("errors") or {}).get("reference_wall_ms") is not None
    }
    for case in document["cases"]:
        errors = case.get("errors")
        source = timings.get(case["case_id"])
        if errors is not None and source is not None:
            errors["reference_wall_ms"] = source["reference_wall_ms"]
            errors["vs"] = source["vs"]


def main(argv: Sequence[str] | None = None) -> int:
    """Print the speed, stage, and cold-start tables of one result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--reference-timings",
        type=Path,
        help="take the dftd3 wall times from this result file (same case ids)",
    )
    args = parser.parse_args(argv)
    document = json.loads(args.result.read_text(encoding="utf-8"))
    if args.reference_timings:
        _override_reference(document, json.loads(args.reference_timings.read_text("utf-8")))
    cases = [c for c in document["cases"] if c["status"] == "ok"]
    failed = [f"{c['case_id']}: {c['status']}" for c in document["cases"] if c["status"] != "ok"]
    d3 = [c for c in cases if not _is_d4(c)]
    d4 = [c for c in cases if _is_d4(c)]
    tables = [("Warm evaluation", _speed(cases))]
    if d3:
        tables.append(("GPU stages (warm native, ms)", _stages(d3)))
    if d4:
        tables.append(("D4 GPU stages (warm native, ms)", _stages(d4, D4_STAGES)))
    tables.append(("Cold start", _cold(cases)))
    for title, table in tables:
        print(f"### {title}\n")
        print("\n".join(table))
        print()
    if failed:
        print("### Not completed\n")
        print("\n".join(f"- {line}" for line in failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
