"""Benchmark result schema ``vanderfast.benchmark/1``."""

import json
import os
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "vanderfast.benchmark/1"


@dataclass
class Timing:
    """Summary of repeated wall-clock samples in milliseconds."""

    median: float
    p10: float
    p90: float
    min: float
    max: float
    samples: list[float]

    @classmethod
    def of(cls, samples: list[float]) -> Timing:
        """Summarise samples (at least one)."""
        ordered = sorted(samples)
        if len(ordered) > 1:
            deciles = statistics.quantiles(ordered, n=10, method="inclusive")
            p10, p90 = deciles[0], deciles[-1]
        else:
            p10 = p90 = ordered[0]
        return cls(statistics.median(ordered), p10, p90, ordered[0], ordered[-1], samples)


@dataclass
class Case:
    """One workload at one size and parameter setting."""

    case_id: str
    family: str
    structure: dict[str, Any]
    parameters: dict[str, Any]
    cutoffs_bohr: dict[str, float]
    skin_angstrom: float
    precision: str
    counts: dict[str, int] = field(default_factory=dict)
    modes: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"


def _command_output(command: list[str]) -> str:
    """Return a command's stdout, or an empty string if it is absent or fails."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:  # the executable is not installed (no git, or no NVIDIA driver)
        return ""
    return completed.stdout.strip()


def _git() -> dict[str, Any]:
    def run(*args: str) -> str:
        return _command_output(["git", *args])

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _gpu() -> dict[str, Any]:
    query = "name,uuid,compute_cap,memory.total,driver_version,clocks.sm,clocks.mem,power.limit"
    output = _command_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )
    if not output:
        return {}
    values = [v.strip() for v in output.splitlines()[0].split(",")]
    return dict(zip(query.split(","), values, strict=False))


def _cpu() -> dict[str, Any]:
    model = platform.processor() or platform.machine()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return {"model": model, "logical_cpus": os.cpu_count()}


def envelope(software: dict[str, Any], reference_data: dict[str, Any]) -> dict[str, Any]:
    """Return the run-level metadata of a result file."""
    return {
        "schema": SCHEMA,
        "started_utc": datetime.now(UTC).isoformat(),
        "git": _git(),
        "software": software,
        "hardware": {
            "gpu": _gpu(),
            "cpu": _cpu(),
            "os": platform.platform(),
        },
        "reference_data": reference_data,
        "cases": [],
    }


def write(path: Path, document: dict[str, Any], cases: list[Case]) -> None:
    """Write a result document with its cases."""
    document["cases"] = [asdict(case) for case in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def validate(document: dict[str, Any]) -> list[str]:
    """Return schema problems (an empty list means valid)."""
    problems = [] if document.get("schema") == SCHEMA else ["wrong schema identifier"]
    problems += [
        f"missing {key}"
        for key in ("git", "software", "hardware", "reference_data", "cases")
        if key not in document
    ]
    problems += [
        f"case {case.get('case_id')}: missing {key}"
        for case in document.get("cases", [])
        for key in ("case_id", "parameters", "cutoffs_bohr", "counts", "modes", "status")
        if key not in case
    ]
    return problems
