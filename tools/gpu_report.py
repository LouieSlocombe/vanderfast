"""Record, or verify, a GPU test run of exact release artifacts.

``record`` hashes every wheel in ``--dist``, reads the pytest JUnit XML from
the run that exercised them, and writes a JSON report with the devices and
build the run used. ``verify`` checks a report against a ``--dist`` directory
before publishing: the same wheels, the same hashes, a GPU in the report, and
no failed or errored tests. Run pytest with ``--require-cuda`` so that a
missing GPU is an error rather than a skip. (The sdist is checked by
``check_sdist.py`` in its own job; it is not what the GPU run installed.)

``verify`` runs on the publishing runner's system Python, so this file sticks
to the standard library and to syntax older Pythons accept.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCHEMA = "vanderfast.gpu-report/1"


def _artifacts(dist: Path) -> dict[str, str]:
    files = sorted(dist.glob("*.whl"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _junit(path: Path) -> dict[str, int]:
    # The file is pytest's own output from the same job, not untrusted input.
    root = ET.parse(path).getroot()  # noqa: S314
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    keys = ("tests", "failures", "errors", "skipped")
    return {k: sum(int(s.get(k, "0")) for s in suites) for k in keys}


def _commit() -> str | None:
    command = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    except OSError:
        result = None
    if result is None or result.returncode != 0:
        return os.environ.get("GITHUB_SHA")
    return result.stdout.strip()


def record(dist: Path, junit: Path, output: Path) -> int:
    """Write the report; return non-zero when the run did not pass."""
    from vanderfast import backend_info  # noqa: PLC0415

    info = backend_info(initialize_cuda=True).to_dict()
    tests = _junit(junit)
    passed = tests["tests"] > 0 and tests["failures"] == 0 and tests["errors"] == 0
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "commit": _commit(),
        "artifacts": _artifacts(dist),
        "tests": tests,
        "passed": passed and bool(info["devices"]),
        "backend": info,
        "python": sys.version.split()[0],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}: {tests} on {[d['name'] for d in info['devices'] or []]}")
    return 0 if report["passed"] else 1


def verify(dist: Path, report_path: Path) -> list[str]:
    """Return the reasons ``report_path`` does not cover the files in ``dist``."""
    report = json.loads(report_path.read_text())
    problems: list[str] = []
    if report.get("schema") != SCHEMA:
        problems.append(f"unknown report schema {report.get('schema')!r}")
    if not report.get("passed"):
        problems.append("the GPU run did not pass")
    if not report.get("backend", {}).get("devices"):
        problems.append("the report lists no CUDA device")
    expected = report.get("artifacts", {})
    actual = _artifacts(dist)
    if not actual:
        problems.append(f"no wheels in {dist}")
    problems += [f"{name} was not tested on a GPU" for name in sorted(actual.keys() - expected)]
    problems += [f"{name} is missing from {dist}" for name in sorted(expected.keys() - actual)]
    problems += [
        f"{name} differs from the tested file"
        for name in sorted(actual.keys() & expected)
        if actual[name] != expected[name]
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record", help="write a report for a finished GPU run")
    recorder.add_argument("--dist", type=Path, required=True)
    recorder.add_argument("--junit", type=Path, required=True)
    recorder.add_argument("--output", type=Path, required=True)
    verifier = commands.add_parser("verify", help="check a report against distributions")
    verifier.add_argument("--dist", type=Path, required=True)
    verifier.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "record":
        return record(args.dist, args.junit, args.output)
    problems = verify(args.dist, args.report)
    for problem in problems:
        print(f"{args.report}: {problem}", file=sys.stderr)
    if not problems:
        print(f"{args.report}: every wheel in {args.dist} passed on a GPU")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
