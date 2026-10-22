"""Command-line interface: ``vanderfast info|methods|verify-data``."""

import argparse
import json
from collections.abc import Sequence

from vanderfast import __version__
from vanderfast.data import load_methods, load_reference_tables
from vanderfast.data.d4 import load_d4_methods, load_d4_reference_tables
from vanderfast.diagnostics import backend_info
from vanderfast.parameters import available_methods
from vanderfast.parameters_d4 import available_d4_methods


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="vanderfast", description="GPU DFT-D3(BJ) and DFT-D4 utilities."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="report build, backend, and data provenance")
    info.add_argument("--json", action="store_true", help="print machine-readable JSON")
    info.add_argument("--cuda", action="store_true", help="also query CUDA devices")
    methods = commands.add_parser("methods", help="list methods with D3(BJ) or D4 parameters")
    methods.add_argument("--model", choices=("d3", "d4"), default="d3", help="parameter table")
    commands.add_parser("verify-data", help="validate the bundled reference tables")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "info":
        report = backend_info(initialize_cuda=arguments.cuda).to_dict()
        if arguments.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for key, value in report.items():
                print(f"{key}: {value}")
        return 0
    if arguments.command == "methods":
        if arguments.model == "d3":
            table = load_methods()["methods"]
            for name in available_methods():
                entry = table[name]
                values = " ".join(f"{key}={entry[key]:<8g}" for key in ("s6", "s8", "a1", "a2"))
                print(f"{name:24s} {values}")
            return 0
        tables_d4 = load_d4_methods()["tables"]
        for name in available_d4_methods():
            entry = tables_d4["bjatm"].get(name) or tables_d4["bj"][name]
            values = " ".join(f"{key}={entry[key]:<10g}" for key in ("s6", "s8", "a1", "a2"))
            print(f"{name:24s} {values}")
        return 0
    tables = load_reference_tables()
    print(f"reference data OK: {tables.source_id} sha256={tables.sha256}")
    d4 = load_d4_reference_tables()
    print(f"D4 reference data OK: {d4.source_id} sha256={d4.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
