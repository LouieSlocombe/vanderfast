"""The benchmark result schema helpers."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_timing_summary_and_round_trip(tmp_path: Path) -> None:
    schema = _load("schema")
    timing = schema.Timing.of([3.0, 1.0, 2.0])
    assert timing.median == 2.0
    assert timing.min == 1.0
    assert schema.Timing.of([5.0]).p90 == 5.0
    document = schema.envelope({"vanderfast": "test"}, {"sha256": "x"})
    case = schema.Case(
        case_id="a/N=1/pair",
        family="f",
        structure={},
        parameters={},
        cutoffs_bohr={},
        skin_angstrom=1.0,
        precision="float64",
    )
    path = tmp_path / "result.json"
    schema.write(path, document, [case])
    loaded = json.loads(path.read_text())
    assert schema.validate(loaded) == []
    assert schema.validate({"schema": "other", "cases": [{}]})


def test_workloads_are_deterministic() -> None:
    workloads = _load("workloads")
    for workload in workloads.WORKLOADS.values():
        a = workload.build(64, 1)
        b = workload.build(64, 1)
        assert (a.positions == b.positions).all()
        assert len(a) > 0


def test_report_renders_every_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _load("report")
    wall = {"median": 2.0, "p10": 1.5, "p90": 250.0}
    document = {
        "cases": [
            {
                "case_id": "w/N=3/atm",
                "status": "ok",
                "counts": {"pair_edges": 6, "atm_edges": 6, "atm_accepted": 1},
                "errors": {"reference_wall_ms": 20.0, "force_max_ev_per_ang": 1e-15},
                "modes": {
                    "warm_native": {"wall_ms": wall, "gpu_stage_ms": {"pair": 0.5, "atm": 1.0}},
                    "warm_ase_ef": {"wall_ms": wall},
                    "cold": {"import_ms": 100.0, "context_ms": 50.0, "first_eval_ms": 5.0},
                    "memory": {"device_bytes": 2**20},
                },
            },
            {
                "case_id": "w/N=3/pair",
                "status": "ok",
                "modes": {"cold": {"status": "error: OSError"}, "warm_native": {}},
            },
            {"case_id": "w/N=9/pair", "status": "error: MemoryError", "modes": {}},
        ]
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document))
    assert report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "| w/N=3/atm | 6 / 6 | 1 | 2 (1.5-250) | 2 | n/a | n/a | 20 | 10x | 1.0e-15 |" in out
    assert "| w/N=3/atm | n/a | n/a | n/a | 0.5 | 1 | n/a | n/a | 1.5 | 2 |" in out
    assert "| w/N=3/atm | 100 | 50 | 5 | 1.0 |" in out
    assert "- w/N=9/pair: error: MemoryError" in out
