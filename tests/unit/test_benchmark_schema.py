"""The benchmark result schema helpers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
    import json  # noqa: PLC0415

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
