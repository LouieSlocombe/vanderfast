"""Run vanderfast benchmarks and write a ``vanderfast.benchmark/1`` result file.

Examples::

    python benchmarks/run.py --suite smoke
    python benchmarks/run.py --workloads mgo-rocksalt --sizes 64,512,4096 --atm both
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import Case, Timing, envelope, write
from workloads import GENERATOR_VERSION, WORKLOADS, perturbed

from vanderfast import resolve_parameters
from vanderfast.ase import D3Calculator
from vanderfast.context import D3Context
from vanderfast.diagnostics import backend_info
from vanderfast.parameters import Cutoffs, D3BJParameters


@dataclass(frozen=True)
class Suite:
    """A named set of workloads, sizes, and repeat counts."""

    workloads: tuple[str, ...]
    sizes: tuple[int, ...]
    repeats: int


ALL_WORKLOADS = ("benzene-cluster", "water-box", "mgo-rocksalt", "skewed-argon", "small-cell")
SUITES = {
    "smoke": Suite(("water-box", "small-cell"), (64,), 3),
    "standard": Suite(ALL_WORKLOADS, (16, 64, 256, 1024, 4096), 10),
    "full": Suite(ALL_WORKLOADS, (16, 64, 256, 1024, 4096, 16384), 20),
}
MODES = ("cold", "warm-native", "warm-ase-ef", "warm-ase-efs", "rebuild", "trajectory", "memory")

COLD_SCRIPT = """
import json, sys, time
t0 = time.perf_counter()
import numpy as np
import vanderfast
from vanderfast.context import D3Context
from vanderfast.parameters import D3BJParameters
t1 = time.perf_counter()
args = json.load(sys.stdin)
ctx = D3Context(np.array(args["numbers"]), D3BJParameters(**args["parameters"]), skin=args["skin"])
t2 = time.perf_counter()
ctx.evaluate(np.array(args["positions"]), cell=np.array(args["cell"]), pbc=args["pbc"],
             properties=("energy", "forces"))
t3 = time.perf_counter()
print(json.dumps({"import_ms": 1e3 * (t1 - t0), "context_ms": 1e3 * (t2 - t1),
                  "first_eval_ms": 1e3 * (t3 - t2)}))
"""


def _cell(atoms: Atoms) -> np.ndarray | None:
    return atoms.cell[:] if atoms.pbc.any() else None


def _structure(atoms: Atoms, workload: str, seed: int) -> dict[str, Any]:
    symbols = atoms.get_chemical_symbols()
    return {
        "n_atoms": len(atoms),
        "species": {s: symbols.count(s) for s in sorted(set(symbols))},
        "pbc": [bool(p) for p in atoms.pbc],
        "cell_angstrom": atoms.cell[:].tolist(),
        "generator": {"name": workload, "version": GENERATOR_VERSION, "seed": seed},
    }


def _time(call: Callable[[int], object], repeats: int, warmup: int) -> list[float]:
    samples = []
    for index in range(warmup + repeats):
        start = time.perf_counter()
        call(index)
        elapsed = 1e3 * (time.perf_counter() - start)
        if index >= warmup:
            samples.append(elapsed)
    return samples


@dataclass
class Run:
    """Shared state of one benchmark case."""

    atoms: Atoms
    parameters: D3BJParameters
    ctx: D3Context
    repeats: int
    warmup: int
    skin: float
    seed: int

    @property
    def cell(self) -> np.ndarray | None:
        """The cell rows in Å, or None for molecules."""
        return _cell(self.atoms)

    def moved(self, index: int) -> np.ndarray:
        """Positions perturbed by up to a twentieth of the skin (no rebuild)."""
        return perturbed(self.atoms.positions, self.skin / 20, self.seed + index)


def mode_warm_native(run: Run) -> dict[str, Any]:
    """Time D3Context energy and forces on perturbed positions without rebuilds."""
    stages: list[dict[str, float]] = []

    def call(index: int) -> None:
        result = run.ctx.evaluate(
            run.moved(index), cell=run.cell, pbc=run.atoms.pbc, properties=("forces",)
        )
        stages.append(result.info["stage_ms"])

    samples = _time(call, run.repeats, run.warmup)
    timed = stages[run.warmup :]
    return {
        "properties": ["energy", "forces"],
        "includes": {"host_transfer": True, "rebuild": False, "python": "D3Context"},
        "wall_ms": Timing.of(samples).__dict__,
        "gpu_stage_ms": {k: float(np.median([s.get(k, 0.0) for s in timed])) for k in timed[-1]},
    }


def mode_ase(run: Run, stress: bool) -> dict[str, Any]:
    """Time the ASE calculator; fail if any call was served from the result cache."""
    calc = D3Calculator(parameters=run.parameters, skin=run.skin)
    moving = run.atoms.copy()
    moving.calc = calc

    def call(index: int) -> None:
        moving.positions = run.moved(index)
        moving.get_forces()
        if stress:
            moving.get_stress()

    before = calc.evaluations
    samples = _time(call, run.repeats, run.warmup)
    if calc.evaluations - before < run.repeats + run.warmup:
        msg = "ASE benchmark timed cached results"
        raise RuntimeError(msg)
    return {
        "properties": ["energy", "forces"] + (["stress"] if stress else []),
        "includes": {"host_transfer": True, "python": "ASE"},
        "wall_ms": Timing.of(samples).__dict__,
    }


def mode_rebuild(run: Run) -> dict[str, Any]:
    """Time evaluations that rebuild the neighbour graph."""

    def call(index: int) -> None:
        del index
        run.ctx.invalidate_neighbors()
        run.ctx.evaluate(
            run.atoms.positions, cell=run.cell, pbc=run.atoms.pbc, properties=("forces",)
        )

    return {"wall_ms": Timing.of(_time(call, run.repeats, run.warmup)).__dict__}


def mode_trajectory(run: Run) -> dict[str, Any]:
    """Time a random walk including the rebuilds it triggers."""
    rng = np.random.default_rng(run.seed)
    x = run.atoms.positions.copy()
    rebuilds = 0
    samples = []
    for _ in range(run.repeats * 5):
        x = x + rng.normal(0.0, 0.05, x.shape)
        start = time.perf_counter()
        result = run.ctx.evaluate(x, cell=run.cell, pbc=run.atoms.pbc, properties=("forces",))
        samples.append(1e3 * (time.perf_counter() - start))
        rebuilds += int(result.info["rebuilt"])
    return {"wall_ms": Timing.of(samples).__dict__, "rebuilds": rebuilds}


def mode_cold(run: Run) -> dict[str, Any]:
    """Time import, context creation, and the first evaluation in a fresh process."""
    payload = {
        "numbers": run.atoms.numbers.tolist(),
        "parameters": {k: v for k, v in run.parameters.as_dict().items() if k != "source"},
        "skin": run.skin,
        "positions": run.atoms.positions.tolist(),
        "cell": run.atoms.cell[:].tolist(),
        "pbc": [bool(p) for p in run.atoms.pbc],
    }
    command = [sys.executable, "-c", COLD_SCRIPT]
    output = subprocess.run(
        command, input=json.dumps(payload), capture_output=True, text=True, check=True
    ).stdout
    result: dict[str, Any] = json.loads(output)
    return result


def _run_mode(handler: Callable[[Run], dict[str, Any]], run: Run) -> dict[str, Any]:
    """Run one mode, recording a failure in place of its timings."""
    try:
        return handler(run)
    except Exception as error:  # noqa: BLE001
        return {"status": f"error: {type(error).__name__}: {error}"}


def run_case(
    workload: str,
    size: int,
    parameters: D3BJParameters,
    *,
    modes: Sequence[str],
    repeats: int,
    warmup: int,
    skin: float,
    seed: int = 1234,
) -> tuple[Case, Atoms]:
    """Benchmark one workload, size, and parameter set; return the case and structure."""
    atoms = WORKLOADS[workload].build(size, seed)
    label = "atm" if parameters.atm else "pair"
    case = Case(
        case_id=f"{workload}/N={len(atoms)}/{label}",
        family=WORKLOADS[workload].family,
        structure=_structure(atoms, workload, seed),
        parameters=parameters.as_dict(),
        cutoffs_bohr=Cutoffs().as_bohr(),
        skin_angstrom=skin,
        precision="float64",
    )
    ctx: D3Context | None = None
    try:
        ctx = D3Context(atoms.numbers, parameters, skin=skin, profile=True)
        run = Run(atoms, parameters, ctx, repeats, warmup, skin, seed)
        first = ctx.evaluate(atoms.positions, cell=run.cell, pbc=atoms.pbc, properties=("forces",))
        keys = ("cn_edges", "pair_edges", "atm_edges", "atm_candidates", "atm_accepted")
        case.counts = {k: int(first.info[k]) for k in keys}
        handlers: dict[str, Callable[[Run], dict[str, Any]]] = {
            "warm-native": mode_warm_native,
            "warm-ase-ef": lambda r: mode_ase(r, stress=False),
            "rebuild": mode_rebuild,
            "trajectory": mode_trajectory,
            "memory": lambda r: dict(r.ctx.diagnostics()["stats"]),
            "cold": mode_cold,
        }
        if atoms.pbc.all():
            handlers["warm-ase-efs"] = lambda r: mode_ase(r, stress=True)
        for mode in modes:
            if mode in handlers:
                case.modes[mode.replace("-", "_")] = _run_mode(handlers[mode], run)
        failed = [name for name, value in case.modes.items() if "status" in value]
        if failed:
            case.status = f"partial: {', '.join(failed)} failed"
    except Exception as error:  # noqa: BLE001
        case.status = f"error: {type(error).__name__}: {error}"
    finally:
        if ctx is not None:
            ctx.close()
    return case, atoms


def reference_errors(atoms: Atoms, parameters: D3BJParameters) -> dict[str, Any]:
    """Time dftd3 on the CPU and compare energies, forces, and stress.

    Called only after every case has been timed, so that dftd3's OpenMP
    worker threads never compete with timed vanderfast calls for the CPU.
    """
    from vanderfast.reference import ReferenceEvaluator, reference_version  # noqa: PLC0415
    from vanderfast.units import (  # noqa: PLC0415
        HARTREE,
        gradient_to_forces,
        length_to_bohr,
        strain_derivative_to_stress,
    )

    cell = _cell(atoms)
    evaluator = ReferenceEvaluator(atoms.numbers, parameters)
    start = time.perf_counter()
    reference = evaluator.evaluate(
        length_to_bohr(atoms.positions), None if cell is None else length_to_bohr(cell), atoms.pbc
    )
    reference_ms = 1e3 * (time.perf_counter() - start)
    properties = ("forces", "stress") if atoms.pbc.all() else ("forces",)
    with D3Context(atoms.numbers, parameters) as ctx:
        ours = ctx.evaluate(atoms.positions, cell=cell, pbc=atoms.pbc, properties=properties)
    if reference.gradient is None or ours.forces is None:
        msg = "missing gradients"
        raise RuntimeError(msg)
    forces = gradient_to_forces(reference.gradient)
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    result: dict[str, Any] = {
        "vs": f"dftd3 {reference_version()} (OMP_NUM_THREADS={threads})",
        "reference_wall_ms": reference_ms,
        "energy_abs_ev": abs(ours.energy - reference.energy * HARTREE),
        "energy_per_atom_ev": abs(ours.energy - reference.energy * HARTREE) / len(atoms),
        "force_max_ev_per_ang": float(np.abs(ours.forces - forces).max()) if len(atoms) else 0.0,
        "force_rms_ev_per_ang": float(np.sqrt(np.mean((ours.forces - forces) ** 2)))
        if len(atoms)
        else 0.0,
    }
    if ours.stress is not None and reference.virial is not None:
        stress = strain_derivative_to_stress(reference.virial, atoms.get_volume())
        result["stress_max_ev_per_ang3"] = float(np.abs(ours.stress - stress).max())
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument("--workloads", help="comma-separated workload names")
    parser.add_argument("--sizes", help="comma-separated atom-count targets")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--atm", choices=("off", "on", "both"), default="both")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skin", type=float, default=1.0, help="Å")
    parser.add_argument("--method", default="pbe")
    parser.add_argument("--reference-max-atoms", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    suite = SUITES[args.suite]
    workloads = args.workloads.split(",") if args.workloads else list(suite.workloads)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(suite.sizes)
    repeats = args.repeats or suite.repeats
    modes = args.modes.split(",")
    atm = {"off": [False], "on": [True], "both": [False, True]}[args.atm]
    info = backend_info(initialize_cuda=True).to_dict()
    document = envelope(
        {
            "vanderfast": version("vanderfast"),
            "numpy": np.__version__,
            "build_info": info["native"],
            "python": sys.version.split()[0],
        },
        info["reference_data"],
    )
    cases: list[Case] = []
    pending: list[tuple[Case, Atoms, D3BJParameters]] = []
    for workload in workloads:
        for size in sizes[:1] if WORKLOADS[workload].fixed_size else sizes:
            for flag in atm:
                parameters = resolve_parameters(method=args.method, damping="bj", atm=flag)
                case, atoms = run_case(
                    workload, size, parameters, modes=modes, repeats=repeats, warmup=args.warmup,
                    skin=args.skin,
                )  # fmt: skip
                cases.append(case)
                if len(atoms) <= args.reference_max_atoms and not case.status.startswith("error"):
                    pending.append((case, atoms, parameters))
                native = case.modes.get("warm_native", {}).get("wall_ms", {}).get("median")
                timing = "n/a" if native is None else f"{native:.2f} ms"
                print(f"{case.case_id:40s} {case.status:6s} warm-native {timing}", flush=True)
    for case, atoms, parameters in pending:
        try:
            case.errors = reference_errors(atoms, parameters)
        except Exception as error:  # noqa: BLE001
            case.errors = {"status": f"error: {type(error).__name__}: {error}"}
        reference = case.errors.get("reference_wall_ms")
        timing = "n/a" if reference is None else f"{reference:.2f} ms"
        print(f"{case.case_id:40s} dftd3 {timing}", flush=True)
    output = (
        args.output or Path(__file__).parent / "results" / f"{args.suite}-{int(time.time())}.json"
    )
    write(output, document, cases)
    print(f"wrote {output}")
    return 0 if all(case.status == "ok" for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
