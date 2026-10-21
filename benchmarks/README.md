# Benchmarks

`run.py` measures vanderfast on deterministic workloads (`workloads.py`) and
writes a `vanderfast.benchmark/1` JSON file (`schema.py`) into `results/`
(ignored by git; curated runs are summarised in `docs/performance.md`).

```bash
python benchmarks/run.py --suite smoke
python benchmarks/run.py --suite standard --atm both
python benchmarks/compare.py results/before.json results/after.json
python benchmarks/report.py results/run.json   # the Markdown tables in docs/performance.md
```

Every case records the hardware, driver, software and build information,
reference-data hash, all damping parameters and cutoffs, skin, precision, edge
and triangle counts, the timing modes, errors against dftd3 (up to
`--reference-max-atoms`), and a status. Failures, out-of-memory conditions,
and timeouts are recorded as a status rather than dropped.

| Mode | What is timed |
| --- | --- |
| `cold` | fresh process: import, context creation (CUDA initialisation and table upload), first evaluation |
| `warm-native` | `D3Context.evaluate` with energy and forces on perturbed positions (no rebuild), plus CUDA-event stage times |
| `warm-ase-ef`, `warm-ase-efs` | the ASE calculator on perturbed positions; the harness checks that no call hit the result cache |
| `rebuild` | an evaluation with a forced neighbour rebuild |
| `trajectory` | a random walk including the rebuilds it triggers |
| `memory` | device bytes, allocation counts, edge capacity |

Pair-only and pair+ATM are measured separately with parameter sets that differ
only in `s9`.
