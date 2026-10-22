# Python API

## Parameters

```python
from vanderfast import Cutoffs, D3BJParameters, resolve_parameters

params = resolve_parameters(method="pbe", damping="bj", atm=True)
custom = D3BJParameters(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0, alp=14.0)
cutoffs = Cutoffs.resolve({"disp2": 25.0})  # Å; omitted keys keep the reference defaults
```

- `resolve_parameters` accepts either `method`, `damping="bj"` and an explicit
  boolean `atm`, or a complete `D3BJParameters`; anything else raises
  `ParameterError`. `atm=True` sets `s9 = 1`; the pair coefficients do not
  change.
- `D3BJParameters` stores `a2` in bohr and the public ATM exponent `alp`; the
  BJ ATM kernel uses `alp + 2`.
- `Cutoffs` stores bohr. The defaults are 40 (CN), 60 (pair), and 40 (ATM) bohr.

## D3Context

```python
from vanderfast import D3Context

with D3Context(numbers, params, device="cuda:0", skin=1.0) as context:
    result = context.evaluate(
        positions, cell=cell, pbc=True, properties=("energy", "forces", "stress")
    )
```

| Argument | Meaning |
| --- | --- |
| `numbers` | atomic numbers, fixed for the context |
| `parameters` | a resolved `D3BJParameters` |
| `device` | `"cuda"`, `"cuda:N"`, or an index |
| `precision` | only `"float64"` |
| `cutoff` | `Cutoffs` (bohr) or a mapping in Å |
| `skin` | neighbour-list skin in Å; lists are reused while every atom stays within half the skin |
| `builder` | `"auto"`, `"brute_force"`, or `"cell_bins"` |
| `reference_data` | path to replacement tables in the bundled format |
| `profile` | record CUDA-event stage timings in `result.info["stage_ms"]` |

`evaluate` takes positions and cell rows in Å and returns a `D3Result` in eV
and Å: `energy`, `energy_pair`, `energy_atm`, `forces`, `strain_derivative`
(eV, any periodicity), `stress` (eV/Å³ in ASE order, 3D cells only), and
`info` (edge and triangle counts, whether the neighbour list was rebuilt, the
builder). `evaluate_atomic` does the same in bohr and hartree.

`set_parameters`, `set_cutoffs`, `invalidate_neighbors`, `diagnostics`, and
`close` manage the context. One context runs one evaluation at a time;
concurrent use raises `ContextBusyError`. Separate contexts are independent.

## DFT-D4

```python
from vanderfast import D4Context, D4Cutoffs, D4Parameters, EEQSolverOptions, resolve_d4_parameters

params = resolve_d4_parameters(method="r2scan")  # ATM on by default, as in dftd4
pair_only = resolve_d4_parameters(method="pbe", atm=False)
custom = D4Parameters(
    s8=0.95948085, a1=0.38574991, a2=4.80688534
)  # s6=1, s9=1, alp=16, ga=3, gc=2, wf=6

with D4Context(numbers, params, charge=-1.0, ewald="upstream") as context:
    result = context.evaluate(
        positions, cell=cell, pbc=True, properties=("energy", "forces", "stress")
    )
    charges = result.charges  # EEQ partial charges in e
```

- `resolve_d4_parameters` takes `method` (dftd4 v4.2.0 names and aliases; any
  `"/basis"` suffix is ignored) and `atm` (default `True`), or a complete
  `D4Parameters`. `available_d4_methods()` lists the names.
- `D4Parameters` stores `a2` in bohr; `alp` is used as is, `(R0/r)^(alp/3)`.
  `ga`, `gc`, and `wf` are honoured: the reference polarisabilities and C6 are
  recomputed when they change.
- `D4Cutoffs` (bohr) defaults to 30 (CN), 60 (pair), and 40 (ATM); the EEQ
  coordination number always uses 25 bohr.

`D4Context` takes the `D3Context` arguments (`numbers`, `parameters`,
`device`, `precision`, `cutoff`, `skin`, `builder`, `reference_data`,
`profile`) plus:

| Argument | Meaning |
| --- | --- |
| `charge` | total charge in e, distributed by the EEQ model |
| `ewald` | periodic charges: `"upstream"` (dftd4's truncated Ewald sums, default) or `"converged"` (complete sums, independent of the cell choice) |
| `solver` | `EEQSolverOptions(method="auto" \| "iterative" \| "direct", tolerance, max_iterations, warm_start)` |

`evaluate` returns a `D4Result` with the `D3Result` fields plus `charges`.
`stress` is available for every periodic input (any periodic flag makes the
system fully periodic, as in dftd4). `info` adds the charge-solver
diagnostics: `eeq_solver` (`"direct"` or `"minres"`), `eeq_iterations` and
`eeq_adjoint_iterations`, `eeq_restarts`, `eeq_backward_error`,
`eeq_matrix_bytes`, `eeq_warm_started`, `ewald_mode`, `ewald_alpha`, and the
numbers of real-space translations and reciprocal vectors. `set_charge`
changes the total charge without rebuilding anything; `set_parameters`,
`set_cutoffs`, `invalidate_neighbors`, `diagnostics`, and `close` work as for
`D3Context`. `evaluate_atomic` and `debug_atomic` work in bohr and hartree.
`evaluate_d4(numbers, positions, ...)` is a one-shot convenience wrapper.

The charge solver needs a dense (N+1)² matrix, or packed tiles of half that
size from 768 atoms (0.4 GB for 10 000 atoms); above half of the device memory
it raises `DeviceOutOfMemoryError`. Periodic gradients also keep nine doubles
per atom pair from the matrix assembly while they fit in the same budget, and
otherwise recompute the lattice sums.

## Errors

All errors derive from `vanderfast.VanderfastError` and a built-in category:
`ParameterError` and `InputError` (ValueError), `UnsupportedFeatureError`
(NotImplementedError), `CudaError` and `CudaUnavailableError` (RuntimeError),
`DeviceOutOfMemoryError` (MemoryError), `ChargeSolverError` (ArithmeticError:
singular or unconverged charge equations), `ContextBusyError`,
`NativeExtensionMissingError` and `ReferenceBackendUnavailableError`
(ImportError).

## Diagnostics

`vanderfast.backend_info(initialize_cuda=True)` and `python -m vanderfast
info --json` report the package version, build (compiler, architectures,
math flags, statically linked runtime), the reference-data identity, and the
CUDA devices, for both the D3 and the D4 tables. `D3Context.diagnostics()`
and `D4Context.diagnostics()` add the resolved parameters, cutoffs, skin (and
for D4 the charge, Ewald mode, and solver options), and runtime counters
(evaluations, rebuilds, allocations, device bytes, edge capacity).
