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
    result = context.evaluate(positions, cell=cell, pbc=True,
                              properties=("energy", "forces", "stress"))
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

## Errors

All errors derive from `vanderfast.VanderfastError` and a built-in category:
`ParameterError` and `InputError` (ValueError), `UnsupportedFeatureError`
(NotImplementedError), `CudaError` and `CudaUnavailableError` (RuntimeError),
`DeviceOutOfMemoryError` (MemoryError), `ContextBusyError`,
`NativeExtensionMissingError` and `ReferenceBackendUnavailableError`
(ImportError).

## Diagnostics

`vanderfast.backend_info(initialize_cuda=True)` and `python -m vanderfast
info --json` report the package version, build (compiler, architectures,
math flags, statically linked runtime), the reference-data identity, and the
CUDA devices. `D3Context.diagnostics()` adds the resolved parameters, cutoffs,
skin, and runtime counters (evaluations, rebuilds, allocations, device bytes,
edge capacity).
