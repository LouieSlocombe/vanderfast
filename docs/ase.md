# ASE calculator

```python
from vanderfast.ase import D3Calculator

atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # three-dimensional periodic cells only
```

Combine with another calculator through `ase.calculators.mixing.SumCalculator`.

| Property | Result |
| --- | --- |
| `energy`, `free_energy` | total dispersion energy (eV) |
| `forces` | eV/Å |
| `stress` | eV/Å³, order xx, yy, zz, yz, xz, xy; raises `PropertyNotImplementedError` without 3D periodicity |

Keywords: `method`, `damping`, `atm` (required with `method`), or `parameters`
(a `D3BJParameters`); `backend` (`"cuda"` or `"reference"` for the dftd3
library), `device`, `precision`, `cutoff` (Å overrides), `skin` (Å).

The calculator keeps one CUDA context per set of atomic numbers. Results are
cached by ASE for unchanged atoms; charges and magnetic moments do not affect
D3 and do not invalidate them. Changing `method`, `damping`, `atm`, or
`parameters` updates the context in place; changing `cutoff` rebuilds the
neighbour list; changing `skin`, `device`, `precision`, or `backend` creates
a new context. Unlike `dftd3.ase.DFTD3(method=...)`, which leaves ATM off
unless requested, `atm` must always be given explicitly.
