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

The calculator keeps one CUDA context per set of atomic numbers. On a
three-dimensional periodic cell, a forces request also computes the stress
(the virial adds a few percent), so `get_stress()` after `get_forces()` costs
nothing extra; energy-only requests skip all derivatives. Results are cached by
ASE for unchanged atoms; charges and magnetic moments do not affect
D3 and do not invalidate them. Changing `method`, `damping`, `atm`, or
`parameters` updates the context in place; changing `cutoff` rebuilds the
neighbour list; changing `skin`, `device`, `precision`, or `backend` creates
a new context. Unlike `dftd3.ase.DFTD3(method=...)`, which leaves ATM off
unless requested, `atm` must always be given explicitly.

## D4Calculator

```python
from vanderfast.ase import D4Calculator

atoms.calc = D4Calculator(method="r2scan")  # ATM on by default, as in dftd4.ase
energy = atoms.get_potential_energy()
charges = atoms.calc.get_property("charges", atoms)  # EEQ charges, e
```

| Property | Result |
| --- | --- |
| `energy`, `free_energy` | total dispersion energy (eV) |
| `forces` | eV/Å |
| `stress` | eV/Å³, order xx, yy, zz, yz, xz, xy; any periodic system |
| `charges` | EEQ partial charges (e) |

Keywords: `method` and `atm` (default `True`), or `parameters` (a
`D4Parameters`); `charge` (total charge in e; by default the sum of
`atoms.get_initial_charges()`, as in `dftd4.ase`); `backend` (`"cuda"` or
`"reference"` for the dftd4 library); `device`, `precision`, `cutoff` (Å
overrides of `cn`, `disp2`, `disp3`), `skin` (Å), `ewald` (`"upstream"` or
`"converged"`), and `solver` (`EEQSolverOptions`).

Any periodic direction makes the system fully periodic over the cell, so a
slab or wire needs vacuum lattice vectors. Without an explicit `charge`,
changing the initial charges invalidates the cached results and updates the
context's total charge in place (no rebuild); with one, initial charges are
ignored. Magnetic moments never matter. The other caching rules are those of
`D3Calculator`.
