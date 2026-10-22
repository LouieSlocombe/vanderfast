"""Compare the CUDA D4 calculator with dftd4's own ASE calculator.

Requires the optional reference library (``pip install vanderfast[reference]``).
"""

import numpy as np
from ase.build import bulk
from dftd4.ase import DFTD4

from vanderfast.ase import D4Calculator


def main() -> None:
    """Evaluate a rattled fluorite cell with both calculators."""
    atoms = bulk("CaF2", "fluorite", a=5.46).repeat(2)
    atoms.rattle(stdev=0.02, seed=1)
    results = {}
    for name, calc in (("vanderfast", D4Calculator(method="pbe")), ("dftd4", DFTD4(method="pbe"))):
        atoms.calc = calc
        results[name] = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
    (e, f, s), (e_ref, f_ref, s_ref) = results["vanderfast"], results["dftd4"]
    print(f"energy        {e:.10f} eV (dftd4 {e_ref:.10f})")
    print(f"|dE|          {abs(e - e_ref):.2e} eV")
    print(f"max |dF|      {np.abs(f - f_ref).max():.2e} eV/Å")
    print(f"max |dstress| {np.abs(s - s_ref).max():.2e} eV/Å³")


if __name__ == "__main__":
    main()
