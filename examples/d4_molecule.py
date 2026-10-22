"""DFT-D4 energy, forces, and EEQ charges of a molecule and an anion."""

import numpy as np
from ase import Atoms
from ase.build import molecule

from vanderfast.ase import D4Calculator

OH_BOND = 1.2  # Å


def acidic_hydrogen(atoms: Atoms) -> int:
    """Index of the hydrogen bonded to an oxygen."""
    oxygens = [a.index for a in atoms if a.symbol == "O"]
    for atom in atoms:
        if atom.symbol == "H" and min(atoms.get_distances(atom.index, oxygens)) < OH_BOND:
            return int(atom.index)
    msg = "no O-H bond"
    raise ValueError(msg)


def main() -> None:
    """Evaluate D4 for acetic acid and its conjugate base (total charge -1)."""
    acid = molecule("CH3COOH")
    acid.calc = D4Calculator(method="r2scan")  # ATM on, as in dftd4
    print(f"acetic acid  {acid.get_potential_energy():.8f} eV")

    # Remove the acidic hydrogen; the total charge comes from the initial charges.
    anion = acid.copy()
    del anion[acidic_hydrogen(acid)]
    charges = np.zeros(len(anion))
    charges[0] = -1.0
    anion.set_initial_charges(charges)
    anion.calc = D4Calculator(method="r2scan")
    print(f"acetate      {anion.get_potential_energy():.8f} eV")
    print(f"max |F|      {abs(anion.get_forces()).max():.3e} eV/Å")
    eeq = anion.calc.get_property("charges", anion)
    for symbol, q in zip(anion.get_chemical_symbols(), eeq, strict=True):
        print(f"  {symbol:2s} {q:+.4f} e")
    print(f"sum of charges {eeq.sum():+.12f} e")


if __name__ == "__main__":
    main()
