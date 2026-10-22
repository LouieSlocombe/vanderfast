"""DFT-D4 stress and charges of a crystal, with both periodic charge models."""

from ase.build import bulk

from vanderfast.ase import D4Calculator


def main() -> None:
    """Compare dftd4's truncated Ewald sums with converged ones for MgO."""
    primitive = bulk("MgO", "rocksalt", a=4.21)
    for ewald in ("upstream", "converged"):
        per_cell = []
        for atoms in (primitive.copy(), primitive.repeat(2)):
            atoms.calc = D4Calculator(method="pbe", ewald=ewald)
            per_cell.append(atoms.get_potential_energy() / (len(atoms) // 2))
        charges = atoms.calc.get_property("charges", atoms)
        stress = atoms.get_stress()
        print(f"ewald={ewald}")
        print(f"  energy per MgO, primitive  {per_cell[0]:.10f} eV")
        print(f"  energy per MgO, 2x2x2      {per_cell[1]:.10f} eV")
        print(f"  Mg charge                  {charges[0]:+.8f} e")
        print(f"  stress xx                  {stress[0]:.6e} eV/Å³")


if __name__ == "__main__":
    main()
