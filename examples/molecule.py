"""Dispersion energy and forces of a molecule with the ASE calculator."""

from ase.build import molecule

from vanderfast.ase import D3Calculator


def main() -> None:
    """Evaluate D3(BJ)+ATM for benzene."""
    atoms = molecule("C6H6")
    atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
    print(f"energy  {atoms.get_potential_energy():.8f} eV")
    print(f"max |F| {abs(atoms.get_forces()).max():.3e} eV/Å")


if __name__ == "__main__":
    main()
