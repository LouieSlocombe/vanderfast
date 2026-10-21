"""Energy, forces, and stress of a periodic crystal."""

from ase.build import bulk

from vanderfast.ase import D3Calculator


def main() -> None:
    """Evaluate D3(BJ)+ATM for rocksalt MgO."""
    atoms = bulk("MgO", "rocksalt", a=4.21, cubic=True).repeat((2, 2, 2))
    atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
    print(f"energy per atom {atoms.get_potential_energy() / len(atoms):.8f} eV")
    print("stress (eV/Å^3, xx yy zz yz xz xy):", atoms.get_stress())


if __name__ == "__main__":
    main()
