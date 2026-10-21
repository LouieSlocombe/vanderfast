"""Add the D3 correction to another ASE calculator and compare with dftd3.

Requires the optional reference library (``pip install vanderfast[reference]``).
"""

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator

from vanderfast.ase import D3Calculator


def main() -> None:
    """Combine EMT with D3(BJ) and check against the dftd3 reference backend."""
    atoms = bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3))
    atoms.calc = SumCalculator([EMT(), D3Calculator(method="pbe", damping="bj", atm=True)])
    total = atoms.get_potential_energy()
    alone = atoms.copy()
    alone.calc = D3Calculator(method="pbe", damping="bj", atm=True)
    dispersion = alone.get_potential_energy()
    alone.calc = D3Calculator(method="pbe", damping="bj", atm=True, backend="reference")
    difference = abs(dispersion - alone.get_potential_energy())
    print(f"EMT + D3 energy        {total:.6f} eV")
    print(f"D3 contribution        {dispersion:.10f} eV")
    print(f"|vanderfast - dftd3|   {difference:.2e} eV")


if __name__ == "__main__":
    main()
