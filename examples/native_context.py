"""Array interface without ASE objects: one persistent context, many geometries."""

import numpy as np
from ase.build import molecule

from vanderfast import D3Context, resolve_parameters


def main() -> None:
    """Scan the separation of a water dimer with one context."""
    monomer = molecule("H2O")
    numbers = np.concatenate([monomer.numbers, monomer.numbers])
    parameters = resolve_parameters(method="b3lyp", damping="bj", atm=True)
    with D3Context(numbers, parameters) as context:
        for separation in (2.6, 2.8, 3.0, 3.5, 4.0):
            positions = np.vstack(
                [monomer.positions, monomer.positions + np.array([0.0, 0.0, separation])]
            )
            result = context.evaluate(positions, properties=("energy", "forces"))
            print(f"R = {separation:.1f} Å  E_disp = {result.energy:+.6f} eV")
        print(context.diagnostics()["stats"])


if __name__ == "__main__":
    main()
