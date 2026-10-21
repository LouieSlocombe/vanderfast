"""Short NVE run of argon with a Lennard-Jones base potential plus D3.

The D3 context persists across steps: device buffers and the neighbour list
(with a 1 Å skin) are reused until an atom moves more than half the skin.
"""

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.mixing import SumCalculator
from ase.md.velocitydistribution import thermalize_momenta
from ase.md.verlet import VelocityVerlet

from vanderfast.ase import D3Calculator


def main() -> None:
    """Run 50 velocity-Verlet steps and report the energy drift."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    d3 = D3Calculator(method="pbe", damping="bj", atm=True, skin=1.0)
    atoms.calc = SumCalculator([LennardJones(sigma=3.4, epsilon=0.0104, rc=8.0, smooth=True), d3])
    thermalize_momenta(atoms, temperature_K=40.0, rng=np.random.default_rng(1))
    dynamics = VelocityVerlet(atoms, timestep=2.0 * units.fs)
    start = atoms.get_total_energy()
    dynamics.run(50)
    print(f"energy drift over 50 steps: {atoms.get_total_energy() - start:.3e} eV")
    print(f"D3 evaluations: {d3.evaluations}")


if __name__ == "__main__":
    main()
