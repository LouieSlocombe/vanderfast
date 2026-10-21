"""Agreement targets.

``ORACLE_*`` bound the NumPy oracle against dftd3 in atomic units. ``TARGET_*``
are the spec's implementation-agreement targets for the CUDA backend in
public units (eV, Å). These are not estimates of D3's physical accuracy.
"""

from typing import Final

ORACLE_ENERGY_ABS: Final = 1e-11  # hartree
ORACLE_ENERGY_REL: Final = 1e-12
ORACLE_GRADIENT_ABS: Final = 1e-11  # hartree/bohr
ORACLE_VIRIAL_ABS: Final = 1e-11  # hartree

TARGET_ENERGY_ABS_EV: Final = 1e-8
TARGET_ENERGY_PER_ATOM_EV: Final = 1e-10
TARGET_FORCE_EV_PER_ANG: Final = 1e-7
TARGET_STRESS_EV_PER_ANG3: Final = 1e-8


def oracle_energy_tolerance(energy: float) -> float:
    """Absolute energy tolerance for oracle-vs-reference comparisons."""
    return ORACLE_ENERGY_ABS + ORACLE_ENERGY_REL * abs(energy)


def target_energy_tolerance(n_atoms: int) -> float:
    """Absolute CUDA-vs-reference energy target in eV."""
    return TARGET_ENERGY_ABS_EV + n_atoms * TARGET_ENERGY_PER_ATOM_EV
