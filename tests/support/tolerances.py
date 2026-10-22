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


D4_ORACLE_ENERGY_ABS: Final = 1e-11  # hartree
D4_ORACLE_ENERGY_REL: Final = 1e-12
D4_ORACLE_CHARGE_ABS: Final = 1e-11  # e
D4_ORACLE_GRADIENT_ABS: Final = 1e-11  # hartree/bohr
D4_ORACLE_VIRIAL_ABS: Final = 1e-10  # hartree


def d4_oracle_energy_tolerance(energy: float) -> float:
    """Absolute energy tolerance for D4 oracle-vs-reference comparisons."""
    return D4_ORACLE_ENERGY_ABS + D4_ORACLE_ENERGY_REL * abs(energy)


# CUDA D4 against dftd4 4.2.0 with the direct charge solver (atomic units).
# Calibrated on every fixture: worst observed 1.1e-13 relative energy,
# 1.9e-14 e, 1.8e-15 hartree/bohr, and 5.2e-15 hartree.
D4_GPU_ENERGY_ABS: Final = 1e-14  # hartree
D4_GPU_ENERGY_REL: Final = 1e-12
D4_GPU_CHARGE_ABS: Final = 1e-12  # e
D4_GPU_GRADIENT_ABS: Final = 1e-12  # hartree/bohr
D4_GPU_VIRIAL_ABS: Final = 1e-12  # hartree


def d4_gpu_energy_tolerance(energy: float) -> float:
    """Absolute energy tolerance for CUDA-vs-reference D4 comparisons."""
    return D4_GPU_ENERGY_ABS + D4_GPU_ENERGY_REL * abs(energy)
