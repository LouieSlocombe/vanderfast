"""Deterministic test structures (positions in Ångström, ASE ``Atoms``).

Every generator is a pure function of its arguments and a fixed seed, so the
committed fixtures can be regenerated exactly. Fixtures store the final
coordinates, so later changes here never silently alter stored references.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.build import bulk, molecule
from ase.spacegroup import crystal

FloatArray = npt.NDArray[np.float64]


def _random_cluster(
    numbers: Sequence[int],
    seed: int,
    min_distance: float = 2.2,
    radius_per_atom: float = 1.4,
) -> Atoms:
    """Place atoms randomly in a sphere with a minimum pair separation."""
    rng = np.random.default_rng(seed)
    radius = radius_per_atom * len(numbers) ** (1.0 / 3.0) + min_distance
    positions: list[FloatArray] = []
    while len(positions) < len(numbers):
        trial = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(trial) > radius:
            continue
        if all(np.linalg.norm(trial - p) >= min_distance for p in positions):
            positions.append(trial)
    return Atoms(numbers=list(numbers), positions=np.array(positions))


def water_dimer(oo_distance: float = 2.91) -> Atoms:
    """Return a hydrogen-bonded water dimer with the given O-O distance."""
    donor: Atoms = molecule("H2O")
    acceptor: Atoms = molecule("H2O")
    acceptor.rotate(90.0, "z")
    acceptor.translate((0.0, 0.0, oo_distance))
    dimer_atoms: Atoms = donor + acceptor
    return dimer_atoms


def benzene_dimer(separation: float = 3.4) -> Atoms:
    """Return a parallel-stacked benzene dimer."""
    lower: Atoms = molecule("C6H6")
    upper: Atoms = molecule("C6H6")
    upper.translate((0.0, 0.0, separation))
    stacked: Atoms = lower + upper
    return stacked


def mixed_fragment() -> Atoms:
    """Return a seeded cluster mixing main-group elements including S, P, Si, Br, I."""
    return _random_cluster([6, 1, 1, 16, 15, 14, 35, 53, 8, 7, 9, 17, 1, 6], seed=11)


def coverage_clusters() -> list[Atoms]:
    """Return seeded clusters that together contain every element Z = 1..103."""
    numbers = list(range(1, 104))
    clusters = []
    for index, start in enumerate(range(0, len(numbers), 12)):
        chunk = numbers[start : start + 12]
        clusters.append(_random_cluster(chunk, seed=100 + index))
    return clusters


def _sphere_points(count: int) -> FloatArray:
    """Return quasi-uniform unit vectors on a sphere (Fibonacci lattice)."""
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = math.pi * (1.0 + math.sqrt(5.0)) * index
    return np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)],
        axis=1,
    )


def shell_cluster(centre: int, shell: int, count: int, radius: float) -> Atoms:
    """Return a central atom inside a spherical shell of ``count`` atoms."""
    points = radius * _sphere_points(count)
    return Atoms(
        numbers=[centre] + [shell] * count,
        positions=np.vstack([np.zeros(3), points]),
    )


def dimer(zi: int, zj: int, distance: float) -> Atoms:
    """Return two atoms on the x axis."""
    return Atoms(numbers=[zi, zj], positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])


def isosceles_trimer(element: int, leg: float, apex_degrees: float) -> Atoms:
    """Return an isosceles triangle with the apex atom at the origin."""
    half = math.radians(apex_degrees) / 2.0
    return Atoms(
        numbers=[element] * 3,
        positions=[
            [0.0, 0.0, 0.0],
            [leg * math.cos(half), leg * math.sin(half), 0.0],
            [leg * math.cos(half), -leg * math.sin(half), 0.0],
        ],
    )


def equilateral_trimer(element: int, side: float) -> Atoms:
    """Return an equilateral triangle."""
    return isosceles_trimer(element, side, 60.0)


def linear_trimer(element: int, spacing: float) -> Atoms:
    """Return three collinear, equally spaced atoms."""
    return Atoms(
        numbers=[element] * 3,
        positions=[[-spacing, 0.0, 0.0], [0.0, 0.0, 0.0], [spacing, 0.0, 0.0]],
    )


def scalene_trimer() -> Atoms:
    """Return a scalene C/N/O triangle."""
    return Atoms(numbers=[6, 7, 8], positions=[[0.0, 0.0, 0.0], [3.1, 0.0, 0.0], [1.2, 2.6, 0.4]])


def simple_cubic(element: int, lattice: float) -> Atoms:
    """Return a one-atom simple cubic cell."""
    return Atoms(numbers=[element], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * lattice, pbc=True)


def one_atom_triclinic() -> Atoms:
    """Return a one-atom, strongly skewed triclinic cell."""
    cell = np.array([[3.9, 0.0, 0.0], [1.2, 3.6, 0.0], [0.7, 1.1, 3.3]])
    return Atoms(numbers=[36], positions=[[0.4, 0.3, 0.2]], cell=cell, pbc=True)


def rocksalt_primitive(lattice: float = 5.64) -> Atoms:
    """Return the two-atom primitive NaCl cell."""
    return bulk("NaCl", "rocksalt", a=lattice)


def co2_crystal() -> Atoms:
    """Return the 12-atom Pa-3 CO2 crystal."""
    return crystal(
        ["C", "O"],
        basis=[(0.0, 0.0, 0.0), (0.118, 0.118, 0.118)],
        spacegroup=205,
        cellpar=[5.624, 5.624, 5.624, 90, 90, 90],
    )


def skewed_triclinic() -> Atoms:
    """Return a four-atom cell with angles of 60, 70, and 115 degrees."""
    return Atoms(
        numbers=[6, 7, 8, 1],
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.15, 0.55], [0.35, 0.7, 0.2], [0.8, 0.85, 0.9]],
        cell=[4.6, 4.2, 4.9, 60.0, 70.0, 115.0],
        pbc=True,
    )


def graphene_slab(vacuum_vector: Sequence[float] = (0.0, 0.0, 15.0)) -> Atoms:
    """Return a two-atom graphene cell periodic in the plane only."""
    a = 2.46
    cell = np.array([[a, 0.0, 0.0], [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0], list(vacuum_vector)])
    return Atoms(
        numbers=[6, 6],
        scaled_positions=[[1.0 / 3.0, 2.0 / 3.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.0]],
        cell=cell,
        pbc=[True, True, False],
    )


def carbon_wire() -> Atoms:
    """Return a two-atom zig-zag chain periodic along z."""
    return Atoms(
        numbers=[6, 6],
        positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 1.25]],
        cell=[[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 2.5]],
        pbc=[False, False, True],
    )


def replicate(atoms: Atoms, repeats: Sequence[int]) -> Atoms:
    """Return a supercell repeated along the lattice vectors."""
    supercell: Atoms = atoms.repeat(tuple(repeats))
    return supercell


MOLECULES: dict[str, Callable[[], Atoms]] = {
    "water": lambda: Atoms(molecule("H2O")),
    "water_dimer": water_dimer,
    "water_dimer_compressed": lambda: water_dimer(2.0),
    "benzene": lambda: Atoms(molecule("C6H6")),
    "benzene_dimer": benzene_dimer,
    "mixed_fragment": mixed_fragment,
}

PERIODIC: dict[str, Callable[[], Atoms]] = {
    "ar_simple_cubic": lambda: simple_cubic(18, 3.4),
    "kr_one_atom_triclinic": one_atom_triclinic,
    "nacl_primitive": rocksalt_primitive,
    "co2_crystal": co2_crystal,
    "skewed_triclinic": skewed_triclinic,
}

PARTIAL: dict[str, Callable[[], Atoms]] = {
    "graphene_slab": graphene_slab,
    "graphene_slab_tilted_vacuum": lambda: graphene_slab((1.0, 0.5, 12.0)),
    "carbon_wire": carbon_wire,
}


__all__ = [
    "MOLECULES",
    "PARTIAL",
    "PERIODIC",
    "benzene_dimer",
    "carbon_wire",
    "co2_crystal",
    "coverage_clusters",
    "dimer",
    "equilateral_trimer",
    "graphene_slab",
    "isosceles_trimer",
    "linear_trimer",
    "mixed_fragment",
    "one_atom_triclinic",
    "replicate",
    "rocksalt_primitive",
    "scalene_trimer",
    "shell_cluster",
    "simple_cubic",
    "skewed_triclinic",
    "water_dimer",
]
