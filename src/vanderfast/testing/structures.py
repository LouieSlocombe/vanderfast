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


def _tetrahedral(centre: int, ligand: int, bond: float) -> Atoms:
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3.0)
    return Atoms(
        numbers=[centre] + [ligand] * 4, positions=np.vstack([np.zeros(3), bond * directions])
    )


def ammonium() -> Atoms:
    """Return NH4+ (tetrahedral, N-H 1.02 Å); total charge +1."""
    return _tetrahedral(7, 1, 1.02)


def hydroxide() -> Atoms:
    """Return OH- (O-H 0.97 Å); total charge -1."""
    return Atoms(numbers=[8, 1], positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])


def sulfate() -> Atoms:
    """Return SO4 2- (tetrahedral, S-O 1.49 Å); total charge -2."""
    return _tetrahedral(16, 8, 1.49)


def acetate() -> Atoms:
    """Return CH3COO- (acetic acid without its acidic proton); total charge -1."""
    acid = molecule("CH3COOH")
    oxygens = acid.positions[np.array(acid.get_chemical_symbols()) == "O"]
    hydrogens = np.flatnonzero(acid.numbers == 1)
    distance = [min(np.linalg.norm(acid.positions[h] - o) for o in oxygens) for h in hydrogens]
    del acid[int(hydrogens[int(np.argmin(distance))])]
    anion: Atoms = Atoms(acid)
    return anion


def hexaaqua_iron() -> Atoms:
    """Return [Fe(H2O)6]3+ (octahedral, Fe-O 2.0 Å, O-H 0.96 Å); total charge +3."""
    numbers = [26]
    positions = [np.zeros(3)]
    half = math.radians(104.5) / 2.0
    for axis in np.vstack([np.eye(3), -np.eye(3)]):
        oxygen = 2.0 * axis
        perpendicular = np.roll(axis, 1)
        numbers.append(8)
        positions.append(oxygen)
        for sign in (1.0, -1.0):
            numbers.append(1)
            positions.append(
                oxygen + 0.96 * (math.cos(half) * axis + sign * math.sin(half) * perpendicular)
            )
    return Atoms(numbers=numbers, positions=np.array(positions))


def uranyl() -> Atoms:
    """Return linear UO2 (U-O 1.77 Å)."""
    return Atoms(
        numbers=[8, 92, 8], positions=[[-1.77, 0.0, 0.0], [0.0, 0.0, 0.0], [1.77, 0.0, 0.0]]
    )


def americium_trichloride() -> Atoms:
    """Return trigonal-planar AmCl3 (Am-Cl 2.6 Å)."""
    angles = np.radians([0.0, 120.0, 240.0])
    ligands = 2.6 * np.stack([np.cos(angles), np.sin(angles), np.zeros(3)], axis=1)
    return Atoms(numbers=[95, 17, 17, 17], positions=np.vstack([np.zeros(3), ligands]))


def hydrogen_cluster_indefinite() -> Atoms:
    """Return a 24-atom hydrogen cluster dense enough (0.3 Å) for an indefinite EEQ matrix."""
    return _random_cluster([1] * 24, seed=19, min_distance=0.3, radius_per_atom=0.35)


def hydrogen_solid() -> Atoms:
    """Return eight hydrogen atoms in a compressed 3.0 Å cubic cell."""
    rng = np.random.default_rng(23)
    return Atoms(
        numbers=[1] * 8, scaled_positions=rng.random((8, 3)), cell=np.eye(3) * 3.0, pbc=True
    )


def magnesia_primitive() -> Atoms:
    """Return the two-atom primitive MgO cell."""
    return bulk("MgO", "rocksalt", a=4.21)


def magnesia_conventional() -> Atoms:
    """Return the eight-atom conventional MgO cell."""
    return bulk("MgO", "rocksalt", a=4.21, cubic=True)


def lithium_hydride() -> Atoms:
    """Return the two-atom primitive LiH cell."""
    return bulk("LiH", "rocksalt", a=4.08)


def lithium_fluoride() -> Atoms:
    """Return the two-atom primitive LiF cell."""
    return bulk("LiF", "rocksalt", a=4.03)


def fluorite() -> Atoms:
    """Return the three-atom primitive CaF2 cell."""
    return bulk("CaF2", "fluorite", a=5.46)


def elongated_cell() -> Atoms:
    """Return a four-atom cell elongated 1:1:4 (exercises the Ewald parameter bisection)."""
    return Atoms(
        numbers=[11, 17, 1, 8],
        scaled_positions=[[0.1, 0.1, 0.1], [0.6, 0.5, 0.2], [0.3, 0.7, 0.6], [0.8, 0.2, 0.85]],
        cell=np.diag([3.2, 3.4, 13.0]),
        pbc=True,
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

IONS: dict[str, tuple[Callable[[], Atoms], float]] = {
    "ammonium": (ammonium, 1.0),
    "hydroxide": (hydroxide, -1.0),
    "acetate": (acetate, -1.0),
    "sulfate": (sulfate, -2.0),
    "hexaaqua_iron": (hexaaqua_iron, 3.0),
    "water_half_cation": (lambda: Atoms(molecule("H2O")), 0.5),
}

IONIC: dict[str, Callable[[], Atoms]] = {
    "mgo_primitive": magnesia_primitive,
    "mgo_conventional": magnesia_conventional,
    "lih_primitive": lithium_hydride,
    "lif_primitive": lithium_fluoride,
    "caf2_primitive": fluorite,
    "elongated_cell": elongated_cell,
}

PARTIAL: dict[str, Callable[[], Atoms]] = {
    "graphene_slab": graphene_slab,
    "graphene_slab_tilted_vacuum": lambda: graphene_slab((1.0, 0.5, 12.0)),
    "carbon_wire": carbon_wire,
}


__all__ = [
    "IONIC",
    "IONS",
    "MOLECULES",
    "PARTIAL",
    "PERIODIC",
    "acetate",
    "americium_trichloride",
    "ammonium",
    "benzene_dimer",
    "carbon_wire",
    "co2_crystal",
    "coverage_clusters",
    "dimer",
    "elongated_cell",
    "equilateral_trimer",
    "fluorite",
    "graphene_slab",
    "hexaaqua_iron",
    "hydrogen_cluster_indefinite",
    "hydrogen_solid",
    "hydroxide",
    "isosceles_trimer",
    "linear_trimer",
    "lithium_fluoride",
    "lithium_hydride",
    "magnesia_conventional",
    "magnesia_primitive",
    "mixed_fragment",
    "one_atom_triclinic",
    "replicate",
    "rocksalt_primitive",
    "scalene_trimer",
    "shell_cluster",
    "simple_cubic",
    "skewed_triclinic",
    "sulfate",
    "uranyl",
    "water_dimer",
]
