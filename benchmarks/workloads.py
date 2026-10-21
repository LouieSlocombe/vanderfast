"""Benchmark workloads: deterministic structure generators with an atom-count target."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule

GENERATOR_VERSION = 1


@dataclass(frozen=True)
class Workload:
    """A named structure family."""

    name: str
    family: str
    build: Callable[[int, int], Atoms]  # (target atom count, seed) -> atoms
    fixed_size: bool = False


def _grid_side(count: int) -> int:
    return max(1, math.ceil(math.cbrt(count)))


def benzene_cluster(target: int, seed: int) -> Atoms:
    """Benzene molecules on a jittered cubic grid (non-periodic, few neighbours)."""
    rng = np.random.default_rng(seed)
    count = max(1, round(target / 12))
    side = _grid_side(count)
    atoms = Atoms()
    template: Atoms = molecule("C6H6")
    for placed, index in enumerate(np.ndindex(side, side, side)):
        if placed == count:
            break
        copy = template.copy()
        copy.rotate(float(rng.uniform(0, 360)), rng.normal(size=3))
        copy.translate(np.array(index) * 7.0 + rng.uniform(-0.5, 0.5, 3))
        atoms += copy
    return atoms


def water_box(target: int, seed: int) -> Atoms:
    """Liquid-density water on a jittered lattice in a cubic periodic box."""
    rng = np.random.default_rng(seed)
    count = max(1, round(target / 3))
    side = _grid_side(count)
    spacing = 29.9 ** (1.0 / 3.0)  # 29.9 Å^3 per molecule at 1 g/cm^3
    atoms = Atoms(cell=np.eye(3) * spacing * side, pbc=True)
    template: Atoms = molecule("H2O")
    for placed, index in enumerate(np.ndindex(side, side, side)):
        if placed == count:
            break
        copy = template.copy()
        copy.rotate(float(rng.uniform(0, 360)), rng.normal(size=3))
        copy.translate((np.array(index) + 0.5) * spacing + rng.uniform(-0.2, 0.2, 3))
        atoms += copy
    return atoms


def mgo_rocksalt(target: int, seed: int) -> Atoms:
    """A dense mixed-element solid: cubic MgO supercells (8 atoms per cell)."""
    del seed
    cells = max(1, round(target / 8))
    side = _grid_side(cells)
    crystal: Atoms = bulk("MgO", "rocksalt", a=4.21, cubic=True).repeat((side, side, side))
    return crystal


def skewed_argon(target: int, seed: int) -> Atoms:
    """A skewed cell: primitive fcc argon (60 degree angles) supercells."""
    del seed
    side = _grid_side(max(1, target))
    crystal: Atoms = bulk("Ar", "fcc", a=5.26).repeat((side, side, side))
    return crystal


def small_cell(target: int, seed: int) -> Atoms:
    """A two-atom cell needing many periodic images."""
    del target, seed
    crystal: Atoms = bulk("NaCl", "rocksalt", a=5.64)
    return crystal


WORKLOADS: dict[str, Workload] = {
    w.name: w
    for w in (
        Workload("benzene-cluster", "molecular", benzene_cluster),
        Workload("water-box", "molecular-liquid", water_box),
        Workload("mgo-rocksalt", "dense-solid", mgo_rocksalt),
        Workload("skewed-argon", "skewed-cell", skewed_argon),
        Workload("small-cell", "many-images", small_cell, fixed_size=True),
    )
}


def perturbed(positions: np.ndarray, amplitude: float, seed: int) -> np.ndarray:
    """Return positions with a uniform random displacement per component (Å)."""
    rng = np.random.default_rng(seed)
    return positions + rng.uniform(-amplitude, amplitude, positions.shape)
