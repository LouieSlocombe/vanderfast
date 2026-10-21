"""Deterministic structure builders."""

from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms

from vanderfast.testing import structures

BUILDERS: dict[str, Callable[[], Atoms]] = {
    **structures.MOLECULES,
    **structures.PERIODIC,
    **structures.PARTIAL,
    "dimer": lambda: structures.dimer(1, 8, 1.0),
    "equilateral": lambda: structures.equilateral_trimer(18, 3.5),
    "linear": lambda: structures.linear_trimer(18, 3.8),
    "scalene": structures.scalene_trimer,
    "shell": lambda: structures.shell_cluster(1, 6, 18, 1.2),
    "replica": lambda: structures.replicate(structures.rocksalt_primitive(), (2, 1, 1)),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builders_are_deterministic(name: str) -> None:
    first, second = BUILDERS[name](), BUILDERS[name]()
    np.testing.assert_array_equal(first.numbers, second.numbers)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.cell[:], second.cell[:])
    assert len(first) > 0


def test_coverage_clusters_contain_every_supported_element() -> None:
    clusters = structures.coverage_clusters()
    numbers = sorted(int(z) for atoms in clusters for z in atoms.numbers)
    assert numbers == list(range(1, 104))
    for atoms in clusters:
        distances = atoms.get_all_distances()[np.triu_indices(len(atoms), 1)]
        assert distances.min() >= 2.2


def test_geometric_builders_have_requested_shapes() -> None:
    apex = structures.isosceles_trimer(18, 3.0, 90.0)
    assert apex.get_distance(1, 2) == pytest.approx(3.0 * np.sqrt(2.0))
    shell = structures.shell_cluster(1, 6, 12, 2.0)
    np.testing.assert_allclose(np.linalg.norm(shell.positions[1:], axis=1), 2.0)
    assert (
        structures.replicate(
            structures.rocksalt_primitive(), (2, 2, 2)
        ).get_global_number_of_atoms()
        == 16
    )
