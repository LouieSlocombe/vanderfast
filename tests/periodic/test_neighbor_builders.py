"""Cell-list and brute-force neighbour builders produce exactly the CPU edge set."""

import numpy as np
import numpy.typing as npt
import pytest
from support import native_helpers as nh

from vanderfast import resolve_parameters
from vanderfast.parameters import Cutoffs

pytestmark = pytest.mark.cuda

PBE = resolve_parameters(method="pbe", damping="bj", atm=False)

PBC_PATTERNS = [
    (True, True, True),
    (True, True, False),
    (False, False, True),
    (False, False, False),
    (True, False, True),
]


def _graph(
    builder: str,
    numbers: npt.ArrayLike,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
    cutoff: float,
) -> set[tuple[int, ...]]:
    cutoffs = Cutoffs(cn=cutoff, disp2=cutoff, disp3=cutoff)
    ctx = nh.context(numbers, PBE, cutoffs, builder=builder)
    debug = ctx.debug(positions, cell, pbc, derivatives=False)
    keys = nh.edge_keys(debug["edge_i"], debug["edge_j"], debug["edge_shift"])
    assert len(keys) == len(debug["edge_i"])  # no duplicates
    return keys


def _cpu(
    positions: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool], cutoff: float
) -> set[tuple[int, ...]]:
    edges = nh.native().enumerate_edges(positions, cell, pbc, cutoff)
    return nh.edge_keys(edges["i"], edges["j"], edges["shift"])


@pytest.mark.parametrize("seed", range(40))
def test_cell_bins_match_the_cpu_enumeration_on_random_cells(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pbc = PBC_PATTERNS[seed % len(PBC_PATTERNS)]
    n = int(rng.integers(1, 30))
    cell = np.diag(rng.uniform(3.0, 16.0, 3)) + rng.uniform(-3.0, 3.0, (3, 3)) * (1 - np.eye(3))
    positions = rng.uniform(-1.5, 2.5, (n, 3)) @ cell  # deliberately unwrapped
    if not any(pbc):
        positions = rng.uniform(-30.0, 30.0, (n, 3))
    cutoff = float(rng.choice([8.0, 15.0, 25.0]))
    expected = _cpu(positions, cell, pbc, cutoff)
    assert _graph("cell_bins", np.full(n, 6), positions, cell, pbc, cutoff) == expected
    assert _graph("brute_force", np.full(n, 6), positions, cell, pbc, cutoff) == expected


@pytest.mark.parametrize(
    ("label", "numbers", "positions", "cell", "pbc", "cutoff"),
    [
        ("tiny-cell", [6], [[0.3, 0.2, 0.1]], np.eye(3) * 3.4, (True, True, True), 60.0),
        (
            "tiny-two-atom",
            [6, 8],
            [[0.3, 0.2, 0.1], [1.9, 1.6, 2.2]],
            np.array([[3.4, 0, 0], [1.0, 3.2, 0], [0.4, 0.6, 3.0]]),
            (True, True, True),
            40.0,
        ),
        (
            "thin-wire",
            [1, 1],
            [[0, 0, 0], [0.5, 0.2, 1.0]],
            np.eye(3) * 2.0,
            (False, False, True),
            50.0,
        ),
    ],
)
def test_cutoffs_much_larger_than_the_cell(
    label: str,
    numbers: list[int],
    positions: list[list[float]],
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
    cutoff: float,
) -> None:
    x = np.asarray(positions, dtype=np.float64)
    expected = _cpu(x, cell, pbc, cutoff)
    assert len(expected) > 100, label
    assert _graph("cell_bins", numbers, x, cell, pbc, cutoff) == expected


def test_large_systems_agree_between_builders() -> None:
    rng = np.random.default_rng(5)
    cell = np.diag([45.0, 49.5, 40.5]) + np.array([[0, 0, 0], [3.0, 0, 0], [2.0, 1.0, 0]])
    positions = rng.uniform(0.0, 1.0, (2000, 3)) @ cell
    pbc = (True, True, True)
    assert _graph("cell_bins", np.full(2000, 8), positions, cell, pbc, 30.0) == _graph(
        "brute_force", np.full(2000, 8), positions, cell, pbc, 30.0
    )
    clusters = np.vstack(
        [rng.uniform(-5, 5, (200, 3)), rng.uniform(-5, 5, (200, 3)) + np.array([300.0, 0.0, 0.0])]
    )
    none = (False, False, False)
    zero = np.zeros((3, 3))
    assert _graph("cell_bins", np.full(400, 1), clusters, zero, none, 20.0) == _cpu(
        clusters, zero, none, 20.0
    )


def test_automatic_builder_selection() -> None:
    small = nh.context([1, 1], PBE)
    result = nh.run(small, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    assert result["info"]["builder"] == "brute_force"
    rng = np.random.default_rng(9)
    cell = np.eye(3) * 60.0
    big = nh.context(np.full(5000, 18), PBE, Cutoffs(cn=10.0, disp2=12.0, disp3=10.0))
    result = nh.run(big, rng.uniform(0, 60.0, (5000, 3)), cell, (True, True, True), gradient=False)
    assert result["info"]["builder"] == "cell_bins"
