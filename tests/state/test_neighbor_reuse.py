"""Neighbour-list reuse: rebuild triggers and equivalence with fresh contexts."""

import dataclasses
from typing import Any

import numpy as np
import pytest
from support import native_helpers as nh
from support.inputs import SHORT_D3

from vanderfast import resolve_parameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
SKIN = 2.0  # bohr


def _crystal() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    atoms = structures.replicate(structures.rocksalt_primitive(), (2, 2, 2))
    return atoms.numbers, atoms.positions * INV_BOHR, atoms.cell[:] * INV_BOHR, (True, True, True)


def _fresh(
    numbers: np.ndarray, x: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool]
) -> dict[str, Any]:
    return nh.run(nh.context(numbers, PBE_ATM, SHORT_D3, skin=SKIN), x, cell, pbc)


def _assert_same(result: dict[str, Any], fresh: dict[str, Any]) -> None:
    assert result["energy"] == pytest.approx(fresh["energy"], rel=1e-12)
    np.testing.assert_allclose(result["gradient"], fresh["gradient"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result["virial"], fresh["virial"], rtol=1e-10, atol=1e-14)


def test_small_moves_reuse_the_graph_and_match_fresh_contexts() -> None:
    numbers, x, cell, pbc = _crystal()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3, skin=SKIN)
    assert nh.run(ctx, x, cell, pbc)["info"]["rebuilt"]
    rng = np.random.default_rng(0)
    moved = x.copy()
    for _ in range(6):
        step = rng.normal(size=x.shape)
        step *= 0.05 * SKIN / np.abs(step).max()
        moved = moved + step
        result = nh.run(ctx, moved, cell, pbc)
        assert not result["info"]["rebuilt"]
        _assert_same(result, _fresh(numbers, moved, cell, pbc))
    assert ctx.stats()["rebuilds"] == 1


def test_crossing_half_the_skin_triggers_one_rebuild() -> None:
    numbers, x, cell, pbc = _crystal()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3, skin=SKIN)
    nh.run(ctx, x, cell, pbc)
    moved = x.copy()
    moved[3, 0] += 0.55 * SKIN
    result = nh.run(ctx, moved, cell, pbc)
    assert result["info"]["rebuilt"]
    _assert_same(result, _fresh(numbers, moved, cell, pbc))
    assert not nh.run(ctx, moved, cell, pbc)["info"]["rebuilt"]


def test_wrapping_an_atom_triggers_a_rebuild_and_preserves_results() -> None:
    numbers, x, cell, pbc = _crystal()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3, skin=SKIN)
    reference = nh.run(ctx, x, cell, pbc)
    wrapped = x.copy()
    wrapped[5] += cell[1]
    result = nh.run(ctx, wrapped, cell, pbc)
    assert result["info"]["rebuilt"]
    _assert_same(result, reference)


def test_cell_cutoff_and_explicit_invalidation_trigger_rebuilds() -> None:
    numbers, x, cell, pbc = _crystal()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3, skin=SKIN)
    nh.run(ctx, x, cell, pbc)
    strained = cell * 1.001
    result = nh.run(ctx, x @ (np.linalg.inv(cell) @ strained), strained, pbc)
    assert result["info"]["rebuilt"]
    p = dataclasses.replace(PBE_ATM, s8=0.8)
    ctx.set_parameters(
        s6=p.s6, s8=p.s8, a1=p.a1, a2=p.a2, s9=p.s9, alp=p.alp,
        cutoff_cn=SHORT_D3.cn, cutoff_disp2=SHORT_D3.disp2, cutoff_disp3=SHORT_D3.disp3,
    )  # fmt: skip
    assert not nh.run(ctx, x @ (np.linalg.inv(cell) @ strained), strained, pbc)["info"]["rebuilt"]
    ctx.set_parameters(
        s6=p.s6, s8=p.s8, a1=p.a1, a2=p.a2, s9=p.s9, alp=p.alp,
        cutoff_cn=SHORT_D3.cn, cutoff_disp2=SHORT_D3.disp2 + 1.0, cutoff_disp3=SHORT_D3.disp3,
    )  # fmt: skip
    assert nh.run(ctx, x, cell, pbc)["info"]["rebuilt"]
    ctx.invalidate_neighbors()
    assert nh.run(ctx, x, cell, pbc)["info"]["rebuilt"]


def test_zero_skin_and_thin_cells_rebuild_every_call() -> None:
    numbers, x, cell, pbc = _crystal()
    no_skin = nh.context(numbers, PBE_ATM, SHORT_D3, skin=0.0)
    nh.run(no_skin, x, cell, pbc)
    assert nh.run(no_skin, x, cell, pbc)["info"]["rebuilt"]
    tiny = nh.context([6], PBE_ATM, SHORT_D3, skin=8.0)
    tiny_cell = np.eye(3) * 3.4
    nh.run(tiny, np.zeros((1, 3)), tiny_cell, pbc)
    assert nh.run(tiny, np.zeros((1, 3)), tiny_cell, pbc)["info"]["rebuilt"]


def test_capacity_grows_when_the_graph_grows() -> None:
    numbers, x, cell, pbc = _crystal()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3, skin=SKIN)
    nh.run(ctx, x, cell, pbc)
    before = ctx.stats()
    compressed = cell * 0.8
    result = nh.run(ctx, x * 0.8, compressed, pbc)
    after = ctx.stats()
    assert after["capacity_edges"] > before["capacity_edges"]
    assert after["allocations"] > before["allocations"]
    _assert_same(result, _fresh(numbers, x * 0.8, compressed, pbc))
