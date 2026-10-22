"""The NumPy D4 oracle: building blocks, upstream replicas, and internal consistency."""

import hashlib
import math
import struct

import numpy as np
import pytest

from vanderfast import _oracle_d4
from vanderfast._oracle_d4 import (
    bordered,
    cap_cn,
    dzeta,
    lattice_points,
    matdet_3x3,
    matinv_3x3,
    reference_alpha,
    reference_c6,
    search_alpha,
    shift_back_abc,
    solve_bordered,
    wigner_seitz_images,
    wrap_to_central_cell,
    zeta,
)
from vanderfast.data.d4 import load_d4_manifest, load_d4_reference_tables
from vanderfast.errors import ChargeSolverError, InputError
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

PBE = D4Parameters(s8=0.95948085, a1=0.38574991, a2=4.80688534)
PBE_PAIR = D4Parameters(s8=0.95948085, a1=0.38574991, a2=4.80688534, s9=0.0)
SHORT = D4Cutoffs(cn=20.0, disp2=25.0, disp3=15.0)


def test_derived_reference_tables_match_the_manifest() -> None:
    """The oracle's polarizabilities and C6 reproduce the generator's derivation bitwise."""
    tables = load_d4_reference_tables()
    derived = load_d4_manifest()["derived"]
    alpha = reference_alpha(tables, derived["ga"], derived["gc"])
    assert hashlib.sha256(alpha.astype("<f8").tobytes()).hexdigest() == derived["refalpha_sha256"]
    blocks = [
        reference_c6(tables, alpha, zi, zj).ravel()
        for zi in range(1, tables.max_z + 1)
        for zj in range(1, zi + 1)
    ]
    flat = np.concatenate(blocks)
    assert flat.size == derived["n_c6"]
    packed = struct.pack(f"<{flat.size}d", *flat)
    assert hashlib.sha256(packed).hexdigest() == derived["refc6_sha256"]


def test_matrix_helpers_follow_mctc() -> None:
    lattice = np.array([[4.0, 0.3, 0.1], [0.2, 5.0, 0.4], [0.0, 0.6, 6.0]])
    assert matdet_3x3(lattice) == pytest.approx(np.linalg.det(lattice), rel=1e-15)
    np.testing.assert_allclose(matinv_3x3(lattice), np.linalg.inv(lattice), rtol=1e-14)


def test_shift_back_abc_wraps_like_upstream() -> None:
    frac = np.array([-0.25, 1.25, 1.0 + 1e-15, 0.5, -3.0, 1.0 - 1e-15])
    np.testing.assert_array_equal(
        shift_back_abc(frac), [0.75, 0.25, 1.0 + 1e-15 - 1.0, 0.5, 0.0, (1.0 - 1e-15) - 1.0]
    )


def test_wrapping_is_a_lattice_translation() -> None:
    cell = np.array([[6.0, 0.0, 0.0], [1.0, 7.0, 0.0], [0.5, 0.8, 8.0]])
    positions = np.array([[-3.0, 2.0, 1.0], [14.0, -5.0, 20.0]])
    wrapped = wrap_to_central_cell(positions, cell)
    shifts = (wrapped - positions) @ np.linalg.inv(cell)
    np.testing.assert_allclose(shifts, np.round(shifts), atol=1e-12)
    frac = wrapped @ np.linalg.inv(cell)
    assert np.all((frac >= -1e-14) & (frac < 1.0))


def test_lattice_points_follow_the_mctc_loop_order() -> None:
    lattice = np.diag([1.0, 2.0, 3.0])
    points = lattice_points(lattice, (1, 1, 1), origin=True)
    assert points.shape == (27, 3)
    np.testing.assert_array_equal(
        points[:5], [[0, 0, 0], [0, 0, 3], [0, 0, -3], [0, 2, 0], [0, -2, 0]]
    )
    assert lattice_points(lattice, (2, 2, 2), origin=True).shape == (125, 3)
    no_origin = lattice_points(lattice, (2, 2, 2), origin=False)
    assert no_origin.shape == (124, 3)
    assert not np.any(np.all(no_origin == 0.0, axis=1))


def test_wigner_seitz_self_images_carry_the_upstream_off_by_one() -> None:
    cubic = np.eye(3) * 5.0
    images = wigner_seitz_images(np.zeros((1, 3)), cubic)
    expected = [[0, 0, 0], [0, 0, 5], [0, 0, -5], [0, 5, 0], [0, -5, -5], [5, 0, 0]]
    np.testing.assert_array_equal(images[0][0], expected)


def test_wigner_seitz_pairs_average_tied_images() -> None:
    cubic = np.eye(3) * 4.0
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    images = wigner_seitz_images(positions, cubic)
    assert images[0][1].shape == (8, 3)
    assert images[1][0].shape == (8, 3)


def test_ewald_parameter_search_and_its_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    lattice = np.eye(3) * 8.0
    alpha = search_alpha(lattice)
    assert alpha == 0.4194304  # the first bisection midpoint already meets the tolerance
    assert search_alpha(np.diag([8.0, 8.0, 40.0])) == pytest.approx(0.14141952, rel=1e-12)
    assert search_alpha(np.eye(3) * 1.0e9) == 0.25
    monkeypatch.setattr(_oracle_d4, "ALPHA_ITERATIONS", -1)
    assert search_alpha(lattice) == 0.25


def test_charge_scaling_branches() -> None:
    c = np.array([2.0, 2.0, 2.0])
    qref = np.array([1.0, 1.0, 1.0])
    qmod = np.array([-0.5, 0.0, 2.0])
    values = zeta(3.0, c, qref, qmod)
    assert values[0] == math.exp(3.0)
    assert values[1] == math.exp(3.0)
    assert values[2] == math.exp(3.0 * (1.0 - math.exp(2.0 * (1.0 - 0.5))))
    slopes = dzeta(3.0, c, qref, qmod)
    assert slopes[0] == 0.0
    assert slopes[1] == 0.0
    assert slopes[2] < 0.0


def test_cn_cap_is_smooth_and_saturates() -> None:
    capped, slope = cap_cn(np.array([0.0, 4.0, 30.0]))
    assert capped[0] == pytest.approx(0.0, abs=1e-15)
    assert capped[2] == pytest.approx(
        math.log1p(math.exp(8.0)) - math.log1p(math.exp(-22.0)), rel=1e-14
    )
    assert slope[0] == pytest.approx(1.0, rel=1e-3)
    assert slope[2] < 1e-9


def test_singular_charge_equations_raise() -> None:
    with pytest.raises(ChargeSolverError, match="singular"):
        solve_bordered(bordered(np.zeros((2, 2))), np.zeros(3))


def test_partial_periodicity_is_promoted_and_needs_a_full_cell() -> None:
    geometry = _oracle_d4.make_geometry(
        [6], [[0.0, 0.0, 0.0]], np.eye(3) * 9.0, (True, False, False)
    )
    assert geometry.pbc.tolist() == [True, True, True]
    flat = np.diag([9.0, 9.0, 0.0])
    with pytest.raises(InputError, match="independent"):
        _oracle_d4.make_geometry([6], [[0.0, 0.0, 0.0]], flat, (True, True, False))
    with pytest.raises(InputError, match="pbc"):
        _oracle_d4.make_geometry([6], [[0.0, 0.0, 0.0]], None, (True, False))


def test_weights_fall_back_when_every_gaussian_underflows() -> None:
    water = structures.water_dimer()
    positions = water.positions * INV_BOHR
    result = _oracle_d4.evaluate(water.numbers, positions, parameters=PBE_PAIR, gradient=False)
    tables = load_d4_reference_tables()
    species = _oracle_d4._species(  # noqa: SLF001
        water.numbers.astype(np.int64), tables, reference_alpha(tables, 3.0, 2.0)
    )
    huge = np.full(water.numbers.size, 80.0)
    weights = _oracle_d4.reference_weights(species, huge, np.zeros(huge.size), 3.0, 2.0, 6.0)
    oxygen = int(np.flatnonzero(water.numbers == 8)[0])
    refs = species.cnref[oxygen, : species.nref[oxygen]]
    expected = (np.abs(refs.max() - refs) < 1e-12).astype(float)
    scale = weights.w[oxygen, : refs.size][expected == 1.0]
    assert np.all(scale > 0.0)
    assert np.all(weights.w[oxygen, : refs.size][expected == 0.0] == 0.0)
    assert np.all(weights.dw_dcn[oxygen] == 0.0)
    assert result.energy < 0.0


def test_counting_conventions_agree() -> None:
    rocksalt = structures.rocksalt_primitive()
    args = (rocksalt.numbers, rocksalt.positions * INV_BOHR)
    kwargs = {"cell": rocksalt.cell[:] * INV_BOHR, "pbc": rocksalt.pbc, "cutoffs": SHORT}
    base = _oracle_d4.evaluate(*args, parameters=PBE, **kwargs)
    upstream = _oracle_d4.evaluate(
        *args,
        parameters=PBE,
        pair_counting="upstream",
        atm_counting="upstream",
        **kwargs,
    )
    assert upstream.energy == pytest.approx(base.energy, rel=1e-13)
    assert upstream.gradient is not None
    assert base.gradient is not None
    np.testing.assert_allclose(upstream.gradient, base.gradient, atol=1e-14)
    assert upstream.strain_derivative is not None
    assert base.strain_derivative is not None
    np.testing.assert_allclose(upstream.strain_derivative, base.strain_derivative, atol=1e-13)


def test_terms_can_be_switched_off() -> None:
    water = structures.water_dimer()
    positions = water.positions * INV_BOHR
    atm_only = D4Parameters(s6=0.0, s8=0.0, a1=0.4, a2=5.0, s9=1.0)
    result = _oracle_d4.evaluate(water.numbers, positions, parameters=atm_only)
    assert result.energy_pair == 0.0
    assert result.energy_atm != 0.0
    assert result.counts["pair_edges"] == 0
    pair_only = _oracle_d4.evaluate(water.numbers, positions, parameters=PBE_PAIR, gradient=False)
    assert pair_only.energy_atm == 0.0
    assert pair_only.gradient is None
    assert pair_only.eeq_adjoint is None


def test_charges_sum_to_the_total_charge_and_respond_to_it() -> None:
    water = structures.water_dimer()
    positions = water.positions * INV_BOHR
    neutral = _oracle_d4.evaluate(water.numbers, positions, parameters=PBE)
    anion = _oracle_d4.evaluate(water.numbers, positions, parameters=PBE, charge=-1.0)
    assert neutral.charges.sum() == pytest.approx(0.0, abs=1e-14)
    assert anion.charges.sum() == pytest.approx(-1.0, abs=1e-14)
    assert anion.polarizabilities.sum() > neutral.polarizabilities.sum()
    assert neutral.ewald_alpha is None
    assert neutral.eeq_matrix.shape == (7, 7)


def test_converged_ewald_is_independent_of_alpha_and_cell_choice() -> None:
    crystal = structures.rocksalt_primitive()
    args = (crystal.numbers, crystal.positions * INV_BOHR)
    kwargs = {"cell": crystal.cell[:] * INV_BOHR, "pbc": crystal.pbc, "cutoffs": SHORT}
    default = _oracle_d4.evaluate(*args, parameters=PBE_PAIR, ewald="converged", **kwargs)
    other = _oracle_d4.evaluate(
        *args,
        parameters=PBE_PAIR,
        ewald="converged",
        ewald_alpha=0.9,
        **kwargs,
    )
    assert other.ewald_alpha == 0.9
    np.testing.assert_allclose(other.charges, default.charges, atol=1e-14)
    assert other.strain_derivative is not None
    assert default.strain_derivative is not None
    np.testing.assert_allclose(other.strain_derivative, default.strain_derivative, atol=1e-15)
    big = crystal.repeat((2, 1, 1))
    supercell = _oracle_d4.evaluate(
        big.numbers,
        big.positions * INV_BOHR,
        parameters=PBE_PAIR,
        cell=big.cell[:] * INV_BOHR,
        pbc=big.pbc,
        cutoffs=SHORT,
        ewald="converged",
    )
    np.testing.assert_allclose(
        supercell.charges.reshape(2, -1), default.charges[None, :].repeat(2, 0), atol=1e-13
    )
    assert supercell.energy / 2 == pytest.approx(default.energy, rel=1e-13)


def test_upstream_ewald_is_cell_dependent() -> None:
    crystal = structures.rocksalt_primitive()
    primitive = _oracle_d4.evaluate(
        crystal.numbers,
        crystal.positions * INV_BOHR,
        parameters=PBE_PAIR,
        cell=crystal.cell[:] * INV_BOHR,
        pbc=crystal.pbc,
        cutoffs=SHORT,
        gradient=False,
    )
    big = crystal.repeat((2, 2, 2))
    supercell = _oracle_d4.evaluate(
        big.numbers,
        big.positions * INV_BOHR,
        parameters=PBE_PAIR,
        cell=big.cell[:] * INV_BOHR,
        pbc=big.pbc,
        cutoffs=SHORT,
        gradient=False,
    )
    assert primitive.ewald_alpha is not None
    spread = np.abs(supercell.charges.reshape(8, -1) - primitive.charges).max()
    assert 1e-7 < spread < 1e-1
