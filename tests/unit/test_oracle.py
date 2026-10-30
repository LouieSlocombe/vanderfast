"""The NumPy oracle: validation, images, weights, counting, and derivatives."""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms
from support.edges import edge_keys, has_duplicates, reversed_edge_keys
from support.fd import agrees_over_sweep, position_gradient, strain_derivative
from support.inputs import SHORT_D3, bohr

from vanderfast import _oracle, resolve_parameters
from vanderfast.data import load_reference_tables, pair_index
from vanderfast.errors import InputError
from vanderfast.parameters import Cutoffs, D3BJParameters
from vanderfast.testing import structures

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
PBE = resolve_parameters(method="pbe", damping="bj", atm=False)


@pytest.mark.parametrize(
    ("numbers", "positions", "cell", "pbc", "match"),
    [
        ([[1]], [[0, 0, 0]], None, (False,) * 3, "one-dimensional"),
        ([1.0], [[0, 0, 0]], None, (False,) * 3, "integer"),
        ([0], [[0, 0, 0]], None, (False,) * 3, "1..103"),
        ([104], [[0, 0, 0]], None, (False,) * 3, "1..103"),
        ([1], [[0, 0]], None, (False,) * 3, "shape"),
        ([1], [[0, 0, np.nan]], None, (False,) * 3, "finite"),
        ([1], [[0, 0, 0]], np.eye(2), (False,) * 3, "3x3"),
        ([1], [[0, 0, 0]], np.eye(3), (True, False), "three"),
        ([1], [[0, 0, 0]], np.zeros((3, 3)), (True, False, False), "independent"),
        ([1], [[0, 0, 0]], [[1, 0, 0], [2, 0, 0], [0, 0, 1]], (True, True, False), "independent"),
    ],
)
def test_geometry_validation(
    numbers: object, positions: object, cell: object, pbc: object, match: str
) -> None:
    with pytest.raises(InputError, match=match):
        _oracle.make_geometry(numbers, positions, cell, pbc)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "pbc", [(True, False, False), (False, True, False), (True, True, False), (False, True, True)]
)
def test_search_basis_completes_with_orthonormal_vectors(pbc: tuple[bool, bool, bool]) -> None:
    cell = np.array([[3.0, 0.2, 0.1], [0.5, 4.0, 0.3], [0.7, 0.4, 5.0]])
    geometry = _oracle.make_geometry([1], [[0.0, 0.0, 0.0]], cell, pbc)
    basis = _oracle.search_basis(geometry)
    active = np.flatnonzero(pbc)
    inactive = np.flatnonzero(~np.asarray(pbc))
    np.testing.assert_array_equal(basis[active], cell[active])
    for k in inactive:
        assert np.linalg.norm(basis[k]) == pytest.approx(1.0)
        for a in active:
            assert abs(basis[k] @ cell[a]) < 1e-12
    assert abs(np.linalg.det(basis)) > 0.0


def test_molecules_have_no_images() -> None:
    geometry = _oracle.make_geometry([1, 1], [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(_oracle.search_basis(geometry), np.eye(3))
    np.testing.assert_array_equal(_oracle.image_ranges(geometry, 10.0), [0, 0, 0])
    empty = _oracle.make_geometry(np.zeros(0, dtype=int), np.zeros((0, 3)))
    assert len(_oracle.directed_edges(empty, 10.0)) == 0


def test_image_ranges_are_complete_for_unwrapped_positions() -> None:
    cell = np.array([[5.0, 0.0, 0.0], [1.5, 4.5, 0.0], [0.5, 1.0, 4.0]])
    wrapped = np.array([[0.2, 0.3, 0.1], [2.5, 2.0, 1.5]])
    shifted = wrapped + np.array([3, -2, 0]) @ cell
    shifted[1] += np.array([-4, 0, 5]) @ cell
    geometry_w = _oracle.make_geometry([6, 8], wrapped, cell, (True, True, True))
    geometry_u = _oracle.make_geometry([6, 8], shifted, cell, (True, True, True))
    edges_w = _oracle.directed_edges(geometry_w, 12.0)
    edges_u = _oracle.directed_edges(geometry_u, 12.0)
    assert len(edges_w) == len(edges_u)
    np.testing.assert_allclose(np.sort(edges_w.r2), np.sort(edges_u.r2), rtol=1e-12)
    ranges = _oracle.image_ranges(geometry_u, 12.0)
    assert np.all(ranges >= _oracle.image_ranges(geometry_w, 12.0))


def test_directed_edges_have_exact_reverse_and_no_self_edge() -> None:
    numbers, positions, cell, pbc = bohr(structures.rocksalt_primitive())
    geometry = _oracle.make_geometry(numbers, positions, cell, pbc)
    edges = _oracle.directed_edges(geometry, 15.0)
    keys = edge_keys(edges.i, edges.j, edges.shift)
    assert not has_duplicates(keys)
    np.testing.assert_array_equal(reversed_edge_keys(edges.i, edges.j, edges.shift), keys)
    same_atom = edges.i == edges.j
    assert not np.any(same_atom & np.all(edges.shift == 0, axis=1))
    assert same_atom.any()  # images of the same atom are kept


def test_weights_are_normalised_with_correct_derivatives() -> None:
    numbers = np.array([1, 6, 26, 91])
    cn = np.array([0.4, 2.7, 3.1, 4.2])
    w, dw = _oracle.reference_weights(numbers, cn)
    tables = load_reference_tables()
    for index, z in enumerate(numbers):
        nref = int(tables.nref[z])
        assert w[index, :nref].sum() == pytest.approx(1.0, rel=1e-14)
        assert np.all(w[index, nref:] == 0.0)
    h = 1e-6
    w_plus, _ = _oracle.reference_weights(numbers, cn + h)
    w_minus, _ = _oracle.reference_weights(numbers, cn - h)
    np.testing.assert_allclose(dw, (w_plus - w_minus) / (2 * h), atol=1e-8)


def test_exceptional_weights_select_maximum_reference() -> None:
    w, dw = _oracle.reference_weights([1, 6], [60.0, 80.0])
    np.testing.assert_array_equal(w[0, :2], [1.0, 0.0])  # H reference CNs are (0.9118, 0)
    np.testing.assert_array_equal(w[1, :5], [0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(dw, 0.0)


def test_tied_maximum_references_all_receive_weight_one() -> None:
    tables = load_reference_tables()
    ref_cn = tables.ref_cn_flat.copy()
    start = int(tables.ref_offset[6])
    ref_cn[start + 3] = ref_cn[start + 4]  # tie the two largest carbon references
    tied = dataclasses.replace(tables, ref_cn_flat=ref_cn)
    w, _ = _oracle.reference_weights([6], [90.0], tied)
    np.testing.assert_array_equal(w[0, :5], [0.0, 0.0, 0.0, 1.0, 1.0])


def test_c6_coefficients_are_symmetric_with_endpoint_derivatives() -> None:
    numbers = np.array([1, 6, 6, 8])
    cn = np.array([0.9, 3.2, 2.1, 1.7])
    w, dw = _oracle.reference_weights(numbers, cn)
    c6, dc6 = _oracle.c6_coefficients(numbers, w, dw)
    assert dc6 is not None
    np.testing.assert_allclose(c6, c6.T, rtol=1e-15)
    h = 1e-6
    for i in range(numbers.size):
        bump = np.zeros(numbers.size)
        bump[i] = h
        w_p, _ = _oracle.reference_weights(numbers, cn + bump)
        w_m, _ = _oracle.reference_weights(numbers, cn - bump)
        c6_p, _ = _oracle.c6_coefficients(numbers, w_p, None)
        c6_m, _ = _oracle.c6_coefficients(numbers, w_m, None)
        numeric = (c6_p - c6_m) / (2 * h)
        # Row i depends on CN_i through the first index (and the diagonal twice).
        expected = dc6[i].copy()
        expected[i] *= 2.0
        np.testing.assert_allclose(numeric[i], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "build",
    [
        structures.water_dimer,
        lambda: structures.simple_cubic(18, 3.4),
        structures.rocksalt_primitive,
        structures.graphene_slab,
        structures.carbon_wire,
    ],
    ids=["water_dimer", "one_atom", "two_atom", "slab", "wire"],
)
def test_directed_and_upstream_counting_agree(build: Callable[[], Atoms]) -> None:
    numbers, positions, cell, pbc = bohr(build())
    kwargs = {"parameters": PBE_ATM, "cell": cell, "pbc": pbc, "cutoffs": SHORT_D3}
    ours = _oracle.evaluate(numbers, positions, **kwargs)
    upstream = _oracle.evaluate(
        numbers, positions, pair_counting="upstream", atm_counting="upstream", **kwargs
    )
    assert ours.energy == pytest.approx(upstream.energy, rel=1e-12, abs=1e-18)
    assert ours.energy_atm == pytest.approx(upstream.energy_atm, rel=1e-11, abs=1e-18)
    np.testing.assert_allclose(ours.atom_energies, upstream.atom_energies, rtol=1e-10, atol=1e-16)
    assert ours.gradient is not None
    assert upstream.gradient is not None
    np.testing.assert_allclose(ours.gradient, upstream.gradient, rtol=1e-9, atol=1e-15)
    assert ours.strain_derivative is not None
    assert upstream.strain_derivative is not None
    np.testing.assert_allclose(
        ours.strain_derivative, upstream.strain_derivative, rtol=1e-10, atol=1e-15
    )


def test_anchored_triangles_count_each_triangle_three_times() -> None:
    numbers, positions, _, _ = bohr(structures.equilateral_trimer(18, 3.5))
    ours = _oracle.evaluate(numbers, positions, parameters=PBE_ATM)
    upstream = _oracle.evaluate(numbers, positions, parameters=PBE_ATM, atm_counting="upstream")
    assert ours.counts["atm_accepted"] == 3
    assert upstream.counts["atm_accepted"] == 1
    assert ours.counts["atm_candidates"] == 3
    assert ours.energy_atm > 0.0


def test_energy_only_path_matches_gradient_path() -> None:
    numbers, positions, cell, pbc = bohr(structures.skewed_triclinic())
    kwargs = {"parameters": PBE_ATM, "cell": cell, "pbc": pbc, "cutoffs": SHORT_D3}
    full = _oracle.evaluate(numbers, positions, **kwargs)
    energy = _oracle.evaluate(numbers, positions, gradient=False, **kwargs)
    assert energy.energy == full.energy
    assert energy.gradient is None
    assert energy.strain_derivative is None
    assert energy.g is None
    assert energy.dweights is None
    assert energy.dc6 is None


def test_empty_and_single_atom_molecules_have_zero_dispersion() -> None:
    empty = _oracle.evaluate(np.zeros(0, dtype=int), np.zeros((0, 3)), parameters=PBE_ATM)
    assert empty.energy == 0.0
    assert empty.gradient is not None
    assert empty.gradient.shape == (0, 3)
    single = _oracle.evaluate([18], [[1.0, 2.0, 3.0]], parameters=PBE_ATM)
    assert single.energy == 0.0
    assert single.counts["pair_edges"] == 0


def test_inactive_terms_are_skipped() -> None:
    numbers, positions, _, _ = bohr(structures.water_dimer())
    atm_only = D3BJParameters(s6=0.0, s8=0.0, a1=0.4, a2=5.0, s9=1.0, alp=14.0)
    result = _oracle.evaluate(numbers, positions, parameters=atm_only)
    assert result.energy_pair == 0.0
    assert result.counts["pair_edges"] == 0
    assert result.energy == result.energy_atm


def test_atm_c6_policy_only_differs_for_tiny_coefficients() -> None:
    tables = load_reference_tables()
    c6 = tables.c6_flat.copy()
    start = int(tables.c6_offset[pair_index(18, 18)])
    c6[start] = 1e-20  # argon has a single reference, so C6(Ar, Ar) becomes tiny
    tiny = dataclasses.replace(tables, c6_flat=c6)
    numbers, positions, _, _ = bohr(structures.equilateral_trimer(18, 3.5))
    skip = _oracle.evaluate(numbers, positions, parameters=PBE_ATM, tables=tiny)
    skip_energy = _oracle.evaluate(
        numbers, positions, parameters=PBE_ATM, tables=tiny, gradient=False
    )
    keep = _oracle.evaluate(
        numbers, positions, parameters=PBE_ATM, tables=tiny, gradient=False, atm_c6_policy="keep"
    )
    assert skip.counts["atm_c6_skipped"] == 3
    assert skip.counts["atm_accepted"] == 0
    assert skip.energy == skip_energy.energy
    assert keep.counts["atm_accepted"] == 3
    assert keep.energy != skip.energy


def test_gradient_matches_finite_differences() -> None:
    numbers, positions, _, _ = bohr(structures.scalene_trimer())
    result = _oracle.evaluate(numbers, positions, parameters=PBE_ATM)

    def energy(x: np.ndarray) -> float:
        return _oracle.evaluate(numbers, x, parameters=PBE_ATM, gradient=False).energy

    assert result.gradient is not None
    ok, errors = agrees_over_sweep(
        result.gradient, lambda h: position_gradient(energy, positions, h), 1e-10
    )
    assert ok, errors


def test_strain_derivative_matches_finite_differences() -> None:
    numbers, positions, cell, pbc = bohr(structures.skewed_triclinic())
    cutoffs = Cutoffs(cn=12.0, disp2=16.0, disp3=12.0)
    result = _oracle.evaluate(
        numbers, positions, parameters=PBE_ATM, cell=cell, pbc=pbc, cutoffs=cutoffs
    )
    assert cell is not None

    def energy(x: np.ndarray, h: np.ndarray) -> float:
        return _oracle.evaluate(
            numbers, x, parameters=PBE_ATM, cell=h, pbc=pbc, cutoffs=cutoffs, gradient=False
        ).energy

    assert result.strain_derivative is not None
    ok, errors = agrees_over_sweep(
        result.strain_derivative,
        lambda h: strain_derivative(energy, positions, cell, h),
        1e-9,
    )
    assert ok, errors
