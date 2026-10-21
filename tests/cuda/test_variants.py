"""Optimised kernels (owner-only, contracted C6, canonical ATM) vs the literal baseline."""

import numpy as np
import pytest
from ase import Atoms
from support import native_helpers as nh

from vanderfast import resolve_parameters
from vanderfast.parameters import Cutoffs, D3BJParameters
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE = resolve_parameters(method="pbe", damping="bj", atm=False)
PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)
CUSTOM_ALP = D3BJParameters(s6=0.95, s8=1.3, a1=0.35, a2=4.6, s9=1.0, alp=11.0)
SHORT = Cutoffs(cn=15.0, disp2=20.0, disp3=15.0)

SYSTEMS = {
    "benzene_dimer": structures.benzene_dimer,
    "coverage": lambda: structures.coverage_clusters()[3],
    "co2_crystal": structures.co2_crystal,
    "skewed": structures.skewed_triclinic,
    "slab": structures.graphene_slab,
    "one_atom": structures.one_atom_triclinic,
}


def _inputs(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    cell = atoms.cell[:] * INV_BOHR if atoms.pbc.any() else None
    return atoms.numbers, atoms.positions * INV_BOHR, cell, atoms.pbc


@pytest.mark.parametrize("parameters", [PBE, PBE_ATM, CUSTOM_ALP], ids=["pair", "atm", "alp11"])
@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_optimized_kernels_match_the_literal_baseline(
    name: str, parameters: D3BJParameters
) -> None:
    numbers, x, cell, pbc = _inputs(SYSTEMS[name]())
    literal = nh.run(nh.context(numbers, parameters, SHORT, kernels="literal"), x, cell, pbc)
    optimized = nh.run(nh.context(numbers, parameters, SHORT), x, cell, pbc)
    assert optimized["energy"] == pytest.approx(literal["energy"], rel=1e-12, abs=1e-16)
    assert optimized["energy_atm"] == pytest.approx(literal["energy_atm"], rel=1e-12, abs=1e-16)
    scale = max(1.0, float(np.abs(literal["gradient"]).max()))
    np.testing.assert_allclose(
        optimized["gradient"], literal["gradient"], rtol=0, atol=1e-12 * scale
    )
    np.testing.assert_allclose(optimized["virial"], literal["virial"], rtol=1e-11, atol=1e-15)
    if parameters.atm:
        assert 3 * optimized["info"]["atm_accepted"] == literal["info"]["atm_accepted"]


def test_atomic_energies_keep_the_upstream_split() -> None:
    numbers, x, cell, pbc = _inputs(structures.co2_crystal())
    flags = tuple(bool(p) for p in pbc)
    literal = nh.context(numbers, PBE_ATM, SHORT, kernels="literal")
    optimized = nh.context(numbers, PBE_ATM, SHORT)
    a = literal.evaluate(x, cell, flags, atom_energies=True)["atom_energies"]
    b = optimized.evaluate(x, cell, flags, atom_energies=True)["atom_energies"]
    np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-16)


def test_owner_only_pair_gradients_are_deterministic() -> None:
    numbers, x, cell, pbc = _inputs(structures.co2_crystal())
    ctx = nh.context(numbers, PBE, SHORT)
    first = nh.run(ctx, x, cell, pbc)
    for _ in range(5):
        again = nh.run(ctx, x, cell, pbc)
        np.testing.assert_array_equal(again["gradient"], first["gradient"])
        np.testing.assert_array_equal(again["virial"], first["virial"])


def test_unknown_kernel_variant_is_rejected() -> None:
    from vanderfast.errors import InputError  # noqa: PLC0415

    with pytest.raises(InputError, match="kernels"):
        nh.context([1], PBE, kernels="fast")
