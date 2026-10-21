"""Parameter resolution, method-name normalisation, and cutoffs."""

import math
from typing import Any

import pytest

from vanderfast import Cutoffs, D3BJParameters, available_methods, resolve_parameters
from vanderfast.errors import ParameterError, UnsupportedFeatureError
from vanderfast.parameters import lookup_method, normalise_method_name
from vanderfast.units import BOHR

PBE: dict[str, Any] = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407, "alp": 14.0}


def test_method_with_atm_sets_s9_and_keeps_pair_coefficients() -> None:
    on = resolve_parameters(method="pbe", damping="bj", atm=True)
    off = resolve_parameters(method="PBE", damping="bj", atm=False)
    assert on.s9 == 1.0
    assert off.s9 == 0.0
    for key, value in PBE.items():
        assert getattr(on, key) == getattr(off, key) == value
    assert on.atm
    assert not off.atm
    assert on == D3BJParameters(**PBE, s9=1.0)
    assert on.source is not None
    assert "pbe" in on.source


def test_explicit_parameters_pass_through_unchanged() -> None:
    params = D3BJParameters(s6=1.0, s8=2.0, a1=0.3, a2=4.0, s9=0.0, alp=16.0)
    assert resolve_parameters(parameters=params) is params
    assert params.atm_exponent == 18.0
    assert params.rs9 == 4.0 / 3.0
    assert params.as_dict()["alp"] == 16.0


def test_pair_and_atm_activity_follow_upstream_epsilon_rules() -> None:
    tiny = 1e-17
    params = D3BJParameters(s6=tiny, s8=-tiny, a1=0.0, a2=0.0, s9=tiny, alp=14.0)
    assert not params.pair
    assert not params.atm
    assert D3BJParameters(s6=0.0, s8=1.0, a1=0.0, a2=0.0, s9=-1.0, alp=1.0).pair


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "pbe", "damping": "bj", "atm": True, "parameters": D3BJParameters(**PBE, s9=0)},
        {"parameters": D3BJParameters(**PBE, s9=0.0), "atm": False},
        {},
        {"method": "pbe", "damping": "bj"},
        {"method": "pbe", "damping": "bj", "atm": 1},
        {"method": "pbe", "damping": "d3bj", "atm": True},
        {"method": "pbe", "atm": True},
    ],
    ids=["mixed", "params+atm", "empty", "no-atm", "int-atm", "spelling", "no-damping"],
)
def test_ambiguous_or_incomplete_input_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ParameterError):
        resolve_parameters(**kwargs)  # type: ignore[arg-type]


def test_parameters_object_type_is_checked() -> None:
    with pytest.raises(ParameterError, match="D3BJParameters"):
        resolve_parameters(parameters={"s6": 1.0})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [("s8", math.nan), ("a2", math.inf), ("alp", 0.0), ("alp", -1.0), ("s6", True), ("s9", "1")],
)
def test_invalid_parameter_values_are_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {**PBE, "s9": 0.0}
    values[field] = value
    with pytest.raises(ParameterError):
        D3BJParameters(**values)  # type: ignore[arg-type]


def test_integer_parameters_are_stored_as_float() -> None:
    params = D3BJParameters(s6=1, s8=1, a1=0, a2=5, s9=0, alp=14)
    assert isinstance(params.s6, float)


@pytest.mark.parametrize(
    ("name", "normalised"),
    [
        ("PBE", b"pbe"),
        ("B3-LYP", b"b3lyp"),
        ("pbe   ", b"pbe"),
        ("  pbe", b"  pbe"),
        ("LCωHPBE", "lcωhpbe".encode()),
        ('"AbC"De', b'"AbC"de'),
    ],
)
def test_normalisation_matches_fortran_get_method_id(name: str, normalised: bytes) -> None:
    assert normalise_method_name(name) == normalised


def test_alias_lookup_and_upstream_case_rules() -> None:
    assert lookup_method("B3LYP5") == "b3lyp"
    assert lookup_method("lcωhpbe") == lookup_method("LCWHPBE") == "lcwhpbe"
    with pytest.raises(ParameterError, match="unknown"):
        lookup_method("LCΩHPBE")  # non-ASCII letters are not case-folded upstream
    with pytest.raises(ParameterError, match="unknown"):
        lookup_method(" pbe")  # leading blanks are significant upstream


def test_unknown_methods_get_suggestions() -> None:
    with pytest.raises(ParameterError, match="did you mean 'pbe0'"):
        lookup_method("pbe00")
    with pytest.raises(ParameterError, match=r"unknown dispersion method 'zzzz'$"):
        lookup_method("zzzz")


def test_methods_without_bj_parameters_are_reported_distinctly() -> None:
    with pytest.raises(ParameterError, match="no D3\\(BJ\\) parameters"):
        resolve_parameters(method="m06", damping="bj", atm=True)


def test_available_methods_are_canonical_and_sorted() -> None:
    names = available_methods()
    assert "pbe" in names
    assert "m06" not in names
    assert list(names) == sorted(names)
    assert len(names) == 161


def test_cutoffs_default_to_reference_values_in_bohr() -> None:
    assert Cutoffs.resolve() == Cutoffs(cn=40.0, disp2=60.0, disp3=40.0)
    mixed = Cutoffs.resolve({"disp2": 20.0})
    assert mixed.cn == 40.0
    assert mixed.disp2 == 20.0 / BOHR
    assert mixed.as_angstrom()["disp2"] == pytest.approx(20.0, rel=1e-15)
    assert mixed.as_bohr()["disp3"] == 40.0


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"width2": 1.0}, UnsupportedFeatureError),
        ({"pair": 1.0}, ParameterError),
        ({"cn": "10"}, ParameterError),
        ({"cn": True}, ParameterError),
        ({"cn": -1.0}, ParameterError),
        ({"disp3": math.inf}, ParameterError),
    ],
)
def test_invalid_cutoffs_are_rejected(overrides: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        Cutoffs.resolve(overrides)  # type: ignore[arg-type]
