"""D4 parameter resolution, functional-name normalisation, and cutoffs."""

import copy
import math
from typing import Any

import pytest

from vanderfast import parameters_d4
from vanderfast.data.d4 import load_d4_methods
from vanderfast.errors import ParameterError, UnsupportedFeatureError
from vanderfast.parameters_d4 import (
    EEQ_CN_CUTOFF,
    D4Cutoffs,
    D4Parameters,
    available_d4_methods,
    lookup_d4_method,
    normalise_d4_method_name,
    resolve_d4_parameters,
)
from vanderfast.units import BOHR

PBE: dict[str, Any] = {"s6": 1.0, "s8": 0.95948085, "a1": 0.38574991, "a2": 4.80688534}


def test_method_defaults_to_atm_and_keeps_pair_coefficients() -> None:
    default = resolve_d4_parameters(method="pbe")
    on = resolve_d4_parameters(method="PBE", atm=True)
    off = resolve_d4_parameters(method="pbe", atm=False)
    assert default == on
    assert on.s9 == 1.0
    assert off.s9 == 0.0
    for key, value in PBE.items():
        assert getattr(on, key) == getattr(off, key) == value
    assert (on.alp, on.ga, on.gc, on.wf) == (16.0, 3.0, 2.0, 6.0)
    assert on.atm
    assert not off.atm
    assert on.source is not None
    assert "bjatm table" in on.source
    assert "bjatm table" in (off.source or "")


def test_dftb_has_separate_pair_only_parameters() -> None:
    on = resolve_d4_parameters(method="dftb3", atm=True)
    off = resolve_d4_parameters(method="dftb3", atm=False)
    assert "bjatm" in (on.source or "")
    assert "(bj table" in (off.source or "")
    assert (on.s8, on.a1, on.a2) == (0.6635015, 0.552324, 4.3537076)
    assert (off.s8, off.a1, off.a2) == (0.4727337, 0.5467502, 4.4955068)


def test_explicit_parameters_pass_through_unchanged() -> None:
    params = D4Parameters(s8=1.0, a1=0.4, a2=5.0, s9=0.0, ga=2.0, gc=1.0)
    assert resolve_d4_parameters(parameters=params) is params
    assert params.atm_exponent == 16.0
    assert params.as_dict()["gc"] == 1.0
    assert params.pair
    assert not D4Parameters(s6=1e-17, s8=0.0, a1=0.0, a2=0.0).pair


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "pbe", "parameters": D4Parameters(**PBE)},
        {"parameters": D4Parameters(**PBE), "atm": True},
        {},
        {"method": "pbe", "atm": 1},
    ],
    ids=["mixed", "params+atm", "empty", "int-atm"],
)
def test_ambiguous_or_incomplete_input_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ParameterError):
        resolve_d4_parameters(**kwargs)  # type: ignore[arg-type]


def test_parameters_object_type_is_checked() -> None:
    with pytest.raises(ParameterError, match="D4Parameters"):
        resolve_d4_parameters(parameters={"s8": 1.0})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("s8", math.nan),
        ("a2", math.inf),
        ("alp", 0.0),
        ("wf", -1.0),
        ("ga", True),
        ("gc", "1"),
    ],
)
def test_invalid_parameter_values_are_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {**PBE}
    values[field] = value
    with pytest.raises(ParameterError):
        D4Parameters(**values)  # type: ignore[arg-type]


def test_integer_parameters_are_stored_as_float() -> None:
    params = D4Parameters(s8=1, a1=0, a2=5, ga=3)
    assert isinstance(params.s8, float)
    assert isinstance(params.ga, float)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PBE", b"pbe"),
        ("B3LYP/def2-TZVP", b"b3lyp"),
        ("wB97X-3c", b"wb97x-3c"),
        ("ωB97X", "ωb97x".encode()),
        ("pbe  ", b"pbe"),
        (" PBE", b" pbe"),
        ("'PBE'x", b"'PBE'x"),
        ("/pbe", b""),
    ],
)
def test_names_normalise_like_dftd4(name: str, expected: bytes) -> None:
    assert normalise_d4_method_name(name) == expected


def test_aliases_resolve_to_canonical_names() -> None:
    assert lookup_d4_method("gga_x_pbe:gga_c_pbe") == "pbe"
    assert lookup_d4_method("r²SCAN-3c") == "r2scan-3c"
    assert lookup_d4_method("B3LYP") == "b3-lyp"
    assert "pbe" in available_d4_methods()
    assert available_d4_methods() == tuple(sorted(available_d4_methods()))


def test_unknown_names_suggest_close_matches() -> None:
    with pytest.raises(ParameterError, match="did you mean"):
        lookup_d4_method("pbe1")
    with pytest.raises(ParameterError, match=r"unknown dispersion method 'zzz'$"):
        lookup_d4_method("zzz")


def test_alias_without_parameters_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    document = copy.deepcopy(load_d4_methods())
    document["aliases"]["made-up"] = "made-up"
    monkeypatch.setattr(parameters_d4, "load_d4_methods", lambda: document)
    with pytest.raises(ParameterError, match="no D4 parameters"):
        lookup_d4_method("made-up")


def test_cutoffs_default_and_resolve_from_angstrom() -> None:
    default = D4Cutoffs.resolve()
    assert default.as_bohr() == {"cn": 30.0, "disp2": 60.0, "disp3": 40.0}
    assert default.cn_radius() == 30.0
    assert D4Cutoffs(cn=20.0).cn_radius() == EEQ_CN_CUTOFF
    resolved = D4Cutoffs.resolve({"disp2": 20.0})
    assert resolved.disp2 == 20.0 / BOHR
    assert resolved.cn == 30.0
    assert resolved.as_angstrom()["disp2"] == pytest.approx(20.0, rel=1e-15)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"width2": 1.0}, UnsupportedFeatureError),
        ({"cn_eeq": 10.0}, ParameterError),
        ({"cn": "10"}, ParameterError),
        ({"cn": True}, ParameterError),
        ({"cn": -1.0}, ParameterError),
    ],
)
def test_invalid_cutoffs_are_rejected(overrides: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        D4Cutoffs.resolve(overrides)
