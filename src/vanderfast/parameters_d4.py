"""DFT-D4 parameter resolution and cutoff configuration.

Parameters are resolved once, either from a functional name in the pinned
dftd4 v4.2.0 table or from a fully explicit parameter object; mixing the two
is rejected. Unlike the D3 resolver, ``atm`` defaults to ``True``: the D4
damping parameters were fitted together with the three-body term, and this
matches ``dftd4.interface.DampingParam(method=..., atm=True)``.
"""

import difflib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Final, Self, cast

from vanderfast import units
from vanderfast.data.d4 import load_d4_methods
from vanderfast.errors import ParameterError, UnsupportedFeatureError
from vanderfast.parameters import DBL_EPSILON

EEQ_CN_CUTOFF: Final = 25.0
"""Cutoff of the EEQ coordination number in bohr (fixed by multicharge)."""

_FIELDS: Final = ("s6", "s8", "a1", "a2", "s9", "alp", "ga", "gc", "wf")


@dataclass(frozen=True, slots=True, kw_only=True)
class D4Parameters:
    """A fully resolved D4 parameter set in atomic units.

    Attributes:
        s8: Scaling of the C8 (dipole-quadrupole) term.
        a1: Scaling of the rational-damping critical radius.
        a2: Offset of the rational-damping critical radius, in bohr.
        s6: Scaling of the C6 (dipole-dipole) term.
        s9: Scaling of the ATM three-body term; zero disables ATM.
        alp: ATM damping exponent, used as is (``(r0/r)^(alp/3)``).
        ga: Charge-scaling height of the reference polarizabilities.
        gc: Charge-scaling steepness (multiplies the chemical hardness).
        wf: Weighting factor of the Gaussian coordination-number weights.
        source: Free-text provenance of the values.
    """

    s8: float
    a1: float
    a2: float
    s6: float = 1.0
    s9: float = 1.0
    alp: float = 16.0
    ga: float = 3.0
    gc: float = 2.0
    wf: float = 6.0
    source: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate that every value is finite and the exponents are positive."""
        for name in _FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"parameter {name} must be a real number, got {value!r}"
                raise ParameterError(msg)
            if not math.isfinite(value):
                msg = f"parameter {name} must be finite, got {value!r}"
                raise ParameterError(msg)
            object.__setattr__(self, name, float(value))
        for name in ("alp", "wf"):
            if getattr(self, name) <= 0.0:
                msg = f"parameter {name} must be positive, got {getattr(self, name)!r}"
                raise ParameterError(msg)

    @property
    def atm(self) -> bool:
        """Whether the ATM term contributes (``|s9| >= DBL_EPSILON``)."""
        return abs(self.s9) >= DBL_EPSILON

    @property
    def pair(self) -> bool:
        """Whether the pair term contributes (not both ``|s6|`` and ``|s8|`` below epsilon)."""
        return not (abs(self.s6) < DBL_EPSILON and abs(self.s8) < DBL_EPSILON)

    @property
    def atm_exponent(self) -> float:
        """Effective ATM damping exponent; D4 uses ``alp`` directly."""
        return self.alp

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a JSON-serialisable mapping."""
        return asdict(self)


def normalise_d4_method_name(name: str) -> bytes:
    """Normalise a functional name exactly as dftd4's ``get_rational_damping_name`` does.

    The name is treated as UTF-8 bytes: everything from the first ``/`` on is
    dropped, ASCII ``A``-``Z`` outside quotes are lowercased, and trailing
    blanks are trimmed. Unlike D3, hyphens are significant; leading blanks are
    significant and non-ASCII letters are not case-folded.
    """
    raw = name.encode("utf-8").split(b"/", 1)[0]
    out = bytearray(raw)
    quote = 0
    for index, byte in enumerate(raw):
        if quote == 0 and byte in {34, 39}:
            quote = byte
            continue
        if quote and byte == quote:
            quote = 0
            continue
        if quote:
            continue
        if 65 <= byte <= 90:  # noqa: PLR2004
            out[index] = byte + 32
    return bytes(out).rstrip(b" ")


def available_d4_methods() -> tuple[str, ...]:
    """Return the canonical names of all functionals with D4 parameters."""
    tables = load_d4_methods()["tables"]
    return tuple(sorted(set(tables["bj"]) | set(tables["bjatm"])))


def lookup_d4_method(name: str) -> str:
    """Return the canonical functional name for an alias.

    Raises:
        ParameterError: If the name is unknown or has no D4 parameters.
    """
    document = load_d4_methods()
    key = normalise_d4_method_name(name)
    aliases: dict[str, str] = document["aliases"]
    for alias, canonical in aliases.items():
        if alias.encode("utf-8") == key:
            if canonical not in available_d4_methods():
                msg = f"no D4 parameters for functional {name!r} in dftd4 v4.2.0"
                raise ParameterError(msg)
            return canonical
    suggestions = difflib.get_close_matches(key.decode("utf-8", "replace"), aliases, n=3)
    hint = f"; did you mean {', '.join(map(repr, suggestions))}?" if suggestions else ""
    msg = f"unknown dispersion method {name!r}{hint}"
    raise ParameterError(msg)


def resolve_d4_parameters(
    *,
    method: str | None = None,
    atm: bool | None = None,
    parameters: D4Parameters | None = None,
) -> D4Parameters:
    """Resolve a complete D4 parameter set.

    Exactly one of two forms is accepted:

    1. ``method`` and optionally ``atm`` (default ``True``). ATM on sets
       ``s9 = 1`` and takes the coefficients fitted with ATM, falling back to
       the pair-only table; ATM off sets ``s9 = 0`` and prefers the pair-only
       table (only the DFTB parametrisations have one), as dftd4 does.
    2. ``parameters``, a fully explicit :class:`D4Parameters`, with no other
       argument.

    Raises:
        ParameterError: For ambiguous, incomplete, or unknown input.
    """
    if parameters is not None:
        if method is not None or atm is not None:
            msg = "pass either explicit parameters or method/atm, not both"
            raise ParameterError(msg)
        if not isinstance(cast("object", parameters), D4Parameters):
            msg = f"parameters must be D4Parameters, got {type(parameters).__name__}"
            raise ParameterError(msg)
        return parameters
    if method is None:
        msg = "a method name or explicit parameters are required"
        raise ParameterError(msg)
    if atm is None:
        atm = True
    if not isinstance(cast("object", atm), bool):
        msg = f"atm must be True or False, got {atm!r}"
        raise ParameterError(msg)
    canonical = lookup_d4_method(method)
    tables = load_d4_methods()["tables"]
    order = ("bjatm", "bj") if atm else ("bj", "bjatm")
    table = next(name for name in order if canonical in tables[name])
    entry = tables[table][canonical]
    return D4Parameters(
        s6=entry["s6"],
        s8=entry["s8"],
        a1=entry["a1"],
        a2=entry["a2"],
        s9=1.0 if atm else 0.0,
        alp=entry["alp"],
        source=f"dftd4 v4.2.0 param.f90: {canonical} ({table} table, atm={atm})",
    )


@dataclass(frozen=True, slots=True)
class D4Cutoffs:
    """Real-space cutoffs in bohr for the D4 coordination number, pairs, and triples.

    The EEQ coordination number uses a fixed 25 bohr cutoff (``EEQ_CN_CUTOFF``)
    and the EEQ Coulomb interaction has no real-space cutoff.
    """

    cn: float = 30.0
    disp2: float = 60.0
    disp3: float = 40.0

    KEYS: ClassVar[tuple[str, ...]] = ("cn", "disp2", "disp3")

    def __post_init__(self) -> None:
        """Validate that every cutoff is finite and positive."""
        for name in self.KEYS:
            value = getattr(self, name)
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0.0):
                msg = f"cutoff {name} must be a positive finite number, got {value!r}"
                raise ParameterError(msg)
            object.__setattr__(self, name, float(value))

    @classmethod
    def resolve(cls, overrides: Mapping[str, float] | None = None) -> Self:
        """Build cutoffs from optional overrides given in Ångström.

        Omitted keys keep the exact reference defaults in bohr.

        Raises:
            UnsupportedFeatureError: For smoothing widths (not supported).
            ParameterError: For unknown keys or invalid values.
        """
        if overrides is None:
            return cls()
        values: dict[str, float] = {}
        for key, value in overrides.items():
            if key in {"width2", "width3"}:
                msg = "smooth cutoff widths are not supported in this release"
                raise UnsupportedFeatureError(msg)
            if key not in cls.KEYS:
                msg = f"unknown cutoff key {key!r}; expected one of {cls.KEYS}"
                raise ParameterError(msg)
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"cutoff {key} must be a number in Ångström, got {value!r}"
                raise ParameterError(msg)
            values[key] = units.cutoff_to_bohr(value)
        return cls(**values)

    def as_angstrom(self) -> dict[str, float]:
        """Return the cutoffs in Ångström."""
        return {key: units.cutoff_to_angstrom(getattr(self, key)) for key in self.KEYS}

    def as_bohr(self) -> dict[str, float]:
        """Return the cutoffs in bohr."""
        return {key: getattr(self, key) for key in self.KEYS}

    def cn_radius(self) -> float:
        """Return the neighbour radius covering both coordination numbers, in bohr."""
        return max(self.cn, EEQ_CN_CUTOFF)


__all__ = [
    "EEQ_CN_CUTOFF",
    "D4Cutoffs",
    "D4Parameters",
    "available_d4_methods",
    "lookup_d4_method",
    "normalise_d4_method_name",
    "resolve_d4_parameters",
]
