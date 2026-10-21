"""D3(BJ) parameter resolution and cutoff configuration.

Parameters are resolved once, either from a method name in the pinned
simple-dftd3 v1.6.0 table or from a fully explicit parameter object. Mixing
the two is rejected. ATM must always be chosen explicitly: unlike the upstream
ASE calculator, vanderfast has no implicit three-body default.
"""

import difflib
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Final, Self, cast

from vanderfast import units
from vanderfast.data import load_methods
from vanderfast.errors import ParameterError, UnsupportedFeatureError

DBL_EPSILON: Final = sys.float_info.epsilon

DAMPING_BJ: Final = "bj"


@dataclass(frozen=True, slots=True, kw_only=True)
class D3BJParameters:
    """A fully resolved D3(BJ) parameter set in atomic units.

    Attributes:
        s6: Scaling of the C6 (dipole-dipole) term.
        s8: Scaling of the C8 (dipole-quadrupole) term.
        a1: Scaling of the BJ critical radius.
        a2: Offset of the BJ critical radius, in bohr.
        s9: Scaling of the ATM three-body term; zero disables ATM.
        alp: Public ATM damping exponent. The ATM kernel uses ``alp + 2``.
        source: Free-text provenance of the values.
    """

    s6: float
    s8: float
    a1: float
    a2: float
    s9: float
    alp: float
    source: str | None = field(default=None, compare=False)

    rs9: ClassVar[float] = 4.0 / 3.0
    """ATM radius scaling fixed by the reference implementation."""

    def __post_init__(self) -> None:
        """Validate that every value is finite and the ATM exponent is positive."""
        for name in ("s6", "s8", "a1", "a2", "s9", "alp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"parameter {name} must be a real number, got {value!r}"
                raise ParameterError(msg)
            if not math.isfinite(value):
                msg = f"parameter {name} must be finite, got {value!r}"
                raise ParameterError(msg)
            object.__setattr__(self, name, float(value))
        if self.alp <= 0.0:
            msg = f"parameter alp must be positive, got {self.alp!r}"
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
        """Effective ATM damping exponent used by the BJ family (``alp + 2``)."""
        return self.alp + 2.0

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a JSON-serialisable mapping."""
        return asdict(self)


def normalise_method_name(name: str) -> bytes:
    """Normalise a method name exactly as simple-dftd3's ``get_method_id`` does.

    The name is treated as UTF-8 bytes: ``-`` is removed, ASCII ``A``-``Z``
    outside quotes are lowercased, and trailing blanks are trimmed. Leading
    blanks are significant and non-ASCII letters are not case-folded.
    """
    raw = name.encode("utf-8").replace(b"-", b"").rstrip(b" ")
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


def available_methods() -> tuple[str, ...]:
    """Return the canonical names of all methods with D3(BJ) parameters."""
    return tuple(sorted(load_methods()["methods"]))


def lookup_method(name: str) -> str:
    """Return the canonical method name for an alias.

    Raises:
        ParameterError: If the name is unknown or has no D3(BJ) parameters.
    """
    table = load_methods()
    key = normalise_method_name(name)
    aliases: dict[str, str] = table["aliases"]
    for alias, canonical in aliases.items():
        if alias.encode("utf-8") == key:
            if canonical not in table["methods"]:
                msg = f"no D3(BJ) parameters for method {name!r} in simple-dftd3 v1.6.0"
                raise ParameterError(msg)
            return canonical
    suggestions = difflib.get_close_matches(key.decode("utf-8", "replace"), aliases, n=3)
    hint = f"; did you mean {', '.join(map(repr, suggestions))}?" if suggestions else ""
    msg = f"unknown dispersion method {name!r}{hint}"
    raise ParameterError(msg)


def resolve_parameters(
    *,
    method: str | None = None,
    damping: str | None = None,
    atm: bool | None = None,
    parameters: D3BJParameters | None = None,
) -> D3BJParameters:
    """Resolve a complete D3(BJ) parameter set.

    Exactly one of two forms is accepted:

    1. ``method``, ``damping="bj"`` and an explicit boolean ``atm``. ATM on
       sets ``s9 = 1``; ATM off sets ``s9 = 0``. The pair coefficients are the
       same in both cases (simple-dftd3 v1.6.0 has no pair-only refits).
    2. ``parameters``, a fully explicit :class:`D3BJParameters`, with no other
       argument.

    Raises:
        ParameterError: For ambiguous, incomplete, or unknown input.
    """
    if parameters is not None:
        if method is not None or damping is not None or atm is not None:
            msg = "pass either explicit parameters or method/damping/atm, not both"
            raise ParameterError(msg)
        if not isinstance(cast("object", parameters), D3BJParameters):
            msg = f"parameters must be D3BJParameters, got {type(parameters).__name__}"
            raise ParameterError(msg)
        return parameters
    if method is None:
        msg = "a method name or explicit parameters are required"
        raise ParameterError(msg)
    if damping != DAMPING_BJ:
        msg = f"damping must be {DAMPING_BJ!r} (the only implemented family), got {damping!r}"
        raise ParameterError(msg)
    if not isinstance(atm, bool):
        msg = "atm must be given explicitly as True or False"
        raise ParameterError(msg)
    canonical = lookup_method(method)
    table = load_methods()
    entry = table["methods"][canonical]
    return D3BJParameters(
        s6=entry["s6"],
        s8=entry["s8"],
        a1=entry["a1"],
        a2=entry["a2"],
        s9=1.0 if atm else 0.0,
        alp=entry["alp"],
        source=f"simple-dftd3 v1.6.0 param.f90: {canonical} (atm={atm})",
    )


@dataclass(frozen=True, slots=True)
class Cutoffs:
    """Real-space cutoffs in bohr for coordination numbers, pairs, and triples."""

    cn: float = 40.0
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
            UnsupportedFeatureError: For smoothing widths (deferred).
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


__all__ = [
    "DAMPING_BJ",
    "DBL_EPSILON",
    "Cutoffs",
    "D3BJParameters",
    "available_methods",
    "lookup_method",
    "normalise_method_name",
    "resolve_parameters",
]
