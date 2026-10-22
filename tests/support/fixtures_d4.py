"""Loading of the committed dftd4 4.2.0 reference fixtures."""

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "dftd4-4.2.0"
FAMILIES = (
    "actinides",
    "cell_dependence",
    "coverage",
    "dimers",
    "eeq_stress",
    "extreme_cn",
    "ions",
    "molecules",
    "partial_pbc",
    "periodic",
    "trimers",
    "zeta",
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class D4FixtureCase:
    """One dftd4 reference case in atomic units."""

    family: str
    name: str
    parameter_set: str
    numbers: npt.NDArray[np.int64]
    positions: FloatArray
    cell: FloatArray | None
    pbc: npt.NDArray[np.bool_]
    charge: float
    parameters: D4Parameters
    cutoffs: D4Cutoffs
    energy: float
    gradient: FloatArray | None
    virial: FloatArray | None
    charges: FloatArray
    cn: FloatArray
    polarizabilities: FloatArray
    c6: FloatArray | None
    known_issue: str | None

    @property
    def id(self) -> str:
        """Stable pytest id."""
        return f"{self.family}/{self.name}/{self.parameter_set}"

    @property
    def periodic(self) -> bool:
        """Whether any direction is periodic (dftd4 then treats all three as periodic)."""
        return bool(self.pbc.any())


def _array(raw: Any, shape: tuple[int, ...] | None = None) -> FloatArray | None:
    if raw is None:
        return None
    values = np.array(raw, dtype=np.float64)
    return values if shape is None else values.reshape(shape)


def _case(family: str, raw: dict[str, Any]) -> D4FixtureCase:
    n = len(raw["numbers"])
    charges = _array(raw["charges"])
    cn = _array(raw["cn"])
    alpha = _array(raw["polarizabilities"])
    if charges is None or cn is None or alpha is None:
        msg = f"fixture {raw['name']} lacks properties"
        raise ValueError(msg)
    return D4FixtureCase(
        family=family,
        name=raw["name"],
        parameter_set=raw["parameter_set"],
        numbers=np.array(raw["numbers"], dtype=np.int64),
        positions=np.array(raw["positions_bohr"], dtype=np.float64).reshape(-1, 3),
        cell=_array(raw["cell_bohr"]),
        pbc=np.array(raw["pbc"], dtype=bool),
        charge=float(raw["charge"]),
        parameters=D4Parameters(**raw["parameters"]),
        cutoffs=D4Cutoffs(**raw["cutoffs_bohr"]),
        energy=float(raw["energy"]),
        gradient=_array(raw["gradient"], (n, 3)),
        virial=_array(raw["virial"], (3, 3)),
        charges=charges,
        cn=cn,
        polarizabilities=alpha,
        c6=_array(raw["c6"], (n, n)),
        known_issue=raw.get("known_issue"),
    )


@functools.cache
def load_header(family: str) -> dict[str, Any]:
    """Return the header line of a fixture family."""
    with (FIXTURE_DIR / f"{family}.jsonl").open(encoding="utf-8") as handle:
        header: dict[str, Any] = json.loads(handle.readline())
    return header


@functools.cache
def load_family(family: str) -> tuple[D4FixtureCase, ...]:
    """Return every case of a fixture family."""
    lines = (FIXTURE_DIR / f"{family}.jsonl").read_text(encoding="utf-8").splitlines()
    return tuple(_case(family, json.loads(line)) for line in lines[1:])


def all_cases() -> list[D4FixtureCase]:
    """Return every fixture case of every family."""
    return [case for family in FAMILIES for case in load_family(family)]


def find_case(family: str, name: str, parameter_set: str) -> D4FixtureCase:
    """Return one named case."""
    for case in load_family(family):
        if case.name == name and case.parameter_set == parameter_set:
            return case
    msg = f"no fixture {family}/{name}/{parameter_set}"
    raise KeyError(msg)
