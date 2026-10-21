"""Loading of the committed dftd3 1.6.0 reference fixtures."""

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vanderfast.parameters import Cutoffs, D3BJParameters

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "dftd3-1.6.0"
FAMILIES = (
    "coverage",
    "dimers",
    "extreme_cn",
    "molecules",
    "partial_pbc",
    "periodic",
    "trimers",
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class FixtureCase:
    """One reference case in atomic units."""

    family: str
    name: str
    parameter_set: str
    numbers: npt.NDArray[np.int64]
    positions: FloatArray
    cell: FloatArray | None
    pbc: npt.NDArray[np.bool_]
    parameters: D3BJParameters
    cutoffs: Cutoffs
    energy: float
    gradient: FloatArray
    virial: FloatArray
    pairwise_additive: FloatArray | None
    known_issue: str | None

    @property
    def id(self) -> str:
        """Stable pytest id."""
        return f"{self.family}/{self.name}/{self.parameter_set}"


def _case(family: str, raw: dict[str, Any]) -> FixtureCase:
    pairwise = raw.get("pairwise_additive")
    return FixtureCase(
        family=family,
        name=raw["name"],
        parameter_set=raw["parameter_set"],
        numbers=np.array(raw["numbers"], dtype=np.int64),
        positions=np.array(raw["positions_bohr"], dtype=np.float64).reshape(-1, 3),
        cell=None if raw["cell_bohr"] is None else np.array(raw["cell_bohr"], dtype=np.float64),
        pbc=np.array(raw["pbc"], dtype=bool),
        parameters=D3BJParameters(**raw["parameters"]),
        cutoffs=Cutoffs(**raw["cutoffs_bohr"]),
        energy=float(raw["energy"]),
        gradient=np.array(raw["gradient"], dtype=np.float64).reshape(-1, 3),
        virial=np.array(raw["virial"], dtype=np.float64),
        pairwise_additive=None if pairwise is None else np.array(pairwise, dtype=np.float64),
        known_issue=raw.get("known_issue"),
    )


@functools.cache
def load_header(family: str) -> dict[str, Any]:
    """Return the header line of a fixture family."""
    with (FIXTURE_DIR / f"{family}.jsonl").open(encoding="utf-8") as handle:
        header: dict[str, Any] = json.loads(handle.readline())
    return header


@functools.cache
def load_family(family: str) -> tuple[FixtureCase, ...]:
    """Return every case of a fixture family."""
    lines = (FIXTURE_DIR / f"{family}.jsonl").read_text(encoding="utf-8").splitlines()
    return tuple(_case(family, json.loads(line)) for line in lines[1:])


def all_cases() -> list[FixtureCase]:
    """Return every fixture case of every family."""
    return [case for family in FAMILIES for case in load_family(family)]


def find_case(family: str, name: str, parameter_set: str) -> FixtureCase:
    """Return one named case."""
    for case in load_family(family):
        if case.name == name and case.parameter_set == parameter_set:
            return case
    msg = f"no fixture {family}/{name}/{parameter_set}"
    raise KeyError(msg)
