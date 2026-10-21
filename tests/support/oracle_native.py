"""A CPU stand-in for ``vanderfast._native`` built on the NumPy oracle.

It implements the native module protocol so the Python context and ASE layers
are fully tested on machines without CUDA.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from vanderfast import _oracle
from vanderfast.data import ReferenceTables, parse_reference_tables
from vanderfast.parameters import Cutoffs, D3BJParameters


def build_info() -> dict[str, str]:
    return {"backend": "oracle"}


def device_count() -> int:
    return 1


def devices() -> list[dict[str, Any]]:
    return [{"index": 0, "name": "oracle", "compute_capability": "0.0"}]


class Context:
    instances = 0

    def __init__(  # noqa: PLR0913
        self,
        numbers: npt.ArrayLike,
        *,
        s6: float,
        s8: float,
        a1: float,
        a2: float,
        s9: float,
        alp: float,
        cutoff_cn: float,
        cutoff_disp2: float,
        cutoff_disp3: float,
        device: int = 0,
        skin: float = 0.0,
        builder: str = "auto",
        reference_data: bytes | None = None,
        reference_identity: str = "",
        collect_timings: bool = False,
    ) -> None:
        Context.instances += 1
        self._numbers = np.asarray(numbers, dtype=np.int64)
        self.device = device
        self.skin = skin
        self.builder = builder
        self.reference_identity = reference_identity
        self.collect_timings = collect_timings
        self.tables: ReferenceTables | None = (
            None if reference_data is None else parse_reference_tables(reference_data)
        )
        self.evaluations = 0
        self.neighbor_invalidations = 0
        self.set_parameters(
            s6=s6,
            s8=s8,
            a1=a1,
            a2=a2,
            s9=s9,
            alp=alp,
            cutoff_cn=cutoff_cn,
            cutoff_disp2=cutoff_disp2,
            cutoff_disp3=cutoff_disp3,
        )

    @property
    def n_atoms(self) -> int:
        return int(self._numbers.size)

    def set_parameters(
        self,
        *,
        s6: float,
        s8: float,
        a1: float,
        a2: float,
        s9: float,
        alp: float,
        cutoff_cn: float,
        cutoff_disp2: float,
        cutoff_disp3: float,
    ) -> None:
        self.parameters = D3BJParameters(s6=s6, s8=s8, a1=a1, a2=a2, s9=s9, alp=alp)
        self.cutoffs = Cutoffs(cn=cutoff_cn, disp2=cutoff_disp2, disp3=cutoff_disp3)

    def invalidate_neighbors(self) -> None:
        self.neighbor_invalidations += 1

    def stats(self) -> dict[str, Any]:
        return {"evaluations": self.evaluations}

    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        gradient: bool = False,
        virial: bool = False,
        atom_energies: bool = False,
    ) -> dict[str, Any]:
        self.evaluations += 1
        result = _oracle.evaluate(
            self._numbers,
            positions,
            parameters=self.parameters,
            cell=cell,
            pbc=pbc,
            cutoffs=self.cutoffs,
            gradient=gradient or virial,
            tables=self.tables,
        )
        return {
            "energy": result.energy,
            "energy_pair": result.energy_pair,
            "energy_atm": result.energy_atm,
            "gradient": result.gradient if gradient else None,
            "virial": result.strain_derivative if virial else None,
            "atom_energies": result.atom_energies if atom_energies else None,
            "info": dict(result.counts),
        }
