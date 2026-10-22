"""A CPU stand-in for ``vanderfast._native`` built on the NumPy oracle.

It implements the native module protocol so the Python context and ASE layers
are fully tested on machines without CUDA.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from vanderfast import _oracle, _oracle_d4
from vanderfast.data import ReferenceTables, parse_reference_tables
from vanderfast.data.d4 import D4ReferenceTables, parse_d4_reference_tables
from vanderfast.parameters import Cutoffs, D3BJParameters
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters


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

    def debug(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        derivatives: bool = True,
    ) -> dict[str, Any]:
        result = _oracle.evaluate(
            self._numbers,
            positions,
            parameters=self.parameters,
            cell=cell,
            pbc=pbc,
            cutoffs=self.cutoffs,
            gradient=derivatives,
            tables=self.tables,
        )
        geometry = _oracle.make_geometry(self._numbers, positions, cell, pbc, self.tables)
        edges = _oracle.directed_edges(
            geometry, max(self.cutoffs.cn, self.cutoffs.disp2, self.cutoffs.disp3) + self.skin
        )

        def pad(values: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64] | None:
            return None if values is None else np.pad(values, ((0, 0), (0, 1)))

        return {
            "cn": result.cn,
            "weights": pad(result.weights),
            "dweights": pad(result.dweights),
            "dEdcn": result.g,
            "edge_i": edges.i.astype(np.int32),
            "edge_j": edges.j.astype(np.int32),
            "edge_shift": edges.shift.astype(np.int32),
        }

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


class D4Context:
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
        ga: float,
        gc: float,
        wf: float,
        cutoff_cn: float,
        cutoff_disp2: float,
        cutoff_disp3: float,
        charge: float = 0.0,
        ewald: str = "upstream",
        solver: str = "auto",
        solver_tolerance: float = 0.0,
        solver_max_iterations: int = 0,
        warm_start: bool = False,
        device: int = 0,
        skin: float = 0.0,
        builder: str = "auto",
        reference_data: bytes | None = None,
        reference_identity: str = "",
        collect_timings: bool = False,
    ) -> None:
        D4Context.instances += 1
        self._numbers = np.asarray(numbers, dtype=np.int64)
        self.charge = charge
        self.ewald = ewald
        self.solver = (solver, solver_tolerance, solver_max_iterations, warm_start)
        self.device = device
        self.skin = skin
        self.builder = builder
        self.reference_identity = reference_identity
        self.collect_timings = collect_timings
        self.tables: D4ReferenceTables | None = (
            None if reference_data is None else parse_d4_reference_tables(reference_data)
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
            ga=ga,
            gc=gc,
            wf=wf,
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
        ga: float,
        gc: float,
        wf: float,
        cutoff_cn: float,
        cutoff_disp2: float,
        cutoff_disp3: float,
    ) -> None:
        self.parameters = D4Parameters(
            s6=s6, s8=s8, a1=a1, a2=a2, s9=s9, alp=alp, ga=ga, gc=gc, wf=wf
        )
        self.cutoffs = D4Cutoffs(cn=cutoff_cn, disp2=cutoff_disp2, disp3=cutoff_disp3)

    def set_charge(self, charge: float) -> None:
        self.charge = charge

    def invalidate_neighbors(self) -> None:
        self.neighbor_invalidations += 1

    def stats(self) -> dict[str, Any]:
        return {"evaluations": self.evaluations}

    def _evaluate(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        gradient: bool,
    ) -> _oracle_d4.D4OracleResult:
        if any(pbc) and not all(pbc):
            msg = "mixed periodicity must be promoted before reaching the native layer"
            raise ValueError(msg)
        return _oracle_d4.evaluate(
            self._numbers,
            positions,
            parameters=self.parameters,
            charge=self.charge,
            cell=cell,
            pbc=pbc,
            cutoffs=self.cutoffs,
            gradient=gradient,
            tables=self.tables,
            ewald="converged" if self.ewald == "converged" else "upstream",
        )

    def debug(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        derivatives: bool = True,
    ) -> dict[str, Any]:
        result = self._evaluate(positions, cell, pbc, derivatives)
        geometry = _oracle_d4.make_geometry(self._numbers, positions, cell, pbc, self.tables)
        edges = _oracle.directed_edges(
            geometry, max(self.cutoffs.cn_radius(), self.cutoffs.disp2, self.cutoffs.disp3)
        )

        def pad(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.pad(values, ((0, 0), (0, 1)))

        return {
            "cn": result.cn,
            "cn_eeq": result.cn_eeq,
            "charges": result.charges,
            "weights": pad(result.weights),
            "weights_q0": pad(result.weights_q0),
            "dweights_dcn": pad(result.dweights_dcn),
            "dweights_dq": pad(result.dweights_dq),
            "dEdcn": result.dedcn,
            "dEdq": result.dedq,
            "edge_i": edges.i.astype(np.int32),
            "edge_j": edges.j.astype(np.int32),
            "edge_shift": edges.shift.astype(np.int32),
        }

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
        result = self._evaluate(positions, cell, pbc, gradient or virial)
        info: dict[str, Any] = dict(result.counts)
        info["eeq_solver"] = "oracle"
        info["eeq_iterations"] = 0
        info["ewald_alpha"] = result.ewald_alpha
        return {
            "energy": result.energy,
            "energy_pair": result.energy_pair,
            "energy_atm": result.energy_atm,
            "gradient": result.gradient if gradient else None,
            "virial": result.strain_derivative if virial else None,
            "atom_energies": result.atom_energies if atom_energies else None,
            "charges": result.charges,
            "info": info,
        }
