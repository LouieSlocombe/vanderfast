from typing import Any

import numpy as np
import numpy.typing as npt

def build_info() -> dict[str, str]: ...
def device_count() -> int: ...
def devices() -> list[dict[str, Any]]: ...
def reference_summary() -> dict[str, Any]: ...
def reference_value(table: str, zi: int, zj: int = 1, a: int = 0, b: int = 0) -> float: ...
def d4_reference_summary() -> dict[str, Any]: ...
def d4_reference_alpha(z: int, ga: float = 3.0, gc: float = 2.0) -> npt.NDArray[np.float64]: ...
def d4_reference_c6(
    zi: int, zj: int, ga: float = 3.0, gc: float = 2.0
) -> npt.NDArray[np.float64]: ...
def enumerate_edges(
    positions: npt.NDArray[np.float64],
    cell: npt.NDArray[np.float64],
    pbc: tuple[bool, bool, bool],
    cutoff: float,
) -> dict[str, npt.NDArray[Any]]: ...

class Context:
    def __init__(
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
        atm_items_per_launch: int = 0,
        kernels: str = "optimized",
    ) -> None: ...
    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        gradient: bool = False,
        virial: bool = False,
        atom_energies: bool = False,
    ) -> dict[str, Any]: ...
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
    ) -> None: ...
    def debug(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        derivatives: bool = True,
    ) -> dict[str, Any]: ...
    def invalidate_neighbors(self) -> None: ...
    def _testing_set_allocation_limit(self, bytes: int) -> None: ...  # noqa: A002
    def _testing_hold(self, seconds: float) -> None: ...
    @property
    def n_atoms(self) -> int: ...
    def stats(self) -> dict[str, Any]: ...

class D4Context:
    def __init__(
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
        atm_items_per_launch: int = 0,
        kernels: str = "optimized",
    ) -> None: ...
    def evaluate(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        gradient: bool = False,
        virial: bool = False,
        atom_energies: bool = False,
    ) -> dict[str, Any]: ...
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
    ) -> None: ...
    def set_charge(self, charge: float) -> None: ...
    def debug(
        self,
        positions: npt.NDArray[np.float64],
        cell: npt.NDArray[np.float64],
        pbc: tuple[bool, bool, bool],
        *,
        derivatives: bool = True,
    ) -> dict[str, Any]: ...
    def invalidate_neighbors(self) -> None: ...
    def _testing_set_allocation_limit(self, bytes: int) -> None: ...  # noqa: A002
    @property
    def n_atoms(self) -> int: ...
    def stats(self) -> dict[str, Any]: ...
