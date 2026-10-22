"""The array-based CUDA evaluation context for DFT-D4 (Å / eV interface).

:class:`D4Context` owns one native CUDA context for D4: device buffers, the
species tables with the reference C6 derived for the chosen ``ga``/``gc``,
the persistent neighbour state, and the EEQ charge solver. Units are
converted exactly once at this boundary, with the recipes of ``dftd4.ase``.

As in dftd4, any periodic flag makes the system fully periodic over the
supplied cell (dispersion and charges alike). A slab or wire therefore needs a
cell whose vacuum vector is long enough for its periodic images to be
negligible, exactly as for a plane-wave code.
"""

import hashlib
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal, Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from vanderfast import units
from vanderfast._native_loader import load_native
from vanderfast.context import (
    MIN_VOLUME,
    PROPERTIES,
    Builder,
    _check_precision,
    _validate_cell,
    _validate_pbc,
    _validate_positions,
    parse_device,
)
from vanderfast.data.d4 import load_d4_reference_tables, parse_d4_reference_tables
from vanderfast.errors import InputError, ParameterError
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters

FloatArray = npt.NDArray[np.float64]

EwaldMode = Literal["upstream", "converged"]
SolverMethod = Literal["auto", "iterative", "direct"]

EWALD_MODES: Final = ("upstream", "converged")
SOLVER_METHODS: Final = ("auto", "iterative", "direct")


class D4NativeContext(Protocol):
    """The subset of ``vanderfast._native.D4Context`` used by this module."""

    @property
    def n_atoms(self) -> int: ...

    def evaluate(
        self,
        positions: FloatArray,
        cell: FloatArray,
        pbc: tuple[bool, bool, bool],
        *,
        gradient: bool = False,
        virial: bool = False,
        atom_energies: bool = False,
    ) -> dict[str, Any]: ...

    def set_parameters(self, **kwargs: float) -> None: ...

    def set_charge(self, charge: float) -> None: ...

    def debug(
        self,
        positions: FloatArray,
        cell: FloatArray,
        pbc: tuple[bool, bool, bool],
        *,
        derivatives: bool = True,
    ) -> dict[str, Any]: ...

    def invalidate_neighbors(self) -> None: ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EEQSolverOptions:
    """Options of the GPU charge solver; the defaults suit every documented use.

    Attributes:
        method: ``"auto"`` (a direct solve for small systems, preconditioned
            MINRES otherwise), ``"iterative"``, or ``"direct"`` (dense LU).
        tolerance: Accepted normwise backward error of the charge equations
            (``None`` keeps the native default, 1e-13).
        max_iterations: Iteration cap of the Krylov solver (``None`` keeps
            the native default).
        warm_start: Start each solve from the previous charges. This saves
            iterations in molecular dynamics, but results then depend in the
            last bits on the call history; off by default.
    """

    method: SolverMethod = "auto"
    tolerance: float | None = None
    max_iterations: int | None = None
    warm_start: bool = False

    def __post_init__(self) -> None:
        """Validate the options."""
        if self.method not in SOLVER_METHODS:
            msg = f"solver method must be one of {SOLVER_METHODS}, got {self.method!r}"
            raise InputError(msg)
        tol = cast("object", self.tolerance)
        if tol is not None and not (
            isinstance(tol, int | float) and math.isfinite(tol) and 0.0 < tol < 1.0
        ):
            msg = f"solver tolerance must be in (0, 1), got {self.tolerance!r}"
            raise InputError(msg)
        cap = cast("object", self.max_iterations)
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            msg = f"max_iterations must be a positive integer, got {self.max_iterations!r}"
            raise InputError(msg)

    def native_kwargs(self) -> dict[str, Any]:
        """Return the options as native keyword arguments (0 keeps a native default)."""
        return {
            "solver": self.method,
            "solver_tolerance": 0.0 if self.tolerance is None else float(self.tolerance),
            "solver_max_iterations": 0 if self.max_iterations is None else self.max_iterations,
            "warm_start": self.warm_start,
        }


@dataclass(frozen=True, eq=False)
class D4AtomicResult:
    """Results in atomic units (hartree, bohr, e)."""

    energy: float
    energy_pair: float
    energy_atm: float
    gradient: FloatArray | None
    virial: FloatArray | None
    atom_energies: FloatArray | None
    charges: FloatArray
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class D4Result:
    """Results in public units (eV, Å, e). Arrays are owned copies.

    Attributes:
        energy: Total dispersion energy (eV).
        energy_pair: Two-body contribution (eV).
        energy_atm: Three-body (ATM) contribution (eV).
        forces: Forces ``-dE/dx`` in eV/Å, if requested.
        strain_derivative: ``dE/dstrain`` in eV for a homogeneous strain of
            positions and cell, if requested.
        stress: ASE Voigt stress (xx, yy, zz, yz, xz, xy) in eV/Å³, if
            requested; available for every periodic system.
        charges: EEQ partial charges in e.
        info: Edge, triangle, and solver counts and rebuild status.
    """

    energy: float
    energy_pair: float
    energy_atm: float
    forces: FloatArray | None
    strain_derivative: FloatArray | None
    stress: FloatArray | None
    charges: FloatArray
    info: dict[str, Any]


def _resolve_d4_cutoffs(cutoff: D4Cutoffs | Mapping[str, float] | None) -> D4Cutoffs:
    if isinstance(cutoff, D4Cutoffs):
        return cutoff
    return D4Cutoffs.resolve(cutoff)


def _validate_d4_numbers(numbers: npt.ArrayLike) -> npt.NDArray[np.int32]:
    z = np.asarray(numbers)
    if z.ndim != 1 or not (z.size == 0 or np.issubdtype(z.dtype, np.integer)):
        msg = "numbers must be a one-dimensional integer array"
        raise InputError(msg)
    tables = load_d4_reference_tables()
    if z.size and (int(z.min()) < 1 or int(z.max()) > tables.max_z):
        msg = f"atomic numbers must be within 1..{tables.max_z} for D4"
        raise InputError(msg)
    return np.ascontiguousarray(z, dtype=np.int32)


def _validate_charge(charge: float) -> float:
    value = cast("object", charge)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        msg = f"charge must be a finite number, got {charge!r}"
        raise InputError(msg)
    return float(value)


def promote_pbc(
    pbc: npt.ArrayLike, cell: npt.ArrayLike | None
) -> tuple[tuple[bool, bool, bool], FloatArray]:
    """Apply the D4 periodicity rule: any periodic flag means fully periodic.

    Returns:
        The effective flags and the validated cell.

    Raises:
        InputError: If a periodic system lacks three independent lattice vectors.
    """
    flags = _validate_pbc(pbc)
    periodic = any(flags)
    h = _validate_cell(cell, flags)
    if not periodic:
        return (False, False, False), h
    gram = h @ h.T
    scale = float(np.prod(np.diag(gram)))
    if scale <= 0.0 or np.linalg.det(gram) <= 1e-12 * scale:
        msg = (
            "D4 treats any periodic direction as fully periodic and needs three linearly "
            "independent lattice vectors; give slabs and wires a vacuum lattice vector"
        )
        raise InputError(msg)
    return (True, True, True), h


def _load_d4_reference_blob(path: str | PathLike[str]) -> tuple[bytes, str]:
    blob = Path(path).read_bytes()
    parse_d4_reference_tables(blob)
    return blob, hashlib.sha256(blob).hexdigest()


class D4Context:
    """A persistent CUDA context evaluating DFT-D4 for one set of atoms.

    Args:
        numbers: Atomic numbers, shape (N,), within 1..103. Fixed for the
            context lifetime.
        parameters: Resolved D4 parameters.
        charge: Total charge in e, distributed by the EEQ model.
        device: ``"cuda"``, ``"cuda:N"``, or a device index.
        precision: Only ``"float64"`` is implemented.
        cutoff: Cutoffs as :class:`D4Cutoffs` (bohr), or a mapping with keys
            ``cn``, ``disp2``, ``disp3`` in Å; omitted keys keep the defaults.
        skin: Neighbour-list skin in Å.
        builder: Neighbour builder selection.
        ewald: Periodic charge model. ``"upstream"`` reproduces dftd4 4.2.0
            (whose charges depend on the choice of cell); ``"converged"`` uses
            complete Ewald sums and is supercell-consistent.
        solver: Charge-solver options.
        reference_data: Optional path to replacement D4 tables in the bundled
            format.
        profile: Record per-stage GPU timings.
    """

    def __init__(
        self,
        numbers: npt.ArrayLike,
        parameters: D4Parameters,
        *,
        charge: float = 0.0,
        device: str | int = "cuda:0",
        precision: str = "float64",
        cutoff: D4Cutoffs | Mapping[str, float] | None = None,
        skin: float = 1.0,
        builder: Builder = "auto",
        ewald: EwaldMode = "upstream",
        solver: EEQSolverOptions | None = None,
        reference_data: str | PathLike[str] | None = None,
        profile: bool = False,
    ) -> None:
        """Create the native context (initialises CUDA on first use)."""
        if not isinstance(cast("object", parameters), D4Parameters):
            msg = "parameters must be a D4Parameters instance"
            raise ParameterError(msg)
        _check_precision(precision)
        length = cast("object", skin)
        if not (isinstance(length, int | float) and math.isfinite(length) and length >= 0.0):
            msg = f"skin must be a finite non-negative length in Å, got {skin!r}"
            raise InputError(msg)
        if ewald not in EWALD_MODES:
            msg = f"ewald must be one of {EWALD_MODES}, got {ewald!r}"
            raise InputError(msg)
        self.numbers = _validate_d4_numbers(numbers)
        self.parameters = parameters
        self.charge = _validate_charge(charge)
        self.cutoffs = _resolve_d4_cutoffs(cutoff)
        self.device = parse_device(device)
        self.precision = precision
        self.skin = float(skin)
        self.ewald: EwaldMode = ewald
        self.solver = solver or EEQSolverOptions()
        blob, identity = (
            (None, "") if reference_data is None else _load_d4_reference_blob(reference_data)
        )
        native = load_native()
        self._native: D4NativeContext | None = native.D4Context(
            self.numbers,
            **self._parameter_kwargs(),
            charge=self.charge,
            ewald=self.ewald,
            **self.solver.native_kwargs(),
            device=self.device,
            skin=units.cutoff_to_bohr(self.skin),
            builder=builder,
            reference_data=blob,
            reference_identity=identity,
            collect_timings=profile,
        )

    def _parameter_kwargs(self) -> dict[str, float]:
        p = self.parameters
        return {
            "s6": p.s6,
            "s8": p.s8,
            "a1": p.a1,
            "a2": p.a2,
            "s9": p.s9,
            "alp": p.alp,
            "ga": p.ga,
            "gc": p.gc,
            "wf": p.wf,
            "cutoff_cn": self.cutoffs.cn,
            "cutoff_disp2": self.cutoffs.disp2,
            "cutoff_disp3": self.cutoffs.disp3,
        }

    @property
    def native(self) -> D4NativeContext:
        """The native context (raises if the context was closed)."""
        if self._native is None:
            msg = "the context has been closed"
            raise InputError(msg)
        return self._native

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.numbers.size)

    def evaluate_atomic(
        self,
        positions: npt.ArrayLike,
        *,
        cell: npt.ArrayLike | None = None,
        pbc: npt.ArrayLike = (False, False, False),
        gradient: bool = False,
        virial: bool = False,
        atom_energies: bool = False,
    ) -> D4AtomicResult:
        """Evaluate with positions and cell in bohr; results in hartree, bohr, and e."""
        flags, h = promote_pbc(pbc, cell)
        x = _validate_positions(positions, self.n_atoms)
        raw = self.native.evaluate(
            x, h, flags, gradient=gradient, virial=virial, atom_energies=atom_energies
        )
        info = dict(raw["info"])
        info["pbc_input"] = _validate_pbc(pbc)
        info["ewald_mode"] = self.ewald if any(flags) else "none"
        return D4AtomicResult(
            energy=float(raw["energy"]),
            energy_pair=float(raw["energy_pair"]),
            energy_atm=float(raw["energy_atm"]),
            gradient=raw["gradient"],
            virial=raw["virial"],
            atom_energies=raw["atom_energies"],
            charges=np.asarray(raw["charges"], dtype=np.float64),
            info=info,
        )

    def evaluate(
        self,
        positions: npt.ArrayLike,
        *,
        cell: npt.ArrayLike | None = None,
        pbc: npt.ArrayLike = (False, False, False),
        properties: Collection[str] = ("energy",),
    ) -> D4Result:
        """Evaluate with positions and cell in Å; results in eV, Å, and e.

        Args:
            positions: Cartesian positions in Å, shape (N, 3).
            cell: Lattice vectors in Å as rows; required when any PBC is set.
            pbc: Periodicity flags; any true flag makes all three periodic.
            properties: Any of ``energy``, ``forces``, ``strain_derivative``,
                ``stress``. The energy and charges are always computed.

        Raises:
            InputError: For unknown properties, or stress without periodicity.
        """
        requested = set(properties)
        unknown = requested - PROPERTIES
        if unknown:
            msg = f"unknown properties {sorted(unknown)}; expected a subset of {sorted(PROPERTIES)}"
            raise InputError(msg)
        flags, h = promote_pbc(pbc, cell)
        volume = abs(float(np.linalg.det(h)))
        if "stress" in requested and (not any(flags) or volume < MIN_VOLUME):
            msg = "stress requires a periodic cell with a non-zero volume"
            raise InputError(msg)
        virial = bool(requested & {"strain_derivative", "stress"})
        raw = self.evaluate_atomic(
            units.length_to_bohr_d4(_validate_positions(positions, self.n_atoms)),
            cell=units.length_to_bohr_d4(h),
            pbc=pbc,
            gradient="forces" in requested,
            virial=virial,
        )
        strain = None if raw.virial is None else raw.virial * units.HARTREE
        stress = None
        if "stress" in requested:
            stress = units.strain_derivative_to_stress_d4(cast("FloatArray", raw.virial), volume)
        return D4Result(
            energy=units.energy_to_ev(raw.energy),
            energy_pair=units.energy_to_ev(raw.energy_pair),
            energy_atm=units.energy_to_ev(raw.energy_atm),
            forces=None if raw.gradient is None else units.gradient_to_forces_d4(raw.gradient),
            strain_derivative=strain if "strain_derivative" in requested else None,
            stress=stress,
            charges=raw.charges.copy(),
            info=raw.info,
        )

    def debug_atomic(
        self,
        positions: npt.ArrayLike,
        *,
        cell: npt.ArrayLike | None = None,
        pbc: npt.ArrayLike = (False, False, False),
        derivatives: bool = True,
    ) -> dict[str, Any]:
        """Return intermediates for development and testing (bohr, e).

        The mapping holds ``cn``, ``cn_eeq``, ``charges``, the charge-scaled
        and neutral weights and their derivatives (N x 8, zero padded),
        ``dEdcn``, ``dEdq``, and the directed edges of the largest view.
        """
        flags, h = promote_pbc(pbc, cell)
        x = _validate_positions(positions, self.n_atoms)
        return self.native.debug(x, h, flags, derivatives=derivatives)

    def set_parameters(self, parameters: D4Parameters) -> None:
        """Replace the parameters, keeping the neighbour state."""
        if not isinstance(cast("object", parameters), D4Parameters):
            msg = "parameters must be a D4Parameters instance"
            raise ParameterError(msg)
        self.parameters = parameters
        self.native.set_parameters(**self._parameter_kwargs())

    def set_cutoffs(self, cutoff: D4Cutoffs | Mapping[str, float] | None) -> None:
        """Replace the cutoffs; the neighbour lists are rebuilt on next use."""
        self.cutoffs = _resolve_d4_cutoffs(cutoff)
        self.native.set_parameters(**self._parameter_kwargs())
        self.native.invalidate_neighbors()

    def set_charge(self, charge: float) -> None:
        """Replace the total charge, keeping every other state."""
        self.charge = _validate_charge(charge)
        self.native.set_charge(self.charge)

    def invalidate_neighbors(self) -> None:
        """Force a neighbour-list rebuild on the next evaluation."""
        self.native.invalidate_neighbors()

    def diagnostics(self) -> dict[str, Any]:
        """Return the resolved configuration and runtime counters."""
        return {
            "n_atoms": self.n_atoms,
            "device": f"cuda:{self.device}",
            "precision": self.precision,
            "parameters": self.parameters.as_dict(),
            "charge": self.charge,
            "ewald": self.ewald,
            "solver": self.solver.native_kwargs(),
            "cutoffs_bohr": self.cutoffs.as_bohr(),
            "cutoffs_angstrom": self.cutoffs.as_angstrom(),
            "skin_angstrom": self.skin,
            "stats": dict(self.native.stats()),
        }

    def close(self) -> None:
        """Release the native context and its device memory."""
        self._native = None

    def __enter__(self) -> Self:
        """Enter a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the context on leaving a ``with`` block."""
        self.close()


def evaluate_d4(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    parameters: D4Parameters,
    *,
    charge: float = 0.0,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    properties: Sequence[str] = ("energy",),
    **options: Any,
) -> D4Result:
    """Evaluate D4 once with a temporary context (convenience for scripts)."""
    with D4Context(numbers, parameters, charge=charge, **options) as context:
        return context.evaluate(positions, cell=cell, pbc=pbc, properties=properties)


__all__ = [
    "D4AtomicResult",
    "D4Context",
    "D4Result",
    "EEQSolverOptions",
    "evaluate_d4",
    "promote_pbc",
]
