"""The array-based CUDA evaluation context (Å / eV interface).

:class:`D3Context` owns one native CUDA context: device buffers, the species
tables, and persistent neighbour state. It converts units exactly once at
this boundary; the native core works in bohr and hartree.
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
from vanderfast.data import load_reference_tables, parse_reference_tables
from vanderfast.errors import InputError, ParameterError, UnsupportedFeatureError
from vanderfast.parameters import Cutoffs, D3BJParameters

FloatArray = npt.NDArray[np.float64]

PROPERTIES: Final = frozenset({"energy", "forces", "strain_derivative", "stress"})
MIN_VOLUME: Final = 1e-10  # Å^3

Builder = Literal["auto", "brute_force", "cell_bins"]


class NativeContext(Protocol):
    """The subset of ``vanderfast._native.Context`` used by this module."""

    @property
    def n_atoms(self) -> int: ...  # noqa: D102

    def evaluate(  # noqa: D102
        self,
        positions: FloatArray,
        cell: FloatArray,
        pbc: tuple[bool, bool, bool],
        *,
        gradient: bool = False,
        virial: bool = False,
        atom_energies: bool = False,
    ) -> dict[str, Any]: ...

    def set_parameters(self, **kwargs: float) -> None: ...  # noqa: D102

    def invalidate_neighbors(self) -> None: ...  # noqa: D102

    def stats(self) -> dict[str, Any]: ...  # noqa: D102


@dataclass(frozen=True, eq=False)
class AtomicResult:
    """Results in atomic units (hartree, bohr)."""

    energy: float
    energy_pair: float
    energy_atm: float
    gradient: FloatArray | None
    virial: FloatArray | None
    atom_energies: FloatArray | None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class D3Result:
    """Results in public units (eV, Å). Arrays are owned copies.

    Attributes:
        energy: Total dispersion energy (eV).
        energy_pair: Two-body contribution (eV).
        energy_atm: Three-body (ATM) contribution (eV).
        forces: Forces ``-dE/dx`` in eV/Å, if requested.
        strain_derivative: ``dE/dstrain`` in eV for a homogeneous strain of
            positions and cell, if requested.
        stress: ASE Voigt stress (xx, yy, zz, yz, xz, xy) in eV/Å³, if
            requested; only for three-dimensional periodic cells.
        info: Edge and triangle counts and rebuild status.
    """

    energy: float
    energy_pair: float
    energy_atm: float
    forces: FloatArray | None
    strain_derivative: FloatArray | None
    stress: FloatArray | None
    info: dict[str, Any]


def parse_device(device: str | int) -> int:
    """Return the CUDA device index for ``"cuda"``, ``"cuda:N"``, or ``N``."""
    value = cast("object", device)
    if isinstance(value, int) and not isinstance(value, bool):
        index = value
    elif value == "cuda":
        index = 0
    elif isinstance(value, str) and value.startswith("cuda:") and value[5:].isdigit():
        index = int(value[5:])
    else:
        msg = f"device must be 'cuda', 'cuda:N', or an index, got {device!r}"
        raise InputError(msg)
    if index < 0:
        msg = f"device index must be non-negative, got {index}"
        raise InputError(msg)
    return index


def _check_precision(precision: str) -> None:
    if precision != "float64":
        msg = f"precision {precision!r} is not implemented; only 'float64' is available"
        raise UnsupportedFeatureError(msg)


def _resolve_cutoffs(cutoff: Cutoffs | Mapping[str, float] | None) -> Cutoffs:
    if isinstance(cutoff, Cutoffs):
        return cutoff
    return Cutoffs.resolve(cutoff)


def _validate_numbers(numbers: npt.ArrayLike) -> npt.NDArray[np.int32]:
    z = np.asarray(numbers)
    if z.ndim != 1 or not (z.size == 0 or np.issubdtype(z.dtype, np.integer)):
        msg = "numbers must be a one-dimensional integer array"
        raise InputError(msg)
    tables = load_reference_tables()
    if z.size and (int(z.min()) < 1 or int(z.max()) > tables.max_z):
        msg = f"atomic numbers must be within 1..{tables.max_z}"
        raise InputError(msg)
    return np.ascontiguousarray(z, dtype=np.int32)


def _validate_pbc(pbc: npt.ArrayLike) -> tuple[bool, bool, bool]:
    flags = np.asarray(pbc)
    if flags.shape == ():
        flags = np.repeat(flags, 3)
    if flags.shape != (3,):
        msg = "pbc must be a boolean or three booleans"
        raise InputError(msg)
    return (bool(flags[0]), bool(flags[1]), bool(flags[2]))


def _validate_cell(cell: npt.ArrayLike | None, pbc: tuple[bool, bool, bool]) -> FloatArray:
    if cell is None:
        if any(pbc):
            msg = "a cell is required for periodic boundary conditions"
            raise InputError(msg)
        return np.zeros((3, 3))
    h = np.asarray(cell, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        msg = "cell must be a finite 3x3 array with lattice vectors as rows"
        raise InputError(msg)
    return h


def _validate_positions(positions: npt.ArrayLike, n_atoms: int) -> FloatArray:
    x = np.asarray(positions, dtype=np.float64)
    if x.shape != (n_atoms, 3):
        msg = f"positions must have shape ({n_atoms}, 3), got {x.shape}"
        raise InputError(msg)
    return x


def _load_reference_blob(path: str | PathLike[str]) -> tuple[bytes, str]:
    blob = Path(path).read_bytes()
    parse_reference_tables(blob)
    return blob, hashlib.sha256(blob).hexdigest()


class D3Context:
    """A persistent CUDA context evaluating D3(BJ) for one set of atoms.

    Args:
        numbers: Atomic numbers, shape (N,). Fixed for the context lifetime.
        parameters: Resolved D3(BJ) parameters.
        device: ``"cuda"``, ``"cuda:N"``, or a device index.
        precision: Only ``"float64"`` is implemented.
        cutoff: Cutoffs as :class:`Cutoffs` (bohr), or a mapping with keys
            ``cn``, ``disp2``, ``disp3`` in Å; omitted keys keep the defaults.
        skin: Neighbour-list skin in Å (candidate lists only; every kernel
            applies the exact cutoff).
        builder: Neighbour builder selection.
        reference_data: Optional path to replacement reference tables in the
            bundled format.
        profile: Record per-stage GPU timings.
    """

    def __init__(
        self,
        numbers: npt.ArrayLike,
        parameters: D3BJParameters,
        *,
        device: str | int = "cuda:0",
        precision: str = "float64",
        cutoff: Cutoffs | Mapping[str, float] | None = None,
        skin: float = 1.0,
        builder: Builder = "auto",
        reference_data: str | PathLike[str] | None = None,
        profile: bool = False,
    ) -> None:
        """Create the native context (initialises CUDA on first use)."""
        if not isinstance(cast("object", parameters), D3BJParameters):
            msg = "parameters must be a D3BJParameters instance"
            raise ParameterError(msg)
        _check_precision(precision)
        length = cast("object", skin)
        if not (isinstance(length, int | float) and math.isfinite(length) and length >= 0.0):
            msg = f"skin must be a finite non-negative length in Å, got {skin!r}"
            raise InputError(msg)
        self.numbers = _validate_numbers(numbers)
        self.parameters = parameters
        self.cutoffs = _resolve_cutoffs(cutoff)
        self.device = parse_device(device)
        self.precision = precision
        self.skin = float(skin)
        blob, identity = (
            (None, "") if reference_data is None else _load_reference_blob(reference_data)
        )
        native = load_native()
        self._native: NativeContext | None = native.Context(
            self.numbers,
            **self._parameter_kwargs(),
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
            "cutoff_cn": self.cutoffs.cn,
            "cutoff_disp2": self.cutoffs.disp2,
            "cutoff_disp3": self.cutoffs.disp3,
        }

    @property
    def native(self) -> NativeContext:
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
    ) -> AtomicResult:
        """Evaluate with positions and cell in bohr; results in hartree and bohr."""
        flags = _validate_pbc(pbc)
        h = _validate_cell(cell, flags)
        x = _validate_positions(positions, self.n_atoms)
        raw = self.native.evaluate(
            x, h, flags, gradient=gradient, virial=virial, atom_energies=atom_energies
        )
        return AtomicResult(
            energy=float(raw["energy"]),
            energy_pair=float(raw["energy_pair"]),
            energy_atm=float(raw["energy_atm"]),
            gradient=raw["gradient"],
            virial=raw["virial"],
            atom_energies=raw["atom_energies"],
            info=dict(raw["info"]),
        )

    def evaluate(
        self,
        positions: npt.ArrayLike,
        *,
        cell: npt.ArrayLike | None = None,
        pbc: npt.ArrayLike = (False, False, False),
        properties: Collection[str] = ("energy",),
    ) -> D3Result:
        """Evaluate with positions and cell in Å; results in eV and Å.

        Args:
            positions: Cartesian positions in Å, shape (N, 3).
            cell: Lattice vectors in Å as rows; required when any PBC is set.
            pbc: Periodicity per lattice vector.
            properties: Any of ``energy``, ``forces``, ``strain_derivative``,
                ``stress``. The energy is always computed.

        Raises:
            UnsupportedFeatureError: If stress is requested without
                three-dimensional periodicity.
        """
        requested = set(properties)
        unknown = requested - PROPERTIES
        if unknown:
            msg = f"unknown properties {sorted(unknown)}; expected a subset of {sorted(PROPERTIES)}"
            raise InputError(msg)
        flags = _validate_pbc(pbc)
        h = _validate_cell(cell, flags)
        volume = abs(float(np.linalg.det(h)))
        if "stress" in requested and (not all(flags) or volume < MIN_VOLUME):
            msg = "stress requires a three-dimensional periodic cell with a non-zero volume"
            raise UnsupportedFeatureError(msg)
        virial = bool(requested & {"strain_derivative", "stress"})
        raw = self.evaluate_atomic(
            units.length_to_bohr(_validate_positions(positions, self.n_atoms)),
            cell=units.length_to_bohr(h),
            pbc=flags,
            gradient="forces" in requested,
            virial=virial,
        )
        strain = None if raw.virial is None else raw.virial * units.HARTREE
        stress = None
        if "stress" in requested:
            stress = units.strain_derivative_to_stress(cast("FloatArray", raw.virial), volume)
        return D3Result(
            energy=units.energy_to_ev(raw.energy),
            energy_pair=units.energy_to_ev(raw.energy_pair),
            energy_atm=units.energy_to_ev(raw.energy_atm),
            forces=None if raw.gradient is None else units.gradient_to_forces(raw.gradient),
            strain_derivative=strain if "strain_derivative" in requested else None,
            stress=stress,
            info=raw.info,
        )

    def set_parameters(self, parameters: D3BJParameters) -> None:
        """Replace the damping parameters, keeping the neighbour state."""
        if not isinstance(cast("object", parameters), D3BJParameters):
            msg = "parameters must be a D3BJParameters instance"
            raise ParameterError(msg)
        self.parameters = parameters
        self.native.set_parameters(**self._parameter_kwargs())

    def set_cutoffs(self, cutoff: Cutoffs | Mapping[str, float] | None) -> None:
        """Replace the cutoffs; the neighbour lists are rebuilt on next use."""
        self.cutoffs = _resolve_cutoffs(cutoff)
        self.native.set_parameters(**self._parameter_kwargs())
        self.native.invalidate_neighbors()

    def diagnostics(self) -> dict[str, Any]:
        """Return the resolved configuration and runtime counters."""
        return {
            "n_atoms": self.n_atoms,
            "device": f"cuda:{self.device}",
            "precision": self.precision,
            "parameters": self.parameters.as_dict(),
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


def evaluate(
    numbers: npt.ArrayLike,
    positions: npt.ArrayLike,
    parameters: D3BJParameters,
    *,
    cell: npt.ArrayLike | None = None,
    pbc: npt.ArrayLike = (False, False, False),
    properties: Sequence[str] = ("energy",),
    **options: Any,
) -> D3Result:
    """Evaluate once with a temporary context (convenience for scripts)."""
    with D3Context(numbers, parameters, **options) as context:
        return context.evaluate(positions, cell=cell, pbc=pbc, properties=properties)
