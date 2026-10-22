"""ASE calculators for the vanderfast D3(BJ) and D4 dispersion corrections.

Example:
    >>> from ase.build import molecule
    >>> from vanderfast.ase import D3Calculator, D4Calculator
    >>> atoms = molecule("H2O")
    >>> atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
    >>> energy = atoms.get_potential_energy()  # doctest: +SKIP
    >>> atoms.calc = D4Calculator(method="pbe")
    >>> energy = atoms.get_potential_energy()  # doctest: +SKIP

Combine with another calculator through ``ase.calculators.mixing.SumCalculator``.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, cast, override

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from vanderfast import units
from vanderfast.context import D3Context, parse_device
from vanderfast.context_d4 import EWALD_MODES, D4Context, EEQSolverOptions
from vanderfast.errors import InputError, UnsupportedFeatureError
from vanderfast.parameters import Cutoffs, D3BJParameters, resolve_parameters
from vanderfast.parameters_d4 import D4Cutoffs, D4Parameters, resolve_d4_parameters
from vanderfast.reference import ReferenceEvaluator
from vanderfast.reference_d4 import D4ReferenceEvaluator

_PARAMETER_KEYS: Final = frozenset({"method", "damping", "atm", "d3_parameters"})
_CONTEXT_KEYS: Final = frozenset({"backend", "device", "precision", "skin"})
_IGNORED_CHANGES: Final = frozenset({"initial_charges", "initial_magmoms"})
_BACKENDS: Final = ("cuda", "reference")


class D3Calculator(Calculator):
    """D3(BJ) dispersion energies, forces, and stress for ASE.

    Args:
        method: Functional name from the simple-dftd3 v1.6.0 table.
        damping: Damping family; only ``"bj"``.
        atm: Whether the ATM three-body term is included. Required with
            ``method``; there is no implicit default.
        parameters: A complete :class:`~vanderfast.D3BJParameters` instead of
            ``method``/``damping``/``atm``.
        backend: ``"cuda"`` (default) or ``"reference"`` (the dftd3 library).
        device: CUDA device for the ``cuda`` backend.
        precision: Only ``"float64"``.
        cutoff: Optional ``{"cn", "disp2", "disp3"}`` overrides in Å.
        skin: Neighbour-list skin in Å for the ``cuda`` backend.
        **kwargs: Passed to :class:`ase.calculators.calculator.Calculator`.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]  # noqa: RUF012
    default_parameters = {  # noqa: RUF012
        "method": None,
        "damping": None,
        "atm": None,
        "d3_parameters": None,
        "backend": "cuda",
        "device": "cuda:0",
        "precision": "float64",
        "cutoff": None,
        "skin": 1.0,
    }

    def __init__(
        self,
        *,
        method: str | None = None,
        damping: str | None = None,
        atm: bool | None = None,
        parameters: D3BJParameters | None = None,
        backend: str = "cuda",
        device: str | int = "cuda:0",
        precision: str = "float64",
        cutoff: Mapping[str, float] | None = None,
        skin: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Resolve the configuration; the CUDA context is created on first use."""
        self._context: D3Context | None = None
        self.evaluations = 0
        self._resolved: D3BJParameters | None = None
        self._cutoffs = Cutoffs()
        Calculator.__init__(
            self,
            method=method,
            damping=damping,
            atm=atm,
            d3_parameters=parameters,
            backend=backend,
            device=device,
            precision=precision,
            cutoff=cutoff,
            skin=skin,
            **kwargs,
        )

    @override
    def set(self, **kwargs: Any) -> dict[str, Any]:
        """Change parameters; clears results and updates native state as needed.

        ``parameters=`` is stored as ``d3_parameters`` because ASE reserves
        the ``parameters`` keyword for parameter files.
        """
        if "parameters" in kwargs:
            kwargs["d3_parameters"] = kwargs.pop("parameters")
        changed: dict[str, Any] = Calculator.set(self, **kwargs)
        if not changed and self._resolved is not None:
            return changed
        p = self.parameters
        if p["backend"] not in _BACKENDS:
            msg = f"backend must be one of {_BACKENDS}, got {p['backend']!r}"
            raise InputError(msg)
        parse_device(p["device"])
        self._resolved = resolve_parameters(
            method=p["method"], damping=p["damping"], atm=p["atm"], parameters=p["d3_parameters"]
        )
        self._cutoffs = Cutoffs.resolve(p["cutoff"])
        keys = set(changed)
        if keys & _CONTEXT_KEYS:
            self._context = None
        elif self._context is not None:
            if keys & _PARAMETER_KEYS:
                self._context.set_parameters(self._resolved)
            if "cutoff" in keys:
                self._context.set_cutoffs(self._cutoffs)
        Calculator.reset(self)
        return changed

    @property
    def resolved_parameters(self) -> D3BJParameters:
        """The fully resolved damping parameters."""
        assert self._resolved is not None  # noqa: S101 (set() always resolves)
        return self._resolved

    @override
    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        """Report changes that affect D3; charges and magnetic moments do not."""
        changes: list[str] = Calculator.check_state(self, atoms, tol)
        return [change for change in changes if change not in _IGNORED_CHANGES]

    def _cuda_context(self, atoms: Atoms) -> D3Context:
        if self._context is None or not np.array_equal(self._context.numbers, atoms.numbers):
            p = self.parameters
            self._context = D3Context(
                atoms.numbers,
                self.resolved_parameters,
                device=p["device"],
                precision=p["precision"],
                cutoff=self._cutoffs,
                skin=p["skin"],
            )
        return self._context

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = tuple(all_changes),
    ) -> None:
        """Evaluate the requested properties and store them in ``results``."""
        Calculator.calculate(self, atoms, properties, system_changes)
        self.evaluations += 1
        atoms = cast("Atoms", self.atoms)
        pbc = np.array(atoms.pbc, dtype=np.bool_)
        want_stress = "stress" in properties
        if want_stress and not pbc.all():
            msg = "stress is only available for three-dimensional periodic cells"
            raise PropertyNotImplementedError(msg)
        want_forces = "forces" in properties or want_stress
        # On a 3D cell the virial costs a few percent of a force evaluation, so
        # compute stress with forces: ASE asks for them in separate calls, and
        # NPT dynamics would otherwise pay for two full evaluations per step.
        want_stress = want_stress or (want_forces and bool(pbc.all()))
        cell = atoms.cell[:] if pbc.any() else None
        if self.parameters["backend"] == "cuda":
            requested = ["energy"] + ["forces"] * want_forces + ["stress"] * want_stress
            result = self._cuda_context(atoms).evaluate(
                atoms.positions, cell=cell, pbc=pbc, properties=requested
            )
            energy, forces, stress = result.energy, result.forces, result.stress
        else:
            energy, forces, stress = self._reference(atoms, cell, pbc, want_forces, want_stress)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if want_forces:
            self.results["forces"] = forces
        if want_stress:
            self.results["stress"] = stress

    def _reference(
        self,
        atoms: Atoms,
        cell: np.ndarray | None,
        pbc: np.ndarray,
        want_forces: bool,
        want_stress: bool,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        evaluator = ReferenceEvaluator(atoms.numbers, self.resolved_parameters, self._cutoffs)
        result = evaluator.evaluate(
            units.length_to_bohr(atoms.positions),
            None if cell is None else units.length_to_bohr(cell),
            pbc,
            gradient=want_forces,
        )
        forces = None if result.gradient is None else units.gradient_to_forces(result.gradient)
        stress = None
        if want_stress and result.virial is not None:
            stress = units.strain_derivative_to_stress(result.virial, atoms.get_volume())
        return units.energy_to_ev(result.energy), forces, stress

    @override
    def todict(self, skip_default: bool = True) -> dict[str, Any]:
        """Return JSON-serialisable calculator parameters."""
        result: dict[str, Any] = Calculator.todict(self, skip_default=skip_default)
        if isinstance(result.get("d3_parameters"), D3BJParameters):
            result["d3_parameters"] = result["d3_parameters"].as_dict()
        return result


_D4_PARAMETER_KEYS: Final = frozenset({"method", "atm", "d4_parameters"})
_D4_CONTEXT_KEYS: Final = frozenset({"backend", "device", "precision", "skin", "ewald", "solver"})


class D4Calculator(Calculator):
    """DFT-D4 dispersion energies, forces, stress, and EEQ charges for ASE.

    Args:
        method: Functional name from the dftd4 v4.2.0 table.
        atm: Whether the ATM three-body term is included (default ``True``
            with ``method``, as in dftd4).
        parameters: A complete :class:`~vanderfast.D4Parameters` instead of
            ``method``/``atm``.
        charge: Total charge in e. ``None`` (default) uses the sum of the
            atoms' initial charges, as ``dftd4.ase`` does.
        backend: ``"cuda"`` (default) or ``"reference"`` (the dftd4 library).
        device: CUDA device for the ``cuda`` backend.
        precision: Only ``"float64"``.
        cutoff: Optional ``{"cn", "disp2", "disp3"}`` overrides in Å.
        skin: Neighbour-list skin in Å for the ``cuda`` backend.
        ewald: Periodic charge model, ``"upstream"`` (dftd4's) or
            ``"converged"`` (``cuda`` backend only).
        solver: Charge-solver options for the ``cuda`` backend.
        **kwargs: Passed to :class:`ase.calculators.calculator.Calculator`.

    As in dftd4, any periodic direction makes the system fully periodic over
    the cell; slabs and wires need vacuum lattice vectors.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges"]  # noqa: RUF012
    default_parameters = {  # noqa: RUF012
        "method": None,
        "atm": None,
        "d4_parameters": None,
        "charge": None,
        "backend": "cuda",
        "device": "cuda:0",
        "precision": "float64",
        "cutoff": None,
        "skin": 1.0,
        "ewald": "upstream",
        "solver": None,
    }

    def __init__(
        self,
        *,
        method: str | None = None,
        atm: bool | None = None,
        parameters: D4Parameters | None = None,
        charge: float | None = None,
        backend: str = "cuda",
        device: str | int = "cuda:0",
        precision: str = "float64",
        cutoff: Mapping[str, float] | None = None,
        skin: float = 1.0,
        ewald: str = "upstream",
        solver: EEQSolverOptions | None = None,
        **kwargs: Any,
    ) -> None:
        """Resolve the configuration; the CUDA context is created on first use."""
        self._context: D4Context | None = None
        self.evaluations = 0
        self._resolved: D4Parameters | None = None
        self._cutoffs = D4Cutoffs()
        Calculator.__init__(
            self,
            method=method,
            atm=atm,
            d4_parameters=parameters,
            charge=charge,
            backend=backend,
            device=device,
            precision=precision,
            cutoff=cutoff,
            skin=skin,
            ewald=ewald,
            solver=solver,
            **kwargs,
        )

    @override
    def set(self, **kwargs: Any) -> dict[str, Any]:
        """Change parameters; clears results and updates native state as needed.

        ``parameters=`` is stored as ``d4_parameters`` because ASE reserves
        the ``parameters`` keyword for parameter files.
        """
        if "parameters" in kwargs:
            kwargs["d4_parameters"] = kwargs.pop("parameters")
        changed: dict[str, Any] = Calculator.set(self, **kwargs)
        if not changed and self._resolved is not None:
            return changed
        p = self.parameters
        if p["backend"] not in _BACKENDS:
            msg = f"backend must be one of {_BACKENDS}, got {p['backend']!r}"
            raise InputError(msg)
        if p["ewald"] not in EWALD_MODES:
            msg = f"ewald must be one of {EWALD_MODES}, got {p['ewald']!r}"
            raise InputError(msg)
        if p["backend"] == "reference" and p["ewald"] != "upstream":
            msg = "the dftd4 reference backend only provides ewald='upstream'"
            raise UnsupportedFeatureError(msg)
        parse_device(p["device"])
        self._resolved = resolve_d4_parameters(
            method=p["method"], atm=p["atm"], parameters=p["d4_parameters"]
        )
        self._cutoffs = D4Cutoffs.resolve(p["cutoff"])
        keys = set(changed)
        if keys & _D4_CONTEXT_KEYS:
            self._context = None
        elif self._context is not None:
            if keys & _D4_PARAMETER_KEYS:
                self._context.set_parameters(self._resolved)
            if "cutoff" in keys:
                self._context.set_cutoffs(self._cutoffs)
        Calculator.reset(self)
        return changed

    @property
    def resolved_parameters(self) -> D4Parameters:
        """The fully resolved D4 parameters."""
        assert self._resolved is not None  # noqa: S101 (set() always resolves)
        return self._resolved

    def total_charge(self, atoms: Atoms) -> float:
        """Return the total charge: the explicit setting or the sum of initial charges."""
        explicit = self.parameters["charge"]
        if explicit is not None:
            return float(explicit)
        return float(atoms.get_initial_charges().sum())

    @override
    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        """Report changes that affect D4; initial charges only without an explicit charge."""
        changes: list[str] = Calculator.check_state(self, atoms, tol)
        ignored = {"initial_magmoms"}
        if self.parameters["charge"] is not None:
            ignored.add("initial_charges")
        return [change for change in changes if change not in ignored]

    def _cuda_context(self, atoms: Atoms, charge: float) -> D4Context:
        if self._context is None or not np.array_equal(self._context.numbers, atoms.numbers):
            p = self.parameters
            self._context = D4Context(
                atoms.numbers,
                self.resolved_parameters,
                charge=charge,
                device=p["device"],
                precision=p["precision"],
                cutoff=self._cutoffs,
                skin=p["skin"],
                ewald=p["ewald"],
                solver=p["solver"],
            )
        elif self._context.charge != charge:
            self._context.set_charge(charge)
        return self._context

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Sequence[str] = tuple(all_changes),
    ) -> None:
        """Evaluate the requested properties and store them in ``results``."""
        Calculator.calculate(self, atoms, properties, system_changes)
        self.evaluations += 1
        atoms = cast("Atoms", self.atoms)
        pbc = np.array(atoms.pbc, dtype=np.bool_)
        periodic = bool(pbc.any())
        want_stress = "stress" in properties
        if want_stress and not periodic:
            msg = "stress is only available for periodic systems"
            raise PropertyNotImplementedError(msg)
        want_forces = "forces" in properties or want_stress
        # The virial costs a few percent of a force evaluation; compute stress
        # with forces so NPT dynamics pays for one evaluation per step.
        want_stress = want_stress or (want_forces and periodic)
        cell = atoms.cell[:] if periodic else None
        charge = self.total_charge(atoms)
        if self.parameters["backend"] == "cuda":
            requested = ["energy"] + ["forces"] * want_forces + ["stress"] * want_stress
            result = self._cuda_context(atoms, charge).evaluate(
                atoms.positions, cell=cell, pbc=pbc, properties=requested
            )
            energy, forces, stress, charges = (
                result.energy,
                result.forces,
                result.stress,
                result.charges,
            )
        else:
            energy, forces, stress, charges = self._reference(
                atoms, cell, pbc, charge, want_forces, want_stress
            )
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["charges"] = charges
        if want_forces:
            self.results["forces"] = forces
        if want_stress:
            self.results["stress"] = stress

    def _reference(
        self,
        atoms: Atoms,
        cell: np.ndarray | None,
        pbc: np.ndarray,
        charge: float,
        want_forces: bool,
        want_stress: bool,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray]:
        evaluator = D4ReferenceEvaluator(
            atoms.numbers, self.resolved_parameters, self._cutoffs, charge=charge
        )
        # Always take dftd4's gradient path: its charges (and hence energy) differ from
        # the energy-only path in the last bits, and dftd4.ase always uses it.
        result = evaluator.evaluate(
            units.length_to_bohr_d4(atoms.positions),
            None if cell is None else units.length_to_bohr_d4(cell),
            pbc,
            gradient=True,
            properties=True,
        )
        gradient = cast("np.ndarray", result.gradient)
        forces = units.gradient_to_forces_d4(gradient) if want_forces else None
        stress = None
        if want_stress and result.virial is not None:
            stress = units.strain_derivative_to_stress_d4(result.virial, atoms.get_volume())
        charges = cast("np.ndarray", result.charges)
        return units.energy_to_ev(result.energy), forces, stress, charges

    @override
    def todict(self, skip_default: bool = True) -> dict[str, Any]:
        """Return JSON-serialisable calculator parameters."""
        result: dict[str, Any] = Calculator.todict(self, skip_default=skip_default)
        if isinstance(result.get("d4_parameters"), D4Parameters):
            result["d4_parameters"] = result["d4_parameters"].as_dict()
        if isinstance(result.get("solver"), EEQSolverOptions):
            result["solver"] = result["solver"].native_kwargs()
        return result
