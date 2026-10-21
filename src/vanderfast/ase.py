"""ASE calculator for the vanderfast D3(BJ) dispersion correction.

Example:
    >>> from ase.build import molecule
    >>> from vanderfast.ase import D3Calculator
    >>> atoms = molecule("H2O")
    >>> atoms.calc = D3Calculator(method="pbe", damping="bj", atm=True)
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
from vanderfast.errors import InputError
from vanderfast.parameters import Cutoffs, D3BJParameters, resolve_parameters
from vanderfast.reference import ReferenceEvaluator

_PARAMETER_KEYS: Final = frozenset({"method", "damping", "atm", "d3_parameters"})
_CONTEXT_KEYS: Final = frozenset({"backend", "device", "precision", "cutoff", "skin"})
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
        elif keys & _PARAMETER_KEYS and self._context is not None:
            self._context.set_parameters(self._resolved)
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
        atoms = cast("Atoms", self.atoms)
        pbc = np.array(atoms.pbc, dtype=np.bool_)
        want_stress = "stress" in properties
        if want_stress and not pbc.all():
            msg = "stress is only available for three-dimensional periodic cells"
            raise PropertyNotImplementedError(msg)
        want_forces = "forces" in properties or want_stress
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
