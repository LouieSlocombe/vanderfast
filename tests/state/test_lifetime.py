"""Context lifetime, result ownership, concurrency, and allocation failures."""

import threading

import numpy as np
import pytest
from support import native_helpers as nh
from support.inputs import SHORT_D3

from vanderfast import resolve_parameters
from vanderfast.errors import ContextBusyError, DeviceOutOfMemoryError
from vanderfast.testing import structures
from vanderfast.units import INV_BOHR

pytestmark = pytest.mark.cuda

PBE_ATM = resolve_parameters(method="pbe", damping="bj", atm=True)


def _system() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    atoms = structures.co2_crystal()
    return atoms.numbers, atoms.positions * INV_BOHR, atoms.cell[:] * INV_BOHR, (True, True, True)


def test_repeated_construction_and_destruction() -> None:
    numbers, x, cell, pbc = _system()
    energies = set()
    for _ in range(20):
        ctx = nh.context(numbers, PBE_ATM, SHORT_D3)
        energies.add(nh.run(ctx, x, cell, pbc)["energy"])
        assert ctx.stats()["device_bytes"] > 0
        del ctx
    assert len(energies) == 1


def test_results_are_owned_copies() -> None:
    numbers, x, cell, pbc = _system()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3)
    first = nh.run(ctx, x, cell, pbc)
    reference = first["gradient"].copy()
    first["gradient"][:] = 123.0
    second = nh.run(ctx, x, cell, pbc)
    np.testing.assert_allclose(second["gradient"], reference, rtol=1e-12, atol=1e-18)


def test_separate_contexts_run_concurrently() -> None:
    numbers, x, cell, pbc = _system()
    serial = nh.run(nh.context(numbers, PBE_ATM, SHORT_D3), x, cell, pbc)["energy"]
    results: list[float] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            ctx = nh.context(numbers, PBE_ATM, SHORT_D3)
            results.extend(nh.run(ctx, x, cell, pbc)["energy"] for _ in range(3))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert results == [serial] * 12


def _run_until_busy(
    ctx: object, x: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool]
) -> None:
    for _ in range(1000):
        nh.run(ctx, x, cell, pbc)


def test_overlapping_use_of_one_context_is_rejected() -> None:
    numbers, x, cell, pbc = _system()
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3)
    started = threading.Event()

    def hold() -> None:
        started.set()
        ctx._testing_hold(0.5)  # noqa: SLF001

    thread = threading.Thread(target=hold)
    thread.start()
    started.wait()
    with pytest.raises(ContextBusyError):
        _run_until_busy(ctx, x, cell, pbc)
    thread.join()
    assert nh.run(ctx, x, cell, pbc)["energy"] < 0.0


def test_allocation_failure_leaves_the_context_usable() -> None:
    numbers, x, cell, pbc = _system()
    reference = nh.run(nh.context(numbers, PBE_ATM, SHORT_D3), x, cell, pbc)
    ctx = nh.context(numbers, PBE_ATM, SHORT_D3)
    ctx._testing_set_allocation_limit(ctx.stats()["device_bytes"] + 1024)  # noqa: SLF001
    with pytest.raises(DeviceOutOfMemoryError):
        nh.run(ctx, x, cell, pbc)
    ctx._testing_set_allocation_limit(2**62)  # noqa: SLF001
    result = nh.run(ctx, x, cell, pbc)
    assert result["energy"] == reference["energy"]
    np.testing.assert_allclose(result["gradient"], reference["gradient"], rtol=1e-12, atol=1e-18)
