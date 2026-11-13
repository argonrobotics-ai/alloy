# SPDX-License-Identifier: Apache-2.0
"""Gap #10b's two opt-in second simulations at their call sites: ``alloy.build(scene, rerun_worlds=n)`` (the fine
re-run pack, ``fine_rerun.py``) and ``alloy.build(scene, clock_worlds=n)`` (the per-world clock's driver,
``world_clock_run.py``). One call per site keeps ``alloy/api/sim.py``, ``alloy/api/_build.py`` and
``alloy/integration/simulation.py`` small. Without the options every hook is one ``getattr`` that finds None.
Laws: tests/coupling/test_world_clock_run.py, test_rerun_worlds_api.py."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger("alloy.coupling")

Factory = Callable[[int], Any]      # n -> an API ``Sim`` of n worlds of the same scene (``_build.build_block``)


# ------------------------------------------------------------------------------------------------ build
def check_slots(slots: Any, n: int, field: str = "rerun_worlds") -> None:
    """``rerun_worlds`` (or ``clock_worlds``) is 0 (off) or an int from 1 to ``n`` - 1 (BuildError otherwise)."""
    from alloy.api.errors import BuildError  # noqa: PLC0415  (the API imports this module's callers)

    if isinstance(slots, bool) or not isinstance(slots, (int, np.integer)) or slots < 0 or (slots and slots >= n):
        raise BuildError(f"{field} must be an int from 0 to worlds - 1 ({n - 1}), got {slots!r}",
                         f"pass {field}=0 (off) or about worlds // 16 ({max(1, n // 16)})", field=field)


def refuse_with(clock_worlds: int, rerun_worlds: int, processes: Any) -> None:
    """``clock_worlds`` works alone and in one process (BuildError naming the field otherwise)."""
    from alloy.api.errors import BuildError  # noqa: PLC0415  (the API imports this module's callers)

    if clock_worlds and (rerun_worlds or processes is not None):
        raise BuildError("clock_worlds works alone and in one process for now",
                         "drop rerun_worlds= and processes=, or clock_worlds=", field="clock_worlds")


def rerun_pack(sim: Any, make: Factory, slots: int, plugins: bool) -> None:
    """``build(..., rerun_worlds=slots)``: the fine pack (alloy.coupling.fine_rerun) on ``sim.engine``, built now so
    an out-of-memory shows at build; off, with a warning, with user plug-ins, and quietly where nothing re-runs."""
    from alloy.api import _say  # noqa: PLC0415
    from alloy.api.errors import BuildError  # noqa: PLC0415
    from alloy.coupling.fine_rerun import FineRerun  # noqa: PLC0415

    if plugins:
        _say.warn("rerun_worlds is off: a user plug-in cannot serve two simulations; re-runs run every world")
        return
    if not callable(getattr(sim.engine.stepper, "rollback_requested", None)):
        log.info("rerun_worlds off: nothing in this scene re-runs a period")
        return
    fr = FineRerun(make, slots)
    if fr.pack() is None:
        raise BuildError(f"the re-run simulation of {slots} worlds did not build: {fr.failed}",
                         "pass a smaller rerun_worlds= or rerun_worlds=0", field="rerun_worlds")
    sim.engine.fine_rerun = fr


def clock_pack(sim: Any, make: Factory, slots: int, plugins: bool) -> None:
    """``build(..., clock_worlds=slots)``: the per-world clock's fine pack and driver on ``sim.engine``, built now;
    off, with a warning that says why, where the scene cannot take it, the moving of a world between the two
    simulations included (tried once here, WP12's W3: not found later, after the second simulation was paid for)."""
    from alloy.api import _say  # noqa: PLC0415
    from alloy.api.errors import BuildError  # noqa: PLC0415
    from alloy.coupling.world_clock_run import install, scene_refusal, uninstall  # noqa: PLC0415

    why = "a user plug-in cannot serve two simulations" if plugins else scene_refusal(sim.engine)
    if why is not None:
        _say.warn(f"clock_worlds is off: {why}; every world keeps the shared exchange step")
        return
    try:
        fine = make(slots)
    except Exception as exc:  # noqa: BLE001 - any build failure (out of memory included) is the user's to size
        raise BuildError(f"the second simulation of {slots} worlds did not build: {type(exc).__name__}: {exc}",
                         "pass a smaller clock_worlds= or clock_worlds=0", field="clock_worlds") from exc
    why = install(sim.engine, fine, slots).transfer_refusal()
    if why is not None:
        uninstall(sim.engine)
        _say.warn(f"clock_worlds is off: {why}; every world keeps the shared exchange step")


# ------------------------------------------------------------------------------------------------ the engine
def begin_period(eng: Any) -> Any:
    """``Simulation.step_period``, first: the residents move (``WorldClockRun.begin_period``). The driver, or None."""
    clock = getattr(eng, "world_clock", None)
    if clock is not None:
        clock.begin_period()
    return clock


def end_period(clock: Any) -> None:
    """``Simulation.step_period``, last: the fine pack's period, the residents mirrored, the merged report."""
    if clock is not None:
        clock.end_period()


def on_reset(eng: Any, world_mask: Any) -> None:
    """``Simulation.reset``, first: the reset worlds leave the fine pack with no copy (the reset writes them)."""
    clock = getattr(eng, "world_clock", None)
    if clock is not None:
        clock.on_reset(world_mask)


# ------------------------------------------------------------------------------------------------ alloy.Sim
def evict(eng: Any, world: int | None = None) -> None:
    """A write through ``alloy.Sim`` (``set_pose``, ``settle``, a held wrench): the residents it touches (``world``;
    None: all) come back to the main simulation first, so the write lands where it is read."""
    clock = getattr(eng, "world_clock", None)
    if clock is not None:
        clock.evict(None if world is None else [int(world)])


def report(eng: Any, out: dict[str, Any]) -> dict[str, Any]:
    """``sim.report()``: ``out`` with the driver's books under ``"world_clock"``."""
    clock = getattr(eng, "world_clock", None)
    if clock is not None:
        out["world_clock"] = clock.report()
    return out


def close(eng: Any) -> None:
    """``sim.close()``: free the second simulation of either option (once: the driver is also ``fine_rerun``)."""
    seen: list[Any] = []
    for extra in (getattr(eng, "fine_rerun", None), getattr(eng, "world_clock", None)):
        if extra is not None and all(extra is not s for s in seen):
            seen.append(extra)
            extra.close()


__all__ = ["begin_period", "check_slots", "clock_pack", "close", "end_period", "evict", "on_reset", "refuse_with",
           "report", "rerun_pack"]
