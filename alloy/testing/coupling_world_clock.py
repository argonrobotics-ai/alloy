# SPDX-License-Identifier: Apache-2.0
"""The two-pack driver's rig (``alloy.build(scene, clock_worlds=n)``, alloy/coupling/world_clock_run.py; gap #10b
P2), shared by tests/coupling/test_world_clock_run.py and test_world_clock_review.py.

4 worlds, a 3 cm box on a sheet on a table, the box's density drawn per world, so one world's own clock need is one
level finer than every other's (test_world_clock_laws.py's rig). ``tripping`` injects a watchdog trip in chosen
periods (the first attempt's report says that world escalated, as test_fine_rerun's rig)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

import alloy
from alloy.coupling.types import WatchState

W, SLOTS, SEED = 4, 1, 0
LIGHT, HEAVY = 500.0, 8000.0
NAMES = ["box", "cloth"]


def scene(lo: float = LIGHT, hi: float = HEAVY) -> alloy.Scene:
    side = 0.03
    scene = alloy.Scene(worlds=W, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=(0, 0, 0.104 + 0.001 + side / 2), density=hi)
    scene.randomize("box", "density", lo, hi)
    return scene


def state(sim: Any) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in sim.state(NAMES).items() if k != "t"}


def level(sim: Any) -> int:
    return int(round(np.log2(sim.dt / float(sim.engine._last_coupling.dt_x))))


def escalated(rep: Any, row: int) -> Any:
    """``rep`` with ``row``'s watch state ESCALATED (the first attempt's report says that world tripped)."""
    ws = np.asarray(rep.watch_state).copy()
    ws[row] = int(WatchState.ESCALATED)
    return dataclasses.replace(rep, watch_state=ws)


def tripping(clock_worlds: int, world: int = 1, period: int | tuple[int, ...] = 2, periods: int = 4,
              setup: Any = None) -> tuple[list[dict], Any]:
    """Identical worlds (nobody resides by need); ``world`` trips once in each ``period`` (the first attempt's report
    says it escalated, as test_fine_rerun's rig). ``setup(sim)`` runs after the build (more injections). Per period:
    states, the coarse level, floor and trips, exchanges, rollbacks, residents."""
    trips = {period} if isinstance(period, int) else set(period)
    opt = {"clock_worlds": clock_worlds} if clock_worlds else {}
    sim = alloy.build(scene(HEAVY, HEAVY), device="cpu", seed=SEED, **opt)
    eng, st = sim.engine, sim.engine.stepper
    report, seen, armed = st.report, set(), []      # armed: only the report taken inside the step trips, not a read

    def tripped() -> Any:
        rep = report()
        if armed and eng.periods not in seen:
            seen.add(eng.periods)
            rep = escalated(rep, world)
        return rep

    st.report = tripped
    if setup is not None:
        setup(sim)
    rows = []
    for _ in range(periods):
        armed[:] = [True] if eng.periods in trips else []
        sim.step(1)
        armed.clear()
        clock = getattr(eng, "world_clock", None)
        rows.append({"state": state(sim), "level": level(sim), "rollbacks": eng.rollbacks,
                     "taken": st.clock_census()["world_exchanges"], "ok": sim.check()["ok"],
                     "floor": int(st._clock.j_floor), "trips": int(getattr(st, "_trips", 0)),
                     "resident": None if clock is None else clock.resident.copy(),
                     "book": None if clock is None else sim.report()["world_clock"],
                     "dt": np.asarray(eng._last_coupling.notes.get("dt_x_world", ()))})
    return rows, sim


__all__ = ["HEAVY", "LIGHT", "NAMES", "SEED", "SLOTS", "W", "escalated", "level", "scene", "state", "tripping"]
