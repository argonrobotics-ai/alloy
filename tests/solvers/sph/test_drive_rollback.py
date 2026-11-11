# SPDX-License-Identifier: Apache-2.0
"""A re-run control period does not run ``sim.drive`` ahead (gaps-now #1, 2026-11-08), CPU.

A coupled scene (a free body beside a liquid: MuJoCo + SPH) re-runs a control period from its start when the coupler
asks for a rollback. The rollback restores the state and the simulation's clock. A drive keeps a clock of its own
(``alloy.api._drive.Drive``), and that clock used to keep the discarded attempts' time, so after every rollback the
driven object jumped ahead by a period's motion. At a tilt pour's start, three rollbacks put the jug 30 ms ahead in
one exchange, and its walls flung corner particles out ([internal note] notes/wp9-runs/sph/kin). Here three rollbacks are
forced in the first period of a jug's tilt about its lip. Two periods later the jug (one kinematic body, its walls
shapes on it) is bitwise where a twin run with no rollbacks put it. Before the fix they were three periods ahead. The
forcing comes in two forms: every world re-run (one world), or world 1 of two escalated. The drive's clock is shared
by all worlds, so both worlds' jugs must match the twin in both forms.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import alloy
from alloy.coupling.types import WatchState

PARTS = ["jug", "jug_floor"]            # one body (a kinematic fixture), by its name and its floor's


def _sim(worlds: int) -> alloy.Sim:
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    sc.add_open_box("jug", inner=(0.05, 0.05, 0.05), wall=0.004, pos=(0.0, 0.0, 0.1), kinematic=True)
    sc.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.3, 0.0, 0.02), mass=0.5)
    sc.add_open_box("tray", inner=(0.04, 0.04, 0.02), wall=0.004, pos=(-0.3, 0.0, 0.0))
    sc.add_liquid("water", size=(0.032, 0.032, 0.008), pos=(-0.3, 0.0, 0.009))
    sim = alloy.build(sc, device="cpu")
    assert callable(getattr(sim.engine.stepper, "rollback_requested", None))         # coupled: MuJoCo + SPH
    sim.step(1)
    sim.drive("jug", angular_velocity=(0.0, math.radians(100), 0.0), pivot=(0.029, 0.0, 0.154), duration=1.0)
    return sim


def _force_rollbacks(monkeypatch, st, n: int, world: int | None) -> None:
    """The stepper's next ``n`` reports ask to re-run the period: every world (``world`` None), or ``world`` alone."""
    left = [n]
    report, wants = st.report, st.rollback_requested

    def report_forced():
        r = report()
        if not left[0]:
            return r
        if world is None:
            return dataclasses.replace(r, notes={**(r.notes or {}), "rerun_all": True})
        ws = np.array(r.watch_state, copy=True)
        ws[world] = int(WatchState.ESCALATED)
        return dataclasses.replace(r, watch_state=ws)

    def wants_forced(r) -> bool:
        if not left[0]:
            return wants(r)
        left[0] -= 1
        return True

    monkeypatch.setattr(st, "report", report_forced)
    monkeypatch.setattr(st, "rollback_requested", wants_forced)


def _poses(sim) -> np.ndarray:
    s = sim.state(PARTS)
    return np.concatenate([np.concatenate([s[f"{p}.pos"], s[f"{p}.quat_xyzw"]], axis=-1) for p in PARTS], axis=-1)


@pytest.mark.parametrize("worlds,rerun", [(1, None), (2, 1)], ids=["every_world", "one_of_two"])
def test_a_kinematic_jug_after_three_rollbacks_lands_where_a_run_without_them_does(monkeypatch, worlds, rerun):
    twin, sim = _sim(worlds), _sim(worlds)
    r0 = twin.engine.rollbacks
    twin.step(1)
    _force_rollbacks(monkeypatch, sim.engine.stepper, 3, rerun)
    r1 = sim.engine.rollbacks
    sim.step(1)
    assert sim.engine.rollbacks == r1 + 3
    assert np.array_equal(_poses(sim), _poses(twin)), np.abs(_poses(sim) - _poses(twin)).max()
    twin.step(1)
    sim.step(1)
    assert twin.engine.rollbacks == r0                     # the twin re-runs nothing on its own
    assert sim.t == twin.t
    assert np.array_equal(_poses(sim), _poses(twin)), np.abs(_poses(sim) - _poses(twin)).max()
    assert sim.check()["ok"], sim.check()["problems"]
