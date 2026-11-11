# SPDX-License-Identifier: Apache-2.0
"""A per-world rollback in a scene with a liquid keeps the worlds that did not escalate (2026-11-08), CPU.

SPH's snapshot arrays include per-particle ones (``sph.kappa_v``, ``sph.kappa_d``, ``sph.phase``,
``particle_flags``). SPH registered its state adapter without their item->world maps, so the rollback snapshot read
them as whole arrays. Every per-world rollback in a scene with a liquid then became a whole-period re-run, and the
worlds that should have been kept were stepped again (WP4's finding from WP9's probe, [internal note]
notes/wp9-runs/sph/kin/twin_check.py). With the maps, the snapshot has no whole arrays, and a world the coupler did
not escalate comes back bitwise as its first attempt, the same as a twin run with no rollback.

The walls' impulse meter (``shape_impulse``) is part of that snapshot too (S4, 2026-11-09): it was not, so a re-run
period counted its walls' impulse twice. With ``ClockPolicy.own_cfl`` forced on, test_sph_mpm_laws' floating slab had one
rollback in 30 periods and its momentum check read 4.96e-3 N s off (bound 2.44e-4): one period's 4.9e-3 counted twice.
Here water settled in its tray, over 10 periods with one re-run in every world, books its weight on the walls (the
meter's z sum against m g t, 2 %) and the same as a twin with no re-run (1 %); a period counted twice reads 10 % high.
(A re-run here is not bitwise, about 5e-5 m in the water, so the twin is compared within that.)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy
from alloy.coupling.types import WatchState


def _sim() -> alloy.Sim:
    sc = alloy.Scene(dt=0.01, worlds=2)
    sc.add_ground()
    sc.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.3, 0.0, 0.02), mass=0.5)
    sc.add_open_box("tray", inner=(0.04, 0.04, 0.02), wall=0.004, pos=(-0.3, 0.0, 0.0))
    sc.add_liquid("water", size=(0.032, 0.032, 0.008), pos=(-0.3, 0.0, 0.009))
    sim = alloy.build(sc, device="cpu")
    assert callable(getattr(sim.engine.stepper, "rollback_requested", None))         # coupled: MuJoCo + SPH
    return sim


def _escalate_world_once(monkeypatch, st, world: int) -> None:
    """The stepper's next report escalates ``world`` alone and asks for a rollback."""
    left = [1]
    report, wants = st.report, st.rollback_requested

    def report_forced():
        r = report()
        if not left[0]:
            return r
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


def test_a_per_world_rollback_with_a_liquid_keeps_the_other_world(monkeypatch):
    twin, sim = _sim(), _sim()
    assert not sim.engine._good.whole_arrays, sim.engine._good.whole_arrays
    twin.step(1)
    _escalate_world_once(monkeypatch, sim.engine.stepper, 1)
    r0 = sim.engine.rollbacks
    sim.step(1)
    assert sim.engine.rollbacks == r0 + 1
    a, b = sim.state(["water", "free"]), twin.state(["water", "free"])
    for key in ("water.points", "free.pos"):
        assert np.array_equal(a[key][0], b[key][0]), (key, np.abs(a[key][0] - b[key][0]).max())
    assert sim.check()["ok"], sim.check()["problems"]


def _rerun_all_once(monkeypatch, st) -> None:
    """The stepper's next report asks to re-run the period in every world (same clock)."""
    left = [1]
    report, wants = st.report, st.rollback_requested

    def report_forced():
        r = report()
        return dataclasses.replace(r, notes={**(r.notes or {}), "rerun_all": True}) if left[0] else r

    def wants_forced(r) -> bool:
        if not left[0]:
            return wants(r)
        left[0] -= 1
        return True

    monkeypatch.setattr(st, "report", report_forced)
    monkeypatch.setattr(st, "rollback_requested", wants_forced)


def test_a_rerun_period_counts_its_walls_impulse_once(monkeypatch):
    from alloy.api._liquid import liquid_solvers

    twin, sim = _sim(), _sim()
    meters = [liquid_solvers(s.engine)[0] for s in (twin, sim)]
    for s, m in zip((twin, sim), meters):
        s.settle(0.2)
        m.reset_shape_meters()
    twin.step(10)
    _rerun_all_once(monkeypatch, sim.engine.stepper)
    r0 = sim.engine.rollbacks
    sim.step(10)
    assert sim.engine.rollbacks == r0 + 1
    weight = float(sim.engine.model.particle_mass.numpy()[meters[1].layout.phase.numpy() >= 1].sum()) * 9.81 * 0.1
    a, b = (float(m.shape_impulse()[:, 2].sum()) for m in (meters[1], meters[0]))
    print(f"walls' impulse z: re-run {a:.5e}, twin {b:.5e}, weight x t {-weight:.5e} N s")
    assert a == pytest.approx(-weight, rel=0.02, abs=0) and a == pytest.approx(b, rel=0.01, abs=0)
