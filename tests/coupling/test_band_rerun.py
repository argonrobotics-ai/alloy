# SPDX-License-Identifier: Apache-2.0
"""The band re-run (alloy/coupling/contact_band.py, R2-E3 follow-up, coordinator 2026-11-08 09:16Z), CPU.

The contact-band bound reaches the clock one period late. A period whose own bound falls under dt_x / 3 (a contact
crossed more than half its band per exchange) is re-run, every world, at the measured bound; the firing is counted.
The band speed is the particle's speed relative to the body point it touches, with the body's own motion at that point
(Newton's record velocity is only a shape's surface velocity). Fast tests: the decision and the body term on stand-in
records, and that IntegratedSimulation turns re-runs on. Slow law: a 5 kg box dropped 0.5 m onto a sheet lands at
3.1 m/s inside a 0.625 ms period; that period re-runs at its own bound. WP11's L4 replay of B3 N220 world 32 is the
GPU row ([internal note])."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.coupling import contact_band as CB
from alloy.coupling.types import ClockPolicy

GAP = 0.003                    # m: the full-surface band
TOP = 0.10                     # m: table top
DROP = 0.5                     # m: the box's fall onto the sheet
BOX_KG = 5.0


def test_a_period_its_own_bound_outran_is_re_run_once():
    """bound < dt_x / 3 fires, once per period, never at the clock's finest step nor for a bound finer than it, only
    with the driver's consent."""
    r = CB.BandRerun()
    assert r.ratio == ClockPolicy().band_rerun == CB.BAND_RERUN == 3.0
    assert not r.judge(3.125e-4, 1.0e-4, 1.0e-5)                     # the driver has not turned it on
    r.enabled = True
    assert r.judge(3.125e-4, 1.0e-4, 1.0e-5) and r.whole() == {"rerun_all": True}
    assert not r.judge(3.125e-4, 3.125e-4 / 3.0, 1.0e-5)             # at dt_x / 3: half the band, no re-run
    assert not r.judge(3.125e-4, 1.0e-5, 3.125e-4)                   # already the finest step: nothing to refine
    assert not r.judge(None, 1.0e-5, 1.0e-5) and r.whole() == {}
    assert r.notes()["unreachable"] == 1                              # the finest-step case above counted
    # test_wiring_fallback's cloth started inside its box: bound 5.95 µs, finest 19.5 µs (0.01 s / 2^9): no re-run
    assert not r.judge(1.5625e-4, 5.95e-6, 0.01 / 2**9) and r.notes()["unreachable"] == 2
    assert r.judge(1.5625e-4, 0.01 / 2**9, 0.01 / 2**9)              # a bound the finest step reaches still fires
    assert r.judge(3.125e-4, 1.0e-4, 1.0e-5)
    r.accept(np.array([5.0]), 3.125e-4, 1.0e-4)
    assert r.begin_attempt() and not r.begin_attempt()               # the next attempt is the re-run, once
    assert not r.judge(1.5625e-4, 1.0e-5, 1.0e-6)                    # one per period
    assert r.notes() == {"rerun": False, "reruns": 1, "unreachable": 2}
    r.new_period()
    assert r.judge(1.5625e-4, 1.0e-5, 1.0e-6)
    off = CB.BandRerun(None)
    off.enabled = True
    assert not off.on and not off.judge(3.125e-4, 1.0e-6, 1.0e-6)


def _band(body_qd, *, shape_body=(0,), shape=0, body_vel=(0.0, 0.0, 0.0), with_bodies=True):
    """One pipeline, one vertex record of particle 0 (at rest) on shape ``shape`` at body point (0.1, 0, 0) of the
    view's body 0, the scene's body 1 (centre of mass at its origin, pose identity, twist ``body_qd`` at the exchange
    start): the period's band speed."""
    dev = "cpu"
    contacts = SimpleNamespace(
        soft_contact_max=1, soft_contact_count=wp.array([1], dtype=wp.int32, device=dev),
        soft_contact_indices=wp.array(np.array([[0, -1, -1]], np.int32), dtype=wp.vec3i, device=dev),
        soft_contact_shape=wp.array([shape], dtype=wp.int32, device=dev),
        soft_contact_body_pos=wp.array(np.array([[0.1, 0.0, 0.0]], np.float32), dtype=wp.vec3, device=dev),
        soft_contact_body_vel=wp.array(np.array([body_vel], np.float32), dtype=wp.vec3, device=dev))
    view = SimpleNamespace(shape_body=wp.array(np.asarray(shape_body, np.int32), dtype=wp.int32, device=dev))
    scene = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=dev),
                            body_qd=wp.array([wp.spatial_vector()] + [wp.spatial_vector(*body_qd)],
                                             dtype=wp.spatial_vector, device=dev))
    model = SimpleNamespace(body_com=wp.zeros(2, dtype=wp.vec3, device=dev))
    entry = SimpleNamespace(state_1=SimpleNamespace(particle_qd=wp.zeros(1, dtype=wp.vec3, device=dev)),
                            solver=SimpleNamespace(model=SimpleNamespace(particle_mass=None)),
                            body_local_to_global=wp.array([1], dtype=wp.int32, device=dev))   # view body 0: scene's 1
    factory = SimpleNamespace(built=[(view, SimpleNamespace(soft_contact_gap=GAP), contacts, None)])
    band = CB.ContactBandBound([(factory, "cloth")], dev)
    band.accumulate({"cloth": entry}, scene if with_bodies else None, model)
    band.end_period()
    return float(band.last_speed[0])


def test_the_band_speed_includes_the_body_points_motion():
    """A box closing on a still sheet at 4 m/s, spinning at 10 rad/s about z: the point 0.1 m off its centre moves
    (0, 1, -4) m/s. Newton's record velocity alone (zero for a rigid box) read 0 before the follow-up."""
    twist = (0.0, 0.0, -4.0, 0.0, 0.0, 10.0)                          # (v, w): Newton's spatial order
    assert _band(twist) == pytest.approx(math.hypot(1.0, 4.0), rel=1e-6, abs=0)
    assert _band(twist, body_vel=(0.0, 0.0, -1.0)) == pytest.approx(math.hypot(1.0, 5.0), rel=1e-6, abs=0)
    assert _band(twist, shape_body=(-1,), body_vel=(0.0, 2.0, 0.0)) == pytest.approx(2.0, rel=1e-6, abs=0)   # static
    assert _band(twist, with_bodies=False, body_vel=(0.0, 2.0, 0.0)) == pytest.approx(2.0, rel=1e-6, abs=0)


def _drop_scene() -> alloy.Scene:
    side = (BOX_KG / 8000.0) ** (1.0 / 3.0)                           # 8,000 kg/m3: a steel-like block, 8.5 cm
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, TOP), pos=(0, 0, TOP / 2), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, TOP + 0.002), resolution=(11, 11), mass=0.03,
                    thickness=0.004, self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=(0, 0, TOP + 0.004 + DROP + side / 2), mass=BOX_KG)
    return scene


def _scheduler(sim):
    st = sim.engine.stepper
    return getattr(st, "base", st)


def test_integrated_simulation_turns_band_reruns_on():
    """The driver re-runs whole periods (rerun_all), so it turns band re-runs on; the falling box's clock is the
    0.625 ms the law below forces past."""
    sim = alloy.build(_drop_scene(), device="cpu")
    sim.step(1)
    assert _scheduler(sim)._band_rerun.on
    assert sim.engine._last_coupling.dt_x == pytest.approx(6.25e-4, rel=1e-12)


def test_replay_period_repeats_the_attempt_and_counts_nothing():
    """``replay_period`` (WP12's MPM grid growth re-runs a period unchanged): the next attempt takes the same dt_x and
    substeps, no rollback or band re-run is counted, nothing escalates, and the discarded attempt's band speeds go."""
    sim = alloy.build(_drop_scene(), device="cpu")
    sim.step(2)
    st, eng = _scheduler(sim), sim.engine
    before = (st._dt_x, dict(st._substeps_now), st._rollbacks, st._band_rerun.total, getattr(st, "_trips", 0))
    st._band.vmax.fill_(9.0)
    st.replay_period()
    dt_x, substeps = st.choose_clock(eng.clock.control_dt, eng._last_coupling)
    assert (dt_x, substeps) == before[:2]
    assert (st._rollbacks, st._band_rerun.total, getattr(st, "_trips", 0)) == before[2:]
    assert float(st._band.vmax.numpy().max()) == 0.0


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_box_landing_on_cloth_reruns_its_period_at_the_band_bound():
    """Law: the box lands at 3.1 m/s inside a 0.625 ms period. That attempt's own band bound is under dt_x / 3 (the
    box crossed more than half the 3 mm band per exchange), so the period re-runs once, at no more than that bound,
    and the run stays healthy. The firing is in the report (band.reruns) and the simulation's rollback count."""
    sim = alloy.build(_drop_scene(), device="cpu")
    rerun = _scheduler(sim)._band_rerun
    fired: list[tuple[float, float, float]] = []
    accept = rerun.accept

    def spy(speeds, dt_x, bound):
        fired.append((float(np.max(speeds)), float(dt_x), float(bound)))
        accept(speeds, dt_x, bound)

    rerun.accept = spy
    landed = None
    for k in range(60):
        sim.step(1)
        if fired and landed is None:
            landed = k
            rep = sim.engine._last_coupling
        if landed is not None and k >= landed + 2:
            break
    assert landed is not None, "no band re-run"
    v, dt_x, bound = fired[0]
    print(f"band re-run at period {landed}: {v:.2f} m/s, dt_x {dt_x:.3g} s, bound {bound:.3g} s; re-run at "
          f"{rep.dt_x:.3g} s; {len(fired)} firing(s), rollbacks {sim.engine.rollbacks}")
    assert len(fired) == 1 and dt_x == pytest.approx(6.25e-4, rel=1e-12)
    assert bound < dt_x / 3.0 and v > 2.5                              # forced: the box's own approach speed
    assert rep.dt_x <= bound and rep.notes["band"]["reruns"] == 1
    assert sim.engine.rollbacks >= 1 and sim.check()["ok"]
