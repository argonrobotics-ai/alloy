# SPDX-License-Identifier: Apache-2.0
"""TUNNEL / FAST_CONTACT tracker and rollback (GAP-E2, WP5 review condition 1): the tracker's arrays are solver state, so
every restore puts them back with the bodies, and a pass that is discarded and re-stepped leaves no flag behind.

R1  snapshot, step through the plate (TUNNEL), restore, run a trajectory that stops short of the plate: no TUNNEL and no
    FAST_CONTACT from the discarded steps.
R2  the converse: a snapshot taken while the centre is inside the plate keeps the entry, so the continued run still
    raises TUNNEL when the ball leaves through the far face.
R3  a coupled scene (cloth under the ball) in which a period is really rolled back and re-run: no flag from the discarded
    pass; then a kept landing flags only its own world.
R4  ``sim.reset()`` clears the flags and the tracking.
R5  the names in a message: past 65,535 judged bodies or static shapes the text says "a body" / "a shape".
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.precision import health_tunnel as HT

ROUTES = ["mjc", "warp"]
R, PLATE_Z = 0.01, 0.3
BITS = {"TUNNEL", "FAST_CONTACT"}


def _plate_scene(heights, route, monkeypatch, plate=0.01):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(worlds=len(heights), dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.5, 0.5, plate), pos=(0, 0, PLATE_Z), static=True)
    scene.add_sphere("ball", R, pos=np.array([[0, 0, PLATE_Z + plate / 2 + R + h] for h in heights]), density=7800.0)
    return alloy.build(scene, device="cpu")


def _tunnel_flags(sim) -> list[set[str]]:
    return [BITS & set(w["flags"]) for w in sim.health()["worlds"]]


def _snapshot(sim):
    """The rollback's own end-of-period snapshot: State, the stepper's arrays (solver adapters) and health's arrays."""
    snap = sim.engine._end_snapshot()
    snap.save(sim.engine.state)
    return snap


@pytest.mark.parametrize("route", ROUTES)
def test_r1_a_discarded_pass_through_the_plate_leaves_no_flag(route, monkeypatch) -> None:
    sim = _plate_scene([5.0], route, monkeypatch)
    snap = _snapshot(sim)
    sim.run(1.4)                                                   # 9.9 m/s at the plate: 12 mm per rigid step
    assert _tunnel_flags(sim) == [{"TUNNEL", "FAST_CONTACT"}], "the discarded pass must really cross"
    snap.restore(sim.engine.state)
    assert _tunnel_flags(sim) == [set()]
    sim.run(0.5)                                                   # the ball falls again from 5 m, 1 s short of the plate
    assert _tunnel_flags(sim) == [set()], "a flag of the discarded pass came back"
    assert sim.state()["ball.pos"][0, 2] > PLATE_Z + 3.0


@pytest.mark.parametrize("route", ROUTES)
def test_r1_the_names_of_a_latched_world_survive_the_latch_restore(route, monkeypatch) -> None:
    """The latch puts a frozen world's tracker back every step; the text of why it froze must not go with it."""
    sim = _plate_scene([5.0], route, monkeypatch)
    sim.run(2.0)
    texts = [p["what"] for p in sim.check()["problems"] if "passed through" in p["what"]]
    assert texts and '"ball" passed through "plate"' in texts[0], sim.check()


def _inside_plate_tracker(sim) -> np.ndarray:
    return HT.watch_of(sim.engine).tr.in_shape.numpy().copy()


@pytest.mark.parametrize("route", ROUTES)
def test_r2_a_snapshot_taken_inside_the_plate_keeps_the_entry(route, monkeypatch) -> None:
    """Drop worlds from a range of heights and stop after each period: the first world whose centre is inside the plate
    (entry tracked) is snapshotted, the run continued, then flags and entry wiped and the snapshot restored: the run again must give the same flags."""
    heights = list(np.linspace(4.5, 6.5, 24))
    sim = _plate_scene(heights, route, monkeypatch)
    watch, found = HT.watch_of(sim.engine), None
    for _ in range(140):
        sim.step(1)
        inside = np.flatnonzero(watch.tr.in_shape.numpy() >= 0)
        flagged = {w for w in range(len(heights)) if "TUNNEL" in _tunnel_flags(sim)[w]}
        inside = [w for w in inside if w not in flagged]
        if inside:
            found = int(inside[0])
            break
    assert found is not None, "no world ended a period with its centre inside the plate (the law would be vacuous)"
    snap = _snapshot(sim)
    sim.run(0.3)
    kept = _tunnel_flags(sim)[found]                               # what the continued run raises with the entry known
    watch.tr.in_shape.fill_(-1)                                    # lose the entry, as a restore that skipped it would
    watch.tr.tunnel.zero_()
    watch.tr.fast.zero_()
    snap.restore(sim.engine.state)
    assert watch.tr.in_shape.numpy()[found] >= 0, "the restore did not bring the entry back"
    sim.run(0.3)
    assert _tunnel_flags(sim)[found] == kept, "the run after the restore differs from the first run"


@pytest.mark.parametrize("route", ROUTES)
def test_r4_reset_clears_the_flags_and_the_tracking(route, monkeypatch) -> None:
    sim = _plate_scene([5.0], route, monkeypatch)
    sim.run(2.0)
    assert "TUNNEL" in _tunnel_flags(sim)[0]
    sim.reset()
    watch = HT.watch_of(sim.engine)
    assert not watch.tr.tunnel.numpy().any() and not watch.tr.fast.numpy().any()
    assert (watch.tr.in_shape.numpy() == -1).all()
    sim.run(0.2)
    assert _tunnel_flags(sim) == [set()]


class _Teleport:
    """Step observer: at step ``at`` put the worlds' ball 2 cm above the plate, falling at 6.3 m/s (7.8 mm per rigid
    step: FAST_CONTACT when it lands on the 1 cm plate, DEGRADED, no TUNNEL; post-step state)."""

    def __init__(self, engine, worlds, at: int) -> None:
        self.engine, self.worlds, self.at, self.fired = engine, worlds, at, False

    def __call__(self, s0, s1, control, contacts, dt) -> None:
        if self.fired or self.engine.steps != self.at:
            return
        m = self.engine.model
        q, qd, bq, bqd = (a.numpy() for a in (s1.joint_q, s1.joint_qd, s1.body_q, s1.body_qd))
        qs, ds, bs = m.joint_q_start.numpy(), m.joint_qd_start.numpy(), m.body_world_start.numpy()
        for w in self.worlds:
            q[qs[w]:qs[w] + 3], qd[ds[w]:ds[w] + 3] = (0.5, 0, 0.335), (0, 0, -6.3)
            bq[bs[w], :3], bqd[bs[w], :3] = (0.5, 0, 0.335), (0, 0, -6.3)
        for a, v in ((s1.joint_q, q), (s1.joint_qd, qd), (s1.body_q, bq), (s1.body_qd, bqd)):
            a.assign(v)
        self.engine.health.reset_reference(s1)                     # a kick is not an ENERGY_JUMP
        self.fired = True


def _coupled_scene(route, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.5, 0.5, 0.01), pos=(0.5, 0, PLATE_Z), static=True)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.002), resolution=(6, 6), mass=0.03, thickness=0.004,
                    self_contact=False)
    scene.add_sphere("ball", R, pos=np.array([[0, 0, 0.0165]] * 3), density=7800.0)       # resting on the cloth
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.timeout(2800)
def test_r3_a_rolled_back_period_in_a_coupled_scene_keeps_only_the_kept_landing(route, monkeypatch) -> None:
    """Worlds 1 and 2 are thrown at the plate mid-period (once; they land on it with a FAST_CONTACT step), and world 1 is
    escalated: the period is rolled back and re-run (this ADMM scene rolls every world back: its multiplier arrays have
    no world axis), without the throw. No kept trajectory came near the plate, so no world may carry a flag. A second
    period throws world 2 alone and keeps it: only world 2 carries FAST_CONTACT."""
    from alloy.testing import integration as I

    sim = _coupled_scene(route, monkeypatch)
    eng = sim.engine
    n = eng.clock.steps_per_period
    eng.add_step_observer(_Teleport(eng, [1, 2], eng.steps + n // 4))
    eng.add_step_observer(I.EscalateOnce(eng, 1, eng.steps + n // 2))
    eng.step_period()
    assert eng.rollbacks >= 1, "the period was not rolled back: the law would be vacuous"
    assert np.all(sim.state()["ball.pos"][:, 2] < 0.1), "a thrown ball survived the rollback"
    assert _tunnel_flags(sim) == [set()] * 3, "a flag of the discarded pass survived"
    eng.add_step_observer(_Teleport(eng, [2], eng.steps + n // 4))
    eng.step_period()
    assert _tunnel_flags(sim) == [set(), set(), {"FAST_CONTACT"}]       # kept: world 2 really landed fast
    assert sim.state()["ball.pos"][2, 2] > 0.3                          # it rests on the plate, not through it
    assert [w["severity"] for w in sim.health()["worlds"]] == ["OK", "OK", "DEGRADED"]


def _fake_watch(count: int) -> HT.TunnelWatch:
    """A watch over ``count`` judged bodies and static shapes of one world, with a key set by the test (no model)."""
    tw = object.__new__(HT.TunnelWatch)
    tw.bodies = [SimpleNamespace(world=0, body=0) for _ in range(count)]
    tw.statics, tw._by_world = np.zeros(count, np.int64), {0: list(range(count))}
    tw.world_count = 1
    tw.model = SimpleNamespace(body_label=["ball"], shape_label=["plate"])
    tw.tr = SimpleNamespace(seen_fast=SimpleNamespace(numpy=lambda: np.zeros(1, np.uint64)),
                            seen_tunnel=SimpleNamespace(numpy=lambda: np.array([(0xFFFF << 16) | 0xFFFF], np.uint64)))
    return tw


@pytest.mark.parametrize("count, named", [(3, False), (65_535, False), (65_536, False), (70_000, False)])
def test_r5_the_clamp_sentinel_names_neither_a_body_nor_a_shape(count, named) -> None:
    """The kernel clamps local indices to 65535: with more of them in a world the code names nothing."""
    assert HT.local_index(0xFFFF, count) is None
    row = _fake_watch(count).notes(0.01)[0]
    assert row["tunnel"] == ("a body", "a shape")


def test_r5_small_indices_still_name_their_body_and_shape() -> None:
    assert HT.local_index(0, 3) == 0 and HT.local_index(2, 3) == 2 and HT.local_index(3, 3) is None
    assert HT.local_index(65_534, 70_000) == 65_534
