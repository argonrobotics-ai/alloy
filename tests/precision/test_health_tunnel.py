# SPDX-License-Identifier: Apache-2.0
"""TUNNEL / FAST_CONTACT (GAP-E2, audit ERR-2): a rigid body passing through a static shape is flagged.

Laws (``health_tunnel.py`` states the rules):
L1  the audit scene (2 cm steel ball dropped 4..12 m onto a 1 cm static plate, 5 mm contact gap, 1.25 ms rigid step),
    on both CPU routes: no world may end with the ball on the far side of the plate without TUNNEL (zero misses); a
    world that tunnelled is INVALID, named in ``sim.check()``, frozen above the plate; the 4 m ball lands and carries
    no TUNNEL (FAST_CONTACT: test_health_tunnel_fast.py).
L2  profile="precision" raises for the 5 m drop.
L3  no false alarms: a gentle drop, a resting box, a ball rolling at 6 m/s, a hollow ring over a static bar, a body
    starting with its centre inside a static box.
L4  a ground plane alone, a ball at 19 m/s: no FAST_CONTACT (a half-space cannot be passed), TUNNEL only if it ends
    wholly under the plane.
L8  FAST_CONTACT: see test_health_tunnel_fast.py (it clears, so those laws read health at every period).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import alloy

ROUTES = ["mjc", "warp"]
HEIGHTS = [4.0, 5.0, 6.0, 8.0, 12.0]
R, PLATE_TOP, PLATE_BELOW = 0.01, 0.305, 0.29
TUNNELLED = [False, True, True, True, True]    # on the base tree these balls end below the plate (audit repro)


def _audit(heights, route, monkeypatch, plate=0.01, **kw):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(worlds=len(heights), dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.5, 0.5, plate), pos=(0, 0, 0.3), static=True)         # 1 cm: top at z = 0.305
    scene.add_sphere("ball", R, pos=np.array([[0, 0, 0.3 + plate / 2 + R + h] for h in heights]), density=7800.0)
    return alloy.build(scene, device="cpu", **kw)


def _unwatched_z(build, monkeypatch, seconds: float, record: str) -> np.ndarray:
    """The same scene run without the tunnel watch (the oracle for where the body really goes): z, (frames, worlds)."""
    from alloy.precision import health_tunnel

    with monkeypatch.context() as m:
        m.setattr(health_tunnel, "watch", lambda solver: None)
        return build().run(seconds, record=[record])[f"{record}.pos"][:, :, 2]


def _flags(sim) -> list[list[str]]:
    return [w["flags"] for w in sim.health()["worlds"]]


def _launch(sim, velocity) -> None:
    """Give world 0's free body ``velocity`` now and take the energy reference again (no ENERGY_JUMP for it)."""
    st = sim.engine.state
    jqd, bqd = st.joint_qd.numpy().copy(), st.body_qd.numpy().copy()
    jqd[:3], bqd[0, :3] = velocity, velocity
    st.joint_qd.assign(jqd)
    st.body_qd.assign(bqd)
    sim.engine.health.reset_reference(st)


@pytest.mark.parametrize("route", ROUTES)
def test_l1_every_ball_that_passes_through_the_plate_is_tunnel_and_frozen_above_it(route, monkeypatch) -> None:
    sim = _audit(HEIGHTS, route, monkeypatch)
    z = sim.run(2.2, record=["ball"])["ball.pos"][:, :, 2]
    flags, health = _flags(sim), sim.health()
    below = z[-1] < PLATE_BELOW
    assert not below.any(), f"a ball ended below the plate: z {z[-1]}"
    for w, tunnelled in enumerate(TUNNELLED):
        assert ("TUNNEL" in flags[w]) == tunnelled, (w, flags[w])
        assert (health["worlds"][w]["severity"] == "INVALID") == tunnelled
    assert z.min(axis=0)[TUNNELLED].min() > PLATE_TOP                               # frozen above the plate
    problems = sim.check()
    assert not problems["ok"]
    for w in np.flatnonzero(TUNNELLED):
        text = [p["what"] for p in problems["problems"] if p["world"] == w and "passed through" in p["what"]]
        assert text and '"ball" passed through "plate"' in text[0]
    # FAST_CONTACT is not sticky (it clears 64 rigid steps after the approach): its laws are in test_health_tunnel_fast.py


def test_l2_the_precision_profile_raises_for_the_5_m_drop(monkeypatch) -> None:
    sim = _audit([5.0], "warp", monkeypatch, profile="precision")
    with pytest.raises(alloy.SimulationError, match="TUNNEL"):      # in the period the ball passes (GAP-E3)
        sim.run(2.2)
    assert sim.t < 1.1                                              # the pass is at 1.01 s: not at the run's end
    assert "TUNNEL" in _flags(sim)[0]


def _scene_with(*adds, route="warp", monkeypatch=None, worlds=1):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    for add in adds:
        add(scene)
    return scene


@pytest.mark.parametrize("route", ROUTES)
def test_l3_a_gentle_drop_and_a_resting_box_carry_no_flag(route, monkeypatch) -> None:
    def parts(scene):
        scene.add_box("plate", size=(0.5, 0.5, 0.01), pos=(0, 0, 0.3), static=True)
        scene.add_sphere("ball", R, pos=(0.0, 0, 0.305 + R + 0.3), density=7800.0)         # v = 2.4 m/s: 3 mm / step
        scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(1.5, 0, 0.2), static=True)
        scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(1.5, 0, 0.426), mass=0.2)

    sim = alloy.build(_scene_with(parts, route=route, monkeypatch=monkeypatch), device="cpu")
    sim.run(1.0)
    assert _flags(sim) == [[]] and sim.check()["ok"]


@pytest.mark.parametrize("route", ROUTES)
def test_l3_a_ball_rolling_at_6_m_per_s_carries_no_flag(route, monkeypatch) -> None:
    sim = alloy.build(_scene_with(lambda s: s.add_sphere("ball", 0.02, pos=(-1.0, 0, 0.0201), density=1000.0),
                                  route=route, monkeypatch=monkeypatch), device="cpu")
    sim.run(0.05)
    _launch(sim, (6.0, 0.0, 0.0))
    tr = sim.run(0.3, record=["ball"])
    assert tr["ball.vel"][-1, 0, 0] > 3.0 and _flags(sim) == [[]]                      # really rolling at speed


RING = """<mujoco><worldbody><body name="ring" pos="0 0 0.8"><freejoint/>
  <geom type="box" pos="0.07 0 0" size="0.02 0.09 0.03" mass="0.1"/><geom type="box" pos="-0.07 0 0" size="0.02 0.09 0.03" mass="0.1"/>
  <geom type="box" pos="0 0.07 0" size="0.09 0.02 0.03" mass="0.1"/><geom type="box" pos="0 -0.07 0" size="0.09 0.02 0.03" mass="0.1"/>
  </body></worldbody></mujoco>"""


@pytest.mark.parametrize("route", ROUTES)
def test_l3_a_hollow_ring_dropped_over_a_static_bar_is_not_judged(route, monkeypatch) -> None:
    def parts(scene):
        scene.add_box("bar", size=(0.04, 0.04, 0.6), pos=(0, 0, 0.3), static=True)
        scene.add_robot("ring", mjcf=RING, pos=(0, 0, 0.8))

    sim = alloy.build(_scene_with(parts, route=route, monkeypatch=monkeypatch), device="cpu")
    tr = sim.run(1.0, record=["ring"])
    assert tr["ring.pos"][-1, 0, 2] < 0.2                                               # it slid down over the bar
    assert "TUNNEL" not in _flags(sim)[0] and "FAST_CONTACT" not in _flags(sim)[0]


@pytest.mark.parametrize("route", ROUTES)
def test_l3_a_body_that_starts_with_its_centre_inside_a_static_box_is_never_tunnel(route, monkeypatch) -> None:
    def parts(scene):
        scene.add_box("block", size=(0.3, 0.3, 0.3), pos=(0, 0, 0.15), static=True)
        scene.add_sphere("ball", 0.02, pos=(0, 0, 0.15), density=1000.0)

    sim = alloy.build(_scene_with(parts, route=route, monkeypatch=monkeypatch), device="cpu")
    sim.run(0.3)
    assert "TUNNEL" not in _flags(sim)[0]


def _ground_ball(route, monkeypatch):
    return alloy.build(_scene_with(lambda s: s.add_sphere("ball", R, pos=(0, 0, 0.5), density=7800.0), route=route,
                                   monkeypatch=monkeypatch), device="cpu")


def _fall_at_19(sim) -> np.ndarray:
    sim.run(0.02)
    _launch(sim, (0.0, 0.0, -19.0))
    return sim.run(0.1, record=["ball"])["ball.pos"][:, 0, 2]


@pytest.mark.parametrize("route", ROUTES)
def test_l4_a_ball_at_19_m_per_s_on_the_ground_alone_is_not_fast_contact(route, monkeypatch) -> None:
    """TUNNEL exactly when the unwatched ball ends wholly under the plane (its centre more than R below it)."""
    sim = _ground_ball(route, monkeypatch)
    _fall_at_19(sim)
    flags = _flags(sim)[0]
    from alloy.precision import health_tunnel

    with monkeypatch.context() as m:
        m.setattr(health_tunnel, "watch", lambda solver: None)
        z = _fall_at_19(_ground_ball(route, monkeypatch))
    assert "FAST_CONTACT" not in flags                       # was expected before the narrowing: a plane cannot be passed
    assert ("TUNNEL" in flags) == bool(z.min() < -R), (flags, float(z.min()))


FAST_HEIGHTS = [4.0, 6.0, 8.0, 12.0, 20.0]


def test_l6_the_flags_are_sticky_under_the_fast_profile(monkeypatch) -> None:
    """Under profile="fast" (a check every 10th exchange; a 3 mm plate, since fast contacts hold a 1 cm one) TUNNEL is
    exactly the worlds whose unwatched ball ends below the plate, and some do (the law is not vacuous). Where the
    frozen world rests is not asserted: fast freezes at its last checked state, which can be past the plate."""
    def build():
        return _audit(FAST_HEIGHTS, "warp", monkeypatch, plate=0.003, profile="fast")
    sim = build()
    sim.run(3.0)
    flags = _flags(sim)
    passed = (_unwatched_z(build, monkeypatch, 3.0, "ball")[-1] < PLATE_BELOW).tolist()
    assert any(passed) and not all(passed)
    for w, tunnelled in enumerate(passed):
        assert ("TUNNEL" in flags[w]) == tunnelled, (w, flags[w])


def _stack_scene(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    scene = alloy.Scene(worlds=64, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(2.0, 2.0, 0.4), pos=(0, 0, 0.2), static=True)
    for i in range(20):
        scene.add_box(f"b{i}", size=(0.08, 0.08, 0.08), pos=(-0.5 + 0.1 * (i % 10), -0.1 + 0.2 * (i // 10), 0.45), mass=0.2)
    return alloy.build(scene, device="cpu")


def _world_steps_per_s(sim, periods: int = 20) -> float:
    sim.step(2)
    s0, t0 = sim.physics_steps, time.perf_counter()
    sim.step(periods)
    sim.state(["b0"])
    return sim.worlds * (sim.physics_steps - s0) / (time.perf_counter() - t0)


def test_the_watch_costs_little(monkeypatch) -> None:
    """L7: 64 worlds, 20 free boxes on a table, CPU warp route, with and without the watch: at most 1.5x."""
    from alloy.precision import health_tunnel

    with_watch = _world_steps_per_s(_stack_scene(monkeypatch))
    monkeypatch.setattr(health_tunnel, "watch", lambda solver: None)
    without = _world_steps_per_s(_stack_scene(monkeypatch))
    print(f"L7 world-steps/s with {with_watch:.0f} without {without:.0f} ratio {without / with_watch:.3f}")
    assert without / with_watch < 1.5

