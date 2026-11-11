# SPDX-License-Identifier: Apache-2.0
"""The inertia floor for light slender free bodies (``alloy.contact.inertia_floor``; coordinator ruling 2026-11-08).

A slender free MuJoCo body below m·r0² gets armature on its 3 rotational dofs up to it (r0 = 9 mm at the 1.25 ms
rigid step), disclosed in one build warning and the route report. Compact and normal-sized bodies keep their
inertia, bitwise. A floored body rolls at the floored rate, (1 + r_g²/R²) / (1 + r0²/R²) of the true one, and the
scene option turns the floor off. The pile it was made for: test_inertia_floor_pile.py."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

import alloy
from alloy.contact import inertia_floor as IF

PEN_R, PEN_L, PEN_M = 0.009, 0.121, 0.0158          # a marker pen's size and mass (YCB 040: 15.8 g)


def _prism(r: float = PEN_R, length: float = PEN_L, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """A 16-sided prism along y, faces wound counter-clockwise from outside."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(a), np.zeros(n), r * np.sin(a)], 1)
    v = np.concatenate([ring - [0.0, length / 2, 0.0], ring + [0.0, length / 2, 0.0]])
    f, c = ConvexHull(v).simplices.copy(), v.mean(0)
    for t in f:
        if np.dot(np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]]), v[t[0]] - c) < 0:
            t[1], t[2] = t[2], t[1]
    return v, f


def _parts_scene(floor: bool = True) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    v, f = _prism()
    s.add_mesh("pen", vertices=v, faces=f, collision="convex", pos=(0.0, 0.0, 0.02), mass=PEN_M)
    s.add_sphere("marble", 0.005, pos=(0.1, 0.0, 0.01), mass=0.004)
    s.add_box("cube", size=(0.015, 0.015, 0.015), pos=(-0.1, 0.0, 0.01), mass=0.01)
    if not floor:
        s.set_inertia_floor(False)
    return s


def test_floor_radius_follows_the_rigid_step():
    assert IF.floor_radius(1.25e-3) == pytest.approx(9e-3)
    assert IF.floor_radius(6.25e-4) == pytest.approx(9e-3)          # flat below 1.25 ms (the sweep)
    assert IF.floor_radius(2.5e-3) == pytest.approx(18e-3)


def test_a_slender_pen_gets_the_floor_and_compact_parts_keep_theirs(caplog):
    """The pen (inertia ratio ~30) gets armature m·r0² − I_min on its 3 angular dofs and nothing on the linear ones;
    a 5 mm marble and a 15 mm cube (below m·r0², but compact) keep theirs. One warning names the pen and the
    opt-out; the route report and sim.routes() carry it; the pile's rk4 choice stands."""
    with caplog.at_level("WARNING", logger="alloy.contact"):
        sim = alloy.build(_parts_scene(), device="cpu")
    mj = sim.engine.parts.solvers["mujoco"]
    m = mj.mj_model
    arm = np.asarray(mj.mjw_model.dof_armature.numpy())[0].reshape(-1, 6)      # the device model (the host copy is the
    # template's on the Warp route; the C route's per-world models carry it too)
    imin, floor = sim.built.report.inertia_floor["pen"]
    assert floor == pytest.approx(PEN_M * IF.floor_radius(1.25e-3) ** 2, rel=1e-6) and imin < floor / 1.5
    assert np.allclose(arm[0, 3:], floor - imin, rtol=1e-5, atol=0.0) and np.all(arm[0, :3] == 0.0)
    assert np.all(arm[1:] == 0.0) and set(sim.built.report.inertia_floor) == {"pen"}
    assert int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4)
    warned = [r.getMessage() for r in caplog.records if "inertia floor" in r.getMessage()]
    assert len(warned) == 1 and "pen x" in warned[0] and "set_inertia_floor(False)" in warned[0]
    assert "inertia floor" in sim.routes()["pen"]["reasons"][-1]


def test_the_scene_option_turns_it_off(caplog):
    with caplog.at_level("WARNING", logger="alloy.contact"):
        sim = alloy.build(_parts_scene(floor=False), device="cpu")
    assert np.all(sim.engine.parts.solvers["mujoco"].mjw_model.dof_armature.numpy() == 0.0)
    assert sim.built.report.inertia_floor == {}
    assert not [r for r in caplog.records if "inertia floor" in r.getMessage()]


def _normal_bodies(floor: bool) -> dict:
    """A 40 mm cylinder rolling down a 10° ramp beside a box dropped tilted onto the ground (it tumbles)."""
    th = math.radians(10.0)
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("ramp", size=(1.0, 0.3, 0.02), pos=(0.0, 0.0, 0.3), rpy=(0.0, th, 0.0), static=True)
    n = np.array([math.sin(th), 0.0, math.cos(th)])
    c = np.array([-0.3 * math.cos(th), 0.0, 0.3 + 0.3 * math.sin(th)]) + n * (0.01 + 0.02)
    s.add_cylinder("roller", 0.02, 0.05, pos=tuple(c), rpy=(math.pi / 2, 0.0, 0.0), mass=0.2)
    s.add_box("brick", size=(0.06, 0.04, 0.02), pos=(0.0, 0.5, 0.1), rpy=(0.4, 0.3, 0.2), mass=0.1)
    if not floor:
        s.set_inertia_floor(False)
    sim = alloy.build(s, device="cpu")
    assert sim.built.report.inertia_floor == {}
    return sim.run(0.5, record=["roller", "brick"])


@pytest.mark.law
def test_bodies_above_the_floor_are_bitwise_unchanged():
    """Neither the 40 mm cylinder rolling down the ramp nor the tumbling box is below the floor or slender, so the
    floor on (the default) and off give the same trajectories, bit for bit."""
    on, off = _normal_bodies(True), _normal_bodies(False)
    keys = [k for k in on if k != "t"]
    assert keys and all(np.array_equal(on[k], off[k]) for k in keys)
    assert float(np.ptp(on["roller.pos"][:, 0, 0])) > 0.05          # it did roll


def _roll(floor: bool) -> float:
    """A pen-sized cylinder (r 9 mm, 121 mm, 15.8 g) rolling down a 10° ramp from rest: its acceleration along the
    slope (a least-squares fit of s = a t²/2 over 0.1-0.4 s)."""
    th = math.radians(10.0)
    s = alloy.Scene(dt=0.01)
    s.add_box("ramp", size=(1.2, 0.3, 0.02), pos=(0.0, 0.0, 0.3), rpy=(0.0, th, 0.0), static=True)
    n, d = np.array([math.sin(th), 0.0, math.cos(th)]), np.array([math.cos(th), 0.0, -math.sin(th)])
    c = np.array([-0.4 * math.cos(th), 0.0, 0.3 + 0.4 * math.sin(th)]) + n * (0.01 + PEN_R)
    s.add_cylinder("pen", PEN_R, PEN_L, pos=tuple(c), rpy=(math.pi / 2, 0.0, 0.0), mass=PEN_M, friction=1.0)
    if not floor:
        s.set_inertia_floor(False)
    sim = alloy.build(s, device="cpu")
    assert ("pen" in sim.built.report.inertia_floor) == floor
    traj = sim.run(0.4, record=["pen"]).world(0)
    t, x = traj["t"], (traj["pen.pos"] - traj["pen.pos"][0]) @ d
    sel = t >= 0.1
    return float(np.sum(x[sel] * t[sel] ** 2) / np.sum(0.5 * t[sel] ** 4))


@pytest.mark.law
def test_a_floored_pen_rolls_at_the_floors_rate():
    """Rolling acceleration g sinθ / (1 + I/(m R²)): off, the solid cylinder's I/(m R²) = 0.5 (1.136 m/s² at 10°);
    with the floor, r0²/R² = 1 (0.852 m/s²), i.e. (1 + 0.5) / (1 + 1) = 0.75 of it: the stated bound."""
    g_s = 9.81 * math.sin(math.radians(10.0))
    off, on = _roll(False), _roll(True)
    r0 = IF.floor_radius(1.25e-3)
    assert off == pytest.approx(g_s / 1.5, rel=0.03)
    assert on == pytest.approx(g_s / (1.0 + (r0 / PEN_R) ** 2), rel=0.03)
    assert on / off == pytest.approx(1.5 / (1.0 + (r0 / PEN_R) ** 2), rel=0.04)


def test_floor_candidates_write_nothing_and_name_the_floored(monkeypatch):
    """F6 R1 reads ``floor_candidates`` before the integrator choice: it leaves body_inertia and joint_armature bit
    for bit as they were, and its names are the build's ``report.inertia_floor`` (floor on). WP3's review."""
    seen, inner = [], IF.floor_candidates

    def spy(b, objs, dt):
        before = (np.array(b.body_inertia, dtype=float), np.array(b.joint_armature, dtype=float))
        out = inner(b, objs, dt)
        after = (np.array(b.body_inertia, dtype=float), np.array(b.joint_armature, dtype=float))
        seen.append((before, after, sorted(f.name for _, _, f in out)))
        return out

    monkeypatch.setattr(IF, "floor_candidates", spy)
    sim = alloy.build(_parts_scene(), device="cpu")
    assert seen
    for before, after, _ in seen:
        for x, y in zip(before, after, strict=True):
            np.testing.assert_array_equal(x, y)
    assert seen[-1][2] == sorted(sim.built.report.inertia_floor) == ["pen"]
