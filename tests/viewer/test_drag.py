# SPDX-License-Identifier: Apache-2.0
"""The mouse-drag tool on the CPU (null backend, a 0.2 kg box on a table in 2 worlds): the spring law (the box hangs
g / w^2 under its target, in the grabbed world only), an off-centre grab turns the box the right way, release removes
the force with one zero wrench, a viewer with nothing grabbed is bit for bit a plain run, static bodies are refused,
the world offset of the viewer is removed from a mouse target, detach clears the force, and the force cap."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

import alloy
import alloy.viewer
from alloy.viewer import _drag_force as dforce
from alloy.viewer._drag import MouseDrag

pytestmark = pytest.mark.compile_heavy      # the first step of a cold cache compiles MuJoCo Warp (over 60 s)

DT = 0.01
MASS = 0.2
REST_Z = 0.425                      # box centre resting on the table (top at 0.4, box 5 cm)
SAG = dforce.G / dforce.OMEGA**2    # a still target holds the body this far above where the spring balances weight


def _scene() -> alloy.Scene:
    scene = alloy.Scene(worlds=2, dt=DT)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=np.array([[0, 0, REST_Z], [0, 0, REST_Z]]), mass=MASS)
    return scene


@pytest.fixture(scope="module")
def sim() -> alloy.Sim:
    return alloy.build(_scene(), device="cpu", seed=0)


@pytest.fixture
def view(sim):
    sim.reset()
    with alloy.viewer.launch(sim, backend="null", drag=True, frames=1_000_000) as v:
        yield v


def _z(sim, world: int = 0) -> float:
    return float(sim.state()["block.pos"][world, 2])


def _row(sim, world: int = 0) -> int:
    return int(sim.index.per_world(sim.index.body("block"), "body")[world])


def _held_force(sim, world: int = 0) -> np.ndarray:
    return np.array(sim.engine.state.body_f.numpy()[_row(sim, world)][:3])


@pytest.fixture
def spy(sim, monkeypatch):
    """The ``apply_wrench`` calls the tool makes: (name, force array)."""
    calls: list[tuple[str, np.ndarray]] = []
    real = sim.apply_wrench

    def record(name, force=(0, 0, 0), torque=(0, 0, 0), **kw):
        calls.append((name, np.array(force, dtype=float)))
        return real(name, force=force, torque=torque, **kw)

    monkeypatch.setattr(sim, "apply_wrench", record)
    return calls


# --- the pure spring -----------------------------------------------------------------------------------------------


def test_spring_maths():
    pose = dforce.BodyPose(np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3), np.zeros(3))
    at_target = dforce.spring_wrench(pose, np.zeros(3), np.zeros(3), 1.0, 1e9)
    assert not at_target.force.any() and not at_target.torque.any()
    pull = dforce.spring_wrench(pose, np.array([0.1, 0, 0]), np.array([0.1, 0, 1.0]), 2.0, 1e9)
    assert pull.force[2] == pytest.approx(2.0 * dforce.OMEGA**2)           # k = m w^2, along the offset
    assert pull.torque[1] == pytest.approx(-0.1 * pull.force[2])             # r x F: the +x side goes up
    fast = dataclasses.replace(pose, lin=np.array([0, 0, 1.0]))
    assert dforce.spring_wrench(fast, np.zeros(3), np.zeros(3), 1.0, 1e9).force[2] == pytest.approx(-2 * dforce.OMEGA)
    assert dforce.force_cap(0.2, False) == pytest.approx(50 * 0.2 * 9.81)
    assert dforce.force_cap(10.0, True) == 200.0 and dforce.force_cap(1.0, True) == 200.0



# --- the law ---------------------------------------------------------------------------------------------------------


def test_grabbed_box_hangs_below_its_target_in_its_world_only(sim, view):
    """Target 0.3 m above the table (z = 0.7): after 2 s the box sits at 0.7 - g / w^2 (6.2 cm) within 2 cm; the other
    world's box stays on the table."""
    drag = MouseDrag.of(view)
    assert drag.grab(view, "block", 0, (0.0, 0.0, 0.7))
    view.run(seconds=2.0, realtime=False)
    hang = _z(sim, 0)
    print(f"hang z={hang:.4f} expected {0.7 - SAG:.4f} (sag {SAG:.4f})")
    assert abs(hang - (0.7 - SAG)) < 0.02
    assert abs(_z(sim, 1) - REST_Z) < 2e-3
    assert sim.check()["ok"]


def test_ungrabbed_world_is_bitwise_a_plain_run(sim, view):
    plain_sim = sim
    plain_sim.reset()
    plain_sim.step(100)
    plain = np.array(plain_sim.state()["block.pos"][1], copy=True)
    view.sim.reset()
    MouseDrag.of(view).grab(view, "block", 0, (0.0, 0.0, 0.7))
    view.run(seconds=1.0, realtime=False)
    assert np.array_equal(sim.state()["block.pos"][1], plain)


def test_off_centre_grab_turns_the_box_corner_up(sim, view):
    """Grab the +x top edge and pull straight up: the box pitches so that edge rises first (rotation about -y)."""
    drag = MouseDrag.of(view)
    assert drag.grab(view, "block", 0, (0.025, 0.0, 0.9), attach_xyz=(0.025, 0.0, REST_Z + 0.025))
    view.run(seconds=0.1, realtime=False)
    quat = sim.state()["block.quat_xyzw"][0]
    assert quat[1] < -0.01, quat                       # q_y = -sin(theta / 2): turned about -y, so +x went up


def test_release_removes_the_force_with_one_zero_wrench(sim, view, spy):
    drag = MouseDrag.of(view)
    drag.grab(view, "block", 0, (0.0, 0.0, 0.7))
    view.run(seconds=1.0, realtime=False)
    assert _z(sim) > 0.55
    spy.clear()
    drag.release()
    view.run(seconds=2.0, realtime=False)
    zeros = [c for c in spy if not c[1].any()]
    assert len(spy) == 1 and len(zeros) == 1, spy
    assert abs(_z(sim) - REST_Z) < 3e-3                 # fell back and rests on the table


# --- nothing grabbed, nothing called -----------------------------------------------------------------------------


def test_viewer_with_nothing_grabbed_is_bitwise_a_plain_run(sim, spy):
    sim.reset()
    sim.run(1.0)
    plain = {k: np.array(v, copy=True) for k, v in sim.state().items()}
    sim.reset()
    assert sim.state()["block.pos"].shape == plain["block.pos"].shape
    with alloy.viewer.launch(sim, backend="null", drag=True, frames=1_000_000) as view:
        view.run(seconds=1.0, realtime=False)
    assert spy == []
    for key, value in sim.state().items():
        if key == "t":
            continue                                     # a reset does not rewind the clock
        assert np.array_equal(np.asarray(value), plain[key]), key


# --- refusals, frames, detach, cap ---------------------------------------------------------------------------------


def test_static_body_grab_is_refused(sim, view, caplog):
    with caplog.at_level(logging.INFO, logger="alloy.viewer"):
        ok = MouseDrag.of(view).grab(view, "table", 0, (0, 0, 1.0))
        missing = MouseDrag.of(view).grab(view, "no_such_thing", 0, (0, 0, 1.0))
    assert ok is False and missing is False
    assert "cannot be grabbed" in caplog.text
    view.step(2)
    assert not _held_force(sim).any()


def test_mouse_target_in_viewer_space_pulls_to_the_right_sim_position(sim, view):
    """World 1 is drawn shifted on the viewer's grid. A ray through its drawn box and a drag to 0.25 m higher, both in
    viewer space, pull world 1's box straight up in sim space (x stays 0), and leave world 0 alone."""
    import warp as wp
    from alloy._vendor.newton.viewer import Picking

    backend = view.backend
    backend.set_world_offsets((1.5, 0.0, 0.0))
    offset = backend.world_offsets.numpy()[1]
    assert abs(offset[0]) + abs(offset[1]) > 0.5
    backend.picking = Picking(sim.engine.model, world_offsets=backend.world_offsets)
    eye = np.array([offset[0], offset[1] - 1.5, 1.0])
    top = np.array([offset[0], offset[1], 0.45])                       # the top face of world 1's box, viewer space
    backend.picking.pick(sim.engine.state, wp.vec3(*eye), wp.vec3(*((top - eye) / np.linalg.norm(top - eye))))
    assert backend.picking.is_picking() and backend.picking.pick_body == _row(sim, 1)
    goal = top + np.array([0.0, 0.0, 0.3])                             # drag 0.3 m up (on the ray: same depth)
    ray = (goal - eye) / np.linalg.norm(goal - eye)
    backend.picking.update(wp.vec3(*eye), wp.vec3(*ray))
    target = eye + ray * np.linalg.norm(top - eye) - offset            # where Newton puts it, in sim space
    view.run(seconds=2.0, realtime=False)
    pos = sim.state()["block.pos"]
    want = target - np.array([0.0, 0.0, SAG + 0.025])                  # hangs below the grabbed top face centre
    print(f"world 1 box at {pos[1]} (expected {want}), world 0 at {pos[0]}")
    assert np.abs(pos[1] - want).max() < 0.03                          # an offset not removed would pull it 0.75 m in x
    assert abs(pos[0, 2] - REST_Z) < 2e-3


def test_detach_clears_a_held_force(sim, spy):
    sim.reset()
    view = alloy.viewer.launch(sim, backend="null", drag=True, frames=1_000_000)
    MouseDrag.of(view).grab(view, "block", 0, (0.0, 0.0, 0.7))
    view.step(3)
    assert _held_force(sim)[2] > 1.0
    view.close()
    assert not _held_force(sim).any() and not spy[-1][1].any()


def test_force_is_capped(sim, view):
    """A target 100 m away: the pull is the cap (50 weights), not 0.2 * w^2 * 100 = 3,000 N."""
    MouseDrag.of(view).grab(view, "block", 0, (0.0, 0.0, 100.0))
    view.step(1)
    size = float(np.linalg.norm(_held_force(sim)))
    assert size == pytest.approx(dforce.force_cap(MASS, False), rel=1e-4)


def test_viewer_without_drag_has_no_tool(sim):
    with alloy.viewer.launch(sim, backend="null", frames=10) as view, pytest.raises(alloy.SceneError):
        MouseDrag.of(view)
