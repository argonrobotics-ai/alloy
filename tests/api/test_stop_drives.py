# SPDX-License-Identifier: Apache-2.0
"""``sim.stop_drives()`` brings a driven kinematic body to rest where it is (WP7's report, 2026-11-10).

No solver integrates a kinematic body, so it kept the last velocity its drive wrote (``body_qd``, ``joint_qd``), and
MPM colliders, contacts and liquid walls read that as a moving wall: a stopped tool kept pushing sand at 0.8-0.9x its
old speed. The engine swaps two states every exchange and in an MPM-only scene no solver writes a kinematic body, so
the fix (``_drive.hold``) writes both; ``set_pose`` wrote one and had the same fault. Laws, on the rigid, liquid and MPM
routes (CPU): after ``stop_drives`` or ``set_pose`` the body is at rest at its pose in both states and stays so
(``.vel`` 0, ``.pos`` unchanged), and sand under a tool stopped mid-press sinks no more than when the drive ends by its
duration.

Law 1 (physics' ruling, WP7's press-and-hold, ``probe_stop/stop.py``): the hold after ``stop_drives`` equals, bit for
bit, the hold after a drive that ends by its duration, steady, the material at rest. Until API-17 (0.1.1,
``test_drive_timing.py``) the duration case kept its velocity one substep past the end and read 3.2 % higher (2.867 N
against 2.776 N), so the law compared with that case's velocity zeroed ("duration0").
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api import _drive


def _rigid(scene: alloy.Scene) -> str:
    scene.add_box("tool", size=(0.2, 0.2, 0.02), pos=(0, 0, 0.1), kinematic=True)
    scene.add_box("cube", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.131), mass=0.1)
    return "tool"


def _liquid(scene: alloy.Scene) -> str:
    scene.add_open_box("tool", inner=(0.06, 0.06, 0.05), wall=0.004, pos=(0, 0, 0.0), kinematic=True)
    scene.add_liquid("water", size=(0.03, 0.03, 0.02), pos=(0, 0, 0.015))
    return "tool"


def _mpm(scene: alloy.Scene) -> str:
    scene.add_box("dough", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.01), material="dough_wheat")
    scene.add_sphere("tool", 0.008, pos=(0, 0, 0.03), kinematic=True)
    return "tool"


SCENES = {"rigid": (_rigid, (0.0, 0.0, 0.05), (0.0, 0.0, 0.5)), "liquid": (_liquid, (0.0, 0.0, 0.02), (0.0, 0.0, 0.0)),
          "mpm": (_mpm, (0.0, 0.0, -0.02), (0.0, 0.0, 0.0))}       # route: (scene, velocity, angular_velocity)


def _rows(sim: alloy.Sim, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``name``'s body rows (one per world) and its free joint's first ``joint_qd`` rows."""
    idx, m = sim.index, sim.engine.model
    joints = idx.per_world(_drive.free_joint(sim, name), "joint")
    return idx.per_world(idx.body(name), "body"), m.joint_qd_start.numpy()[joints]


def _assert_held(sim: alloy.Sim, name: str, pose: np.ndarray) -> None:
    """At ``pose`` and at rest in BOTH of the engine's states: it swaps them every exchange, and in an MPM-only scene
    no solver writes a kinematic body, so a row left in the other state comes back every second exchange."""
    bodies, starts = _rows(sim, name)
    for st in (sim.engine.state, sim.engine._next):
        jqd = st.joint_qd.numpy()
        assert not st.body_qd.numpy()[bodies].any() and not any(jqd[s:s + 6].any() for s in starts)
        assert np.allclose(st.body_q.numpy()[bodies], pose, atol=1e-6, rtol=0)
    assert not sim.state([name])[f"{name}.vel"].any()       # what the API reads: AUDIT-STACK API-16


def _driven(route: str) -> tuple[alloy.Sim, str]:
    """A two-world CPU scene on ``route`` whose kinematic "tool" is under a drive that never ends, 3 periods in."""
    add, velocity, angular = SCENES[route]
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    name = add(scene)
    sim = alloy.build(scene, device="cpu")
    sim.drive(name, velocity=velocity, angular_velocity=angular)
    sim.step(3)
    bodies, starts = _rows(sim, name)
    jqd = sim.engine.state.joint_qd.numpy()
    assert np.allclose(sim.engine.state.body_qd.numpy()[bodies][:, 2], velocity[2])   # driving: the wall moves
    assert np.allclose([jqd[s + 2] for s in starts], velocity[2])
    assert np.allclose(sim.engine.state.body_qd.numpy()[bodies][:, 3:], angular)
    return sim, name


@pytest.mark.compile_heavy
@pytest.mark.parametrize("route", sorted(SCENES))
def test_stop_drives_leaves_the_body_at_rest_where_it_is(route):
    sim, name = _driven(route)
    bodies, _ = _rows(sim, name)
    pose = sim.engine.state.body_q.numpy()[bodies]
    pos = sim.state([name])[f"{name}.pos"].copy()
    sim.stop_drives()
    _assert_held(sim, name, pose)                            # was the drive's last velocity before the fix
    sim.step(3)
    _assert_held(sim, name, pose)                            # and stays so: nothing writes it back
    assert np.array_equal(sim.state([name])[f"{name}.pos"], pos)
    assert not [o for o in sim.engine._observers if isinstance(o, _drive.Drive)]
    sim.close()


@pytest.mark.compile_heavy
@pytest.mark.parametrize("route", sorted(SCENES))
def test_set_pose_holds_the_new_pose_at_rest_in_both_states(route):
    """``set_pose`` wrote ``engine.state`` alone: the other state kept the old pose and the drive's speed, which came
    back every second exchange (physics' check of WP7's rig: a set_pose hold drifted 2.99 -> 4.55 N)."""
    sim, name = _driven(route)
    bodies, _ = _rows(sim, name)
    target = sim.state([name])[f"{name}.pos"] + np.array([0.0, 0.0, 0.002])
    sim.set_pose(name, pos=target)                           # every world: it removes the running drive
    pose = sim.engine.state.body_q.numpy()[bodies]
    _assert_held(sim, name, pose)
    sim.step(3)
    _assert_held(sim, name, pose)
    assert np.allclose(sim.state([name])[f"{name}.pos"], target, atol=1e-6, rtol=0)
    sim.close()


def _sand_sink(stop: bool) -> np.ndarray:
    """Settled sand in a tray; a 2 cm tool set 0.5 mm over its surface presses at 5 cm/s, 2.5 mm in, and the press
    ends by ``stop_drives`` mid-press (``stop``) or by the drive's own ``duration``; then a 0.2 s hold. Returns how far
    each grain under the tool sank during the hold (m)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tray", inner=(0.05, 0.05, 0.03), wall=0.004, pos=(0, 0, 0), static=True)
    scene.add_box("bed", size=(0.05, 0.05, 0.016), pos=(0, 0, 0.012), material="sand_silica_dry")
    scene.add_box("tool", size=(0.02, 0.02, 0.01), pos=(0, 0, 0.05), kinematic=True)
    sim = alloy.build(scene, device="cpu")
    sim.step(20)                                             # the sand settles
    p = sim.state(["bed"], world=0)["bed.points"]
    surface = float(p[(np.abs(p[:, 0]) < 0.01) & (np.abs(p[:, 1]) < 0.01), 2].max())
    sim.set_pose("tool", pos=(0, 0, surface + 0.0055))
    sim.step(1)
    sim.drive("tool", velocity=(0, 0, -0.05), duration=float("inf") if stop else 0.06)
    sim.step(6)                                              # 3 mm: the 0.5 mm gap, then 2.5 mm into the sand
    if stop:
        sim.stop_drives()
    z0 = sim.state(["bed"], world=0)["bed.points"]
    sim.step(20)
    z1 = sim.state(["bed"], world=0)["bed.points"]
    assert sim.check()["ok"]
    sim.close()
    under = (np.abs(z0[:, 0]) < 0.01) & (np.abs(z0[:, 1]) < 0.01)
    return z0[under, 2] - z1[under, 2]


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_sand_under_a_tool_stopped_mid_press_sinks_no_more_than_with_duration():
    """AUDIT-STACK API-16: stopped with ``stop_drives`` the tool kept its 5 cm/s. On this rig (CPU) the sand under it
    then sank 0.60 mm on average in 0.2 s (2.5 mm at most) against -0.03 mm (0.016 mm) when the drive ended by its
    duration; with the fix both read the same, bit for bit."""
    stopped, ended = _sand_sink(True), _sand_sink(False)
    assert stopped.mean() <= ended.mean() + 5e-5, (stopped.mean(), ended.mean())
    assert stopped.max() <= ended.max() + 2e-4, (stopped.max(), ended.max())
