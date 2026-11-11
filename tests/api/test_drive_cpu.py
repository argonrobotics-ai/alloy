# SPDX-License-Identifier: Apache-2.0
"""Kinematic objects on CPU: drive (lift, tilt a fixture), set_pose, settle -> reset."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.api as alloy


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("tray", size=(0.2, 0.2, 0.02), pos=(0, 0, 0.1), kinematic=True)
    scene.add_box("cube", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.131), mass=0.1)
    scene.add_open_box("bin", inner=(0.06, 0.06, 0.04), wall=0.004, pos=(0.4, 0, 0), kinematic=True)
    s = alloy.build(scene, device="cpu")
    yield s
    s.close()


@pytest.mark.compile_heavy
def test_settle_then_reset_returns_to_the_settled_state(sim):
    sim.settle(0.3)
    z = float(sim.state(["cube"], world=0)["cube.pos"][2])
    assert z == pytest.approx(0.13, abs=2e-3)                  # resting on the tray
    sim.step(5).reset()
    assert float(sim.state(["cube"], world=0)["cube.pos"][2]) == pytest.approx(z, abs=1e-6)


@pytest.mark.compile_heavy
def test_drive_lifts_the_tray_and_the_cube_rides(sim):
    sim.drive("tray", velocity=(0, 0, 0.1), duration=0.5)
    traj = sim.run(0.8, record=["tray", "cube"])
    assert float(traj["tray.pos"][-1, 0, 2]) == pytest.approx(0.15, abs=2e-3)   # 0.1 m/s x 0.5 s, then holds
    assert float(traj["cube.pos"][-1, 0, 2]) == pytest.approx(0.18, abs=3e-3)
    assert sim.check()["ok"]


@pytest.mark.compile_heavy
def test_drive_tilts_a_fixture_about_a_pivot_and_set_pose(sim):
    pivot = (0.434, 0.0, 0.044)                                 # the bin's +x rim
    sim.drive("bin", angular_velocity=(0, -math.radians(40), 0), pivot=pivot, duration=0.5)
    sim.step(60)
    q = sim.state(["bin_floor"], world=0)["bin_floor.quat_wxyz"]
    assert 2 * math.degrees(math.acos(min(1.0, abs(float(q[0]))))) == pytest.approx(20.0, abs=1.0)
    sim.stop_drives().set_pose("tray", pos=(0, 0, 0.3))
    assert np.allclose(sim.state(["tray"], world=0)["tray.pos"], [0, 0, 0.3], atol=1e-6)
    with pytest.raises(alloy.UnknownNameError, match="static"):
        sim.drive("ground", velocity=(0, 0, 1))


@pytest.mark.compile_heavy
def test_a_drive_per_period_replaces_the_last_one_and_moves_the_same(sim):
    """WP1's whisk review: one drive per control period piled up 1,600 observers by 4 s (each step slower)."""
    sim.stop_drives().set_pose("tray", pos=(0, 0, 0.2))
    for _ in range(40):
        sim.drive("tray", velocity=(0.0, 0.1, 0.0), duration=sim.dt)
        sim.step(1)
    drives = [o for o in sim.engine._observers if type(o).__name__ == "Drive"]
    assert [d.name for d in drives] == ["tray"]                             # the newest only, not 40
    pos = sim.state(["tray"], world=0)["tray.pos"]
    assert pos[1] == pytest.approx(40 * sim.dt * 0.1, abs=1e-5) and pos[2] == pytest.approx(0.2, abs=1e-6)
    sim.stop_drives()
