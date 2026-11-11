# SPDX-License-Identifier: Apache-2.0
"""End to end on CPU: build, per-world starts, run with a target ramp, named state, trajectory, health, check."""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy

SLIDER = ('<mujoco><worldbody><body name="carriage" pos="0 0 0.3">'
          '<joint name="z" type="slide" axis="0 0 1" range="-0.5 0.5" damping="20"/>'
          '<geom type="box" size="0.02 0.02 0.02" mass="0.2" contype="0" conaffinity="0"/></body></worldbody>'
          '<actuator><position joint="z" kp="400" kv="20"/></actuator></mujoco>')


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.04, 0.04, 0.04), pos=[[0.0, 0.0, 0.1], [0.05, 0.0, 0.1]], mass=0.1)
    scene.add_robot("slider", mjcf=SLIDER, pos=(0.3, 0.0, 0.0))
    s = alloy.build(scene, device="cpu")
    yield s
    s.close()


@pytest.mark.compile_heavy
def test_names_and_start_state(sim):
    names = sim.names()
    assert "cube" in names["bodies"] and "slider/carriage" in names["bodies"] and names["static"] == ["ground"]
    assert "z" in names["joints"] and "slider/z" in names["joints"]
    st = sim.state(["cube"])
    assert st["cube.pos"].shape == (2, 3) and st["cube.quat_wxyz"].shape == (2, 4)
    assert np.allclose(st["cube.pos"][:, 0], [0.0, 0.05], atol=1e-6)        # per-world starts applied
    assert np.allclose(st["cube.quat_wxyz"], [1, 0, 0, 0], atol=1e-6)


@pytest.mark.compile_heavy
def test_run_ramp_and_trajectory(sim, tmp_path):
    traj = sim.run(0.4, record=["cube", "z"], targets={"z": (0.0, 0.05)})
    assert traj.frames == 41 and traj["cube.pos"].shape == (41, 2, 3) and traj["z.q"].shape == (41, 2)
    assert traj["z.q"][-1] == pytest.approx([0.05, 0.05], abs=0.02)            # the slider followed the ramp
    z = traj["cube.pos"][-1, :, 2]
    assert np.allclose(z, 0.02, atol=2e-3)                                     # the cube landed and rests
    assert traj.meta["physics_steps"] > 0 and traj.meta["health"]["ok"] and traj.meta["mpm_grid"] == []
    path = traj.world(1).save(tmp_path / "t.npz")
    assert np.load(path)["cube.pos"].shape == (41, 3)
    assert sim.check()["ok"] and sim.t == pytest.approx(0.4)


@pytest.mark.compile_heavy
def test_unknown_name_lists_the_names(sim):
    with pytest.raises(alloy.UnknownNameError) as info:
        sim.state(["cub"])
    assert "cube" in info.value.fix
    with pytest.raises(alloy.UnknownNameError, match="static"):
        sim.state(["ground"])


@pytest.mark.compile_heavy
def test_seed_reaches_randomized_draws():
    def xs(seed: int) -> np.ndarray:
        scene = alloy.Scene(worlds=3, dt=0.01)
        scene.add_ground()
        scene.add_box("b", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.1))
        scene.randomize("b", "x", -0.05, 0.05)
        s = alloy.build(scene, device="cpu", seed=seed)
        try:
            first = s.state(["b"])["b.pos"][:, 0].copy()
            s.step(2).reset()
            assert np.array_equal(s.state(["b"])["b.pos"][:, 0], first)      # reset redraws at the same seed
            return first
        finally:
            s.close()

    a, b, c = xs(1), xs(1), xs(2)
    assert np.array_equal(a, b) and not np.allclose(a, c) and np.all(np.abs(a) <= 0.05)


@pytest.mark.compile_heavy
def test_actuator_names_and_unknown_joint(sim):
    assert sim.names()["actuators"] == ["slider/z", "z"]
    with pytest.raises(alloy.UnknownNameError, match="no joint named 'cube'"):
        sim.set_targets({"cube": 0.1})
