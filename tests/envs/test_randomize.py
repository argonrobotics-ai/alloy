# SPDX-License-Identifier: Apache-2.0
"""Random starts on CPU: jittered objects and joints differ per world and per reset, stay where they were put, are
reproducible from the seed, and a masked reset leaves the other worlds alone."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import EnvError, VectorEnv, jitter_body, jitter_joint
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task


def _yaw_deg(q_xyzw: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arctan2(q_xyzw[:, 2], q_xyzw[:, 3]))


@pytest.fixture(scope="module")
def env():
    task = gripper_lift_task()
    task.randomize = [jitter_body("cube", pos=(0.02, 0.0, 0.0), yaw_deg=30), jitter_joint("gripper/lift", by=0.02)]
    e = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu", backend="numpy")
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_jitter_ranges_hold_and_seed(env):
    env.reset(seed=3)
    s = env.state
    x, yaw, lift = s["cube.pos"][:, 0].copy(), _yaw_deg(s["cube.quat_xyzw"]), s["gripper/lift.q"].copy()
    assert np.all(np.abs(x) <= 0.02) and np.ptp(x) > 1e-3 and np.all(np.abs(yaw) <= 30.0) and np.ptp(yaw) > 1.0
    assert np.all(np.abs(lift) <= 0.02) and np.ptp(lift) > 1e-3
    assert np.allclose(s.start["cube.pos"], s["cube.pos"])                        # rise() counts from here
    targets = env.sim.engine.control.joint_target_q.numpy()[env.layout.joint("gripper/lift").target_q[:, 0]]
    assert np.allclose(targets, lift)                                             # the servo holds the new start
    for _ in range(10):
        env.step(np.zeros((4, 3), np.float32))
    s = env.state
    assert np.allclose(s["cube.pos"][:, 0], x, atol=1e-3) and np.allclose(_yaw_deg(s["cube.quat_xyzw"]), yaw, atol=0.5)
    env.reset(seed=3)
    assert np.allclose(env.state["cube.pos"][:, 0], x) and np.allclose(env.state["gripper/lift.q"], lift)
    env.reset(seed=4)
    assert not np.allclose(env.state["cube.pos"][:, 0], x)


@pytest.mark.compile_heavy
def test_masked_reset_jitters_only_masked(env):
    env.reset(seed=0)
    before = env.state["cube.pos"].copy()
    env.reset(options={"reset_mask": np.array([True, False, False, True])})
    after = env.state["cube.pos"]
    assert np.allclose(after[[1, 2]], before[[1, 2]]) and not np.allclose(after[[0, 3], 0], before[[0, 3], 0])


def test_bad_jitters(env):
    with pytest.raises(EnvError, match="half-ranges"):
        jitter_body("cube", pos=(-0.1, 0, 0))
    with pytest.raises(EnvError, match="by >= 0"):
        jitter_joint("gripper/lift", by=-1)
    from alloy.envs.randomize import Randomizer

    with pytest.raises(EnvError, match="not a free object"):
        Randomizer([jitter_body("gripper/palm", pos=(0.01, 0, 0))], env.layout, env.sim.engine, env.xp)


@pytest.mark.compile_heavy
def test_masked_jitter_keeps_other_worlds_targets(env):
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32))     # raise every lift target
    rows = env.layout.joint("gripper/lift").target_q[:, 0]
    before = env.sim.engine.control.joint_target_q.numpy()[rows].copy()
    env.reset(options={"reset_mask": np.array([True, False, False, False])})
    after = env.sim.engine.control.joint_target_q.numpy()[rows]
    assert np.allclose(after[1:], before[1:]) and abs(after[0] - env.state["gripper/lift.q"][0]) < 1e-6


CUP = """<mujoco><worldbody><body name="cup" pos="0 0 0.06">
  <joint name="x" type="slide" axis="1 0 0" range="-0.1 0.1" damping="50"/>
  <geom type="box" size="0.032 0.032 0.003" pos="0 0 -0.03" mass="0.1"/>
  <geom type="box" size="0.003 0.032 0.03" pos="0.032 0 0" mass="0.02"/>
  <geom type="box" size="0.003 0.032 0.03" pos="-0.032 0 0" mass="0.02"/>
  <geom type="box" size="0.032 0.003 0.03" pos="0 0.032 0" mass="0.02"/>
  <geom type="box" size="0.032 0.003 0.03" pos="0 -0.032 0" mass="0.02"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="200"/></actuator></mujoco>"""


def _cup_env(carry: tuple[str, ...], fill: str = "water", yaw: float = 0.0) -> VectorEnv:
    """2 worlds: the cup on its slide (turned ``yaw`` rad about z), holding water or a small cloth, jittered by up
    to 3 cm at each reset."""
    import alloy.api as alloy
    from alloy.envs import ActionSpec, Task

    sc = alloy.Scene(dt=0.02)
    sc.add_ground()
    sc.add_robot("pourer", mjcf=CUP, rpy=(0.0, 0.0, yaw))
    if fill == "water":
        sc.add_liquid("water", size=(0.04, 0.04, 0.012), pos=(0.0, 0.0, 0.037))  # on the cup's floor (top z 0.033)
    else:
        sc.add_cloth("sheet", size=(0.04, 0.04), pos=(0.0, 0.0, 0.036), resolution=5, mass=0.005)
    task = Task(obs=["pourer"], actions=ActionSpec(["pourer/x"], mode="delta", delta=0.005), max_steps=50,
                randomize=[jitter_joint("pourer/x", by=0.03, carry=carry)])
    return VectorEnv(sc, task, worlds=2, device="cpu", backend="numpy", seed=0)


def _centre(env: VectorEnv, name: str) -> np.ndarray:
    return np.asarray(env.sim.state([name])[f"{name}.points"], np.float64).mean(-2)       # (W, 3)


def _slide(env: VectorEnv) -> np.ndarray:
    return np.asarray(env.state["pourer/x.q"], np.float64).reshape(-1)


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("carry", "carried"), [(("water",), True), ((), False)])
def test_a_jittered_cup_carries_its_water(carry, carried):
    """carry= takes a particle object: the water moves with the cup's slide (before, it stayed where the cup was and
    a 3 cm jitter left 20-41 % of it outside the cup after 10 steps). Without carry: left behind, as before."""
    env = _cup_env(carry)
    start = _centre(env, "water")
    env.reset(seed=1)
    x, moved = _slide(env), _centre(env, "water") - start
    env.close()
    assert np.abs(x).max() > 0.01                                                # each world, by its own offset
    assert np.allclose(moved[:, 0], x if carried else 0.0, atol=1e-5) and np.allclose(moved[:, 1:], 0.0, atol=1e-5)


@pytest.mark.compile_heavy
def test_a_partial_reset_moves_only_the_reset_worlds_water():
    env = _cup_env(("water",))
    start = _centre(env, "water")
    env.reset(seed=1)
    before, x_before = _centre(env, "water"), _slide(env)
    env.reset(options={"reset_mask": np.array([True, False])})
    after, x = _centre(env, "water"), _slide(env)
    env.close()
    assert x[0] != x_before[0] and np.isclose(after[0, 0] - start[0, 0], x[0], atol=1e-5)  # from its start, not again
    assert x[1] == x_before[1] and np.array_equal(after[1], before[1])                     # untouched


@pytest.mark.compile_heavy
def test_a_turned_slide_carries_its_water_along_its_own_axis():
    yaw = 0.6
    env = _cup_env(("water",), yaw=yaw)
    start = _centre(env, "water")
    env.reset(seed=1)
    x, moved = _slide(env), _centre(env, "water") - start
    env.close()
    axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])                             # the slide's x, turned about z
    assert np.abs(x).max() > 0.01 and np.allclose(moved, x[:, None] * axis[None, :], atol=1e-5)


@pytest.mark.compile_heavy
def test_a_jittered_tray_carries_its_cloth():
    env = _cup_env(("sheet",), fill="cloth")
    start = _centre(env, "sheet")
    env.reset(seed=1)
    x, moved = _slide(env), _centre(env, "sheet") - start
    env.close()
    assert np.abs(x).max() > 0.01 and np.allclose(moved[:, 0], x, atol=1e-5) and np.allclose(moved[:, 1:], 0.0,
                                                                                           atol=1e-5)


def test_carry_names_what_it_can_move():
    task = gripper_lift_task()
    task.randomize = [jitter_joint("gripper/lift", by=0.01, carry=["nothing"])]
    with pytest.raises(EnvError, match="free object or a particle object"):
        VectorEnv(gripper_lift_scene(2), task, worlds=2, device="cpu", backend="numpy")
