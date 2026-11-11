# SPDX-License-Identifier: Apache-2.0
"""Delta and end-effector targets stay near the measured joints, so a blocked joint does not wind its target up.

``ActionSpec(mode="delta")`` and ``EEActionSpec`` add each step's change to the last TARGET. Against a table, a part
or a door stop the target ran on while the joint stood still, and the servo pushed harder every step: in a 100-step
scripted Panda push with end-effector actions, fingers went 1-2 cm into the table and cube (a training audit). Now
the last target is first pulled back toward the joint's measured position (``ActionWriter.held``): within 5 x
``delta`` per joint for joint deltas; for an arm under end-effector actions the arm's lead shrinks as one vector until
every joint is within 0.03 rad (the hand keeps its direction; a clip per joint bent its path, and a scripted 10 g
pick ended over 2 cm off its part).

A servo that lags its target by more than the window at full speed now moves slower under full actions (the pusher of
``examples/envs/02_custom_task.py``, kp 400: 91 steps to its goal, 47 with no window); one that keeps up moves the same.

Measured (CPU, the Panda only, 2 worlds, seeds 0-3): after a 3 s press on the table an up command lifts the hand 1 cm
in 4 steps, in 12-23 with no window (seed 0, the test's: 23). The audit's push (64 worlds, seeds 9001 and 11-15,
both CPU routes): with the arm packs' firm contacts 0 INVALID with and without the window (also one L4, the audit's
own torch repro, three runs), so the window no longer prevents a freeze on the shipped arms; at MuJoCo's default
contact softness 34-45 with no window, 14-22 with it (1-7 with a clip per joint, not shipped: it bends the path).
The 0.03 rad is set for the Panda's shipped gains. The scripted examples (pick-and-place, peg, drawer, pour,
end-effector at 64 worlds) end with the same outcomes as with no window, though the window binds in many steps.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, EnvError, Task, VectorEnv
from alloy.envs.diffik import EEActionSpec

SLIDER = """<mujoco><worldbody><body name="cart">
  <joint name="x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="20"/>
  <geom type="box" size="0.02 0.02 0.02" mass="0.5"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="100"/></actuator></mujoco>"""
DELTA, WALL_GAP = 0.01, 0.03          # m per step at action 1; free travel before the cart's face meets the wall
STEPS = 5                             # the default window, in delta steps (the literal, not the module constant)


def _slider_env(window=None, wall: bool = True) -> VectorEnv:
    sc = alloy.Scene(dt=0.02)
    sc.add_ground()
    sc.add_robot("slider", mjcf=SLIDER, pos=(0.0, 0.0, 0.1))
    if wall:
        sc.add_box("wall", size=(0.02, 0.2, 0.2), pos=(0.02 + WALL_GAP + 0.01, 0.0, 0.1), static=True)
    task = Task(obs=["slider"], actions=ActionSpec(["slider/x"], mode="delta", delta=DELTA, window=window),
                max_steps=1000)
    return VectorEnv(sc, task, worlds=2, device="cpu", backend="numpy")


def _push(env: VectorEnv, actions) -> tuple[np.ndarray, np.ndarray]:
    """(target, measured x) per world after stepping ``actions`` (a list of (W, 1) arrays)."""
    env.reset(seed=0)
    for a in actions:
        env.step(np.asarray(a, np.float32))
    w = env.writer
    target = np.asarray(env.xp.view(w.target)[w.rows])[:, 0]
    return target, np.asarray(env.state["slider/x.q"]).reshape(-1)


@pytest.mark.timeout(300)                     # two builds: over 60 s on a loaded host
def test_a_blocked_slide_keeps_its_target_within_the_window_of_the_joint():
    push = [np.ones((2, 1))] * 40                                   # 40 cm asked, the wall 3 cm away
    for window, near in ((None, True), (float("inf"), False)):
        env = _slider_env(window)
        target, x = _push(env, push)
        env.close()
        assert np.all(x < WALL_GAP + 0.003)                         # the wall holds the cart
        if near:
            assert np.all(target - x <= (STEPS + 1) * DELTA + 1e-6)          # the window, then this step's delta
            assert np.all(target - x > 0.5 * STEPS * DELTA)         # still pressing on the wall
        else:
            assert np.all(target - x > 0.3)                         # as before: the target ran on 37 cm


@pytest.mark.timeout(300)
def test_the_window_holds_a_target_on_both_sides_and_leaves_one_inside_alone():
    env = _slider_env(wall=False)
    env.reset(seed=0)
    w, xp = env.writer, env.xp
    assert np.allclose(np.asarray(w.window), STEPS * DELTA)
    q = np.asarray(xp.view(w.engine.state.joint_q)[w.q_rows], np.float32)
    for side in (1.0, -1.0):                                         # far ahead, far behind
        assert np.allclose(np.asarray(w.held(xp.asarray(q + side), w.window)), q + side * STEPS * DELTA)
    inside = (q + 0.5 * STEPS * DELTA).astype(np.float32)
    assert np.array_equal(np.asarray(w.held(xp.asarray(inside), w.window)), inside)
    env.close()


@pytest.mark.timeout(300)
def test_a_servo_that_keeps_up_moves_the_same_bit_for_bit():   # small random moves, no wall
    rng = np.random.default_rng(0)
    moves = [rng.uniform(-1, 1, (2, 1)) for _ in range(30)]
    runs = []
    for window in (None, float("inf")):
        env = _slider_env(window, wall=False)
        runs.append(_push(env, moves))
        env.close()
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


@pytest.mark.timeout(300)
def test_only_position_servos_get_a_window():
    for actuator, windowed in (('<motor joint="x" gear="1"/>', False),
                               ('<general joint="x" gainprm="2000" biastype="affine" biasprm="0 -2000 -100"/>', True)):
        sc = alloy.Scene(dt=0.02)
        sc.add_robot("slider", mjcf=SLIDER.replace('<position joint="x" kp="2000" kv="100"/>', actuator))
        task = Task(obs=["slider"], actions=ActionSpec(["slider/x"], kind="ctrl", mode="delta", delta=DELTA))
        env = VectorEnv(sc, task, worlds=2, device="cpu", backend="numpy")
        assert env.writer.measured_np.tolist() == [windowed]          # a motor's control is a force, not a position
        assert np.isfinite(np.asarray(env.writer.window)).tolist() == [windowed]
        env.close()


@pytest.mark.parametrize("window", [0.0, -0.01, [0.01, 0.0]])
def test_a_bad_window_raises_naming_it(window):
    with pytest.raises(EnvError) as e:
        ActionSpec(["a/x"], mode="delta", window=window).validate()
    assert e.value.field == "window"
    with pytest.raises(EnvError) as e:
        EEActionSpec(body="arm/hand", joints=["arm/j1"], window=-1.0).validate()
    assert e.value.field == "window"


@pytest.mark.compile_heavy
def test_the_arm_lead_shrinks_to_the_window_as_one_vector():
    import dataclasses

    from alloy.envs.tasks import panda_reach_task, panda_scene

    spec = EEActionSpec(body=HAND, joints=[f"panda/actuator{k}" for k in range(1, 8)])
    task = dataclasses.replace(panda_reach_task(max_steps=100), actions=spec, name="PandaHeld")
    env = VectorEnv(panda_scene(2, 0, props=False), task, worlds=2, device="cpu", backend="numpy")
    env.reset(seed=0)
    w, xp = env.writer, env.xp
    q = np.asarray(xp.view(w.engine.state.joint_q)[w.inner.q_rows], np.float32)
    lead = np.random.default_rng(0).uniform(-1.0, 1.0, q.shape).astype(np.float32)
    lead *= 0.2 / np.abs(lead).max(axis=1, keepdims=True)                  # 0.2 rad on the largest joint
    held = np.asarray(w._held(xp.asarray(q + lead))) - q
    env.close()
    assert np.allclose(np.abs(held).max(axis=1), 0.03, atol=1e-5)          # the window, exactly
    assert np.allclose(held, lead * 0.03 / 0.2, atol=1e-5)                 # the same direction: one shrink


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_hand_pressed_on_the_table_lifts_as_soon_as_it_is_sent_up_cpu():
    windup, held = _steps_to_lift(window=float("inf")), _steps_to_lift()
    print(f"up steps to lift 1 cm after a 3 s press: {windup} with no window, {held} with it")
    assert held <= 6 and windup >= 10                    # one robot, the Panda; seeds 0-3: 4 with it, 12-23 without


@pytest.mark.slow
@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
def test_a_scripted_ee_push_into_the_table_leaves_few_worlds_invalid_cpu():
    windup, held = _scripted_push(64, 9001, window=float("inf")), _scripted_push(64, 9001)
    print(f"INVALID of 64: {windup} with no window, {held} with it")
    assert held <= 10 and (windup < 10 or 3 * held <= windup)      # the firm arm packs: 0 both ways


def _steps_to_lift(**ee) -> int:
    """The Panda's hand sent down 2 cm a step for 150 steps (onto the table), then up: steps until it is 1 cm higher."""
    import dataclasses

    from alloy.envs.tasks import panda_reach_task, panda_scene

    spec = EEActionSpec(body=HAND, joints=[f"panda/actuator{k}" for k in range(1, 8)], **ee)
    task = dataclasses.replace(panda_reach_task(max_steps=1000), actions=spec, name="PandaPress")
    env = VectorEnv(panda_scene(2, 0, props=False), task, worlds=2, device="cpu", backend="numpy")
    env.reset(seed=0)
    for _ in range(150):
        env.step(np.tile(np.float32([0, 0, -1]), (2, 1)))
    low = np.array(env.state[f"{HAND}.pos"])[:, 2]
    for k in range(1, 61):
        env.step(np.tile(np.float32([0, 0, 1]), (2, 1)))
        if np.all(np.array(env.state[f"{HAND}.pos"])[:, 2] > low + 0.01):
            env.close()
            return k
    env.close()
    return 61                                                      # not within 60 steps


# --- the audit's scripted push, on the NumPy backend -----------------------------------------------------------------

CUBE = ('<mujoco><worldbody><body name="box"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.1" '
        'friction="0.6"/></body></worldbody></mujoco>')
BOX, HAND, TCP_OFFSET, GOAL_DX = "cube/box", "panda/hand", 0.1034, 0.2


def _push_scene():
    from alloy.cards.library import default_root
    from alloy.envs.mjcf import without_bodies

    path = without_bodies(default_root() / "robots" / "emb_e2_panda" / "scene.xml", ["box", "cylinder", "mug"])
    s = alloy.Scene(dt=0.02)
    s.add_robot("panda", path)
    s.add_robot("cube", mjcf=CUBE, pos=(0.45, 0.0, 0.02))
    s.randomize("cube", "mass_scale", 0.5, 1.5)
    s.randomize("cube", "friction", 0.3, 1.0)
    s.randomize("panda", "kp_scale", 0.8, 1.2)
    for f, lo, hi in (("start_x", -0.1, 0.1), ("start_y", -0.1, 0.1), ("start_yaw", -0.5, 0.5)):
        s.randomize("cube", f, lo, hi, when="reset")
    return s


def _tcp(st) -> np.ndarray:
    q = np.asarray(st[f"{HAND}.quat_wxyz"])
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    zaxis = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    return np.asarray(st[f"{HAND}.pos"]) + TCP_OFFSET * zaxis


def _push_env(worlds: int, seed: int, **ee) -> VectorEnv:
    from alloy.envs import jitter_joint
    from alloy.envs.realism import ActionDelay, ObsNoise
    from alloy.envs.tasks import PANDA_HOME

    obs = ["panda", BOX, "action"]
    task = Task(name="PandaPushCube", obs={"policy": obs, "critic": obs}, max_steps=100,
                actions=EEActionSpec(body=HAND, joints=[f"panda/actuator{k}" for k in range(1, 8)], delta=0.04, **ee),
                randomize=[jitter_joint(f"panda/{j}", by=0.05, at=v) for j, v in PANDA_HOME.items()],
                layers=[ObsNoise(0.002), ActionDelay((0, 1))])
    return VectorEnv(_push_scene(), task, worlds=worlds, seed=seed, backend="numpy", autoreset="disabled",
                     device="cpu")


def _scripted_push(worlds: int, seed: int, **ee) -> int:
    """INVALID worlds after the audit's 100-step scripted push: hand behind the cube, down, push 20 cm in +x."""
    env = _push_env(worlds, seed, **ee)
    env.reset(seed=seed)
    goal = np.array(env.state.start[f"{BOX}.pos"])
    goal[:, 0] += GOAL_DX
    for _ in range(100):
        p, c = _tcp(env.state), np.array(env.state[f"{BOX}.pos"])
        pre = c + 0.0
        pre[:, 0] -= 0.06
        pre[:, 2] = 0.03
        above = pre + np.array([0.0, 0.0, 0.07])
        xy_ok = np.linalg.norm(p[:, :2] - pre[:, :2], axis=-1) < 0.015
        push = xy_ok & (p[:, 2] < 0.04)
        tgt = np.where(push[:, None], goal + np.array([0.0, 0.0, 0.005]), np.where(xy_ok[:, None], pre, above))
        env.step(np.clip((tgt - p) / 0.04, -1, 1).astype(np.float32))
    invalid = len(env.sim.health()["invalid_worlds"])
    env.close()
    return invalid
