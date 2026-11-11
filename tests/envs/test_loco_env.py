# SPDX-License-Identifier: Apache-2.0
"""The Go2 velocity environment end to end on CPU (NumPy backend, 4 worlds): spaces, projected gravity,
accelerometer, foot contact, commands, pushes, termination and autoreset, action mapping, determinism, health and
import hygiene. Parts that need no simulation are in ``test_loco_terms.py``."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alloy.envs.loco import GO2, Go2VelocityConfig, Pushes, VelocityCommands, go2_velocity, go2_velocity_task
from alloy.envs.loco import terms as T
from alloy.envs.loco.env import LocoEnv, stand_action

W = 4
DT = 0.02
ROOT = Path(__file__).resolve().parents[2]
PUSH = Pushes(every_s=0.5, duration_s=0.1, force=(80.0, 80.0, 0.0))
pytestmark = pytest.mark.compile_heavy


def _build(pushes: Pushes | None) -> LocoEnv:
    cfg = Go2VelocityConfig(pushes=pushes, commands=VelocityCommands(standing=0.0))
    return go2_velocity(num_envs=W, device="cpu", backend="numpy", config=cfg)


@pytest.fixture(scope="module")
def env():
    e = _build(None)
    yield e
    e.close()


def _stand(env: LocoEnv, steps: int, seed: int = 0) -> None:
    env.reset(seed=seed)
    zero = stand_action(env)
    for _ in range(steps):
        env.step(zero)


def _keys(labels: list[str]) -> list[str]:
    out: list[str] = []
    for label in labels:
        key = re.sub(r"\[\d+\]$", "", label)
        if not out or out[-1] != key:
            out.append(key)
    return out


# --- spaces and labels --------------------------------------------------------------------------------------------

def test_observation_groups_have_the_documented_widths_and_labels(env) -> None:
    """Policy (W, 55), critic (W, 79); labels list the configured keys in order; critic = policy + privileged."""
    obs, info = env.reset(seed=0)
    assert env.obs_groups == ("policy", "critic") and list(obs) == ["policy", "critic"]
    assert obs["policy"].shape == (W, 55) and obs["critic"].shape == (W, 79)
    assert np.allclose(obs["critic"][:, :55], obs["policy"])
    names, critic_names = env.obs_names(), env.obs_names("critic")
    policy_keys = [k if isinstance(k, str) else k[0] for k in T.POLICY_OBS]
    privileged_keys = [k if isinstance(k, str) else k[0] for k in T.PRIVILEGED_OBS]
    assert len(names) == 55 and len(critic_names) == 79
    assert _keys(names) == policy_keys and _keys(critic_names) == policy_keys + privileged_keys


def test_step_returns_groups_and_the_documented_info_keys(env) -> None:
    """A step returns the same two groups; info carries command, invalid, success, targets, terms (no group copy)."""
    env.reset(seed=0)
    obs, _, _, _, info = env.step(stand_action(env))
    assert obs["policy"].shape == (W, 55) and obs["critic"].shape == (W, 79)
    assert {"command", "invalid", "success", "targets", "terms"} <= set(info) and "obs_groups" not in info
    for gone in ("observation_groups", "obs_group_names"):
        assert not hasattr(env, gone)
    assert not hasattr(env.config, "critic_obs")


def test_spaces_and_action_names(env) -> None:
    """Observation Dict of policy (W, 55) and critic (W, 79), action Box(-4, 4, (12,)) per world, 12 joint labels."""
    assert env.single_observation_space["policy"].shape == (55,)
    assert env.single_observation_space["critic"].shape == (79,)
    assert env.observation_space["policy"].shape == (W, 55) and env.observation_space["critic"].shape == (W, 79)
    assert env.action_space.shape == (W, 12) and env.single_action_space.shape == (12,)
    assert np.all(env.single_action_space.low == -4.0) and np.all(env.single_action_space.high == 4.0)
    assert env.action_names() == GO2.joint_names("go2") and len(env.action_names()) == 12


# --- sensors ------------------------------------------------------------------------------------------------------

def test_gravity_points_down_in_the_base_frame_when_upright(env) -> None:
    """At reset the base is level: projected gravity is (0, 0, -1) within 0.02."""
    env.reset(seed=0)
    assert np.allclose(env.state["base.gravity_b"], [0.0, 0.0, -1.0], atol=0.02)


def test_standing_robot_reads_g_up_and_no_rotation(env) -> None:
    """After 1 s of zero actions: accelerometer (0, 0, 9.81) within 0.5 m/s^2, gyroscope under 0.1 rad/s."""
    _stand(env, 50)
    s = env.state
    assert np.allclose(s["base.accel_b"], [0.0, 0.0, 9.81], atol=0.5)
    assert np.abs(s["base.ang_vel_b"]).max() < 0.1


def test_standing_feet_touch_the_floor(env) -> None:
    """After 1 s standing, all four feet are in contact. The soles sit 13-14 mm below z = 0 (the Go2's soft foot
    contact sinks that far under the robot's weight), so the window is [-0.02, 0.005] rather than [-0.003, 0.005]."""
    _stand(env, 50)
    s = env.state
    assert (s["feet.contact"] == 1).all()
    assert (s["feet.height"] > -0.02).all() and (s["feet.height"] < 0.005).all()


def test_lifted_robot_has_no_contact_then_lands_with_air_time(env) -> None:
    """Lifted 0.2 m: no foot touches; falling back, a touchdown appears on a step whose air time is positive."""
    _stand(env, 50)
    lifted = env.state["base.height"] + 0.2
    env.sim.set_pose("go2", pos=np.stack([np.zeros(W), np.zeros(W), lifted], 1))
    env.step(stand_action(env))
    assert (env.state["feet.contact"] == 0).all()
    landed = np.zeros(W, bool)
    for _ in range(40):
        env.step(stand_action(env))
        s = env.state
        landed |= ((s["feet.touchdown"] == 1) & (s["feet.air_time"] > 0)).any(1)
    assert landed.all()


# --- commands -----------------------------------------------------------------------------------------------------

def test_step_info_command_is_the_state_command(env) -> None:
    """The command reported by a step is the one in `env.state` right after it."""
    env.reset(seed=0)
    _, _, _, _, info = env.step(stand_action(env))
    assert np.array_equal(env.state["command"], info["command"])


def test_masked_reset_redraws_only_the_masked_worlds(env) -> None:
    """Resetting worlds 1 and 3 changes their commands and leaves 0 and 2 alone."""
    env.reset(seed=5)
    before = env.state["command"].copy()
    env.reset(options={"reset_mask": np.array([False, True, False, True])})
    after = env.state["command"]
    assert np.array_equal(after[[0, 2]], before[[0, 2]])
    assert (np.abs(after[[1, 3]] - before[[1, 3]]).max(1) > 1e-6).all()


# --- action mapping -----------------------------------------------------------------------------------------------

def _targets_after(env: LocoEnv, joint: int, value: float) -> np.ndarray:
    env.reset(seed=0)
    a = stand_action(env)
    a[:, joint] = value
    return env.step(a)[4]["targets"]


def test_action_one_moves_a_joint_target_by_a_quarter_radian(env) -> None:
    """Action 1.0 on a calf joint writes 0.25 rad there and 0 on the other 11 joints."""
    t = _targets_after(env, 2, 1.0)
    want = np.zeros((W, 12))
    want[:, 2] = 0.25
    assert np.allclose(t, want, atol=1e-6)


def test_oversized_actions_clip_at_four_then_at_the_joint_range(env) -> None:
    """Action 10 acts as 4 (target 1.0 rad on a thigh, range to 1.6); on a hip it stops at its 0.9472 rad limit."""
    assert np.allclose(_targets_after(env, 1, 10.0)[:, 1], 1.0, atol=1e-6)
    assert np.allclose(_targets_after(env, 0, 10.0)[:, 0], 0.9472, atol=1e-4)


def test_joint_torque_follows_the_servo_law_and_motor_limit(env) -> None:
    """torque = clip(50 (target - q) - 0.5 qd, +-limit) from the state's own q, qd and target."""
    _targets_after(env, 2, 1.0)
    s = env.state
    want = np.clip(GO2.kp * (s["joints.target"] - s["joints.pos"]) - GO2.kd * s["joints.vel"],
                   -np.array(GO2.torque_limit), np.array(GO2.torque_limit))
    assert np.allclose(s["joints.torque"], want, atol=1e-3) and np.abs(s["joints.torque"][:, 2]).max() > 1.0


# --- termination and reset ----------------------------------------------------------------------------------------

def test_flipped_world_terminates_and_autoresets_clean(env) -> None:
    """Flipping world 0 upside down ends it: terminated, an episode report, then upright again with zeroed
    action buffers and a new command; the other worlds go on."""
    _stand(env, 10)
    old_command = env.commands.value.copy()
    env.sim.set_pose("go2", pos=(0.0, 0.0, 0.4), quat_wxyz=(0.0, 1.0, 0.0, 0.0), world=0)
    a = stand_action(env) + 1.0
    _, _, terminated, truncated, info = env.step(a)
    s = env.state
    assert terminated.tolist() == [True, False, False, False] and not truncated.any()
    assert info["_episode"][0] and "episode" in info
    assert 0.25 <= float(s["base.height"][0]) <= 0.30 and s["base.gravity_b"][0, 2] < -0.9
    assert not s["action"][0].any() and not s["action.prev"][0].any() and np.abs(s["action"][1]).min() == 1.0
    assert np.abs(s["command"][0] - old_command[0]).max() > 1e-6
    assert np.array_equal(s["command"][1:], old_command[1:])


def test_episode_length_follows_episode_seconds() -> None:
    """`episode_s=0.2` at dt 0.02 gives a 10-step limit (no simulation needed)."""
    assert go2_velocity_task(Go2VelocityConfig(episode_s=0.2)).max_steps == 10
    assert go2_velocity_task().max_steps == 1000


# --- determinism and health ---------------------------------------------------------------------------------------

def test_same_seed_gives_the_same_commands_and_first_observation(env) -> None:
    """Two resets with seed 3 agree on commands and on the first observation."""
    obs1, _ = env.reset(seed=3)
    cmd1 = env.commands.value.copy()
    obs2, _ = env.reset(seed=3)
    assert np.allclose(cmd1, env.commands.value, atol=1e-6)
    assert all(np.allclose(obs1[g], obs2[g], atol=1e-6) for g in obs1)


def test_hundred_zero_action_steps_stay_healthy(env) -> None:
    """After the stepping above and 100 more standing steps the engine's health check passes."""
    _stand(env, 100)
    assert env.sim.check()["ok"]


# --- pushes (second environment, pushes every 0.5 s) --------------------------------------------------------------

@pytest.fixture(scope="module")
def push_run():
    """Record 40 zero-action steps: the held push force and the base's world velocity before each step."""
    e = _build(PUSH)
    e.reset(seed=0)
    mass = float(e.sim.engine.model.body_mass.numpy().reshape(W, -1).sum(1)[0])
    force, vel = [], [e.state["base.lin_vel_w"].copy()]
    for _ in range(40):
        e.step(stand_action(e))
        force.append(e.pusher.force.copy())
        vel.append(e.state["base.lin_vel_w"].copy())
    e.close()
    return mass, np.array(force), np.array(vel)


def test_push_force_is_held_for_five_steps_from_tick_25(push_run) -> None:
    """Zero before step 25; nonzero on steps 25-29 only; within +-80 N; different between worlds."""
    _, force, _ = push_run
    active = (np.abs(force).max((1, 2)) > 0)
    assert np.flatnonzero(active).tolist() == [25, 26, 27, 28, 29]
    assert (np.abs(force[25:30, :, 0:2]) <= 80).all() and (force[25:30, :, 2] == 0).all()
    assert len({tuple(r) for r in force[25].round(2)}) == W


def test_push_moves_the_base_along_the_force_with_f_t_over_m_speed(push_run) -> None:
    """Planar velocity change over the push points along the force (cosine > 0.8), 0.3-1.5 x F t / m in size."""
    mass, force, vel = push_run
    dv = vel[30, :, 0:2] - vel[25, :, 0:2]
    expect = force[25, :, 0:2] * (5 * DT) / mass
    cos = (dv * expect).sum(1) / (np.linalg.norm(dv, axis=1) * np.linalg.norm(expect, axis=1))
    ratio = np.linalg.norm(dv, axis=1) / np.linalg.norm(expect, axis=1)
    assert 12.0 < mass < 20.0
    assert (cos > 0.8).all(), cos
    assert ((ratio > 0.3) & (ratio < 1.5)).all(), ratio


def test_env_without_pushes_never_pushes(env) -> None:
    """A `pushes=None` environment keeps its push force zero over more than one push period."""
    _stand(env, 30)
    assert not env.state["push.force"].any() and not env.pusher.force.any()


# --- imports ------------------------------------------------------------------------------------------------------

BLOCKED = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "gymnasium", "rsl_rl"):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)", name=name)

sys.meta_path.insert(0, Block())
import alloy.envs.loco as loco
assert loco.go2_velocity and loco.terms and loco.LocoEnv
import alloy.envs.loco.train as train
assert train.train and train.evaluate and train.ppo_config and train.main
print("ok")
"""


def test_loco_imports_without_torch_rsl_rl_or_gymnasium() -> None:
    """`import alloy.envs.loco` and `alloy.envs.loco.train` need neither PyTorch, rsl_rl nor Gymnasium."""
    res = subprocess.run([sys.executable, "-c", BLOCKED], cwd=ROOT, capture_output=True, text=True, timeout=300,
                         check=False)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]
