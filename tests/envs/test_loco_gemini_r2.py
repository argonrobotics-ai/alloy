# SPDX-License-Identifier: Apache-2.0
"""Gemini audit round 2, the Go2 envs (CPU):

- T1-016: the command was resampled inside the step before the reward, so a resample step scored the policy against
  a command it had not seen, and an episode ending on a resample step showed the curriculum a fresh command; the
  curriculum also compared the walk with the last command times the episode time. Now the reward reads the command
  the policy saw, finished worlds draw at their reset, and the curriculum compares with the commanded travel.
- T1-023: with ``ActionDelay``, step 1 got its own command at once (an absolute writer filled the delay with the
  first action) and ``joints.target`` showed the undelayed target. Now the delay holds the start target and the
  state shows what reached the servos.
- T1-022: ``loco.train.evaluate`` counted time-limit ends as falls and moved the rough curriculum between commands."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.envs.loco import go2_rough, go2_velocity
from alloy.envs.loco.commands import VelocityCommands
from alloy.envs.loco.go2 import Go2VelocityConfig
from alloy.envs.loco.rough import Go2RoughConfig
from alloy.envs.realism import ActionDelay

ZERO = np.zeros((1, 12), np.float32)


def _capture_reward_commands(env) -> list[np.ndarray]:
    """Wrap the env's rewarder to record the command each reward was computed with."""
    seen, rewarder = [], env.rewarder

    def spy(s):
        seen.append(np.array(s["command"]))
        return rewarder(s)
    env.rewarder = spy
    return seen


@pytest.mark.compile_heavy
def test_the_reward_reads_the_command_the_policy_saw_across_a_resample():
    cfg = Go2VelocityConfig(pushes=None, commands=VelocityCommands(resample_s=0.1))      # a new draw every 5 steps
    env = go2_velocity(num_envs=4, device="cpu", backend="numpy", config=cfg)
    try:
        env.reset(seed=3)
        seen = _capture_reward_commands(env)
        shown = [np.array(env.commands.value)]
        for _ in range(12):
            shown.append(np.array(env.step(np.zeros((4, 12), np.float32))[4]["command"]))
        assert any(not np.array_equal(a, b) for a, b in zip(shown, shown[1:]))        # resamples happened
        for k, cmd in enumerate(seen):                  # step k's reward used the command shown before step k
            assert np.array_equal(cmd, shown[k]), k                                    # was: the new draw
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_the_curriculum_compares_the_walk_with_the_commanded_travel():
    cfg = Go2RoughConfig(pushes=None, episode_s=0.2)                                  # 10-step episodes
    env = go2_rough(num_envs=3, device="cpu", backend="numpy", config=cfg)
    try:
        env.reset(seed=0)
        env.commands.set((0.5, 0.0, 0.0))
        for _ in range(5):
            env.step(np.zeros((3, 12), np.float32))
        env.commands.set((0.0, 0.0, 0.0))                                             # a frozen robot's last draw: 0
        for _ in range(4):
            env.step(np.zeros((3, 12), np.float32))
        travel = np.linalg.norm(env.asked, axis=1)
        assert np.allclose(travel, 0.5 * 5 * float(env.sim.dt), rtol=0.05)             # was: 0 x t = 0, no demotion
        levels = env.levels.copy()
        env.levels = np.maximum(levels, 1)
        before = env.levels.copy()
        env.step(np.zeros((3, 12), np.float32))                                       # the 10th step ends the episode
        assert np.array_equal(env.levels, before - 1)                                 # stood still: down one level
        assert float(np.abs(env.asked).max()) == 0.0                                  # the new episode starts at 0
    finally:
        env.commands.release()
        env.close()


@pytest.mark.compile_heavy
def test_action_delay_holds_the_start_target_and_the_state_shows_what_was_written():
    import dataclasses

    from alloy.envs.loco.env import LocoConfig, LocoEnv
    from alloy.envs.loco.go2 import GO2, ROBOT, go2_scene, go2_velocity_task

    cfg = Go2VelocityConfig(pushes=None)
    task = dataclasses.replace(go2_velocity_task(cfg), layers=[ActionDelay(2)])
    loco = LocoConfig(robot=GO2, name=ROBOT, commands=cfg.commands, pushes=None, action_scale=cfg.action_scale)
    env = LocoEnv(go2_scene(1, 0, cfg.dt), task, loco, worlds=1, device="cpu", backend="numpy")
    try:
        env.reset(seed=0)
        start = np.array(env.state["joints.target"])
        written = []
        for a in (2.0, -2.0, 0.0, 0.0):
            info = env.step(np.full((1, 12), a, np.float32))[4]
            written.append(np.array(info["targets"]))
            assert np.allclose(env.state["joints.target"], written[-1])               # was: the undelayed target
        assert np.allclose(written[0], start) and np.allclose(written[1], start)       # was: step 1's command at once
        assert not np.allclose(written[2], start) and not np.allclose(written[3], written[2])   # +0.5 then -0.5 rad
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_evaluate_counts_falls_not_time_limits_and_holds_the_curriculum():
    pytest.importorskip("rsl_rl")
    torch = pytest.importorskip("torch")
    from alloy.envs.loco.train import evaluate
    from alloy.envs.wrappers import wrap

    def stand(obs):
        return torch.zeros((obs["policy"].shape[0], 12))

    env = go2_velocity(num_envs=2, device="cpu", backend="torch", config=Go2VelocityConfig(pushes=None, episode_s=0.4))
    venv = wrap(env, "rsl_rl")
    try:
        rows = evaluate(venv, stand, seconds=1.0)                                     # 50 steps over 20-step episodes
        assert all(r["worlds_fell"] == 0 for r in rows)                               # was: every truncation a fall
    finally:
        venv.close()
    rough = go2_rough(num_envs=3, device="cpu", backend="torch", config=Go2RoughConfig(pushes=None, episode_s=0.2))
    rv = wrap(rough, "rsl_rl")
    try:
        rough.levels = torch.full_like(rough.levels, 2)
        evaluate(rv, stand, seconds=0.4)
        assert rough.levels.tolist() == [2, 2, 2] and rough.curriculum is True        # was: 2 -> 1 -> 0
    finally:
        rv.close()
