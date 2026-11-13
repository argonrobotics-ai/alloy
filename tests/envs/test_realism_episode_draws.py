# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T1-012: sim-to-real layers draw afresh at every episode, also when every world ends together.

A task with no randomize rows never advances the env's own stream, and the layers used to reseed from it at every
reset of all worlds: each simultaneous truncation replayed the same per-world delays and the same "white" noise. Now
the layers reseed only after ``env.reset(seed=...)``, so a seeded reset still repeats. CPU, a gantry gripper."""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs import checks as C
from alloy.envs.realism import ActionDelay, ObsDelay, ObsNoise
from alloy.envs.robots import gantry_actions, gantry_gripper

W, STEPS = 4, 4


def _env() -> tuple[VectorEnv, ObsDelay, ActionDelay]:
    scene = alloy.Scene(worlds=W, dt=0.02)
    scene.add_ground()
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.025), mass=0.1)
    scene.add_robot("gripper", mjcf=gantry_gripper(), pos=(0.0, 0.0, 0.2))
    obs_delay, act_delay = ObsDelay((0, 3)), ActionDelay((0, 3))
    task = Task(obs=["gripper", "cube"], actions=gantry_actions("gripper", yaw=False),
                reward={"lift": (1.0, C.lifted("cube", by=0.05))}, success=C.lifted("cube", by=0.05),
                end_on_success=False, max_steps=STEPS, layers=[ObsNoise(0.05), obs_delay, act_delay])
    return VectorEnv(scene, task, worlds=W, device="cpu", backend="numpy"), obs_delay, act_delay


def _episode(env: VectorEnv, first_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero actions until every world truncates together: (observations, whether all truncated at the end)."""
    traj, obs, trunc = [first_obs.copy()], first_obs, None
    for _ in range(STEPS):
        obs, _, _, trunc, _ = env.step(np.zeros((W, env.single_action_space.shape[0]), np.float32))
        traj.append(obs.copy())
    return np.stack(traj), np.asarray(trunc)


@pytest.mark.compile_heavy
def test_layers_draw_new_delays_and_noise_each_episode_and_repeat_on_a_seeded_reset():
    env, obs_delay, act_delay = _env()
    try:
        obs, _ = env.reset(seed=0)
        delays, trajs = [(obs_delay.k.tolist(), act_delay.k.tolist())], []
        for _ in range(3):
            traj, trunc = _episode(env, obs)
            assert trunc.all()                                       # every world ended together (autoreset)
            trajs.append(traj)
            obs = traj[-1]                                           # the reset worlds' first observation
            delays.append((obs_delay.k.tolist(), act_delay.k.tolist()))
        assert len({str(d) for d in delays}) > 1                     # was: one draw replayed at every reset
        assert float(np.abs(trajs[1][1:] - trajs[0][1:]).max()) > 1e-3   # was: 0.0, the same noise each episode
        assert float(np.abs(trajs[2][1:] - trajs[1][1:]).max()) > 1e-3

        again, _ = env.reset(seed=0)                                 # a seeded reset still repeats the first episode
        assert (obs_delay.k.tolist(), act_delay.k.tolist()) == delays[0]
        assert np.array_equal(_episode(env, again)[0], trajs[0])
    finally:
        env.close()
