# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T1-009: a seeded reset of the Go2 envs repeats a fresh env's start; other resets keep training state.

``go2_velocity``: ``reset(seed=S)`` during a push kept the held base wrench (about 100 N at t = 0) and the push
timer. ``go2_rough``: ``reset(seed=S)`` advanced the level curriculum from the last episode, so robots spawned on other
tiles (6 m away). Now a seeded reset restarts the timer, lifts the push and restores the start levels; an autoreset
or an unseeded reset keeps the timer and the curriculum's levels. CPU, numpy backend."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.envs.loco import go2_rough, go2_velocity
from alloy.envs.loco.go2 import Go2VelocityConfig
from alloy.envs.loco.pushes import Pushes
from alloy.envs.loco.rough import Go2RoughConfig

QUICK_PUSHES = Pushes(every_s=0.1, duration_s=0.04)          # a push every 5 control steps, held for 2


def _rollout(env, n: int) -> list[np.ndarray]:
    return [env.step(np.zeros((env.num_envs, 12), np.float32))[0]["policy"].copy() for _ in range(n)]


@pytest.mark.compile_heavy
def test_a_seeded_reset_during_a_push_repeats_the_first_episode():
    env = go2_velocity(num_envs=2, device="cpu", backend="numpy", config=Go2VelocityConfig(pushes=QUICK_PUSHES))
    try:
        first, _ = env.reset(seed=42)
        run1 = _rollout(env, 7)
        assert env.pusher.active                                      # step 6 started a push; step 7 holds it
        again, _ = env.reset(seed=42)
        assert float(np.abs(env.state["push.force"]).max()) == 0.0    # was: the held push at t = 0
        assert np.array_equal(again["critic"], first["critic"])
        assert all(np.array_equal(a, b) for a, b in zip(_rollout(env, 7), run1))
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_rough_seeded_reset_restores_start_levels_and_other_resets_keep_the_curriculum():
    cfg = Go2RoughConfig(pushes=None, episode_s=0.1)                 # 5-step episodes: autoresets come quickly
    env = go2_rough(num_envs=6, device="cpu", backend="numpy", config=cfg)
    try:
        env.reset(seed=42)
        start, pos = env.levels.copy(), env.state["go2/base.pos"].copy()
        top = env.terrain.levels - 1
        env.levels = np.full_like(env.levels, top)                    # as if the curriculum had promoted every world
        env.reset()                                                   # unseeded: the curriculum's levels stand
        assert int(env.levels.min()) >= top - 1                       # at most one level down, never back to start
        env.levels = np.full_like(env.levels, top)
        for _ in range(5):                                            # the 5th step ends every episode: autoreset
            env.step(np.zeros((6, 12), np.float32))
        assert int(env.levels.min()) >= top - 1
        env.reset(seed=42)                                            # seeded: the fresh env's start again
        assert np.array_equal(env.levels, start)                      # was: levels moved by the last episode
        assert float(np.abs(env.state["go2/base.pos"] - pos).max()) < 1e-6   # was: spawned 6 m away
    finally:
        env.close()
