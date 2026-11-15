# SPDX-License-Identifier: Apache-2.0
"""A world that freezes in a scene whose worlds reset together is counted and penalised once, and every RL wrapper
ends its episode once, as terminated, at the common end (audit T1-014); rsl_rl's staggered episode lengths leave such
a scene's one episode clock alone (audit T2-021).

Before: such a world added 1 to ``env.invalid_episodes`` and earned ``invalid_penalty`` at every step until
``max_steps`` (4 counts and -8 for one freeze 4 steps before the end); skrl reported it terminated at every step and
SB3 reported a done with no episode info at every step; and rsl_rl's ``init_at_random_ep_len`` staggered the clocks
into resets of single worlds, which such a scene refuses (EnvError). A scene resets together when its engine state
has arrays with no per-world axis (sand and pastes); the env reads that once at build as ``env.together``, set here
on a quick rigid scene so the test runs in seconds.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.envs import VectorEnv
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

W, STEPS, FREEZE = 3, 6, 2                  # world 2 freezes on step 2 (0-based) of a 6-step episode


def _env(backend: str = "numpy", **task_kw) -> VectorEnv:
    task = dataclasses.replace(gripper_lift_task(), max_steps=STEPS, invalid_penalty=-2.0, **task_kw)
    env = VectorEnv(gripper_lift_scene(W), task, worlds=W, device="cpu", backend=backend, seed=0)
    env.together = True                     # what a scene of sand or paste reads at build (``per_world_reset``)
    return env


def _throw_cube(env: VectorEnv, world: int = 2) -> None:
    """World ``world``'s cube at 500 m/s: the engine freezes that world (INVALID) on the next step."""
    eng = env.sim.engine
    qd = eng.state.joint_qd.numpy()
    qd[env.layout.world_of["qd"].tolist().index(world) + 2] = 500.0
    eng.state.joint_qd.assign(qd)


def _episode(step, env: VectorEnv, zeros) -> list[tuple]:
    """One whole episode, freezing world 2 before step FREEZE: what ``step`` returned at every step."""
    out = []
    for k in range(STEPS):
        if k == FREEZE:
            _throw_cube(env)
        out.append(step(zeros))
    return out


@pytest.mark.compile_heavy
@pytest.mark.parametrize("on_invalid", ["truncate", "terminate"])
def test_a_frozen_world_is_counted_and_penalised_once(on_invalid):
    env = _env(on_invalid=on_invalid)
    env.reset(seed=0)
    steps = _episode(env.step, env, np.zeros((W, 3), np.float32))
    rewards = [float(s[1][2]) for s in steps]
    assert rewards[FREEZE] == -2.0 and rewards[FREEZE + 1:] == [0.0] * (STEPS - FREEZE - 1)
    assert env.invalid_episodes.tolist() == [0, 0, 1]
    for _obs, _rew, term, trunc, _info in steps[:-1]:                 # nothing ends before the common end
        assert not term.any() and not trunc.any()
    _obs, _rew, term, trunc, info = steps[-1]
    frozen_ends = [False, False, on_invalid == "terminate"]
    assert term.tolist() == frozen_ends and trunc.tolist() == [not t for t in frozen_ends]
    assert info["episode"]["invalid"].tolist() == [False, False, True]
    _obs, rew, *_ = env.step(np.zeros((W, 3), np.float32))             # the next episode: a fresh record
    assert env.invalid_episodes.tolist() == [0, 0, 1] and float(rew[2]) != -2.0
    env.close()


@pytest.mark.compile_heavy
def test_sb3_ends_a_frozen_world_once_with_its_episode_info():
    pytest.importorskip("stable_baselines3")
    from alloy.envs.wrappers import wrap

    venv = wrap(_env(), "sb3")
    venv.reset()
    steps = _episode(lambda a: venv.step(a), venv.env, np.zeros((W, 3), np.float32))
    for _obs, _rew, dones, infos in steps[:-1]:
        assert not dones.any() and infos == [{}] * W
    _obs, _rew, dones, infos = steps[-1]
    assert dones.tolist() == [True] * W and all("terminal_observation" in i for i in infos)
    assert [i["TimeLimit.truncated"] for i in infos] == [True, True, False]   # the frozen one: no bootstrap
    venv.close()


@pytest.mark.compile_heavy
def test_skrl_ends_a_frozen_world_once_as_terminated():
    pytest.importorskip("skrl")
    from alloy.envs.wrappers import wrap

    env = wrap(_env(backend="torch"), "skrl")
    env.reset()
    steps = _episode(env.step, env._env, env.zero_actions())
    for _obs, _rew, term, trunc, _info in steps[:-1]:
        assert not term.any() and not trunc.any()
    _obs, _rew, term, trunc, _info = steps[-1]
    assert term.flatten().tolist() == [False, False, True] and trunc.flatten().tolist() == [True, True, False]
    env.close()


@pytest.mark.compile_heavy
def test_rsl_rl_ends_a_frozen_world_once_without_a_time_out():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from alloy.envs.wrappers import wrap

    venv = wrap(_env(backend="torch"), "rsl_rl")
    steps = _episode(venv.step, venv.env, torch.zeros((W, venv.num_actions)))
    for _obs, _rew, dones, _extras in steps[:-1]:
        assert not dones.any()
    _obs, _rew, dones, extras = steps[-1]
    assert dones.tolist() == [1] * W and extras["time_outs"].tolist() == [True, True, False]
    venv.close()


@pytest.mark.compile_heavy
def test_rsl_rl_random_episode_lengths_keep_the_one_clock(caplog):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from alloy.envs.wrappers import wrap

    venv = wrap(_env(backend="torch"), "rsl_rl")
    venv.episode_length_buf = torch.tensor([4, 0, 3, 1][:W])           # what init_at_random_ep_len writes
    assert venv.episode_length_buf.tolist() == [0] * W
    assert "keep one clock" in caplog.text
    dones = [venv.step(torch.zeros((W, venv.num_actions)))[2].tolist() for _ in range(STEPS)]
    assert dones == [[0] * W] * (STEPS - 1) + [[1] * W]                 # every world ends at max_steps, together
    venv.close()
