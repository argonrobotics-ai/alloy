# SPDX-License-Identifier: Apache-2.0
"""TRAIN-FIX item 4 (training audit TA-3 F2, TA-4 F2, TA-8 F2 and F8): a frozen (INVALID) world gives a policy
nothing, the wrappers never bootstrap its value, the count of frozen episodes survives the autoreset, the env's build
seed holds through the rsl_rl wrapper, and ObsNoise works on a flat observation list. CPU, the GripperLift scene."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.envs import VectorEnv
from alloy.envs import checks as C
from alloy.envs.realism import ObsNoise
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

W = 3
SQUEEZE = np.array([0.0, 1.0, 1.0], np.float32)


ALWAYS = C.height("cube") > -10.0                 # a success check that holds in every world, frozen or not


def _env(seed: int = 0, backend: str = "numpy", autoreset: str = "same_step", **task_kw) -> VectorEnv:
    task = dataclasses.replace(gripper_lift_task(), **task_kw)
    return VectorEnv(gripper_lift_scene(W), task, worlds=W, device="cpu", backend=backend, seed=seed,
                     autoreset=autoreset)


def _throw_cube(env: VectorEnv, world: int) -> None:
    """World ``world``'s cube at 500 m/s: the engine freezes that world (INVALID) on the next step."""
    eng = env.sim.engine
    qd = eng.state.joint_qd.numpy()
    qd[env.layout.world_of["qd"].tolist().index(world) + 2] = 500.0
    eng.state.joint_qd.assign(qd)


@pytest.fixture(scope="module")
def env():
    e = _env(invalid_penalty=-2.0, success=ALWAYS, end_on_success=False)
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_a_frozen_world_earns_only_its_penalty_and_shows_zeros(env):
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.tile(SQUEEZE, (W, 1)))
    before = env.invalid_episodes.copy()
    _throw_cube(env, 2)
    obs, reward, term, trunc, info = env.step(np.tile(SQUEEZE, (W, 1)))
    assert info["invalid"].tolist() == [False, False, True]
    assert reward[2] == -2.0 and reward[0] != -2.0                      # the penalty alone, no task reward
    assert all(float(v[2]) == 0.0 for v in info["terms"].values())
    assert info["success"].tolist() == [True, True, False]            # the check holds; a frozen world never succeeds
    assert not info["episode"]["success"][2]
    assert info["episode"]["invalid"].tolist() == [False, False, True]
    assert np.all(info["final_obs"][2] == 0.0) and np.any(info["final_obs"][0] != 0.0)
    assert np.any(obs[2] != 0.0)                                         # the reset world's first observation
    assert (env.invalid_episodes - before).tolist() == [0, 0, 1]
    env.step(np.tile(SQUEEZE, (W, 1)))
    assert (env.invalid_episodes - before).tolist() == [0, 0, 1]         # kept across the autoreset


@pytest.mark.compile_heavy
def test_rsl_rl_never_bootstraps_a_frozen_world():
    pytest.importorskip("rsl_rl")
    from alloy.envs.wrappers import wrap

    venv = wrap(_env(backend="torch"), "rsl_rl")
    venv.step(_zeros(venv))
    _throw_cube(venv.env, 1)
    _obs, _rew, dones, extras = venv.step(_zeros(venv))
    assert dones.tolist() == [0, 1, 0] and extras["time_outs"].tolist() == [False, False, False]
    assert venv.tally.take_invalid() == 1 and venv.tally.take_invalid() == 0
    venv.close()


@pytest.mark.compile_heavy
def test_the_build_seed_holds_through_the_rsl_rl_wrapper():
    pytest.importorskip("rsl_rl")
    from alloy.envs.wrappers import wrap

    starts = []
    for seed in (1, 2, 1):
        venv = wrap(_env(seed=seed, backend="torch"), "rsl_rl")
        starts.append(venv.get_observations()["policy"].clone())
        venv.close()
    assert not bool((starts[0] == starts[1]).all())                     # seed 2 is not seed 1 (was: both seed 0)
    assert bool((starts[0] == starts[2]).all())                         # the same seed repeats on the CPU


@pytest.mark.compile_heavy
def test_obs_noise_on_a_flat_observation_list():
    plain, noisy = _env(), _env(layers=[ObsNoise(0.01)])
    a, _ = plain.reset(seed=0)
    b, _ = noisy.reset(seed=0)
    assert a.shape == b.shape == (W, len(plain.obs_names()))
    assert 0.0 < float(np.abs(b - a).max()) < 0.1
    plain.close()
    noisy.close()


@pytest.mark.compile_heavy
def test_next_step_autoreset_counts_a_frozen_episode_once():
    env = _env(autoreset="next_step")
    env.reset(seed=0)
    env.step(np.tile(SQUEEZE, (W, 1)))
    _throw_cube(env, 0)
    for _ in range(3):                                                  # frozen, then restarted on the next call
        env.step(np.tile(SQUEEZE, (W, 1)))
    assert env.invalid_episodes.tolist() == [1, 0, 0]
    env.close()


@pytest.mark.compile_heavy
def test_sb3_never_bootstraps_a_frozen_world():
    pytest.importorskip("stable_baselines3")
    from alloy.envs.wrappers import wrap

    venv = wrap(_env(), "sb3")
    venv.reset()
    venv.step(np.zeros((W, 3), np.float32))
    _throw_cube(venv.env, 1)
    _obs, _rew, dones, infos = venv.step(np.zeros((W, 3), np.float32))
    assert dones.tolist() == [False, True, False]
    assert infos[1]["TimeLimit.truncated"] is False and "terminal_observation" in infos[1]
    venv.close()


@pytest.mark.compile_heavy
def test_skrl_never_bootstraps_a_frozen_world():
    pytest.importorskip("skrl")
    from alloy.envs.wrappers import wrap

    env = wrap(_env(backend="torch"), "skrl")
    env.reset()
    env.step(env.zero_actions())
    _throw_cube(env._env, 2)
    _obs, _rew, term, trunc, _info = env.step(env.zero_actions())
    assert term.flatten().tolist() == [False, False, True] and trunc.flatten().tolist() == [False, False, False]
    env.close()


def _zeros(venv):
    import torch

    return torch.zeros((venv.num_envs, venv.num_actions), device=venv.device)
