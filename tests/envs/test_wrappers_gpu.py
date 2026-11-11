# SPDX-License-Identifier: Apache-2.0
"""The RL-library wrappers on a GPU: rsl_rl and skrl keep observations, rewards and actions as CUDA tensors, and
each library's train function runs a few iterations there. Skips cleanly without CUDA, PyTorch or the library."""

from __future__ import annotations

import pytest

from alloy.envs import make

pytestmark = [pytest.mark.gpu, pytest.mark.compile_heavy]
WORLDS = 16


@pytest.fixture(scope="module")
def cuda_env():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    env = make("GripperLift-v0", num_envs=WORLDS, device="cuda:0", backend="torch")
    yield env
    env.close()


def test_rsl_rl_tensors_stay_on_the_gpu(cuda_env):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from alloy.envs.wrappers import wrap

    venv = wrap(cuda_env, "rsl_rl")
    obs = venv.get_observations()
    assert obs["policy"].is_cuda and obs.batch_size == torch.Size([WORLDS])
    obs, reward, dones, extras = venv.step(torch.zeros(WORLDS, venv.num_actions, device=venv.device))
    assert obs["policy"].is_cuda and reward.is_cuda and dones.is_cuda and extras["time_outs"].is_cuda


def test_skrl_tensors_stay_on_the_gpu(cuda_env):
    torch = pytest.importorskip("torch")
    pytest.importorskip("skrl")
    from alloy.envs.wrappers import wrap

    venv = wrap(cuda_env, "skrl")
    obs, _ = venv.reset()
    assert obs.is_cuda
    out = venv.step(torch.zeros(WORLDS, 3, device=obs.device))
    assert out[0].is_cuda and out[1].is_cuda and out[1].shape == (WORLDS, 1)


@pytest.mark.parametrize(("lib", "module"), [("rsl_rl", "rsl_rl"), ("sb3", "stable_baselines3"), ("skrl", "skrl")])
def test_train_runs_on_the_gpu(lib, module, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip(module)
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from alloy.envs.wrappers.train import train

    summary = train("GripperLift-v0", lib=lib, num_envs=WORLDS, iterations=3, device="cuda:0",
                    log_dir=str(tmp_path / lib), report_every=3)
    assert summary["samples"] == 3 * 32 * WORLDS and summary["checkpoint"]
