# SPDX-License-Identifier: Apache-2.0
"""PyTorch backend (zero-copy views of the engine's arrays), Gymnasium integration and the GPU path.

Skips cleanly where PyTorch / Gymnasium / CUDA are missing."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make


@pytest.fixture(scope="module")
def torch_env():
    pytest.importorskip("torch")
    e = make("GripperLift-v0", num_envs=2, device="cpu", backend="torch")
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_torch_tensors_share_engine_memory(torch_env):
    import torch

    env = torch_env
    obs, _ = env.reset()
    assert isinstance(obs, torch.Tensor) and obs.dtype == torch.float32 and obs.shape == (2, 22)
    bq = env.sim.engine.state.body_q
    view = env.xp.view(bq)
    assert view.data_ptr() == bq.ptr                                         # zero-copy
    obs, r, term, trunc, info = env.step(torch.zeros((2, 3)))
    assert isinstance(r, torch.Tensor) and term.dtype == torch.bool and info["success"].dtype == torch.bool
    obs, _ = env.reset(options={"reset_mask": torch.tensor([True, False])})
    assert env.state["steps"].tolist() == [0, 1]


@pytest.mark.compile_heavy
def test_gymnasium_vector_env_and_make_vec():
    gym = pytest.importorskip("gymnasium")
    from alloy.envs import register_gymnasium

    ids = register_gymnasium()
    assert "alloy/GripperLift-v0" in ids and "alloy/GripperLift-v0" in gym.registry
    env = gym.make_vec("alloy/GripperLift-v0", num_envs=2, vectorization_mode="vector_entry_point", device="cpu",
                       backend="numpy")
    try:
        assert isinstance(env.unwrapped, gym.vector.VectorEnv) and env.num_envs == 2
        obs, _ = env.reset(seed=1)
        assert obs.shape == (2, 22) and env.observation_space.contains(obs)
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert r.shape == (2,) and env.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    finally:
        env.close()


@pytest.mark.gpu
def test_gpu_torch_state_and_camera():
    torch = pytest.importorskip("torch")
    env = make("GripperLiftCamera-v0", num_envs=4, device="cuda:0")
    try:
        obs, _ = env.reset(seed=0)
        assert obs["state"].is_cuda and obs["state"].shape == (4, 22)
        assert obs["front"].is_cuda and obs["front"].dtype == torch.uint8 and obs["front"].shape == (4, 64, 64, 3)
        a = torch.zeros((4, 3), device="cuda")
        a[:, 1:] = 1.0
        for _ in range(5):
            obs, r, term, trunc, info = env.step(a)
        assert r.is_cuda and float(obs["front"].float().std()) > 1.0             # an image, not a blank
        env.reset(options={"reset_mask": torch.tensor([False, True, False, False], device="cuda")})
        assert env.state["steps"].tolist() == [5, 0, 5, 5]
        assert np.isfinite(env.xp.to_numpy(obs["state"])).all()
    finally:
        env.close()


@pytest.mark.gpu
def test_gpu_panda_reach_steps():
    torch = pytest.importorskip("torch")
    env = make("PandaReach-v0", num_envs=8, device="cuda:0")
    try:
        obs, _ = env.reset(seed=0)
        assert env.writer.kind == "ctrl" and env.single_action_space.shape == (7,) and obs.shape[0] == 8
        for _ in range(5):
            obs, r, term, trunc, info = env.step(torch.zeros((8, 7), device="cuda"))
        assert not info["invalid"].any() and torch.isfinite(obs).all()
        assert float(info["terms"]["distance"].max()) < 0.6                     # targets are within reach
    finally:
        env.close()
