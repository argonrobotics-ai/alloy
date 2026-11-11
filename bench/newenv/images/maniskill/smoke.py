# SPDX-License-Identifier: Apache-2.0
"""ManiSkill smoke: PickCube-v1, 16 GPU-parallel envs, state obs, 100 steps (default render backend: needs the host's libnvidia-gl for Vulkan, see README); one JSON line."""
import json
import time

t0 = time.perf_counter()
import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers the tasks)
import torch

t_import = time.perf_counter() - t0
N, STEPS = 16, 100
t0 = time.perf_counter()
env = gym.make("PickCube-v1", num_envs=N, obs_mode="state", sim_backend="gpu")
obs, _ = env.reset(seed=0)
t_build = time.perf_counter() - t0
act = torch.zeros((N,) + env.action_space.shape[1:], device=obs.device)
env.step(act)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(STEPS):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
torch.cuda.synchronize()
wall = time.perf_counter() - t0
assert torch.isfinite(obs).all()
print(json.dumps({"arm": "maniskill", "envs": N, "steps": STEPS, "device": str(obs.device),
                  "steps_per_s": round(STEPS / wall, 1), "env_steps_per_s": round(N * STEPS / wall),
                  "t_import_s": round(t_import, 1), "t_build_s": round(t_build, 1)}))
