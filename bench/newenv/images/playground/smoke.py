# SPDX-License-Identifier: Apache-2.0
"""Playground 1.0.1 smoke: CartpoleBalance (reaf-sim + MuJoCo Warp), 16 worlds, reset + 100 steps; one JSON line."""
import json
import time

t0 = time.perf_counter()
import torch
import warp as wp
from mujoco_playground.rl import env_registry

t_import = time.perf_counter() - t0
N, STEPS = 16, 100
t0 = time.perf_counter()
env = env_registry.get("CartpoleBalance").env_builder(nworld=N, device="cuda:0")
env.reset()
t_build = time.perf_counter() - t0
spec = env.action_spec()
action = {k: torch.zeros(v.shape, dtype=torch.float32, device="cuda:0") for k, v in spec.items()}  # action_spec() is a dict
env.step(action)
wp.synchronize()
t0 = time.perf_counter()
for _ in range(STEPS):
    out = env.step(action)
wp.synchronize()
wall = time.perf_counter() - t0
print(json.dumps({"arm": "playground", "envs": N, "steps": STEPS, "action_shape": {k: list(v.shape) for k, v in spec.items()},
                  "steps_per_s": round(STEPS / wall, 1), "env_steps_per_s": round(N * STEPS / wall),
                  "t_import_s": round(t_import, 1), "t_build_s": round(t_build, 1)}))
