# SPDX-License-Identifier: Apache-2.0
"""A ready-made Gym environment with random actions: every world steps at once, finished worlds restart by themselves.

Run: python examples/envs/01_random_actions.py [num_envs] [steps]     (GPU if there is one; try 1024 worlds there)
Prints one JSON line: speed, episodes finished, mean return.
"""
import json
import sys
import time

from alloy.envs import make

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100

env = make("GripperLift-v0", num_envs=num_envs)            # torch tensors when PyTorch is installed, else NumPy
obs, info = env.reset(seed=0)
print("observation", env.single_observation_space, "first columns", env.obs_names()[:4])
print("actions", env.single_action_space, env.action_names())

episodes, returns, t0 = 0, 0.0, time.perf_counter()
for _ in range(steps):
    actions = env.action_space.sample()                     # (num_envs, 3) in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(actions)
    if "_episode" in info:                                  # these worlds just finished and were reset
        done = env.xp.to_numpy(info["_episode"])
        episodes += int(done.sum())
        returns += float(env.xp.to_numpy(info["episode"]["r"])[done].sum())
wall = time.perf_counter() - t0
print(json.dumps({"device": env.device, "backend": env.xp.name, "num_envs": num_envs, "steps": steps,
                  "env_steps_per_s": round(steps / wall, 1), "world_steps_per_s": round(steps * num_envs / wall),
                  "episodes": episodes, "mean_return": round(returns / max(episodes, 1), 3)}))
env.close()
