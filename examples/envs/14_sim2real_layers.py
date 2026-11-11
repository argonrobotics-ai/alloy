# SPDX-License-Identifier: Apache-2.0
"""Sim-to-real layers on GripperLift: observation noise, observation delay, observation history and action delay.

Run: python examples/envs/14_sim2real_layers.py [num_envs] [device]     (default 64 worlds; GPU if there is one)
e.g. ``python examples/envs/14_sim2real_layers.py 4 cpu`` (under a minute after the first kernel build).
Prints one JSON line: the observation shape of each group (the policy's is 3 steps wide, the critic's stays clean),
each world's drawn action delay, the noise std measured on one column (set to 0.02), and the first column labels.
The layers change what the policy sees and what the writer receives, never the simulation:
docs/guide/rl.md, alloy/envs/realism.py.
"""
import dataclasses
import json
import sys

import numpy as np

from alloy.envs import VectorEnv
from alloy.envs.realism import ActionDelay, ObsDelay, ObsHistory, ObsNoise, obs_names
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

GROUPS = {"policy": ["gripper", "action"], "critic": ["gripper", "cube", "action"]}
COLUMN, STD = "gripper/lift.q", 0.02


def np_(x):
    """Any array the env returns (NumPy or torch) as NumPy."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def build(num_envs, device):
    layers = [ObsNoise({COLUMN: STD, "action": 0.05}), ObsDelay(1), ObsHistory(3), ActionDelay((0, 2))]
    task = dataclasses.replace(gripper_lift_task(max_steps=50), obs=GROUPS, layers=layers)
    return VectorEnv(gripper_lift_scene(num_envs), task, worlds=num_envs, device=device)


def measure_noise(env, steps=50):
    """Std of (policy - critic) on the noisy column: both read the same joint, the critic has no noise. The
    policy sees it one step late, so compare against the critic's previous step and keep the first 49 steps."""
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    names, critic_names = env.obs_names(), env.obs_names("critic")
    col, ccol = names.index(COLUMN), critic_names.index(COLUMN)
    width, diffs, prev = len(names), [], np_(obs["critic"])[:, ccol]
    for _ in range(steps):
        a = rng.uniform(-1, 1, (env.num_envs, env.writer.size)).astype(np.float32)
        obs, _, _, _, info = env.step(a)
        newest = np_(obs["policy"])[:, 2 * width + col]
        if "final_obs" not in info:
            diffs.append(newest - prev)
        prev = np_(obs["critic"])[:, ccol]
    return float(np.std(np.concatenate(diffs)))


def main(num_envs=64, device=None):
    env = build(num_envs, device)
    obs, _ = env.reset(seed=0)
    delays = np_(env.layers[3].k).tolist()
    measured = measure_noise(env)
    print(json.dumps({"obs_shapes": {g: list(np_(v).shape) for g, v in obs.items()}, "action_delay": delays,
                      "noise_std_set": STD, "noise_std_measured": round(measured, 4),
                      "obs_names_head": obs_names(env)[:4]}))
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64, sys.argv[2] if len(sys.argv) > 2 else None)
