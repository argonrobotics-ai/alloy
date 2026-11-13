# SPDX-License-Identifier: Apache-2.0
"""PickPlace-v0: pick a YCB object off the table and drop it in a bin, with a scripted policy (no learning).

The policy reads named state (``env.state["obj.com"]``, the gripper palm) and sends delta actions for the gripper's
x, y, z and fingers. Run: python examples/envs/05_pick_place.py [num_envs] [object card]
(object cards: alloy.envs.library_tasks.PICK_OBJECTS; the first build fetches the object's files)
Prints ``in_bin``, the share of worlds whose object ends in the bin. The script never checks that the grasp
held and does not succeed in every world. On the CPU: 64 of 64 worlds (CPU runs repeat bit for bit). On one L4:
62 and 64 of 64 worlds in two runs and 254 of 256 in one; runs with the same seed differ there, because contacts on
a GPU do not repeat bit for bit. Before the gripper's force cap (0.1.0): 4 of 4, 8 of 8 and 64 of 64 on the CPU;
on one L4 4 of 4, 8 of 8, 60 to 62 of 64 in six runs and 249 and 250 of 256.
"""
import json
import sys

import numpy as np

from alloy.envs import make

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
obj = sys.argv[2] if len(sys.argv) > 2 else "ycb_077_rubiks_cube"
env = make("PickPlace-v0", num_envs=num_envs, obj=obj)
xp = env.xp
env.reset(seed=0)

BIN = np.array([0.18, 0.0])
# "carry" is 45 steps of at most 1 cm: the farthest start is 0.33 m from the bin (30 steps let go short of it)
PHASES = [("over", 25), ("down", 25), ("grip", 15), ("up", 20), ("carry", 45), ("drop", 20)]


def policy(phase: str) -> np.ndarray:
    s = env.state
    palm, o = xp.to_numpy(s["gripper/palm.pos"]), xp.to_numpy(s["obj.com"])
    a = np.zeros((num_envs, 4), np.float32)
    a[:, 3] = 1.0 if phase in ("grip", "up", "carry") else -1.0                  # close / open the fingers
    if phase in ("over", "down"):
        a[:, :2] = np.clip((o[:, :2] - palm[:, :2]) / 0.01, -1, 1)
    if phase == "down":
        a[:, 2] = np.clip((np.maximum(o[:, 2] + 0.04, 0.075) - palm[:, 2]) / 0.01, -1, 1)
    if phase == "up":
        a[:, 2] = 1.0
    if phase == "carry":
        a[:, :2] = np.clip((BIN - palm[:, :2]) / 0.01, -1, 1)
    return a


for phase, steps in PHASES:
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(xp.asarray(policy(phase)))
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(xp.asarray(np.zeros((num_envs, 4), np.float32)))
print(json.dumps({"object": obj, "worlds": num_envs, "in_bin": float(xp.to_numpy(info["success"]).mean())}))
env.close()
