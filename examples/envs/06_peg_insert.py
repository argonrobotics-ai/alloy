# SPDX-License-Identifier: Apache-2.0
"""PegInsert-v0: the gripper holds a 10 mm square peg above a block; insert it into a 10.5 mm hole (scripted policy).

The policy aligns the peg's tip (not its centre: the peg can tilt a few degrees in the fingers) over the hole, then
lowers it. Precise fit: rigid contacts step at 0.25 ms, so use a GPU. Run: python examples/envs/06_peg_insert.py [n]
"""
import json
import sys

import numpy as np

from alloy.envs import make
from alloy.envs.library_tasks import BLOCK, peg_tip

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
env = make("PegInsert-v0", num_envs=num_envs)
xp = env.xp
env.reset(seed=0)
for _ in range(120):
    tip = xp.to_numpy(peg_tip(env.state))
    a = np.zeros((num_envs, 4), np.float32)
    a[:, 3] = 1.0                                                   # keep squeezing the peg
    err = -tip[:, :2]                                               # the hole is at x = y = 0
    a[:, :2] = np.clip(err / 0.004, -1, 1)
    above = tip[:, 2] - BLOCK[2]
    aligned = np.abs(err).max(1) < 0.0003
    a[:, 2] = np.where(aligned | (above > 0.006), -0.5, np.where(above < 0.001, 0.2, 0.0))
    obs, reward, terminated, truncated, info = env.step(xp.asarray(a))
print(json.dumps({"worlds": num_envs, "inserted": float(xp.to_numpy(info["success"]).mean()),
                  "mean_depth_mm": round(1e3 * float(xp.to_numpy(info["terms"]["depth"]).mean()), 1)}))
env.close()
