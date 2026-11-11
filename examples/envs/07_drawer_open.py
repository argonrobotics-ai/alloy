# SPDX-License-Identifier: Apache-2.0
"""DrawerOpen-v0: grasp a RoboCasa cabinet drawer's handle and pull it open (scripted policy).

The gripper faces the cabinet: its "z" slide moves toward / away from the cabinet and "y" up / down; the fingers
close vertically on the horizontal handle bar. Pull slowly (a fast pull slips off the bar).
Run: python examples/envs/07_drawer_open.py [num_envs]
"""
import json
import sys

import numpy as np

from alloy.envs import make
from alloy.envs.library_tasks import grasp_point, grasp_target

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
env = make("DrawerOpen-v0", num_envs=num_envs)
xp = env.xp
env.reset(seed=0)
for k in range(110):
    s = env.state
    err = xp.to_numpy(grasp_target(s) - grasp_point(s))           # world x, y (toward the cabinet), z
    a = np.zeros((num_envs, 4), np.float32)
    if k < 50:                                                      # reach (fingers around the bar), then close
        a[:, 0], a[:, 1], a[:, 2] = (np.clip(e / 0.005, -1, 1) for e in (err[:, 0], err[:, 2], -err[:, 1]))
        a[:, 3] = -1.0 if k < 35 else 1.0
    else:                                                           # pull back slowly, keep the grip
        a[:, 2], a[:, 3] = 0.3, 1.0
    obs, reward, terminated, truncated, info = env.step(xp.asarray(a))
print(json.dumps({"worlds": num_envs, "opened": float(xp.to_numpy(info["success"]).mean()),
                  "mean_opening_m": round(float(xp.to_numpy(info["terms"]["open"]).mean()), 3)}))
env.close()
