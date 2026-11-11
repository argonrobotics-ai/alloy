# SPDX-License-Identifier: Apache-2.0
"""Pour-v0: water pours from a spout (Scene.add_pour, 100 ml at 100 ml/s); move a bowl under the stream to catch it.

Liquids reset all worlds together (``env.together``): every episode ends at max_steps. A GPU is strongly advised.
Run: python examples/envs/08_pour.py [num_envs]
"""
import json
import sys

import numpy as np

from alloy.envs import make

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
env = make("Pour-v0", num_envs=num_envs)
xp = env.xp
env.reset(seed=0)
for _ in range(env.task.max_steps):
    bowl = xp.to_numpy(env.state["tray/bowl.pos"])
    action = np.clip(-bowl[:, :2] / 0.01, -1, 1).astype(np.float32)    # the spout is above x = y = 0
    obs, reward, terminated, truncated, info = env.step(xp.asarray(action))
print(json.dumps({"worlds": num_envs, "caught_fraction": round(float(xp.to_numpy(info["terms"]["in_bowl"]).mean()), 3),
                  "success": float(xp.to_numpy(info["success"]).mean())}))
env.close()
