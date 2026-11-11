# SPDX-License-Identifier: Apache-2.0
"""Camera observations: every world renders a 64x64 image each step, next to the state vector.

Run: python examples/envs/03_camera_obs.py [out_dir]       (a GPU makes this fast; CPU works with few worlds)
Writes worlds.png (the first 4 worlds' images side by side) and prints one JSON line.
"""
import json
import sys
from pathlib import Path

import numpy as np

from alloy.camera import write_png
from alloy.envs import make

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_camera_obs")
env = make("GripperLiftCamera-v0", num_envs=4)
obs, info = env.reset(seed=0)
print({k: tuple(v.shape) for k, v in obs.items()})        # {"state": (4, 22), "front": (4, 64, 64, 3)}

action = np.tile([0.0, 1.0, 1.0], (4, 1)).astype(np.float32)   # squeeze ...
for step in range(45):
    if step == 20:
        action[:, 0] = 1.0                                       # ... then lift
    obs, reward, terminated, truncated, info = env.step(env.xp.asarray(action))

images = env.xp.to_numpy(obs["front"])                           # (4, 64, 64, 3) uint8
out.mkdir(parents=True, exist_ok=True)
write_png(out / "worlds.png", np.concatenate(list(images), axis=1))
print(json.dumps({"image": str(out / "worlds.png"), "mean_pixel": round(float(images.mean()), 1),
                  "cube_rise_m": [round(float(v), 3) for v in env.xp.to_numpy(info["terms"]["rise"])]}))
env.close()
