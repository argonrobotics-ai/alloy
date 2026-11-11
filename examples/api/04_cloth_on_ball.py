# SPDX-License-Identifier: Apache-2.0
"""A cotton cloth falls onto a static ball and drapes over it; saves vertex frames and triangles.

Run: python examples/api/04_cloth_on_ball.py [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_cloth")
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_sphere("ball", 0.2, pos=(0, 0, 0.4), static=True)
scene.add_cloth("cloth", size=(0.8, 0.8), pos=(0, 0, 0.7), resolution=25, mass=0.15)   # 25 x 25 vertices
sim = alloy.build(scene)
print("routes:", {k: v["route"] for k, v in sim.routes().items()})
traj = sim.run(2.5, record=["cloth"], every=5).world(0)          # (frames, vertices, 3)
traj.save(out / "traj.npz", faces=sim.faces("cloth"))
v = traj["cloth.points"][-1]
dist = np.linalg.norm(v - np.array([0, 0, 0.4]), axis=1) - 0.2
print(json.dumps({"vertices": int(v.shape[0]), "top_z": float(v[:, 2].max()), "min_ball_gap_m": float(dist.min()),
                  "below_equator": float((v[:, 2] < 0.4).mean()), "health_ok": sim.check()["ok"]}))
