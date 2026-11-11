# SPDX-License-Identifier: Apache-2.0
"""Suture practice pads: a soft silicone fat layer and a stiffer silicone skin layer, side by side.

Two soft solids do not collide with each other yet, so keep them apart.

Run: python examples/soft/soft_food_tissue_pad.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("fat", "tissue_pad_fat", pos=(0, 0, 0.0055))       # 100 x 80 x 10 mm
scene.add_object("skin", "tissue_pad_skin", pos=(0.12, 0, 0.0025))  # 100 x 80 x 4 mm
sim = alloy.build(scene)
traj = sim.run(0.5, record=["skin"], every=10).world(0)
print(json.dumps({"routes": {k: v["route"] for k, v in sim.routes().items()},
                  "skin_top_z_m": float(traj["skin.points"][-1][:, 2].max()), "health_ok": sim.check()["ok"]}))
