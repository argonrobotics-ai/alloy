# SPDX-License-Identifier: Apache-2.0
"""A tomato and a peach (soft solids, sourced whole-fruit values) rest side by side on the ground.

Run: python examples/soft/soft_food_fruit.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("tomato", "fruit_tomato", pos=(0.0, 0, 0.033))     # 62 mm, about 0.11 kg
scene.add_object("peach", "fruit_peach", pos=(0.1, 0, 0.031))       # 58 mm, about 0.10 kg
sim = alloy.build(scene)
traj = sim.run(0.5, record=["tomato", "peach"], every=10).world(0)
print(json.dumps({"routes": {k: v["route"] for k, v in sim.routes().items()},
                  "tomato_lowest_z_m": float(traj["tomato.points"][-1][:, 2].min()),
                  "peach_lowest_z_m": float(traj["peach.points"][-1][:, 2].min()), "health_ok": sim.check()["ok"]}))
