# SPDX-License-Identifier: Apache-2.0
"""A cucumber (soft solid, sourced flesh values) lies on the ground; prints its route, lowest point and health.

Run: python examples/soft/soft_food_cucumber.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("cucumber", "cucumber_slicing", pos=(0, 0, 0.025))   # 210 x 44.5 mm, about 0.22 kg
sim = alloy.build(scene)
traj = sim.run(0.5, record=["cucumber"], every=10).world(0)
pts = traj["cucumber.points"][-1]
print(json.dumps({"route": sim.routes()["cucumber"]["route"], "lowest_z_m": float(pts[:, 2].min()),
                  "health_ok": sim.check()["ok"]}))
