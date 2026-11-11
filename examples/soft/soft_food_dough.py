# SPDX-License-Identifier: Apache-2.0
"""A ball of wheat dough (elastoplastic paste) dropped onto the ground; it slumps a little and stays put.

Run: python examples/soft/soft_food_dough.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("dough", "dough_ball", pos=(0, 0, 0.045))          # 80 mm ball, about 0.29 kg
sim = alloy.build(scene)
traj = sim.run(0.5, record=["dough"], every=10).world(0)
pts = traj["dough.points"][-1]
print(json.dumps({"route": sim.routes()["dough"]["route"], "top_z_m": float(pts[:, 2].max()),
                  "health_ok": sim.check()["ok"]}))
