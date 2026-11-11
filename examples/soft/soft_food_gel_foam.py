# SPDX-License-Identifier: Apache-2.0
"""A 10 % ballistic gelatin block and a flexible polyurethane foam pad, each a soft solid with sourced values.

Run: python examples/soft/soft_food_gel_foam.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("gel", "gel_block_ballistic", pos=(0.0, 0, 0.027))     # 80 x 60 x 50 mm, about 0.25 kg
scene.add_object("foam", "foam_pad_pu", pos=(0.15, 0, 0.0145))         # 100 x 100 x 25 mm, about 6 g
sim = alloy.build(scene)
traj = sim.run(0.5, record=["gel", "foam"], every=10).world(0)
print(json.dumps({"routes": {k: v["route"] for k, v in sim.routes().items()},
                  "gel_top_z_m": float(traj["gel.points"][-1][:, 2].max()),
                  "foam_top_z_m": float(traj["foam.points"][-1][:, 2].max()), "health_ok": sim.check()["ok"]}))
