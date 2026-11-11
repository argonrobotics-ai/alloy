# SPDX-License-Identifier: Apache-2.0
"""A wet cellulose kitchen sponge (114 x 64 x 25 mm), a soft solid: dropped on the table, it settles and rests.

Run: python examples/soft/soft_food_sponge.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("sponge", "sponge_cellulose", pos=(0.0, 0, 0.03))      # about 64 g wrung out (an estimate)
sim = alloy.build(scene)
traj = sim.run(0.5, record=["sponge"], every=10).world(0)
print(json.dumps({"route": sim.routes()["sponge"]["route"], "bottom_z_m": float(traj["sponge.points"][-1][:, 2].min()),
                  "top_z_m": float(traj["sponge.points"][-1][:, 2].max()), "health_ok": sim.check()["ok"]}))
