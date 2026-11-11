# SPDX-License-Identifier: Apache-2.0
"""A cup of water from the card library in one line: 200 ml as a cylinder inside the library cup tumbler_pp.

The fill card is centred on its own origin, so it goes at the cup's pos + (0, 0, 0.0232): its bottom sits 0.5 mm
above the cup's 1.5 mm floor. A round fill starts at rest in a round cup; a square block in it would splash on start.

Run: python examples/soft/soft_fluid_cup.py [seconds]   (GPU if present, else CPU)
"""
import json
import sys

import numpy as np

import alloy

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

scene = alloy.Scene()
scene.add_ground()
scene.add_object("cup", "tumbler_pp", pos=(0, 0, 0), static=True)
scene.add_object("water", "water_tumbler_200ml", pos=(0, 0, 0.0232))     # the one line: a cup of water

sim = alloy.build(scene)
p = sim.run(seconds, record=["water"], every=10).world(0)["water.points"][-1]
inside = (np.hypot(p[:, 0], p[:, 1]) < 0.0388) & (p[:, 2] > 0.0015)
print(json.dumps({"particles": int(len(p)), "fraction_in_cup": float(inside.mean()),
                  "level_mm": round(float(np.percentile(p[:, 2], 95)) * 1e3, 1), "health_ok": sim.check()["ok"]}))
