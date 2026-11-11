# SPDX-License-Identifier: Apache-2.0
"""A box of liquid or grains from the card library in one line, then let it settle and report where it is.

Each fill card fills the open box liquid_box_56 (56 x 56 mm inside, 80 mm deep) when both sit at the same pos:
    water_box_200ml, glycerol_80_box_100ml, oil_olive_box_150ml        (SPH liquids)
    honey_box_150ml                                                    (MPM paste: thicker than 1 Pa s)
    sauce_ketchup_box_150ml, sauce_mayonnaise_box_150ml                (MPM pastes with a yield stress)
    yogurt_box_150ml, peanut_butter_box_150ml, batter_bread_box_150ml  (shear-thinning, with a yield stress)
    batter_cake_box_150ml, puree_fruit_box_150ml,
    tomato_concentrate_box_150ml                                       (shear-thinning power-law pastes)
    sand_box_200ml, rice_box_200ml                                     (MPM grains)
The box is square so a fill starts at rest; a square fill in a round cup collapses and splashes on start.

Run: python examples/soft/soft_fluid_fills.py [fill card] [seconds]   (GPU if present, else CPU)
"""
import json
import sys

import numpy as np

import alloy

fill = sys.argv[1] if len(sys.argv) > 1 else "water_box_200ml"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

scene = alloy.Scene()
scene.add_ground()
scene.add_object("box", "liquid_box_56", pos=(0, 0, 0), static=True)   # the container
scene.add_object("fill", fill, pos=(0, 0, 0))                            # the one line: a box of it, same pos

sim = alloy.build(scene)
p = sim.run(seconds, record=["fill"], every=10).world(0)["fill.points"][-1]
inside = (np.abs(p[:, 0]) < 0.028) & (np.abs(p[:, 1]) < 0.028) & (p[:, 2] > 0.004)
print(json.dumps({"fill": fill, "particles": int(len(p)), "fraction_in_box": float(inside.mean()),
                  "level_mm": round(float(np.percentile(p[:, 2], 95)) * 1e3, 1), "health_ok": sim.check()["ok"]}))
