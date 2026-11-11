# SPDX-License-Identifier: Apache-2.0
"""Pour 200 ml of water from a nozzle into a round cup (a static fixture of box walls) and let it settle.

Pour liquid into containers with add_pour (a stream from a nozzle). A block of liquid dropped from above splashes
several percent over the rim when it lands: a flat liquid face hitting a flat floor makes real pressure jets.
Run: python examples/api/05_pour_liquid.py [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_pour")
scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_open_cylinder("cup", inner_radius=0.04, depth=0.10, wall=0.004, material="pp_homopolymer")
scene.add_pour("water", 200e-6, pos=(0, 0, 0.15), rate=2e-4)      # 200 ml at 200 ml/s, 5 cm above the rim
sim = alloy.build(scene)
print("liquid: %.1f ml" % (sim.liquid_volume("water") * 1e6))
traj = sim.run(2.5, record=["water"], every=20).world(0)        # water.active marks poured particles
traj.save(out / "traj.npz", particle_volume_m3=sim.particle_volume("water"))
p = traj["water.points"][-1]
inside = (np.hypot(p[:, 0], p[:, 1]) < 0.04) & (p[:, 2] < 0.104)
print(json.dumps({"particles": int(len(p)), "poured": float(traj["water.active"][-1].mean()),
                  "fraction_in_cup": float(inside.mean()), "health_ok": sim.check()["ok"]}))
