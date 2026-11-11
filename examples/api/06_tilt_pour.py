# SPDX-License-Identifier: Apache-2.0
"""Pour from a jug by tilting it: a kinematic open box of water turns about its lip over a cup.

The jug is kinematic (it moves only by sim.drive, as if held), and the water starts at rest inside it. Where to put the
cup: the stream leaves the lip at about sqrt(g h), with h the liquid's height above the lip (~0.3 m/s here), and lands
about v * sqrt(2 dz / g) beyond the lip after falling dz (~4 cm here). So the cup's axis goes a few cm past the lip,
not under it: here a cup 4 cm past the lip catches about 95 % of the water, the same cup centred under the lip
about a third. Tilt slowly so the stream stays thin.
Run: python examples/api/06_tilt_pour.py [out_dir]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_tilt_pour")
JUG_X, JUG_Z, INNER, WALL = -0.06, 0.12, 0.05, 0.004
LIP_X, LIP_Z = JUG_X + INNER / 2 + WALL, JUG_Z + WALL + INNER    # the jug's +x rim, outer top edge
CUP_X, CUP_R, CUP_DEPTH = LIP_X + 0.04, 0.035, 0.06

scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_open_box("jug", inner=(INNER, INNER, INNER), wall=WALL, pos=(JUG_X, 0, JUG_Z), kinematic=True)
scene.add_liquid("water", size=(0.044, 0.044, 0.02), pos=(JUG_X, 0, JUG_Z + WALL + 0.011))   # ~40 ml in the jug
scene.add_open_cylinder("cup", inner_radius=CUP_R, depth=CUP_DEPTH, wall=WALL, pos=(CUP_X, 0, 0))
sim = alloy.build(scene)
sim.settle(0.2)
# positive rotation about +y lowers the +x lip side: 40 deg/s about the lip for 2.25 s (90 deg)
sim.drive("jug", angular_velocity=(0, math.radians(40), 0), pivot=(LIP_X, 0, LIP_Z), duration=2.25)
traj = sim.run(3.5, record=["water"], every=20).world(0)
traj.save(out / "traj.npz", particle_volume_m3=sim.particle_volume("water"))
p = traj["water.points"][-1]
in_cup = (np.hypot(p[:, 0] - CUP_X, p[:, 1]) < CUP_R) & (p[:, 2] < WALL + CUP_DEPTH)
print(json.dumps({"water_ml": round(sim.liquid_volume("water") * 1e6, 1), "fraction_in_cup": float(in_cup.mean()),
                  "retired": sim.health()["retired_particles"], "health_ok": sim.check()["ok"]}))
