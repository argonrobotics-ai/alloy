# SPDX-License-Identifier: Apache-2.0
"""A free-base robot placed per world: a carton with a hinged flap (an MJCF with <freejoint/> on its root body) starts
at its own spot in every world, with a randomized x; sim.set_pose then lifts every carton (all links follow) and
drops it again. Robot shapes collide like every scene object: contacts start 5 mm apart (contact_gap=).

Run: python examples/api/08_robot_per_world.py [worlds]
"""
import json
import sys

import numpy as np

import alloy

CARTON = """<mujoco><worldbody><body name="box"><freejoint/>
  <geom type="box" size="0.05 0.04 0.03" mass="0.1"/>
  <body name="flap" pos="0.05 0 0.03"><joint name="hinge" type="hinge" axis="0 1 0" range="-1.6 0.2" damping="0.01"/>
    <geom type="box" size="0.03 0.04 0.002" pos="0.03 0 0" mass="0.01"/></body>
  </body></worldbody></mujoco>"""

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
starts = np.zeros((N, 3))
starts[:, 1] = 0.25 * np.arange(N)                               # one row of cartons, 25 cm apart in y
starts[:, 2] = 0.031                                             # resting on the ground (box half height 3 cm)
scene = alloy.Scene(worlds=N, dt=0.01)
scene.add_ground()
scene.add_robot("carton", mjcf=CARTON, pos=starts)               # a free base: one start per world
scene.randomize("carton", "x", -0.02, 0.02)                      # and x drawn per world, seeded by build(seed=)
sim = alloy.build(scene, seed=1)
start = sim.state(["carton"])["carton.pos"]
sim.run(0.5)
lifted = start + np.array([0.0, 0.0, 0.1])
sim.set_pose("carton", pos=lifted)                               # every world's carton, flap and all, 10 cm up
up = sim.state(["carton", "carton/flap"])
traj = sim.run(1.0, record=["carton"], every=10)
print(json.dumps({"worlds": N, "start_x": [round(float(x), 4) for x in start[:, 0]],
                  "flap_above_box_m": round(float((up["carton/flap.pos"] - up["carton.pos"])[0, 2]), 4),
                  "landed_z": [round(float(z), 4) for z in traj["carton.pos"][-1, :, 2]], "ok": sim.check()["ok"]}))
