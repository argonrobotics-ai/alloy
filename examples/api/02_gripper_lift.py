# SPDX-License-Identifier: Apache-2.0
"""A two-finger gripper (an MJCF string with position actuators) grasps a cube, lifts it 12 cm and holds it.

Joints are driven with sim.set_targets({...}); bodies are read by name ("gripper/finger_left").
Run: python examples/api/02_gripper_lift.py [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

import alloy

GRIPPER = """<mujoco>
  <worldbody>
    <body name="palm">
      <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0.5" damping="40"/>
      <geom type="box" size="0.04 0.06 0.01" mass="0.3" friction="1.0"/>
      <body name="finger_left" pos="0 0.055 -0.04">
        <joint name="close_left" type="slide" axis="0 -1 0" range="0 0.05" damping="5"/>
        <geom type="box" size="0.012 0.005 0.025" mass="0.03" friction="1.0"/>
      </body>
      <body name="finger_right" pos="0 -0.055 -0.04">
        <joint name="close_right" type="slide" axis="0 1 0" range="0 0.05" damping="5"/>
        <geom type="box" size="0.012 0.005 0.025" mass="0.03" friction="1.0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="lift" kp="3000" kv="150"/>
    <position joint="close_left" kp="400" kv="10"/>
    <position joint="close_right" kp="400" kv="10"/>
  </actuator>
</mujoco>"""

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_gripper_lift")
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=(0, 0, 0.03), mass=0.15, friction=1.0)
scene.add_robot("gripper", mjcf=GRIPPER, pos=(0, 0, 0.075))   # palm 7.5 cm up: finger centres at cube height
sim = alloy.build(scene)
print("names:", sim.names())

record = ["cube", "gripper/finger_left", "gripper/finger_right"]
parts = [sim.run(0.6, record=record, targets={"close_left": 0.03, "close_right": 0.03}),  # squeeze (stops at the cube)
         sim.run(1.0, record=record, targets={"lift": (0.0, 0.12)}),                     # raise 12 cm smoothly
         sim.run(2.5, record=record)]                                                    # hold (targets stay)
traj = {k: np.concatenate([p[k][(1 if i else 0):] for i, p in enumerate(parts)]) for k in parts[0]}
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "traj.npz", **traj)
z = traj["cube.pos"][:, 0, 2]
print(json.dumps({"lift_m": float(z[-1] - z[0]), "min_z_last_2s": float(z[-200:].min()),
                  "health_ok": sim.check()["ok"]}))
