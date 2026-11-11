# SPDX-License-Identifier: Apache-2.0
"""One scene, many worlds, a different joint target in each: a sweep run as a single batched simulation.

Shows: worlds=N, a (worlds,) target array in set_targets, per-world state, world-steps per second.
Run: python examples/basics/03_per_world_targets.py [worlds]
"""
import json
import sys
import time

import numpy as np

import alloy

W = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PENDULUM = """<mujoco><compiler angle="radian"/><worldbody><body name="bob" pos="0 0 1.0">
  <joint name="swing" type="hinge" axis="0 1 0" damping="0.5"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="0.5"/></body></worldbody>
  <actuator><position joint="swing" kp="40" kv="2"/></actuator></mujoco>"""
scene = alloy.Scene(worlds=W, dt=0.01)
scene.add_robot("pend", mjcf=PENDULUM)
sim = alloy.build(scene)
targets = np.linspace(-1.0, 1.0, W)                       # one target angle per world (radians)
sim.set_targets({"pend/swing": targets})
sim.step(1)                                               # warm-up (kernel compile happens on the first step)
s0, t0 = sim.physics_steps, time.perf_counter()
sim.step(199)
q = sim.state(["pend/swing"])["pend/swing.q"]             # (worlds,); reading waits for the device
wall = time.perf_counter() - t0
print(json.dumps({"worlds": W, "target": np.round(targets[:4], 3).tolist(), "reached": np.round(q[:4], 3).tolist(),
                  "max_error_rad": round(float(np.abs(q - targets).max()), 4),
                  "world_steps_per_s": round(W * (sim.physics_steps - s0) / wall), "ok": sim.check()["ok"]}))
