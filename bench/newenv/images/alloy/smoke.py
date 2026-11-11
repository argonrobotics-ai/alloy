# SPDX-License-Identifier: Apache-2.0
"""Alloy smoke: AGENTS.md section 6 scene at 16 worlds (spheres falling on the ground), 100 control steps; one JSON line."""
import json
import time

import numpy as np

t0 = time.perf_counter()
import alloy

t_import = time.perf_counter() - t0
W, STEPS = 16, 100
scene = alloy.Scene(worlds=W, dt=0.01)
scene.add_ground()
xs = np.stack([np.linspace(-0.2, 0.2, W), np.zeros(W), np.full(W, 0.3)], axis=1)
scene.add_sphere("ball", 0.03, pos=xs, material="steel_carbon")
scene.randomize("ball", "friction", 0.6, 1.0)
t0 = time.perf_counter()
sim = alloy.build(scene, seed=0)
sim.step(1)
t_build = time.perf_counter() - t0
s0, t0 = sim.physics_steps, time.perf_counter()
sim.step(STEPS)
z = sim.state(["ball"])["ball.pos"][:, 2]
wall = time.perf_counter() - t0
assert np.isfinite(z).all() and sim.check()["ok"]
print(json.dumps({"arm": "alloy", "version": alloy.__version__, "envs": W, "steps": STEPS, "device": str(sim.device) if hasattr(sim, "device") else "?",
                  "steps_per_s": round(STEPS / wall, 1), "env_steps_per_s": round(W * STEPS / wall),
                  "world_physics_steps_per_s": round(W * (sim.physics_steps - s0) / wall),
                  "t_import_s": round(t_import, 1), "t_build_s": round(t_build, 1), "ball_z_min": round(float(z.min()), 4)}))
