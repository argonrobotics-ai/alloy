# SPDX-License-Identifier: Apache-2.0
"""Many parallel worlds: the same box drop in N worlds, each with its own start (pos as an (N, 3) array), plus a
randomized mass; reports world-steps per second while stepping (compile excluded by a warm-up step).

Run: python examples/api/03_many_worlds.py [worlds]
"""
import json
import sys
import time

import numpy as np

import alloy

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 256
starts = np.zeros((N, 3))
starts[:, 0] = 0.002 * (np.arange(N) % 21 - 10)                 # x from -2 cm to +2 cm
starts[:, 2] = 0.30
scene = alloy.Scene(worlds=N, dt=0.01)
scene.add_ground()
scene.add_box("box", size=(0.05, 0.05, 0.05), pos=starts, mass=0.2)
scene.randomize("box", "density", 1400.0, 1800.0)                # drawn per world, seeded by build(seed=)
print(alloy.estimate(scene, seconds=2.0)["wall_time_s"], "s estimated")
sim = alloy.build(scene, seed=7)
sim.step(1)                                                      # warm-up: kernel compile / graph capture
steps0, t0 = sim.physics_steps, time.perf_counter()
traj = sim.run(2.0, record=["box"], every=5)
wall = time.perf_counter() - t0
steps = sim.physics_steps - steps0
x0 = traj["box.pos"][0, :, 0]
print(json.dumps({"worlds": N, "physics_steps": steps, "wall_s": wall, "world_steps_per_s": N * steps / wall,
                  "start_x_ok": bool(np.allclose(x0, starts[:, 0], atol=1e-4)),
                  "rest_z_mean": float(traj["box.pos"][-1, :, 2].mean()), "invalid": sim.health()["invalid_worlds"]}))
