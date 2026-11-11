# SPDX-License-Identifier: Apache-2.0
"""Plan a big run before building it: estimated time, memory and first-run compile for several world counts.

Shows: alloy.estimate(scene, worlds, seconds), and a measured short run to compare (measure=True).
Run: python examples/basics/07_estimate_cost.py
"""
import json

import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.2), mass=0.1)
scene.add_sphere("ball", 0.03, pos=(0.1, 0, 0.2))
rows = {}
for worlds in (1, 64, 1024):
    e = alloy.estimate(scene, worlds, seconds=5.0)
    rows[worlds] = {k: e[k] for k in ("wall_time_s", "world_steps_per_s", "memory_bytes", "compile_s_first_run")}
measured = alloy.estimate(scene, 4, seconds=0.2, measure=True)
print(json.dumps({"estimates": rows, "measured_4_worlds": {k: measured.get(k) for k in ("wall_time_s", "basis")}},
                 default=str))
