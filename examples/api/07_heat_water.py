# SPDX-License-Identifier: Apache-2.0
"""Heat water in a pan: the pan's floor is a heater held at 80 °C; read the water's temperature as it warms.

Temperatures are in kelvin (20 °C = 293.15 K). `temperature=` on a liquid makes it take heat; `set_heater` turns a
static box (here the open box's floor) into a heater. Heat runs `time_scale` times faster than motion: with dt 0.01 s
and 60, each step heats for 0.6 s, so 1.5 s of motion is 90 s of heating. Hot water does not rise in this model (no
buoyancy from temperature), so it stays layered: warmest at the floor.
Run: python examples/api/07_heat_water.py [out_dir]
"""
import json
import sys
from pathlib import Path

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_heat_water")
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0, 0, 0.02))       # floor top at z = 0.025
scene.add_liquid("water", size=(0.07, 0.07, 0.02), pos=(0, 0, 0.0355), temperature=293.15)
scene.set_heater("pan_floor", temperature=353.15)                                    # or power=500 (W)
scene.set_heat(time_scale=60.0)
sim = alloy.build(scene)
traj = sim.run(1.5, record=["water"], every=25).world(0)                           # water.temperature: (frames, n)
traj.save(out / "traj.npz")
heat = sim.heat("water")
print(json.dumps({"mean_C_per_frame": [round(float(t) - 273.15, 1) for t in traj["water.temperature"].mean(axis=1)],
                  "now_C": {k[:-2]: round(float(heat[k][0]) - 273.15, 1) for k in ("mean_K", "min_K", "max_K")},
                  "heater_area_m2": float(sim.plugins.read()["particle_thermal"]["heater_area_m2"][0, 0]),
                  "health_ok": sim.check()["ok"]}))
