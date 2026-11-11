# SPDX-License-Identifier: Apache-2.0
"""Drop a box onto a table and save its trajectory (positions, wxyz quaternions) to an .npz file.

Run: python examples/api/01_drop_box.py [out_dir]     (GPU if present, else CPU; first run compiles kernels)
"""
import json
import sys
from pathlib import Path

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_drop_box")
scene = alloy.Scene(dt=0.005)                                    # control period: 5 ms per sim.step()
scene.add_ground()
scene.add_box("table", size=(0.8, 0.8, 0.70), pos=(0, 0, 0.35), static=True)   # top at z = 0.70
scene.add_box("box", size=(0.08, 0.08, 0.08), pos=(0, 0, 0.90), mass=0.5)     # full edge lengths, kg
sim = alloy.build(scene)
traj = sim.run(1.5, record=["box"])                              # frames every 5 ms incl. t = 0
w0 = traj.world(0)                                               # drop the world axis
w0.save(out / "traj.npz")                                        # keys: t, box.pos, box.quat_wxyz, ...
rest = float(w0["box.pos"][-1, 2])
print(json.dumps({"rest_height_m": rest, "expected_m": 0.74, "health_ok": sim.check()["ok"]}))
