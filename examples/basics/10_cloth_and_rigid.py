# SPDX-License-Identifier: Apache-2.0
"""Cloth and rigid bodies in one scene: a sheet lands on a table, then a ball drops onto the sheet.

Shows: routing (sim.routes(): rigid and cloth go to different solvers, coupled), cloth points + faces, saving a mesh
sequence you can view, and reading a rigid body that rests on cloth.
Run: python examples/basics/10_cloth_and_rigid.py [out_dir]     (GPU: seconds; CPU: about 10 min, mostly cloth)
"""
import json
import sys
from pathlib import Path

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_cloth_rigid")
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.4, 0.4, 0.3), pos=(0, 0, 0.15), static=True)
scene.add_cloth("sheet", size=(0.5, 0.5), pos=(0, 0, 0.35), resolution=15, mass=0.08)
scene.add_sphere("ball", 0.03, pos=(0.05, 0.0, 0.6), mass=0.05)
sim = alloy.build(scene)
traj = sim.run(1.0, record=["sheet", "ball"], every=10).world(0)
traj.save(out / "cloth_rigid.npz", faces=sim.faces("sheet"))
pts = traj["sheet.points"][-1]
print(json.dumps({"routes": {n: r["route"] for n, r in sim.routes().items()},
                  "sheet_vertices": int(pts.shape[0]), "sheet_min_z": round(float(pts[:, 2].min()), 3),
                  "ball_z": round(float(traj["ball.pos"][-1, 2]), 3), "ok": sim.check()["ok"]}))
