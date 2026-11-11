# SPDX-License-Identifier: Apache-2.0
"""Recording: chosen names every k periods, the Trajectory dict, its meta, saving to .npz and reading it back.

Shows: run(record=[...], every=k), traj.meta (wall time, physics steps, health), traj.world(w), traj.save,
consecutive runs continuing one simulation.
Run: python examples/basics/11_record_trajectory.py [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_record")
scene = alloy.Scene(worlds=3, dt=0.01)
scene.add_ground()
scene.add_sphere("ball", 0.05, pos=np.array([[0, 0, 0.3], [0, 0, 0.6], [0, 0, 0.9]]), restitution=0.6)
sim = alloy.build(scene)
first = sim.run(0.5, record=["ball"], every=5)            # frames at t = 0, 0.05, ..., 0.5
second = sim.run(0.5, record=["ball"], every=5)           # continues from t = 0.5
path = first.world(2).save(out / "ball_world2.npz")
back = np.load(path)
print(json.dumps({"keys": sorted(first), "pos_shape": list(first["ball.pos"].shape), "t_second_run": [
    round(float(second["t"][0]), 3), round(float(second["t"][-1]), 3)], "saved_keys": sorted(back.files),
    "meta": {k: first.meta[k] for k in ("dt", "worlds", "periods", "physics_steps")},
    "health_ok": first.meta["health"]["ok"]}))
