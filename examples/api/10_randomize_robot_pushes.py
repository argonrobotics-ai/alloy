# SPDX-License-Identifier: Apache-2.0
"""Timed pushes and random starts: the shipped Go2 stands in 8 worlds, each pushed on its base at its own random
times with its own random force, and each starting at its own place and heading after a reset.

scene.add_push(body, force, every, duration) pushes a body: after a wait drawn in every=(lo, hi) seconds, a force
drawn in +-force (N, per world axis) is held for duration seconds, then the world waits again. sim.pushes() is the
force held now. scene.randomize("go2", "start_yaw", lo, hi, when="reset") (also start_x, start_y, start_z) moves a
free-base robot's start at every reset. The robot only stands on its servos here: no policy balances it, so the
pushes are kept small. Building the Go2 prints many NewtonWarning lines about geom margins: they are harmless.
Run: python examples/api/10_randomize_robot_pushes.py
"""
import numpy as np

import alloy
from alloy.cards.library import default_root

W = 8
scene = alloy.Scene(worlds=W, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")     # the file brings its own floor
scene.add_push("go2/base", force=(30.0, 30.0, 0.0), every=(0.5, 1.5), duration=0.1)   # N; seconds between pushes
scene.randomize("go2", "start_x", -1.0, 1.0, when="reset")          # m from the start position, at every reset
scene.randomize("go2", "start_yaw", -3.14, 3.14, when="reset")      # rad about the vertical, at every reset
sim = alloy.build(scene, seed=0)

count, first, largest, speed = np.zeros(W, int), np.full(W, np.nan), np.zeros(W), np.zeros(W)
was_on = np.zeros(W, bool)
for k in range(300):                                                # 3 s
    sim.step(1)
    force = sim.pushes()["go2/base"]                                # (worlds, 3) N held in this period
    on = np.abs(force).sum(axis=1) > 0
    started = on & ~was_on
    count += started
    first = np.where(started & np.isnan(first), sim.t, first)
    largest = np.maximum(largest, np.linalg.norm(force, axis=1))
    if sim.t > 0.4:                                                 # after the robot has settled on its feet
        speed = np.maximum(speed, np.linalg.norm(sim.state(["go2/base"])["go2/base.vel"][:, :2], axis=1))
    was_on = on

print("world  pushes  first_push_s  largest_push_N  top_speed_m_s")
for w in range(W):
    print(f"{w:5d}  {count[w]:6d}  {first[w]:12.2f}  {largest[w]:14.1f}  {speed[w]:13.3f}")
print("healthy:", sim.check()["ok"])
sim.reset()                                                         # every world gets a new start
state = sim.state(["go2/base"])
quat = state["go2/base.quat_wxyz"]
print("after sim.reset(): start_x  ", sim.randomized()["go2.start_x"].round(2))
print("                   base x   ", state["go2/base.pos"][:, 0].round(2))
print("                   start_yaw", sim.randomized()["go2.start_yaw"].round(2))
print("                   heading  ", (2 * np.arctan2(quat[:, 3], quat[:, 0])).round(2))
sim.run(1.0)
print("healthy after 1 s from the new starts:", sim.check()["ok"])
