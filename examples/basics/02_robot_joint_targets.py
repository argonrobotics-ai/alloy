# SPDX-License-Identifier: Apache-2.0
"""A robot arm from a file (a shipped MJCF model) moves joints to targets, smoothly, and reports where its hand is.

Shows: add_robot from a path, sim.names(), "<robot>/<joint>" names, ramped targets in run(), set_targets, body poses.
Run: python examples/basics/02_robot_joint_targets.py
"""
import json
from pathlib import Path

import alloy

ARM = Path(__file__).resolve().parents[2] / "library" / "robots" / "emb_e2_panda" / "scene.xml"
scene = alloy.Scene(dt=0.01)
scene.add_robot("panda", ARM)          # this file brings its own floor and table: no add_ground() (they would overlap)
sim = alloy.build(scene)
joints = [j for j in sim.names()["joints"] if j.startswith("panda/joint")]
start = {j: float(sim.state([j], world=0)[j + ".q"]) for j in joints}
# joint4's range is about [-3.07, -0.07] rad (see the robot file): bend it further negative, inside its range
goal = {**start, "panda/joint2": start["panda/joint2"] + 0.4, "panda/joint4": start["panda/joint4"] - 0.6}
sim.run(1.5, record=[], targets={j: (start[j], goal[j]) for j in joints})   # ramp every joint over 1.5 s
sim.set_targets(goal)                                                        # then hold
sim.step(50)
q = {j: round(float(sim.state([j], world=0)[j + ".q"]), 3) for j in ("panda/joint2", "panda/joint4")}
hand = sim.state(["panda/hand"], world=0)["panda/hand.pos"]
print(json.dumps({"joints": joints, "q": q, "goal": {j: round(goal[j], 3) for j in q},
                  "hand_pos_m": [round(float(v), 3) for v in hand], "ok": sim.check()["ok"]}))
