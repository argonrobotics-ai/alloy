# SPDX-License-Identifier: Apache-2.0
"""Blocks of different materials slide down the same static ramp; friction decides which ones stop.

Shows: library materials (alloy.materials()), a per-object friction override, set_friction for one pair, rpy.
Soft materials (rubbers, foams) would make a block a soft body (slower); a rigid material with friction= gives grip.
Run: python examples/basics/01_materials_friction.py
"""
import json
import math

import alloy

ANGLE = math.radians(25)                                  # tan(25 deg) = 0.47: mu above that holds, below slides
scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_box("ramp", size=(0.6, 1.0, 0.02), pos=(0, 0, 0.2), rpy=(0, ANGLE, 0), static=True)
blocks = {"steel": dict(material="steel_carbon"), "ptfe": dict(material="ptfe"),
          "acetal": dict(material="pom_acetal"), "grippy": dict(material="abs", friction=0.9)}
for i, (name, kw) in enumerate(blocks.items()):
    y = -0.375 + 0.25 * i
    # 15 cm up-slope from the ramp centre (+x is downhill), lifted along the normal by half block + half ramp
    x, z = -0.15 * math.cos(ANGLE), 0.2 + 0.15 * math.sin(ANGLE)
    nx, nz = math.sin(ANGLE) * 0.031, math.cos(ANGLE) * 0.031      # ramp normal (sin, 0, cos), 1 mm clear
    scene.add_box(name, size=(0.04, 0.04, 0.04), pos=(x + nx, y, z + nz), rpy=(0, ANGLE, 0), **kw)
scene.set_friction("ptfe", "ramp", 0.05)                   # one pair: slippery whatever the materials say
sim = alloy.build(scene)
start = sim.state(world=0)
sim.run(1.0, record=[])
end = sim.state(world=0)
moved = {n: round(float(math.dist(start[n + ".pos"], end[n + ".pos"])), 3) for n in blocks}
mu = {m["name"]: m["friction"] for m in alloy.materials() if m["name"] in ("steel_carbon", "ptfe", "pom_acetal", "abs")}
print(json.dumps({"moved_m": moved, "material_friction": mu, "routes": {n: r["route"] for n, r in sim.routes().items()}, "ok": sim.check()["ok"]}))
