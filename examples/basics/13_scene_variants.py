# SPDX-License-Identifier: Apache-2.0
"""Scene variants: four copies of one table scene with the free objects moved and each look given another texture of
its family (``alloy.vary``), then one variant built and run like any scene.

Run: python examples/basics/13_scene_variants.py [out_dir]       (CPU or GPU; about 1 min on a CPU)
The same seed gives the same variants. ``out_dir/variants.json`` lists what each variant moved and swapped; the
command line does the same with JSON in and out: ``alloy vary scene.json --n 4 --pos 0.02 --yaw 15 --seed 3``.
"""

import json
import sys
from pathlib import Path

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_variants")

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")   # stays put
scene.add_box("a", size=(0.04, 0.04, 0.04), pos=(-0.06, 0, 0.42), material="wood_beech")
scene.add_box("b", size=(0.04, 0.04, 0.04), pos=(0.06, 0, 0.42), material="abs")

# free objects move within +-2 cm and +-15 degrees; each look takes another texture of its family (wood -> wood)
variants = alloy.vary(scene, 4, pos=0.02, yaw_deg=15, seed=3, out=out)    # a list of alloy.Scene
manifest = json.loads((out / "variants.json").read_text())
for row in manifest["variants"]:
    moves = {name: f"{m['dx'] * 1e3:+.1f} mm, {m['dy'] * 1e3:+.1f} mm, {m['yaw_deg']:+.1f} deg"
             for name, m in row["moves"].items()}
    print(row["file"], moves, row["textures"]["a"])      # e.g. {'wood_beech': 'polyhaven:kitchen_wood'}

sim = alloy.build(variants[2])                           # a variant is an ordinary scene
traj = sim.run(0.5, record=["a", "b"])
print("a rests at", traj["a.pos"][-1, 0].round(3), "ok:", sim.check()["ok"])
