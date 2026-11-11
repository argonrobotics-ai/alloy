# SPDX-License-Identifier: Apache-2.0
"""Your own card directory: a new material and object written as YAML cards (with sources), used from Python.

Shows: card layout (materials/ objects/ scenes/), quantities with units and notes, alloy.load(<scene card>) looking up
your cards first and the shipped library second, and Scene.load to add Python objects to a card scene.
Run: python examples/basics/12_custom_cards.py [card_dir]
"""
import json
import sys
from pathlib import Path

import alloy

root = Path(sys.argv[1] if len(sys.argv) > 1 else "my_cards")
CARDS = {
    "materials/oak_board.yaml": """schema: alloy.material/1
name: oak_board
density: {value: 720 kg/m^3, source: measured, note: "weighed 57.6 g for an 80 cm3 offcut"}
surface:
  mu_static: {value: 0.55, source: measured, note: "tilt test on plywood, 29 deg"}
  mu_dynamic: {value: 0.45, source: model, note: "about 0.8 x static for dry wood"}
  restitution: {value: 0.4, source: measured, note: "drop from 30 cm, rebound 4.8 cm"}
elastic:
  youngs: {value: 11 GPa, source: catalog, note: "oak along the grain, 10-13 GPa"}
  poisson: {value: 0.35, source: catalog, note: "hardwoods 0.3-0.4"}
""",
    "objects/puck.yaml": """schema: alloy.object/1
name: puck
shape: {primitive: {kind: cylinder, size: [0.035, 0.0125]}}   # radius, HALF height (cards use half sizes)
material: oak_board
""",
    "scenes/puck_drop.yaml": """schema: alloy.scene/1
robots: []
objects:
  - {object: puck, pose: [0, 0, 0.2, 0, 0, 0, 1]}            # x y z qx qy qz qw
  - {object: puck, pose: [0.1, 0, 0.3, 0, 0, 0, 1]}
worlds: 1
control_dt: 0.01
""",
}
for rel, text in CARDS.items():
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)
scene = alloy.Scene.load(root / "scenes" / "puck_drop.yaml")   # your cards first, then the shipped library
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(-0.1, 0, 0.1), material="oak_board")
sim = alloy.build(scene)
sim.run(1.0, record=[])
print(json.dumps({"names": scene.names(), "z": {n: round(float(v[2]), 4) for n, v in sim.state(world=0).items()
                                                if n.endswith(".pos")}, "ok": sim.check()["ok"]}))
