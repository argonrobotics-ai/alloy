# SPDX-License-Identifier: Apache-2.0
"""Cut a soft block along any blade path: its card opts in with `topology: {cut_model: copies}`, and a kinematic
knife with a `blade:` edge slices it, with no pre-split seams.

Every tet is cuttable: the cut follows the knife's edge, here 1.7 mm off the block's mesh planes. The knife is a
kinematic object (`motion: kinematic`) whose `blade:` block marks its cutting edge. The default cut model,
`cut_model: seams`, cuts only along pre-split seams.
Prints how many tets the knife cut, how far the two pieces have parted, and the health check.
- The knife meets no cutting resistance in this model, and only mesh nodes meet its faces. On a mesh this coarse
  (5 mm cells) no node lies within its thickness, so the 1.2 mm blade parts the pieces by about 0.1 mm, not 1.2 mm.
- Cut where each piece keeps two layers of mesh nodes, as here (the knife crosses the block's long axis, 5 mm cells).
  A piece left with one layer has nothing under its cut side and falls over
  (docs/known_gaps/cutting-copies-coarse-mesh.md).
Run: python examples/topology/cut_anywhere.py [card_dir]
"""
import json
import sys
from pathlib import Path

import alloy
from alloy.export import frame_export

root = Path(sys.argv[1] if len(sys.argv) > 1 else "cut_cards")
CUT_X = 0.0017                      # [m] the knife's plane, between the block's mesh planes (x = 0 and 5 mm)
CARDS = {
    "materials/agar_cut.yaml": """schema: alloy.material/1
name: agar_cut
density: {value: 1010 kg/m^3, source: model, note: "agar gel"}
surface:
  mu_static: {value: 0.3, source: model, note: "wet gel"}
  mu_dynamic: {value: 0.25, source: model, note: "wet gel"}
  restitution: {value: 0.05, source: model, note: "gel"}
elastic:
  youngs: {value: 50 kPa, source: model, note: "soft agar"}
  poisson: {value: 0.45, source: model, note: "near incompressible"}
failure:
  fracture_strain: {value: 0.3, source: model, note: "gel"}
""",
    "objects/agar_block.yaml": """schema: alloy.object/1
name: agar_block
shape: {primitive: {kind: box, size: [0.01, 0.0075, 0.0075]}}   # half sizes: a 20 x 15 x 15 mm block
material: agar_cut
budget: {tets: 216}
topology: {cut_model: copies}                                   # cuttable along any blade path
""",
    "objects/knife.yaml": """schema: alloy.object/1
name: knife
shape: {primitive: {kind: box, size: [0.06, 0.0125, 0.0006]}}   # a 1.2 mm steel plate
material: stainless_304
motion: kinematic                                               # moved by sim.drive
features:
  thickness: {value: 0.0012, source: model, note: "1.2 mm blade"}
  min_wall: {value: 0.0012, source: model, note: "1.2 mm blade"}
topology: {}
blade:                                                          # the cutting edge, in the plate's frame
  edge_a: [-0.06, -0.0125, 0.0]
  edge_b: [0.06, -0.0125, 0.0]
""",
    "scenes/cut_block.yaml": f"""schema: alloy.scene/1
robots: []
objects:
  - {{object: agar_block, pose: [0, 0, 0.0089, 0, 0, 0, 1]}}   # resting on its 1.4 mm particle radius
  - {{object: knife, pose: [{CUT_X}, 0, 0.0309, 0.5, 0.5, 0.5, 0.5]}}   # across x, edge down, 2 mm above the block
worlds: 1
control_dt: 0.01
""",
}
for rel, text in CARDS.items():
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)
scene = alloy.Scene.load(root / "scenes" / "cut_block.yaml")
scene.add_ground()
sim = alloy.build(scene)
start = frame_export(sim, [0])[0]                                # frame export: the mesh with its copy vertices marked
nodes = start.topology.mesh_vertex_active["agar_block"]         # True: a mesh node (the rest: copies, asleep until cut)
x0 = start.meshes["agar_block"][nodes, 0]
sim.drive("knife", velocity=(0, 0, -0.05), duration=0.33)        # down through the block, stop 1.9 mm above the ground
sim.run(0.4)
end = frame_export(sim, [0])[0]
x1 = end.meshes["agar_block"][nodes, 0]
left, right = x0 < CUT_X, x0 > CUT_X                             # the nodes of the two pieces
parted = (x1[right].min() - x1[left].max()) - (x0[right].min() - x0[left].max())
print(json.dumps({"tets_cut": end.topology_version >> 40, "pieces_parted_mm": round(1e3 * float(parted), 2),
                  "health_ok": sim.check()["ok"]}))
