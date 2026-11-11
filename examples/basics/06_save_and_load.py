# SPDX-License-Identifier: Apache-2.0
"""A scene as data: to JSON and back, to YAML cards on disk and back, and an edit of a loaded scene.

Shows: Scene.to_json / from_dict, Scene.save(dir) -> card files, alloy.load(file), Scene.load + add more objects.
Run: python examples/basics/06_save_and_load.py [out_dir]
"""
import json
import sys
from pathlib import Path

import alloy

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out_save_load")
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_cylinder("can", 0.033, 0.12, pos=(0, 0, 0.06), material="steel_tinplate", mass=0.35)
scene.add_sphere("ball", 0.02, pos=(0.0, 0.0, 0.3), material="steel_carbon")

out.mkdir(parents=True, exist_ok=True)
(out / "scene.json").write_text(scene.to_json())                      # plain data, keeps everything
again = alloy.Scene.from_dict(json.loads((out / "scene.json").read_text()))
card = scene.save(out / "cards")                                      # YAML cards: materials/ objects/ scenes/
sim = alloy.load(card)                                                # build straight from the card file
sim.run(0.5, record=[])
edited = alloy.Scene.load(card)                                       # an editable copy of the card scene
edited.add_box("lid", size=(0.08, 0.08, 0.005), pos=(0, 0, 0.2))
print(json.dumps({"json_names": again.names(), "card_file": str(card), "card_scene_names": sim.names(),
                  "edited_names": edited.names(), "ball_z": round(float(sim.state(world=0)["ball.pos"][2]), 3)}))
