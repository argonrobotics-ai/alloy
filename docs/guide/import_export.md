# Import and export: MJCF, URDF, USD

- **Robots** load directly: `scene.add_robot(name, "robot.xml" | "robot.urdf" | "robot.usd")` or
  `alloy.load("robot.xml")`. Use this for anything with joints you will drive.
  USD files need the `usd` extra (`pip install "alloy-engine[usd]"`); MJCF and URDF need nothing extra. Writing USD
  (`export_scene`) needs nothing extra.
- **Import to cards** (`alloy.io.import_file(path, out_dir)`): turns an MJCF / URDF / USD file into Alloy cards
  (`objects/`, `materials/`, `scenes/<name>.yaml`) you can edit, check and reuse. `res.unmapped` lists what the cards
  could not hold; `res.issues` the card validation results. Then `alloy.load(out_dir + "/scenes/<name>.yaml")`.
- **Export** (`alloy.io.export_scene(scene, out_dir)`): writes MJCF (`<name>.xml`, loaded once in MuJoCo as a
  check), URDF (`urdf/<object>.urdf` + `urdf/index.json`) and USD (`<name>.usda`) of world 0. Alloy-only features
  (cloth, liquid, precision contact) are listed in `res.unmapped`. A scene saved from Python exports from its file:
  `export_scene(scene.save("my_scene"), out_dir)`.
- Command line, JSON out: `python -m alloy.io import robot.xml --out my_library`,
  `python -m alloy.io export --scene demo_peg_d12 --out exported`.

```python
import json
import alloy
import alloy.io

res = alloy.io.export_scene("demo_peg_d12", "exported")          # a library scene by name
files = res.to_json()["files"]
print(json.dumps(files))                                         # mjcf, urdf, usd paths
print([row["item"] for row in res.to_json()["unmapped"]][:5])    # what the formats could not hold
sim = alloy.load(files["mjcf"])                                  # the exported MJCF, back in Alloy
print(sim.names()["bodies"])
```

Runnable: `examples/io/import_file.py`, `examples/io/export_scene.py`.
