# Your own objects: alloy create (and alloy audit)

`alloy.create` turns your own object into a checked object card: a mesh file (OBJ, STL, GLB; or MJCF / URDF / USD
with its joints), one photo, or a text description. It infers or takes the real size, builds a watertight physics
mesh and convex parts, picks a material and mass with their sources, runs a drop test, and says which values are
estimates and what a person should verify.

It needs the optional `import` extra (mesh processing): `pip install "alloy-engine[import]"`
(`install.md`).

```python
import alloy
import alloy.create
from alloy.api.meshes import open_cylinder

v, f = open_cylinder(inner_radius=40, depth=90, wall=3)        # a cup modelled in millimetres
with open("cup.obj", "w") as fh:
    fh.writelines(f"v {x} {y} {z}\n" for x, y, z in v)
    fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)
r = alloy.create.create(mesh="cup.obj", name="my_cup", material="ceramic", out="my_objects")
print(r.ok, r.estimates, r.verify[:2])                        # what is a guess, and what to check first

scene = alloy.Scene(dt=0.01, roots=["my_objects"])           # your card folder, searched before the library
scene.add_ground()
scene.add_object("cup", "my_cup", pos=(0, 0, 0.05))
sim = alloy.build(scene)
sim.run(0.5, record=[])
print(sim.state(world=0)["cup.pos"], sim.check()["ok"])
```

- Command line (JSON result): `python -m alloy.create --mesh cup.stl --material plastic --out my_objects`,
  `--photo bowl.jpg --name yellow_bowl`, `--text "a red ceramic coffee mug with a handle"`.
- Size: `size=` (longest side in m) or `units=` (`m`, `cm`, `mm`, `in`, `ft`); otherwise inferred from the object's
  category. Mass: `mass=` with `mass_source=` (for example "kitchen scale") turns the estimate into a measured value.
- Photo and text need the mesh generator; its set-up, licences and every option: `alloy/create/README.md`.
- `alloy.audit.audit(scene_file, rounds=3, episodes=16, library=["my_objects"])` stress-tests a whole scene (check
  rounds plus adversarial episodes) and reports what breaks, as JSON.
- Runnable: `examples/create/` (mesh, photo, text, and create -> compose -> audit).
