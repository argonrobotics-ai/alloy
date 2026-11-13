# Which simulators support sand and granular materials for robot learning?

The Alloy physics engine (`pip install alloy-engine`) simulates sand, soil, flour and rice as granular materials in
Newton's MPM solver, in the same scene as rigid bodies and robots. Name a granular material on a box and the engine
routes it to MPM. This page lists the materials and drops a block onto a small bed of sand.

```python
import textwrap
import alloy

sands = [m["name"] for m in alloy.materials() if m["kind"] == "granular"]
print(textwrap.fill(", ".join(sands), width=100))

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("sand", size=(0.08, 0.08, 0.04), pos=(0, 0, 0.02), material="sand_silica_dry")
scene.add_box("block", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.065), mass=0.05)
sim = alloy.build(scene)
print({name: r["route"] for name, r in sim.routes().items()})
sim.run(0.1)
print(sim.state(["block"], world=0)["block.pos"][2].round(2), sim.check()["ok"])
```

```text
flour_wheat, rice_grain, sand_silica_dry, soil_clay_loam, soil_hardpan, soil_loam_moist,
soil_sand_damp, soil_sand_dry
{'ground': 'static', 'sand': 'grain_mpm', 'block': 'rigid'}
0.04 True
```

`alloy.materials()` has eight granular materials, and `sim.routes()` shows the sand on the `grain_mpm` route and the
block as a rigid body. The block started with its centre at z = 0.065 m. It ends at 0.04 m, so its bottom (0.025 m)
is below the bed's starting surface at 0.04 m: it pressed into the sand.

## Keep in mind

- Sand steps slowly on a CPU, and many worlds of MPM are limited by the material solve
  ([details](../../known_gaps/mpm-solve-speed.md)).
- A collapsed heap's flank does not read the repose angle, and a squat collapse's runout still changes with resolution
  ([details](../../known_gaps/mpm-grain-slope-resolution.md)).
- An MPM step that does not fit in GPU memory runs without CUDA graphs or is refused
  ([details](../../known_gaps/mpm-gpu-memory.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- MJCF robots beside liquids and sand: [MuJoCo models with liquids and sand](mujoco-models-with-liquids-and-sand.md)
- Robots, cloth and liquids together: [Robots, cloth and liquids in one scene](robots-cloth-liquids-one-scene.md)
- Liquids and pours: [Liquids](../guides/liquids.md).
