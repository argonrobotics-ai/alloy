# How do I simulate squeezing a soft object between two fingers?

In the Alloy physics engine (`pip install alloy-engine`) a soft material makes an object a soft body that deforms
under contact. Give a box a soft material, close two fingers on it with `sim.drive`, and read its width back from
its points. This page squeezes a 4 cm foam cube; its fingers are two moving boxes, not a robot hand.

## 1. Squeeze a foam cube between two fingers

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("pad", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.02), material="foam_pu_flexible")
scene.add_box("left", size=(0.04, 0.01, 0.04), pos=(0, -0.026, 0.02), kinematic=True)
scene.add_box("right", size=(0.04, 0.01, 0.04), pos=(0, 0.026, 0.02), kinematic=True)
sim = alloy.build(scene)
print(sim.routes()["pad"]["route"])
```

```text
soft_vbd
```

`sim.routes()` says the cube went to the soft-body solver (`soft_vbd`), not the rigid one. The two fingers are
kinematic boxes: they follow the motion you give them and push the cube.

## 2. Close the fingers and read the width

```python
def width(sim):
    pts = sim.state(["pad"], world=0)["pad.points"]
    return float(pts[:, 1].max() - pts[:, 1].min())

before = width(sim)
sim.drive("left", velocity=(0, 0.06, 0), duration=0.15)
sim.drive("right", velocity=(0, -0.06, 0), duration=0.15)
sim.step(15)
print("width before and after (m):", round(before, 3), round(width(sim), 3))
print(sim.check()["ok"])
```

```text
width before and after (m): 0.04 0.035
True
```

The fingers are driven 9 mm each in 0.15 s, and the cube ends 5 mm narrower: the foam gives way. `"pad.points"`
holds the cube's mesh vertices, so the width is the spread of their y values. Material cards set how stiff a soft
object is (`alloy.materials()` lists them).

## Keep in mind

- Soft bodies step much slower than rigid ones, most of all on the CPU. A soft part that must stay rigid takes
  `rigid=True`.
- A very small soft body can rest on one particle
  ([details](../../known_gaps/soft-small-body-rests-on-one-particle.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Soft bodies from Python: [answer](../answers/soft-body-manipulation-python.md).
- Deformable object manipulation: [answer](../answers/deformable-object-manipulation.md).
- Robots in Alloy: [Robots](../guides/robots.md).
