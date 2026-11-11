# Cloth

Add a sheet of fabric with `scene.add_cloth`, drop it over an object, and read where every vertex ends up.

## 1. Drop a sheet over a stand

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)         # top at z = 0.3
scene.add_cloth("sheet", size=(0.5, 0.5), pos=(0, 0, 0.5), resolution=10, mass=0.1)
sim = alloy.build(scene)
print(sim.routes()["sheet"]["route"])
```

```text
cloth_vbd
```

The sheet is flat, 50 cm square, 20 cm above the stand, with 11 x 11 vertices. `sim.routes()` says it runs on the
cloth solver.

## 2. Run it and measure the drape

```python
sim.run(1.0)
points = sim.state(["sheet"], world=0)["sheet.points"]
print(points.shape, (0.3 - points[:, 2].min()).round(1))
```

```text
(121, 3) 0.2
```

`"sheet.points"` holds one row per vertex. The number is how far the lowest vertex hangs
below the stand top, in metres.
After 1 s the sheet has fallen onto the stand and its edges hang 20 cm down the sides.

## 3. Save a picture

```python
sim.render(width=640, height=480, path="cloth.png")
print(sim.check()["ok"])
```

```text
True
```

![A cloth sheet draped over a stand](../img/cloth.png)

## Limits

- A sheet thinner than the 2 mm default needs a smaller `dt`.
- There is no cloth from a mesh. To cover a shape, drape a flat sheet sized to it from a little above.
- Cloth is slow on the CPU; keep the resolution low while you try things out.
- Sheets that fall onto each other can pass through
  ([details](../../known_gaps/cloth-falling-sheets-pass-through.md)).
- A thin cloth (2 mm) slips out of a robot pinch ([details](../../known_gaps/cloth-robot-grasp-carry.md)).

## Next

- Pour water instead: [Liquids](liquids.md).
- Move a robot beside cloth: [Robots](robots.md).
- Read the health checks: [World health](health.md).
