# Camera

Render a picture of a scene, place a named camera, read depth and ids, and pick the fast or the full camera.

## 1. Render a picture

```python
import alloy
import numpy as np

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.45), mass=0.2)
scene.add_sphere("ball", 0.05, pos=(0.2, 0, 0.45), mass=0.2)
sim = alloy.build(scene)
sim.run(0.5)

image = sim.render(width=640, height=480, path="camera.png")
print(image.shape, image.dtype)
```

```text
(480, 640, 3) uint8
```

![A block and a ball on a table](../img/camera.png)

`sim.render` frames world 0 automatically and returns the picture as a `(height, width, 3)` array of 8-bit colour.
`path=` also saves it as a PNG. Look at a new scene once before a long run.

## 2. Place a camera

```python
cam = alloy.Camera("front", look_at=((1.2, -1.0, 0.9), (0, 0, 0.4)), width=160, height=120)
views = sim.render(cam)
print(views.shape, views.dtype)
```

```text
(1, 120, 160, 3) uint8
```

`look_at=(eye, target)` puts the camera at `eye` looking at `target`. A camera renders every world, so the array has a
leading world axis: `(worlds, height, width, 3)`.

## 3. Read depth

```python
depth = sim.render(cam, output="depth")
seen = depth[np.isfinite(depth)]
print(depth.shape, depth.dtype, "nearest", seen.min().round(2), "m")
print("share of pixels with no hit:", round(float(np.isinf(depth).mean()), 2))
```

```text
(1, 120, 160) float32 nearest 1.06 m
share of pixels with no hit: 0.3
```

`output="depth"` gives the distance in metres for each pixel, and `inf` where nothing is hit. Use
`output="segmentation"` for an integer id per pixel.

## 4. Tell objects apart

```python
from alloy.camera.aov import instance_labels

ids = sim.render(cam, output="instance_id")
names = instance_labels(sim)
print(ids.shape, ids.dtype, [names[i] for i in np.unique(ids)])
```

```text
(1, 120, 160) int32 ['background', 'ground', 'table', 'block', 'ball']
```

`"instance_id"` gives every shape an id of its own, static ones too. In `"segmentation"` the ground and the table
share id 1.

## 5. Fast camera and full camera

Everything above used the fast camera: a studio sun and sky, made for many worlds. In an outside test on one
NVIDIA L4 it rendered about 1,000 colour images per second at 128 px over 256 to 1,024 worlds.

```python
photo = sim.render(width=320, height=240, quality="full")
print(photo.shape, photo.dtype)
```

```text
(240, 320, 3) uint8
```

`quality="full"` is the full camera: materials, lights and soft shadows, for pictures that should look real. It is
far slower: 21 to 24 images per second in the same test (128 px, 64 or 256 worlds, 32 samples per pixel), and it
gives colour only, not depth or ids.

Lights and colours are one look for the whole render call: every world gets the same look. Each new look compiles
again, about 100 s the first time, so keep one look for a run.

## Keep in mind

- There is no sensor model: no noise, and no range limit other than the camera's clip distances.
- A glass object renders as an opaque surface ([glass is opaque](../../known_gaps/camera-glass-opaque.md)).
- The camera images of the ready-made training environments are small, bright and grainy
  ([env camera images](../../known_gaps/camera-env-images.md)).

## Next

- Add a robot to look at: [Robots](robots.md).
- Train from state: [Train with RL](rl.md).
- Check a run before you trust it: [World health](health.md).
