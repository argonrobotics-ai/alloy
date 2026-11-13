# How do I render a camera for thousands of parallel worlds?

In the Alloy physics engine, one `alloy.Camera` renders every world in a single call and returns an array shaped
`(worlds, height, width, 3)`. `cam.render_torch(sim)` keeps the images on the GPU for training. Install with
`pip install alloy-engine`; the example below runs on the CPU with four worlds.

```python
import numpy as np
import alloy

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
starts = np.array([[x, 0, 0.45] for x in (-0.2, -0.1, 0.1, 0.2)])
scene.add_sphere("ball", 0.05, pos=starts, mass=0.2)           # one start per world
sim = alloy.build(scene, device="cpu")
sim.run(0.2)

cam = alloy.Camera("front", look_at=((1.2, -1.0, 0.9), (0, 0, 0.4)), width=64, height=64)
images = sim.render(cam)
print(images.shape, images.dtype)
depth = sim.render(cam, output="depth")
print(depth.shape, depth.dtype)
```

```text
(4, 64, 64, 3) uint8
(4, 64, 64) float32
```

Each world has its own picture, so a policy can see four different scenes from one call. On a GPU with PyTorch,
`cam.render_torch(sim)` returns the images as a torch tensor on the simulation's device.

In Alloy's public camera benchmark, Alloy at 1 sample per pixel rendered the cartpole scene at 128 px
over 1,024 worlds at 124,095 images/s on one RTX PRO 6000 Blackwell, with physics on. Alloy's camera default is 16
samples per pixel, which ran the same scene at 12,515 images/s, about ten times slower. The MuJoCo Warp renderer is
faster on that scene: 269,159 images/s. Set the samples per pixel with `spp=` on the camera, for example
`alloy.Camera(..., spp=4)`: fewer samples are faster and noisier.

## Keep in mind

- The camera images of the ready-made training environments are small, bright and grainy
  ([env camera images](../../known_gaps/camera-env-images.md)).
- A glass object renders as an opaque surface ([glass is opaque](../../known_gaps/camera-glass-opaque.md)).
- The benchmark figures are for one RTX PRO 6000 with physics on; they are not CPU speeds.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- [Camera](../guides/camera.md)
- [Train with RL](../guides/rl.md)
- [Robots](../guides/robots.md)
- Related: [the Gymnasium vector API](gymnasium-vector-api.md),
  [GPU-parallel engines for RL](../answers/gpu-parallel-engines-for-rl.md).
