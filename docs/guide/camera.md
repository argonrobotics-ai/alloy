# Camera

`alloy.camera` is a path-traced camera that runs on the same device as the simulation and renders every world at
once. The quick look is one call on a built `Sim`:

```python
import alloy

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
scene.add_open_box("bin", inner=(0.12, 0.12, 0.06), wall=0.005)
scene.add_sphere("ball", 0.02, pos=(0.0, 0.0, 0.2), material="steel_carbon")
sim = alloy.build(scene)
sim.run(0.5, record=[])
img = sim.render(width=320, height=240, path="look.png")      # world 0, framed automatically: (240, 320, 3) uint8
cam = alloy.Camera("front", look_at=((0.4, -0.4, 0.3), (0.0, 0.0, 0.03)), width=128, height=128, fov=45, spp=16)
batch = sim.render(cam)                                        # every world: (worlds, 128, 128, 3) uint8 numpy
print(img.shape, batch.shape)
```

- `sim.render(width=, height=, world=0, spp=, look_at=(eye, target), path=)`: one image of one world, saved as PNG
  when `path` is given. Use it to check a scene by eye before a long run.
- `alloy.Camera(name, look_at=(eye, target) | pose=..., width, height, fov (degrees), spp, denoise=None)`: a camera
  you keep and render every step. `sim.render(cam)` returns numpy; `cam.render_torch(sim)` returns a torch tensor on
  the GPU for training (`cam.render(sim)` gives the Warp array, overwritten by the next render: copy what you keep).
- `attach_to="<body name>"` mounts the camera on a body (a wrist camera); its `pose` is then in the body's frame.
- `spp` (samples per pixel) trades noise for time: 4-16 for training images, 64+ for a clean still;
  `denoise="auto"` makes a few samples look like many.
- Denoising needs a CUDA GPU. `denoise="auto"` uses the learned denoiser when PyTorch is installed
  (`pip install torch`) and its weights are there, else Intel Open Image Denoise if you fetched it
  (`python -m alloy.camera.oidn fetch`), else nothing. When it falls back it says so on the console, with the reason,
  once per process. `denoise="learned"` and `denoise="oidn"` raise instead of falling back.
  - The learned weights (18 MB a file) download on first use from the Alloy library on the Hugging Face Hub (dataset
    `argonrobotics-ai/alloy-library`, folder `denoiser/`; no account needed). They are checked against the sha256
    in the package and cached in `~/.cache/alloy/denoiser` (`$ALLOY_CACHE/denoiser`). They are Apache-2.0, trained
    on our own Cycles renders (`alloy/camera/denoiser/WEIGHTS.md`).
  - Offline: a cached file needs no network. Fill the cache ahead of time on a machine with network
    (`python -m alloy.camera.denoiser.weights`), or copy the `.pt` files into the cache folder.
    `ALLOY_DENOISER_URL=<folder or URL>` reads them from your own copy; `ALLOY_DENOISER_URL=""` turns downloads off.
    Without the weights, `"auto"` falls back as above and `"learned"` raises, naming the missing file.
  - On a scene of your own, `"auto"` uses the general weights (`v3`), trained on Alloy's scenes; the speeds below use
    weights tuned for each test scene. On four new scenes the general weights matched Open Image Denoise at 128 px
    and fell behind it at 640x480 from 64 spp (`alloy/camera/denoiser/WEIGHTS.md`). Nothing checks the quality on your
    scene: compare a few frames with a high-`spp` render before you rely on it, or use `denoise="oidn"`.
- `output="linear"` gives float32 linear colour instead of uint8, `output="all"` adds albedo and normal images;
  `output="depth"`, `"segmentation"`, `"body_id"` and `"world_id"` give depth and id images instead of colour, and
  a tuple such as `("rgb", "depth")` gives several at once (next section).
- The first render compiles the renderer's kernels; later renders are fast. Images are lit by a studio sun and sky
  by default; `quality="full"` and `look=` (a look file) give Blender-style materials and lights, without Blender.
- Scanned library objects (YCB, Google Scanned Objects, Objaverse and the other scans `alloy fetch` downloads) are
  drawn with their own photo texture, in the quick look and with `quality="full"` alike. The camera reads the scan's
  `visual.glb` (its mesh, UVs and base-colour image) beside the card's mesh, for a scene built from an `alloy.Scene`.
  - A look that names the object (`look=`, an item look, or the card's own object look) still decides its colour.
  - Decoding the image needs Pillow, which the `import` extra brings (fetching a scan needs that extra anyway).
    Without it, or for a glb whose size does not match the object's mesh, the object keeps a flat colour and the
    console says why, once.
  - Images over 2,048 texels on a side are reduced at load (`ALLOY_CAMERA_TEXTURE_MAX`).
  - The image is multiplied by the glb's base-colour factor, except a factor of exactly (0.4, 0.4, 0.4) beside an
    image, which counts as 1: that is the grey trimesh writes when the source gave no colour (in one count of 23
    fetched scans, 22 of their 24 textured materials carried it), and taken literally it would draw them 2.5x dark.
  - Cost: on one L4, a table with four YCB scans rendered 2-3 % slower with their textures (fast camera, 16 spp:
    723 to 709 images/s at 128 px over 256 worlds, 43.5 to 42.2 at 640 px over 16 worlds; one VM, median of 3 runs);
    a scene without scans renders as fast as before. On a CPU the difference was within run-to-run noise (a scene
    without scans, which the change does not touch, moved 3 % between the two runs).
  - A scan that simulates as a soft body (a soft material, such as the YCB apple and banana) shows its photo too,
    with two limits. Its texture is fixed at the build pose, so a deformed soft body stretches the texture with it.
    And its particle surface has no vertices along the texture's seams, so thin streaks can show there: on the YCB
    apple 4.5 % of the surface triangles straddle a seam, on the banana 2.8 %. Seen head-on at a seam, a test ball's
    streak covers 12 % of it; away from seams the soft ball matches its rigid twin within 4 of 255 levels on average.
  - Primitives, robots and your own meshes have no photo texture and are drawn as before. For a texture on your own
    mesh, see `alloy.camera.meshes.TexturedMesh`.
- A scene variant's item looks (`alloy.vary`: texture swaps and tints) render with the full-material tracer. With
  `camera_scene(sim, quality="fast")` the fast tracer draws them instead, each as its colour, or its colour map tiled
  over the shape (projected along each face's axis, one tile per `scale_m`), without bump, roughness or the full
  lights: a Gym environment's cameras render that way (`docs/guide/envs.md`). In the fast camera a ground plane with
  no colour of its own is grey, also beside a robot whose parts carry their own colours. A variant texture that is
  not on disk is drawn by both tracers in a flat colour of its family (wood brown, plastic grey, ...; a material with
  no family keeps its preset colour), with one warning naming `python -m alloy.looks --fetch-textures`.
- More (materials, lights, mounting a camera on a robot link, scene packs): the docstrings in `alloy/camera/`.

## Depth and segmentation

Ask `sim.render` for a depth image or an id image instead of (or beside) colour. One run of this script builds four
worlds, renders them from above and prints what the images hold:

```python
import numpy as np
import alloy
from alloy.camera import Camera
from alloy.camera.aov import segment_labels

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)             # top at z = 0.4
scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(-0.1, 0, 0.45), mass=0.2)            # on the table, top at 0.5
scene.add_sphere("ball", 0.04, pos=(0.12, 0, 0.46), mass=0.1)
sim = alloy.build(scene)
sim.step(30)                                                    # let them settle

cam = Camera("top", look_at=((0, 0, 1.4), (0, 0, 0), (0, 1, 0)), width=64, height=64, fov=40, spp=4)
out = sim.render(cam, output=("rgb", "depth", "segmentation"))  # a dict of numpy arrays
depth, seg = out["depth"], out["segmentation"]
print(out["rgb"].shape, depth.shape, depth.dtype, seg.shape, seg.dtype)
print("centre depth, world 0: %.3f m" % depth[0, 32, 32])      # the table top, 1.4 - 0.4 m away
names = segment_labels(sim)                                     # names[id] for every segmentation id
print({int(i): names[i] for i in np.unique(seg)})
```

It prints:

```
(4, 64, 64, 3) (4, 64, 64) float32 (4, 64, 64) int32
centre depth, world 0: 1.000 m
{1: 'no_body', 2: 'block', 3: 'ball'}
```

`sim.render(cam, output="depth")` returns one numpy array; a tuple returns a dict of numpy arrays by name (a colour name
in the tuple, `"rgb"`, `"linear"`, `"albedo"` or `"normal"`, also traces colour). `cam.render(sim, output=...)` gives
Warp arrays on the sim's device (one array, or an object with `.rgb`, `.depth`, ...) that the next render overwrites:
copy what you keep. A runnable script that also saves a picture: `examples/camera/depth_segmentation.py`.

| Name | dtype, shape | Meaning | Special values |
|---|---|---|---|
| `"depth"` | float32 `(worlds, H, W)` | metres from the camera to the first surface, along the optical axis (not along the ray) | `inf`: nothing hit |
| `"segmentation"` | int32 `(worlds, H, W)` | 0 nothing hit, 1 a surface that is not on a body (ground, static objects, cloth, liquid), 2 + k the k-th body of its world: the same id in every world | `segment_labels(sim)[id]` names it |
| `"body_id"` | int32 `(worlds, H, W)` | the body's index in the simulation's body arrays | -1 not on a body, -2 nothing hit |
| `"world_id"` | int32 `(worlds, H, W)` | the world the surface belongs to | -2 nothing hit; -1 an instance the model shares between worlds (not in a scene built with `alloy.Scene`) |
| `"instance_id"` | int32 `(worlds, H, W)` | 0 nothing hit, 1 + k the k-th drawn shape of its world: the same id in every world | `instance_labels(sim)[id]` names it |
| `"semantic"` | int32 `(worlds, H, W)` | 0 nothing hit, 1 a shape with no class, 2 + k the k-th class you gave | needs `set_semantics(sim, {...})` first; it returns the names |
| `"flow"` | float32 `(worlds, H, W, 2)` | pixels the surface point moved since this camera's previous flow render: (right, down) | NaN: nothing hit, or not tracked (below) |

Row 0 is the top of the image. Each pixel is one ray through its centre, so the images have no sampling noise and,
for the same simulation state, repeat bit for bit, whatever `spp` and the seed are. RGB images are the same as
without these outputs, and an output with no colour name traces no colour at all, so a depth-only render costs less
than an RGB one (Speed, below). These outputs run on the CPU and on a GPU.

A scene built with `alloy.Scene` gives every world its own ground and static objects, so their pixels read that
world's index in `"world_id"`, never -1. To pick surfaces that are not on a body, use `segmentation == 1` or
`body_id == -1`.

The first render in a new install compiles the camera's kernels: in one cold-install test (one run each) it took
44 s on a 14-core CPU and 1 min 45 s on one L4. They are cached, so later runs start in seconds.

### Depth to a point cloud

Pixel `(row, col)` looks along the direction `(x, y, 1)` in the camera frame (x right, y up, z forward), with
`x = u * hx`, `y = v * hy`, where `(u, v)` runs from -1 to 1 across the image and `(hx, hy)` are the half extents of
the image plane at distance 1 from `alloy.camera.pose.image_extent(fov, width, height)`. Depth is the z of the point, so
the point is `(x * depth, y * depth, depth)`. Then the camera's rotation and eye place it in the world:

```python
import numpy as np
import alloy
from alloy.camera import Camera
from alloy.camera.pose import image_extent

scene = alloy.Scene(worlds=1, dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(-0.1, 0, 0.45), mass=0.2)
sim = alloy.build(scene)
sim.step(30)
eye, target, up = np.array([0.3, -0.2, 1.4]), np.array([0.0, 0.0, 0.4]), np.array([0.0, 0.0, 1.0])
W = H = 64
FOV = 40.0
cam = Camera("view", look_at=(tuple(eye), tuple(target)), width=W, height=H, fov=FOV, spp=4)
depth = sim.render(cam, output="depth")[0]                       # (H, W), metres along the optical axis

hx, hy = image_extent(FOV, W, H)                                 # half extents of the image plane at z = 1
u = (np.arange(W) + 0.5) / W * 2 - 1                             # pixel centres, -1 (left) .. +1 (right)
v = 1 - (np.arange(H) + 0.5) / H * 2                             # +1 (top row) .. -1 (bottom row)
x, y = np.meshgrid(u * hx, v * hy)                               # ray direction (x, y, 1) per pixel
pts = np.stack([x * depth, y * depth, depth], axis=-1)           # camera frame: x right, y up, z forward
pts = pts[np.isfinite(depth)]                                    # drop pixels that hit nothing

f = (target - eye) / np.linalg.norm(target - eye)                # camera forward in the world
r = np.cross(f, up) / np.linalg.norm(np.cross(f, up))            # right
R = np.stack([r, np.cross(r, f), f], axis=1)                     # columns: right, up, forward
world = eye + pts @ R.T                                          # (N, 3) points in world coordinates
top = world[(abs(world[:, 0] + 0.1) < 0.04) & (abs(world[:, 1]) < 0.04) & (world[:, 2] > 0.45)]
print(len(pts), "points; block top: z = %.3f m (expect 0.5), x = %.3f (expect -0.1)" % (top[:, 2].mean(), top[:, 0].mean()))
```

It prints `4096 points; block top: z = 0.500 m (expect 0.5), x = -0.099 (expect -0.1)`. The `look_at` here has the world
z axis up; the three lines that build `R` repeat the camera's own rule (right = forward x up), so they hold for any
`eye`, `target` and `up`.

### Torch, for training

`cam.render_torch(sim, output=("rgb", "depth", "segmentation"))` returns a dict of torch tensors by name, on the sim's
device: CUDA tensors on a GPU, without a copy through the host, and CPU tensors on a CPU machine. It needs PyTorch
(`pip install torch`, or the `envs` extra, which brings it); the rest of this page does not.

```python
import alloy
from alloy.camera import Camera
import torch  # optional: only render_torch needs PyTorch

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_sphere("ball", 0.04, pos=(0, 0, 0.5), mass=0.1)
sim = alloy.build(scene)
cam = Camera("top", look_at=((0, 0, 1.4), (0, 0, 0), (0, 1, 0)), width=64, height=64, spp=4)
sim.step(1)
obs = cam.render_torch(sim, output=("rgb", "depth", "segmentation"))   # dict of tensors, one entry per name
print({k: (tuple(v.shape), v.dtype) for k, v in obs.items()})
```

The tensors are the camera's own buffers: the next render overwrites them, so `.clone()` what you keep.

### Instance ids, classes and optical flow

Three more outputs, by the same pixel-centre ray. `"instance_id"` tells shapes apart, static ones too (in
`"segmentation"` the ground and the table both read 1). `"semantic"` gives each shape the class you name for it.
`"flow"` says how far each surface point moved in the image since the last time you asked this camera for flow.

```python
import numpy as np
import alloy
from alloy.camera import Camera
from alloy.camera.aov import instance_labels, set_semantics

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)             # top at z = 0.4
scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(-0.1, 0, 0.45), mass=0.2)
scene.add_sphere("ball", 0.04, pos=(0.12, 0, 0.44), mass=0.1)
sim = alloy.build(scene)
sim.step(30)
cam = Camera("top", look_at=((0, 0, 1.4), (0, 0, 0), (0, 1, 0)), width=64, height=64, fov=40, spp=4)

shapes = instance_labels(sim)                                    # shapes[id] for every instance id
classes = set_semantics(sim, {"block": "toy", "ball": "toy", "table": "furniture"})   # classes[id]
out = sim.render(cam, output=("instance_id", "semantic", "flow"))                     # the first flow reads 0
print({int(i): shapes[i] for i in np.unique(out["instance_id"])})
print({int(i): classes[i] for i in np.unique(out["semantic"])})

x = sim.state(world=0)["block.pos"]
sim.set_pose("block", pos=(x[0] + 0.02, x[1], x[2]))             # move the block 2 cm along +x (image right)
out = sim.render(cam, output=("instance_id", "flow"))
flow = out["flow"]                                               # (worlds, H, W, 2): pixels moved, (right, down)
block = out["instance_id"][0] == shapes.index("block")
print("block: %.2f px right, %.2f px down" % tuple(np.median(flow[0][block], axis=0)))
print("table: %.2f px" % np.abs(flow[0][out["instance_id"][0] == shapes.index("table")]).max())
```

It prints:

```
{1: 'ground', 2: 'table', 3: 'block', 4: 'ball'}
{1: 'unlabeled', 2: 'toy', 3: 'furniture'}
block: 1.95 px right, 0.00 px down
table: 0.00 px
```

- **Instance ids.** One id per drawn shape, in the scene's order, the same in every world. A body with several
  shapes (a robot link with two geoms, the five boxes of an `add_open_box`) has one id per shape; a robot's shapes
  carry the labels of its file (`instance_labels(sim)` lists them).
- **Classes.** `set_semantics(sim, {pattern: class, ...})` takes names or `fnmatch` patterns (`*`, `?`,
  case-sensitive). A key matches a shape by its own label (the names `instance_labels` returns) or by a name of
  the body it is on (`sim.names()["bodies"]`), so `"arm/*"` labels every link of the robot `arm` and `"bin_*"` the
  parts of an open box called `bin`. The first key that matches wins; shapes no key matches read 1. It returns
  `["background", "unlabeled", <your classes in the order given>]`. Calling it again replaces the classes; asking
  for `"semantic"` before calling it raises `CameraError`.
- **Flow.** `flow[w, row, col] = (dx, dy)` in pixels, x to the right and y down: the surface point now at that pixel
  was at `(col - dx, row - dy)` in the previous render of this camera that asked for `"flow"`. Each camera keeps
  its own previous poses, and only a render that asks for flow moves them on, so flow spans whatever happened in
  between: steps, `set_pose`, a `sim.reset()` (then it reads the jump back to the start), or a camera you moved.
  The first flow render of a camera reads 0. It is computed from the poses of rigid bodies (robot links and
  kinematic objects included), static shapes and the camera, not from images: exact for rigid motion up to float32
  rounding (on the CPU, the tests hold it within 0.002 px on 16-pixel images).
- **Where flow is NaN.** Pixels that hit nothing; cloth, soft bodies, films and liquid (their surface motion is not
  tracked); and a point that was behind the camera at the previous render. `np.isfinite(flow[..., 0])` is the valid
  mask. A point that was hidden or outside the image before still gets a value (where it was), with no occlusion
  flag.

### Speed

Colour images, at the camera's image-quality settings (for each scene the fewest samples, with the learned denoiser,
that pass its quality gate against reference renders), 128 x 128 pixels:
- render only, 1,024 worlds, one RTX PRO 6000: 1,182 images per second on a small object scene (5 samples per pixel)
  and 1,258 on a two-arm robot scene (40 samples per pixel);
- with physics stepping between renders: the robot scene 117 images per second (512 worlds, one L4) and 685 (1,024
  worlds, one RTX PRO 6000); the object scene 181 and 1,202. So a robot scene with physics in the loop is under 1,000
  images per second today. A scene with glass and liquid has no setting that passes the gate yet, so no speed.

Depth and segmentation: one run on one NVIDIA L4 (24 GB), 1,024 worlds, one camera per world, render only (no physics
step between renders), no colour traced, after a warm-up; images per second at 64 x 64 / 128 x 128 / 256 x 256
pixels. These rates are not comparable to a colour image rate (one ray per pixel, no shading, no denoiser), and they
are provisional: one run, not yet reproduced by a second measurement and not measured with physics in the loop.

| Output (no colour traced) | Franka arm at a table (`emb_e2_panda`) | Go2 on a plane (`emb_e8_go2`) |
|---|---|---|
| `"depth"` | 70,786 / 16,111 / 4,133 | 327,866 / 91,974 / 26,092 |
| `("depth", "segmentation")` | 70,477 / 16,093 / 4,129 | 318,572 / 89,975 / 25,563 |

In the same run, adding depth and segmentation to a colour render cost 4-7 % at `spp=4` and 14-22 % at `spp=1`, and
depth alone ran 14-24 times faster than colour at `spp=4`. Those colour renders were raw `spp=1` and `spp=4` without
the denoiser, below the image-quality settings above, so their rates are not camera speeds to quote. The first call
compiles one more kernel (2.3 s here). Not measured: depth with physics in the loop, the CPU, the OptiX backend, other
GPUs, and the spread between runs.

### Not covered

- A full-material camera (`quality="full"`, `look=`, scene packs of version 2, or a sim whose items carry looks from
  `alloy.vary`) raises `CameraError` for these outputs.
- With a live liquid on the OptiX backend these outputs raise `CameraError`: use the default `backend="warp"`.
- `sim.render()` without a Camera (the quick look) gives colour only; `sim.render(output="depth")` without a Camera
  raises `CameraError` telling you to pass one.
- Transparent surfaces (glass, liquid) are hit, not seen through: the depth is the glass surface.
- No anti-aliasing: the edges of depth and id images are hard, while RGB edges are filtered.
- Cloth, liquid, soft bodies and films have no `"segmentation"` id of their own: they read 1, like static objects,
  and their `"flow"` is NaN. In `"instance_id"` each has an id of its own, named by its label, and in `"semantic"`
  it reads 1 unless a key matches that label (checked for cloth; not yet for liquid, soft bodies and films).
- `cam.render_torch` returns `"instance_id"`, `"semantic"` and `"flow"` like the other outputs (`"flow"` as
  `(worlds, H, W, 2)`). No speed is quoted for these three outputs yet.
- A cable that steps as a chain is rigid links: one instance id per link (labelled `shape_0`, ...), classed by the
  cable's name (`"cord*"`), with flow. A clamped cable (`rod_vbd`) was not checked.
- No flow for deforming surfaces, no occlusion or disocclusion mask, and no forward flow (to the next frame).
- Classes are set per simulation after the build, by name; they are not part of a scene file or an object card.
- A robot is not one id: each link is a body with its own id.
- No sensor model: no noise, and no range limit other than the camera's `clip` distances (0.01 m to 1000 m by
  default; a surface outside them reads as nothing hit).
- In `"segmentation"` static objects share one id (the ground, the table and every other static object all read 1;
  `"instance_id"` tells them apart). No lens distortion and no intrinsics matrix (the pinhole `fov` only).
- Environment observations (`alloy.envs` cameras) are still RGB only.
