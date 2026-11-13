# Tactile images (GelSight / DIGIT style)

`alloy.camera.tactile` renders what the camera inside a vision-based tactile sensor sees: the gel pressed by the
objects it touches, coloured by a calibrated look-up table, with marker dots that move as the gel shears. Every world
is rendered at once on the GPU (CPU works, slowly).

```python
from alloy.camera.tactile import Augment, TactileSensor

sensor = TactileSensor("left_gel", attach_to="gripper/finger_left",
                       pose=((0.0, -0.005, 0.0), (0.7071068, 0.0, 0.0, 0.7071068)),   # gel on the inner face
                       preset="gelsight", width=320, augment=Augment.strong(seed=0))
img = sensor.render(sim)                       # wp.array uint8 (worlds, 240, 320, 3)
img_t = sensor.render_torch(sim)               # the same as a torch tensor on the GPU
out = sensor.render(sim, output="all")         # + out.press (worlds, H, W) mm, out.markers, out.motion
sensor.reset(worlds=done, seed=episode)        # after a (masked) world reset: those worlds' touches and augmentation
```

- **Pose:** the gel surface centre in the `attach_to` body frame; +Z points out of the gel toward what it touches,
  +X is image right, +Y image down (the view of the camera behind the gel).
- **Presets:** `"gelsight"` (18.9 x 14.2 mm, markers) and `"digit"` (17.0 x 12.8 mm, no markers), from the
  MIT-licensed Taxim calibrations. `markers=True/False` overrides; `width=` sets the size (aspect kept).
- **Depth:** `out.press` is the depth image in the sensor frame: how far each touching shape reaches into the
  undeformed gel (mm), ray-cast exactly against the collision shapes. Cloth and soft-body surfaces, MPM grains
  and liquid particles press too, from their particles (`deformables=False` turns that off). Give the pad a soft
  contact (MJCF `solref`, e.g. `"0.03 1"`) so objects sink about as far as into a real gel; a stiff pad gives faint
  images.
- **Markers:** pushed outward by the press and dragged by the touching object (stick, then slip past `slip_mm`).
  An engine's own marker field can drive them instead (`TactileRenderer.render(..., marker_disp=...)`).
- **Augmentation:** `Augment.strong(seed)` randomises lighting, gel colour, tone, sensor offset, noise, worn spots
  and missing markers per world (key for sim-to-real, per TacSL).
- **No simulation needed:** `TactileRenderer("digit", width=160, num_worlds=N).render(press_mm)` turns any
  (N, H, W) press map in mm into images.

Speed on one L4, 1,024 worlds, full pipeline: about 35,700 images/s at 160 x 120, 8,700 at 320 x 240 and 2,200
at 640 x 480 (`python -m alloy.camera.tactile.bench --worlds 1024`).

Engine gel fields (the `alloy.plugins.tactile.GelSensor` plug-in: the engine's own gel penetration and marker
displacements) render through `EngineTactile.from_gel_sensor(gel).render(gel.outputs())`
(`alloy.camera.tactile.engine`; example: `python examples/tactile/engine_gel.py`).
Check against real sensor images: `python -m alloy.camera.tactile.validate --pack <Taxim>/calibs/dataPack.npz`.

Run the example: `python examples/tactile/gripper_touch.py --out tactile_gripper` (a gripper squeezes a rod; the
left finger's gel image is saved as a PNG strip).
