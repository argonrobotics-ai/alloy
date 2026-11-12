# Alloy render: ARB v1 bundles, look files, and the Cycles CLI

This document is the contract between clip builders and the renderer. A builder runs Alloy physics and writes an
**ARB bundle** with `alloy.render.RenderRecorder`. It then writes a **look file** with the camera, lights and
materials, and runs `python -m alloy.render.cli render`. The CLI renders the bundle in Blender Cycles on spot L4 VMs
and returns PNG frames and an h264 mp4.

- **Physics stays physics.** The renderer never moves an object.
  - It interpolates between recorded frames only for motion blur sub-steps, within one frame.
  - It meshes liquid particles into a surface.
  - It smooths and subdivides cloth or cable meshes for display.
  - All of this is display only (SC honesty rule 2).
- **Blender runs as a separate process** (`blender -b -P ...`). Alloy code never imports `bpy`, which is GPL. The
  Blender-side scripts in `alloy/render/blender/` are standalone and read only the bundle and the look file.
- **Pins:** Blender 4.5 LTS with Cycles on OptiX; the VM image is `common-cu129-ubuntu-2204-nvidia-580`.

## 1. Producing a bundle

```python
from alloy.render import RenderRecorder

rec = RenderRecorder(model, "out/c5_take07", fps=120.0, worlds=[3], meta={"clip": "C5", "take": 7})
rec.add_points("honey", particle_ids)                     # name a particle set (SPH liquid, MPM grains, ...)
rec.add_curve_from_bodies("harness", r"harness/seg_\d+")   # a CHAIN cable rendered as one smooth tube
rec.add_curve_from_particles("thread", particle_ids)       # a particle rod, in order
rec.add_instancer("rice", r"rice_\d+")                     # many one-shape bodies -> one instanced object
for k in range(n_frames):
    ...                                                    # step physics to time t
    rec.capture(state, t)                                  # copies device -> host once per captured frame
rec.close()                                                # writes scene.json + frames/*.npy
```

- **`fps`** is the capture rate. Capture at the rate you want to render; the look file picks every Nth frame. To
  play a 120 fps sim at 30 fps, use `frames.step` = 4 for real time and 1 for 4× slow motion.
- **`worlds`** lists the worlds to export. Each exported world is placed at its own origin, so pick the best take of
  your 16 worlds and export only that one. For a multi-world grid shot, pass several worlds and `grid_m=spacing`.
- **Auto-detection runs in `close()`** for anything not declared explicitly:
  - **Shapes.** Every shape with `ShapeFlags.VISIBLE` and not `SITE` becomes a shape. Build with
    `parse_visuals=True` so robots get their visual meshes. Collision-only proxies are skipped, except that an
    object with no visual geom at all is exported with its collision shapes, flagged `collision_fallback`.
    MJCF tables and floors are often only collision boxes.
  - **Surfaces from triangles.** Each connected component of `model.tri_indices` (VBD cloth, sheets, cardboard)
    becomes a surface `cloth_<k>`.
  - **Surfaces from tetrahedra.** The boundary faces of each connected component of `model.tet_indices` (soft
    solids, a loaf) become a surface `soft_<k>`.
  - **Leftover particles.** Particles not covered by the above or by `add_*` become points `particles_<k>`, one set
    per world.
- **Leftover particles.**
  - Pass `leftover_points=False` to drop every particle no `add_*` set covers. Use it when a grain instancer draws 1
    in 8 MPM particles and the rest would add GBs.
  - `rec.skip_particles(ids)` drops specific ids from auto-detection.
  - Only particles that some exported set reads are copied each frame.
- **Cuts and tears** need nothing extra. Alloy's topology is fixed: cut lines are pre-split, with duplicated
  vertices joined by twin bonds (`alloy.topology`).
  - The recorder tags a surface face `cut` (tag 1) when all its vertices are split twins, so a loaf's crumb can get a
    different material from its crust. A builder may re-tag cut faces 2, 3, ... (e.g. rim, flesh, seed core, seeds);
    every tag >= 1 counts as cut for the weld, and the tag picks the surface's material list entry.
  - The renderer welds twins that are still within 0.02 mm, so an intact seam shows no crease.
- **Labels are how materials attach.** Give every shape and body a meaningful label (`newton` `shape_label` /
  `body_label`; MJCF geom and body names). Name every particle set and surface you declare.
- **Size.** Frames are float32, stored as uncompressed `.npy` files so they can be memory-mapped. For example, 200k
  SPH particles × 450 frames come to about 1 GB of positions plus 1 GB of velocities. The recorder keeps captures in
  host RAM until `close()`.

## 2. Bundle layout (ARB v1)

```
<bundle>/
  scene.json                  static description (below)
  frames/t.npy                [T] float64, capture time [s]
  frames/body_q.npy           [T, B, 7] float32, world pose per exported body: px py pz qx qy qz qw (Warp order)
  frames/surf_<name>.npy      [T, V, 3] float32, surface vertex positions [m]
  frames/pts_<name>.npy       [T, N, 3] float32, point positions [m]
  frames/vel_<name>.npy       [T, N, 3] float32, point velocities [m/s] (points only)
  frames/curve_<name>.npy     [T, K, 3] float32, centreline points [m], in order
  frames/inst_<name>.npy      [T, K, 7] float32, instance poses (pose of the source body)
  meshes/<id>.npz             vertices [V,3] f32 (shape-local, before scale), faces [F,3] i32, optional uv [V,2] f32
  surfaces/<name>.npz         faces [F,3] i32 (local), src [V] i32 (model particle ids), face_tag [F] u8, twins [P,2]
```

`scene.json` has these fields. All poses are in world space [m]. Exported world `w` is shifted by
`world_offsets[w]`.

| Field | Content |
|---|---|
| `arb` | `1` |
| `meta` | Builder `meta`, plus `alloy_commit`, `created`, `device`, `world_count` of the source model |
| `frames` | `{"count": T, "fps": f}` |
| `worlds` | Exported world ids, with `world_offsets` as a list of `[x, y, z]` |
| `bodies[]` | `{"id", "label", "world"}`. `id` is the column in `body_q` |
| `shapes[]` | `{"id", "label", "body" (-1 static), "world", "type", "size", "xform", "scale", "mesh", "color", "opacity"}`. `type` is one of `box sphere capsule cylinder cone ellipsoid plane mesh`. `size` follows Newton's `shape_scale`: box half-extents; sphere `[r]`; capsule, cylinder and cone `[r, half_height]` along local Z; ellipsoid semi-axes; plane `[width, length]` with 0 meaning infinite. `xform` is the shape pose in the body frame. `color` is sRGB 0-1, from `shape_color` |
| `meshes{}` | `{id: {"file", "verts", "faces", "uv"}}`. Identical meshes are stored once |
| `surfaces[]` | `{"name", "kind": "cloth or soft", "world", "file", "verts", "faces", "tags": ["outer", "cut"]}` |
| `points[]` | `{"name", "world", "count", "radius"}`. `radius` is the median `particle_radius` |
| `curves[]` | `{"name", "world", "count", "radius", "from": "bodies or particles"}` |
| `instancers[]` | `{"name", "world", "count", "mesh" or "shape_type" + "size", "xform"}`. `xform` is the source shape in its body |

## 3. Look file (`look.json`, written per clip by the builder)

```json
{
  "look": 1,
  "render": {"resolution": [1080, 1920], "samples": 512, "noise_threshold": 0.01, "denoiser": "OIDN",
             "view": "AgX", "look": "AgX - Medium High Contrast", "exposure": 0.0,
             "motion_blur": {"shutter_s": 0.0167}, "bounces": {"total": 12, "transmission": 16, "volume": 2},
             "caustics": {"reflective": false, "refractive": true}, "film_transparent": false},
  "frames": {"start": 0, "end": -1, "step": 4},
  "cameras": {
    "vertical":   {"lens_mm": 85, "sensor_mm": 36, "fstop": 2.8, "focus": {"label": "nut"},
                   "keys": [{"t": 0.0, "eye": [0.35, -0.42, 0.30], "target": [0.0, 0.0, 0.08]},
                            {"t": 12.0, "eye": [0.33, -0.40, 0.29], "target": [0.0, 0.0, 0.08]}]},
    "horizontal": {"resolution": [1920, 1080], "lens_mm": 60, "fstop": 4.0, "focus": {"distance": 0.45},
                   "keys": [{"t": 0.0, "eye": [0.5, -0.5, 0.35], "target": [0.0, 0.0, 0.06]}]}
  },
  "world": {"hdri": "polyhaven:brown_photostudio_02", "strength": 1.0, "rotation_deg": 40, "show_hdri": false,
            "background_color": [0.02, 0.02, 0.02]},
  "lights": [{"type": "area", "eye": [0.4, 0.3, 0.6], "target": [0, 0, 0.05], "size_m": 0.6, "watts": 8,
              "kelvin": 5200}],
  "materials": {
    "table":  {"preset": "marble_white", "texture": "polyhaven:marble_01", "scale_m": 0.6},
    "honey":  {"preset": "honey"},
    "nut":    {"preset": "zinc_plated", "roughness": 0.3}
  },
  "assign": [{"match": "^table", "material": "table"}, {"match": "honey", "material": "honey"},
             {"match": "nut", "material": "nut"}],
  "liquids":   {"honey": {"voxel_m": 0.0006, "radius_scale": 1.6, "smooth": 2, "material": "honey"}},
  "points":    {"rice_mpm": {"instance": "rice_grain", "scale": 1.0, "material": "rice"}},
  "instances": {"rice": {"material": "rice"}},
  "curves":    {"harness": {"material": "pvc_cable_black", "resolution": 3}},
  "surfaces":  {"towel": {"subdivide": 1, "thickness_m": 0.003, "materials": ["cotton_terry"]},
                "loaf":  {"materials": ["bread_crust", "bread_crumb"]}},
  "hide": ["_collision$"]
}
```

- **`assign`.** First match wins. Each regex (Python `re.search`) is tried against the shape label, then the body
  label, then the surface, points, curve or instancer name. An unmatched shape gets `plastic_matte` tinted with its
  `color`. `cli check` lists every unmatched label.
- **`cameras`.** Pick one with `--camera <name>`.
  - **`keys`** are time-keyed eye and target positions. A Catmull-Rom spline runs through them, with times in bundle
    seconds; one key gives a still camera. Use it for the slight drift.
  - **`focus`** is one of `{"label": regex}` (follows that shape or body's origin), `{"point": [x, y, z]}` or
    `{"distance": m}`.
  - **`fstop`** sets depth of field; `"fstop": 0` turns it off.
  - **`sensor_mm`** is the sensor width (36 = full frame).
- **`render.motion_blur.shutter_s`** is the exposure in *bundle* seconds, so blur matches real motion at any `step`.
  For example, 1/60 s is a 180° shutter at 30 fps real time. In slow motion, use the shutter of the high-speed
  camera you are imitating (1/240 s at 4×). Every object is sub-stepped between recorded frames; liquids blur from
  their velocities. Set it to `null` to turn blur off.
- **`liquids.<points name>`** meshes a point set into a smooth surface per frame:
  - Points become a level set (`radius_scale` × particle radius, voxel `voxel_m`).
  - Then `smooth` passes of mesh smoothing.
  - Then the material.
  - The voxel should be about half the particle spacing or smaller.
- **`liquids.<name>.contact`** handles liquid in a container (water in a glass, honey in a cup).
  `[{"match": "<container shape/body/skin regex>", "distance_m": 1.5 r (default), "overlap_m": 0.0002}]`.
  - Liquid vertices within `distance_m` of the container are snapped onto its surface and `overlap_m` into the wall,
    fading out between 1× and 2× the distance. This removes the air gap and the crumpled refraction through the
    glass.
  - Smoothing then runs, and the wet vertices are snapped again.
  - Moving containers work, with each motion step re-meshed (about +6 s/frame).
  - A rule that matches nothing is an error.
  - The free top surface keeps particle-scale ripples; raise `smooth` or `radius_scale` with `erode` if they show.
- **Grains.**
  - **`instances.<instancer>`** renders bodies collected with `add_instancer`. Their shape geometry is instanced at
    each body pose.
  - **`points.<name>.instance`** places a preset grain mesh (`rice_grain`, `sand_grain`, `sphere`) at every point,
    with a stable random rotation per point id.
    - `"rotation": "flat"` gives a random yaw with tilt within `max_tilt_deg` (default 15), for grains lying in a
      bed.
    - `"random"` (the default) allows any orientation.
- **Surfaces.**
  - `subdivide` applies render-time Catmull-Clark levels.
  - `thickness_m` gives a solidified shell for cloth and card.
    - The offset is clamped to the local edge length at sharp folds (`thickness_clamp`, default 1.0).
    - Even offset is off for surfaces by default and on for skins. `"even_offset": true` turns it on, but it throws
      spikes at folds flattened to about 180°.
  - `materials[i]` is used for face tag `i` (0 outer, 1 cut).
- **`bevels`** (display only; collision stays sharp): `[{"match": regex, "bevel_m": 0.0003, "segments": 3}]`.
  - Each matched box, cylinder or mesh shape gets rounded edges, angle-limited at 30°. The regex is tried against the
    shape label, then the body label, and the first match wins.
  - Moulded plastic reads as real at 0.2–0.5 mm; machined metal at 0.05–0.2 mm.
- **`skins`** (display only): one rest-pose mesh that follows N rigid bodies by blend skinning. For example, a
  snap-fit lid made of 49 rigid strips renders as one smooth shell, the way a cable tube does.
  ```json
  "skins": {"lid": {"mesh": "extra/lid_rest.obj", "weights": "extra/lid_weights.json", "rest_frame": 0,
                    "material": "pp_clear", "thickness_m": 0.0007, "subdivide": 0, "hide_bodies": true}}
  ```
  - **Paths** are relative to the bundle root. Put the files in `<bundle>/extra/` so the farm syncs them.
  - **The OBJ** is in world coordinates of the exported world at `rest_frame`, before the grid offset.
  - **Weights** are JSON `{"bodies": [body labels], "weights": [[[body_index, w], ...] per vertex]}`, or `.npz` with
    `labels [N]`, `idx [V,K]`, `w [V,K]`. Weights are normalised.
  - **Per frame and motion-blur sub-step:** `v = Σ w_i · T_i(t) · T_i(rest)⁻¹ · v0`, with T from `body_q`.
    Dual-quaternion blending is used if linear blending collapses at sharp bends.
  - **`hide_bodies`** (default true) hides every shape of the listed bodies.
  - **`thickness_m`** solidifies the shell, so clear plastic refracts through two surfaces.
- **`render.white_balance`** controls the colour cast.
  - **`"auto"` (default)** renders a small 18 % grey ball at the camera target under the scene's HDRI and lights,
    then sets Blender's white balance so that ball is neutral. This works like camera auto white balance and adds
    about 3 s per job.
  - **`{"kelvin": K, "tint": t}`** sets it by hand; tint 10 is daylight.
  - **`"off"`** disables it.
  - Warm HDRIs and lights below 6500 K otherwise read orange. The applied value is in `timing.jsonl`.
- **`ground`** (`{"z": -0.7527, "material": "floor", "size_m": 4000, "center": [x, y]}`) adds one render-only
  plane under every world, centred on the world grid unless `center` is given. In a multi-world grid the
  per-world floors overlap and shadow each other (black patches), so hide them (`hide: ["/floor"]`) and use
  this; with `world.fade` on the same material it dissolves into the background with no horizon line.
- **`props`** add static, render-only set dressing (out-of-focus backgrounds).
  ```json
  "props": [{"model": "polyhaven:wooden_bowl_01", "pose": [0.4, 0.3, 0, 0, 0, 0, 1], "scale": 1.0, "res": "1k"},
            {"model": "file:extra/jar.glb", "pose": [...], "material": "glass_clear"}]
  ```
  - Poly Haven glTF models are CC0 and cached. `file:` paths are bundle-relative `.glb`, `.gltf` or `.obj`.
  - A failed download logs a warning and skips the prop.
- **Textures replace the preset's base colour.** The preset's other values and the look's overrides stay. For
  example, `"roughness"` replaces the texture's roughness map. Add `"tex_tint": true` to keep `base_color` and use
  the texture only for light/dark variation.
- **`color_gain`** scales a material's final base colour (texture or not): `0.707` darkens it by half a stop, e.g. a
  floor whose light pool reads hot.
- **Hero models (display only)** go on the same sim bodies. They replace the proxies in the picture and change no
  physics.
  - **Skin meshes** (`skins.<n>.mesh`) may be `.obj`, `.glb` or `.gltf`.
    - A glTF keeps its own PBR materials, UVs and textures. `material_map {"<glTF material>": "<look material>"}`
      replaces named ones.
    - OBJ `usemtl` groups map through `materials {"<usemtl>": "<look material>"}`.
  - **Rigid attach:** `skins.<n>.body = "<label>"` means no weights file is needed. The mesh is in world coordinates
    at `rest_frame`, or in the body's frame with `"local": true`.
  - **Weights files** may carry `points` (`[[x,y,z], ...]`). Each mesh vertex then takes the weights of its nearest
    point, which helps for glb files whose vertex order you don't control.
  - **Grid replicas.** A skin replaces its bodies' shapes in the hero world only. `skins.<n>.replicas` sets what the
    other worlds of a `worlds="all"` grid show for those bodies: `"proxy"` (default: their own proxy shapes),
    `"skin"` (this skin's mesh, instanced per replica body), `{"mesh": "extra/part_lod.glb"}` (a lighter mesh in the
    same frame as `mesh`, e.g. ~1,200 triangles per world) or `"none"` (drop them). `"skin"` and `{"mesh"}` need a
    rigid `body` skin.
- **Own art.** `materials.<m>.texture = "file:extra/print.png"` (PNG/JPG/EXR, bundle-relative) is UV-mapped by
  default. `"texture_alpha": true` lays it as a decal over `base_color`.
- **Planar projection.** `"mapping": "planar"` projects the image along `planar_axis` instead, for surfaces with
  no UVs (deforming surfaces, cut faces):
  - The image is centred on `planar_origin`, `scale_m` across, with its v (up) direction along `planar_up` projected
    into the plane. `planar_up` defaults to +z, or +y when the axis is near z. Point `planar_axis` out of the printed
    face: the image reads unmirrored seen from that side.
  - Coordinates are the texture space: world metres of the first frame (`rest_p`) on deforming surfaces, so the
    print sticks to the deforming mesh; object space on rigid shapes.
  - Beyond the image the edge pixels repeat. Faces parallel to the axis get streaks, so assign the material only to
    the faces that face the axis (e.g. a per-face-tag material).
  - Example: `{"preset": "dough", "texture": "file:extra/slice.png", "mapping": "planar",
    "planar_axis": [1, 0, 0], "planar_origin": [0.42, 0.0, 0.81], "scale_m": 0.04}`.
- **Cable UVs.** `curves.<n>.uv_scale_m` (default 0.01) sets the u extent: u = arc length from the first frame /
  `uv_scale_m`, and v runs around the tube. `uv_twist` sets turns per metre.
  - A jacket print repeats along the cable and does not swim. It can still roll with cable twist, because only the
    centreline is recorded.
  - Tube ends are capped.
- **Keyed material values.** `materials.<m>.keys = [{"t": s, "emission": [r,g,b], "emission_strength": x, ...}]`
  with `"interp": "constant" | "linear"`.
  - Keyable: emission, emission_strength, alpha, transmission, metallic, coat, sheen, subsurface, ior, specular,
    anisotropic.
  - Use the event time from the sim (e.g. the latch click) so the LED is honest.
- **`render.glare`** gives lens bloom on emitters: `{"type": "fog_glow" | "bloom", "threshold": 1.0, "size": 6-9,
  "strength": 1.0}`.
- **`materials.<m>.imperfection = "ambientcg:<id>"`** (e.g. `Fingerprints002`) varies roughness at
  `imperfection_scale_m` (0.1) and `imperfection_strength` (0.3), for macro shots.
- **Macro presets:** polycarbonate_clear, gold_plated, copper_wire, pvc_jacket_matte, abs_switch_housing, led_green,
  led_amber. At 100 mm, f/5.6 and 0.1 m, depth of field is about 2 mm, so focus on the contact.
- **Textures** are CC0 only: `polyhaven:<id>` and `ambientcg:<AssetID>`.
  - They are fetched on the VM into a cache and mapped by triplanar box projection in metres (`scale_m` = tile size).
  - Meshes with UVs use them when the preset has no texture.
- **Body-chain cables.** Shapes whose `scene.json` entry has `"curve": <name>` belong to a body-chain cable. They are
  hidden, because the tube replaces them, and `check` does not ask for a material for them.
- **Presets** are listed in `alloy/render/blender/presets.py`. Any of `base_color` (sRGB), `roughness`, `metallic`,
  `transmission`, `ior`, `subsurface` (+ `subsurface_radius_m`), `sheen`, `coat`, `anisotropic`,
  `absorption_color` + `absorption_m` (liquids and glass) and `emission` overrides a preset value.

- **Units and ranges.**
  - `watts` is Blender light power. About 5–10 W at 0.5 m suits exposure 0.
  - `sensor_mm` is the sensor width along the image's long side.
  - `frames.end` is inclusive, and negative values count from the end. The CLI's `--frames A:B` excludes B.
- **Extra keys** the Blender side accepts:
  - `liquids.<name>.erode`, which shrinks the level set after smoothing;
  - `instances.<name>.instance`, which replaces an instancer's shape with a preset grain mesh;
  - `skins.<name>.blend` (`"dqs"` by default, or `"lbs"`);
  - `curves.<name>.radius_m`;
  - camera `blades` (aperture shape);
  - `world.res` (HDRI resolution, `"2k"`/`"4k"`);
  - `world.background_from_hdri` (`false` by default): with `show_hdri` off, camera rays see a flat colour as
    bright as the environment's mean, instead of `background_color`. A large ground plane's horizon then fades into
    the background, where a fixed `background_color` leaves a hard line (exposure moves with the lights);
  - `world.fade` (`{"material": "floor", "start_m": 110, "depth_m": 500}`, needs `show_hdri` off): to camera
    rays, that material turns transparent between `start_m` and `start_m + depth_m` from the camera (smoothstep),
    so a large ground plane dissolves into exactly the background (no horizon line, whatever the exposure or white
    balance); light, shadow and reflection rays still see it. Keep `start_m` beyond the farthest object that must
    stay crisp; the camera's far clip (200 m) moves out past the fade's end;
  - any Principled override on a material.
- **Liquid motion blur** comes from a per-vertex `velocity` attribute sampled from the particles, so the surface is
  meshed once per frame.

## 4. CLI

```
python -m alloy.render.cli check  <bundle> --look look.json [--camera NAME]
python -m alloy.render.cli still  <bundle> --look look.json --camera NAME --frame K [--samples 128] [--vm NAME]
python -m alloy.render.cli render <bundle> --look look.json --camera NAME --out DIR [--vms 4] [--frames A:B]
                                  [--samples S] [--zone Z] [--keep-vms] [--reuse-vm NAME ...]
                                  [--region-cap 8] [--stage gs://BASE | --no-stage] [--frames-gcs gs://PREFIX|auto]
python -m alloy.render.cli encode DIR|gs://PREFIX --fps 30 [--crf 16] [--out clip.mp4]
```

- **`check`** validates the bundle and look file without a GPU. It prints frame count, time span, entity counts,
  unmatched labels and estimated render cost, and exits with code 1 on any error.
- **`still`** renders one frame, for look development. It reuses `--vm` if given, otherwise it starts one spot L4 and
  leaves it up for 30 min idle.
- **`render`** splits the frame list across `--vms` spot L4 VMs, named `alloy-render-<bundle>-<k>` with labels
  `purpose=alloy, role=wp20-render`.
  - Each VM gets the bundle and assets by rsync. It renders its contiguous range on OptiX with persistent data, then
    sends PNGs back to `DIR/frames/`.
  - A preempted VM's missing frames are re-queued on a new VM. Frames already in `DIR/frames/` are skipped, so a
    rerun resumes.
  - At the end it encodes `DIR/clip.mp4` (h264, yuv420p, `--fps` = 30 by default) and writes `DIR/render_log.json`
    (frames, seconds per frame, GPU-hours, spot cost, Blender and Alloy commits).
  - VMs are deleted at the end unless `--keep-vms`.
- **Frames off the host (`--frames-gcs`).** Long or large renders fill the controller's disk (a 1,191-frame
  2560x1440 clip is ~6 GB of PNG). With `--frames-gcs gs://PREFIX` (or `auto` =
  `gs://<gcp-project>-alloy-render/frames/<out dir name>/frames`) each VM uploads its own frames to
  the prefix every 10 s and marks them (render VMs have the `storage-rw` scope, and the default compute account
  may write only to the `<gcp-project>-alloy-render` bucket, so use a prefix there); the controller only counts the
  marks, pulls nothing, and resumes from what is already in the prefix. The clip is encoded by streaming the
  objects into ffmpeg (`gcloud storage cat`), so `DIR` ends up holding only `clip.mp4`, `look.json` and
  `render_log.json`. `encode gs://PREFIX --out X.mp4` re-encodes the same way.
- **Bundles in GCS (`render gs://BUCKET/path/bundle ...`).** `check` and `render` accept a bundle that lives only in
  GCS. The controller keeps a sparse mirror in `~/.cache/alloy_render/mirrors/<name>` (everything but the frame
  arrays, which get only their .npy headers as sparse files, so a 10 GB grid costs ~0.2 GB of disk), and every VM
  pulls the real bundle straight from the gs:// path (always staged; never rsynced from the mirror). With
  `--frames-gcs`, a render then touches no large data on the controller at all.
- **Spot L4 stockouts are normal.**
  - Workers that come up claim frame batches from a shared pool, and missing VMs are retried every ~2 min and join
    late.
  - 31 L4 zones are tried, US first. A zone that reports a stockout is skipped for 10 min.
  - The job aborts only if no VM has existed for 20 min.
  - `--reuse-vm NAME` adds an existing VM as an extra worker; it is never deleted.
  - Compiled OptiX kernels are cached once and pushed to new VMs.
- **Scale-out (32–48 VMs).** Preemptible L4 quota is 8 per region.
  - `--region-cap` (default 8) limits this job's VMs per region. New VMs go to the region with the fewest.
  - A quota error cools the whole region for 10 min; a stockout cools one zone.
  - Above 8 VMs (or with `--stage`), the bundle, look, Blender scripts, assets and kernel cache upload once to
    `gs://<gcp-project>-alloy-render/stage/<job>-<run>/`, and each VM pulls from there (~26 s). Without
    staging, the controller rsyncs to every VM.
  - `render_log.json` records regions, quota hits, late joiners and per-VM setup time.
  - Expect ~3 min per VM to boot plus ~22 s per 1080×1920 frame at 256 spp. 450 frames on 48 VMs take ~8–10 min
    when capacity exists.
- **Baked image.** `scripts/render_image.sh` builds the `alloy-render-blender45` image family: the DLVM image plus
  OptiX libraries and Blender, on a 50 GB disk.
  - The farm uses it whenever it exists. A VM is then ready about 100 s after the create request, against about
    7 min with the stock image plus install. Measured: 101.7 s mean setup.
  - Rebuild it after a Blender or startup-script change. Old images can be deleted.
- **First render on a fresh VM** spends about 170 s compiling OptiX kernels.
- **Measured on one L4 at 1080×1920:** 64 spp 7 s, 128 spp 14 s, 256 spp 22 s per frame. Scene build takes 0.5 s.
  A 1,024-world replica grid at 1920×1080 and 128 spp takes 2 s/frame.
- **`encode`** re-encodes frames, for example at a different fps for the slow-motion part of an edit.

## 5. Versioning

`arb` and `look` are integers. A change that breaks an existing bundle or look file bumps the number. The CLI
refuses a version it does not know.
