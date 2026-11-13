# Frame export

`alloy.export` hands picked worlds of a running simulation to a viewer you write, as plain numpy arrays. (Alloy's own
viewer, [viewer.md](viewer.md), pauses and steps a run, with opt-in joint sliders and mouse dragging.) It sends body
poses, cloth, film and soft-body vertices, downsampled liquid and MPM particles, and rod centre lines. It never slows
or changes the run: stepping with the export on gives the same state, bit for bit, as stepping with it off.

```python
import alloy
from alloy.export import FrameExporter

scene = alloy.Scene(worlds=64, dt=0.01)
scene.add_ground()
scene.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)          # in topology.statics
scene.add_cloth("sheet", size=(0.4, 0.4), pos=(0, 0, 0.5), resolution=16, mass=0.05)  # meshes["sheet"]
sim = alloy.build(scene)

exporter = FrameExporter(sim, worlds=[0, 17, 63], max_points=20000)   # at most 8 picked worlds
for _ in range(100):
    sim.step(1)
    exporter.capture()                     # training thread, between steps: never waits
    frames = exporter.poll()               # any thread: None, or one Frame per picked world
```

The one-shot form waits for its copy: `from alloy.export import frame_export; frame_export(sim, [0, 3])`.

The example drapes the sheet over a static stand. It is sized for a GPU: a similar 64-world sheet took 16 s per
simulated second on one L4, after the first compile. On a CPU 64 cloth worlds are slow: 0.7 s per step for one world,
26 s per step for 64, about 43 min for the example's 100 steps on 32 cores, so use `worlds=1` there. A scene with free
bodies exports their poses in the same way. Beside cloth, though, a free body sets the cloth's exchange clock, near
the sheet or not (`docs/guide/deformables.md`, Cost). With a free 0.2 kg box in this scene, a period took 64 physics
steps instead of 8. On one L4 that came to 1.8-3.5 s per period, and the first run compiled for 10 minutes.

## What a frame holds
- **Identity:** `world`, `period` (control periods so far), `t` (seconds), and `topology_version`.
- **Bodies:** `body_pos` [B, 3] and `body_quat_xyzw` [B, 4], float32, for every body of the world.
  - These are raw body frames, the same as `"<name>.body_pos"`. A scanned object's `"<name>.pos"` is its anchor point
    instead.
  - Quaternions are in xyzw order, as the key says (and `quat_order` is `"xyzw"`). Shapes give `pos` and
    `quat_xyzw`.
- **Meshes:** `meshes[name]` [V, 3] holds every vertex of a cloth, film sheet or VBD soft body.
- **Particles:** `particles[name]` is `{"pos": [k, 3], "active": [k] bool}` for liquids and MPM materials.
  - k is a fixed subset of the world's particles: a sorted random choice with a fixed seed, the same every frame
    and spread evenly through the material (an even stride would draw an MPM lattice as stripes).
  - All sets together stay within `max_points`, sharing it by size.
- **Rods:** `rods[name]` [N + 1, 3] is the centre line: every segment's start, then the last end.
- **Topology:** `topology` arrives on a world's first frame and again whenever `topology_version` changes. In other
  frames it is None. It holds:
  - `body_names` and `body_active`;
  - `body_shapes`: each body's shapes in its body frame;
  - `statics`: the ground, tables and bins, with their world poses;
  - both list the shapes the camera draws: the visible ones, plus the collision shapes of a moving link that has no
    visible shape (the Gym tasks' gantry gripper, whose palm and fingers Newton's import leaves unmarked);
  - `mesh_faces`, indexing the mesh vertices, and `mesh_thickness`: the vertices are the mid-surface, so offset a
    sheet by half its thickness to draw it resting on what it touches;
  - `mesh_vertex_active`, and for soft bodies `mesh_tets` with `mesh_tet_active` (see Topology version);
  - per particle set: route, material, material id, count, the subset's indices and radii;
  - per rod: the segment count, its route, the bodies it is made of (indices into `body_names`), and
    `drawn_by_bodies`: True when those bodies have visible shapes (a MuJoCo chain's capsules), so a viewer can
    skip the polyline.
- **Shape sizes** are full sizes, as in `alloy.Scene`:
  - box: `size` (full edges);
  - sphere: `radius`;
  - capsule, cylinder and cone: `radius` and `height` (a capsule's straight part between its caps, as `add_capsule`);
  - ellipsoid: `size` (full axes);
  - plane: `size`, the full width and length (0 means infinite);
  - mesh: `vertices` (already scaled) and `faces`.

## Topology version
Cuts, tears and crumbles never rebuild a mesh's index arrays. Instead:
- a tear or bond cut breaks bonds on a mesh that was split at build;
- a crumble retires bodies;
- a virtual-node cut moves a tet's material to fragment tets allocated at build and wakes their copy vertices.

So a world's `topology_version` is `broken + retired * 2**20 + cut * 2**40`:
- `broken`: its broken bonds;
- `retired`: its retired bodies;
- `cut`: its cuttable whole tets that a split has emptied. A split empties exactly one, even when one side is too thin
  and folds into the other.

The weights keep a change in one count from cancelling a change in another. Each count stays below its field (bonds
and bodies under 2\*\*20, cuttable tets under 2\*\*23; the export refuses a larger scene), so the version fits an
int64. It rises at every event, never runs back within an episode, and returns to its start on `sim.reset()`.

To draw a cut soft body, use the topology part's masks:
- `mesh_tets` and `mesh_tet_active`: draw a tet only while it is active. This hides dormant fragments before a split
  and the whole tet after it.
- `mesh_vertex_active`: False marks a dormant copy vertex.

## How it stays out of the way
- `capture()` is the only method that issues GPU work. Call it between steps from the thread that steps.
  - It launches small gathers of the picked worlds on the simulation's own stream, after the step.
  - It queues one asynchronous copy into pinned host memory on a separate stream.
  - While the device is recording a CUDA graph it refuses, issues nothing and counts the refusal in
    `exporter.refused`.
- `poll()` makes no CUDA call at all, so a publisher thread can call it at any time without disturbing a graph
  capture.
- A frame becomes pollable at the first `capture()` (or `flush()`) after its copy has landed.
- There are two staging slots. A frame nobody polled gives way to a newer one, and if both slots are still copying the
  new frame is dropped. `exporter.dropped` counts both cases. The run never waits on the viewer.
- `exporter.resend_topology()` (any thread, no CUDA call) makes every picked world's next frame carry its topology
  again, for a viewer that reconnects. Keep one exporter per pick set. Create and drop exporters in the training
  thread, because freeing their buffers is GPU work, and do not let another thread hold the last reference.
- `examples/export/rerun_live.py` logs frames to Rerun from a publisher thread.
- A Sim built with `processes=` steps its worlds in worker processes and cannot be exported; build the exported run
  without it.

This is a live stream for viewers. To write a scene to MJCF, URDF or USD files, use `alloy.io.export_scene`
(`docs/guide/import_export.md`).
