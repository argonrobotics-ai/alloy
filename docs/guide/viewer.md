# Viewer

`alloy.viewer` shows a running simulation: in a browser tab, in a desktop window, in Rerun, or as a USD file. It is
built on the Newton viewers that ship inside Alloy (Apache-2.0). It only reads the sim: stepping with a viewer
attached gives the same state, bit for bit, as stepping without it (checked on the CPU, on both rigid routes, and on
one L4 GPU). `import alloy.viewer` needs no optional package; the backend you launch does (table below), and a missing
one raises `alloy.FeatureUnavailableError` whose `fix` is the pip line.

## Install
```bash
pip install "viser>=1.1.1"           # browser viewer (default): works on a remote or headless machine
pip install pyglet imgui_bundle      # desktop window ("gl"): needs a display
pip install rerun-sdk                # "rerun"
pip install usd-core                 # "usd": writes a time-sampled USD file
```
These packages are optional: installing Alloy, with or without extras, does not bring them.

## 60 seconds
```python
import alloy
import alloy.viewer

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)
sim = alloy.build(scene)
view = alloy.viewer.launch(sim)                    # backend="viser"
print("open:", view.url)                           # http://localhost:8080
view.run(seconds=10)                               # 10 s; view.run() goes on until the viewer closes or Ctrl-C
```
Open the address in a browser. On a cloud machine forward the port first: `ssh -L 8080:localhost:8080 host`, then open
`http://localhost:8080`. The first build of a new scene compiles kernels with nothing printed meanwhile: about 30 s
for this scene on a 14-core CPU and about 2 min on a machine with one L4 GPU; later runs of the same scene take about
20 s to start (each measured once). `view.run()` steps in real time (one control period per frame); when a step and
its drawing take longer than the control period, the run is slower than real time. `view.run(realtime=False)` steps as
fast as it can. The same, from a shell, for any scene `alloy.load` takes (a library scene name, a YAML card, an MJCF
or URDF path):

```bash
python -m alloy.viewer task_emb_e2_panda --backend viser --port 8080 --device cpu
```
Options: `--worlds N` (build N worlds), `--show 0,3` (draw only these), `--seconds S` (stop after S simulated
seconds), `--paused` (start paused), `--no-contacts`, `--joints` (joint sliders), `--drag` (mouse dragging), `--path
FILE` (usd or rerun output). The CPU build of `task_emb_e2_panda` takes about 25 s, `demo_peg_d12` about 80 s (a
precision scene). Runnable versions: `examples/viewer/01_view_scene.py`, `02_pause_step_contacts.py` (both take
`--backend null --seconds 0.2` to run with no output).

## Controls
| Backend | Control |
|---|---|
| viser | "Simulation" folder: "Pause" checkbox and "Step" button (one control period, enabled while paused). "Example Options" folder (Alloy's own): slider "World shown (-1: all)" and checkboxes "Alloy contact points", "Alloy contact forces". "Visualization" checkboxes for Newton's own overlays (visual and collision shapes, wireframe, ground, joints, centres of mass, inertia boxes). Mouse: orbit, pan and zoom the camera. |
| gl | Space pause / resume; `.` step one period while paused; `H` hide the panel; `F` frame the camera on the scene; Esc close. `W A S D` move the camera, `Q` / `E` down / up; mouse looks around. The panel has the same Pause and Step. |
| rerun, usd | No controls of Alloy's: Rerun's own timeline; the USD file opens in any USD viewer. |

Checked in a headless browser on the viser backend (4 worlds, CPU): Pause, Step (0.01 s per click at `dt=0.01`), the
world slider and both contact boxes. The desktop window gets the same three Alloy controls in its side panel; they
were not clicked there (no display on the test machines).

Not working through Alloy: the "Reset" button (viser: greyed out; gl: not shown), the "Enable Picking" checkbox
without `drag=True` (it grabs nothing then; with it, see Dragging), and the panels' "Show Contacts", "Contact
Normals", "Contact Modes" and "Contact Forces"
boxes, which belong to Newton's own contact lists that Alloy does not fill. Alloy's contact overlay is drawn
separately: use the two "Alloy contact" boxes, `contacts=` or `view.show_contacts(...)`.

## Your own loop
`view.run()` owns the loop. To keep yours, step the sim and draw when you want to:
```python
import alloy
import alloy.viewer

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.3), mass=0.2)
sim = alloy.build(scene)
view = alloy.viewer.launch(sim, backend="viser", paused=True)
for i in range(1000):
    sim.step(1)
    if i % 10 == 0:
        view.sync()                # draw the sim's current state; nothing is stepped
view.pause(); view.resume(); print(view.paused)
view.step(10)                      # advance 10 control periods and draw, also while paused
```
- `view.run(seconds=None, realtime=True)` ends when the viewer closes, after `seconds`, or on Ctrl-C. `seconds` counts
  simulated time stepped plus time spent paused (a paused frame counts 1/60 s), so a paused `run(seconds=...)` in a
  script returns after about `seconds` of wall time without stepping. With `seconds=None` a paused run waits for the
  browser's Step or Pause to be unticked.
- `sync()` costs one copy of the shown worlds: for a training loop call it every N steps, not every step.
- `with alloy.viewer.launch(sim) as view:` closes it for you. `view.frames` counts frames drawn.

## Worlds
`launch(sim, worlds=[0, 2])` draws those worlds only (all are simulated); `view.show_worlds([0, 3])` changes the pick
and `view.worlds` reads it; in the browser, the "World shown" slider picks one world or all (a list of
several is Python only). `view.show_worlds` and `view.show_contacts` may be called from another thread while
`view.run()` draws. Drawing costs time and the sim runs slower with a viewer attached, so for many worlds show
few (`worlds=[0]`). Contacts are read for the shown worlds only. Measured on one CPU core with a block on a
table in each world: `alloy.viewer.contacts(sim, 0)` takes 0.1-0.3 ms and one `view.sync()` of world 0 about 1 ms with
2 or 64 worlds; syncing all 64 worlds takes 6-9 ms.

## Contacts
Contact points and force arrows are drawn for rigid bodies and robots (the MuJoCo route). `contacts=False` or
`--no-contacts` turns them off; `view.show_contacts(points=True, forces=True)` toggles them. An arrow's length grows
with the square root of the force (four times the force, twice the arrow), so light and heavy contacts both show.
An arrow shows the force on `body_b`, or on `body_a` when `body_b` is a fixed object (a ball on a table gets an arrow
pushing the ball up); read exact forces from the data:
```python
import alloy
import alloy.viewer

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.43), mass=0.2)
scene.add_sphere("ball", 0.03, pos=(0.15, 0, 0.44), mass=0.1)
sim = alloy.build(scene)
sim.step(100)
print(alloy.viewer.force_on(sim, "block"))         # net contact force on the block: (0, 0, m g) = (0, 0, 1.962) N
print(alloy.viewer.force_on(sim, "table")[2])      # what the table carries: -(0.2 + 0.1) * 9.81 = -2.943 N
c = alloy.viewer.contacts(sim, world=0)            # every contact; or view.contacts(world=0)
print(len(c["depth"]), list(zip(c["body_a"], c["body_b"])))
```
`c` holds, per contact, `pos` (n, 3) m, `normal` (n, 3), `force` (n, 3) N, `depth` (n,) m and the names `body_a`,
`body_b`. Signs: `normal` points from `body_a` toward `body_b`, and `force` is the force **on `body_b`** (the force on
`body_a` is its negative). Which object is `a` and which is `b` is the solver's choice and differs from pair to pair
(a block on a table may read table-block, a ball beside it ball-table), so never sum `c["force"]` over mixed pairs:
`alloy.viewer.force_on(sim, name, world=0)` sums one object's contacts with the right signs. `depth` is positive when
the shapes overlap. Contacts are those of the last physics step; a pair listed just before touching (robot shapes:
from 5 mm apart) has a negative depth and zero force.

## Joint sliders
`alloy.viewer.launch(sim, joints=True)` (or `python -m alloy.viewer <scene> --joints`) adds a "Joint targets (all
worlds)" group to the panel of the browser viewer and the desktop window: one slider per joint that a position
actuator drives (MJCF `<position>`, or a `<general>` servo as in MuJoCo Menagerie files), at most 32, in
`sim.names()["joints"]` order. A slider runs over the joint's range (a joint with no finite range: ±π for a hinge, ±1
m for a slide, around where it starts) and starts at the joint's current target (a `<general>` servo: its current
position in world 0). Moving it calls `sim.set_targets({joint: value})` before the next step, so the value applies to
every world and the joint goes there as to any target (a large jump makes a stiff servo kick: move slowly). It stays
until changed. Nothing is sent while the sliders are left alone: the run is bit for bit the one without them (tested
on the CPU on a one-hinge arm: a slider at 0.8 rad brings the joint to 0.808 rad in every world; the Panda example was
run without a window). Not covered: not run on a GPU; joints no actuator drives and velocity servos (no slider), one
target per world, a servo's `ctrlrange` wider than the joint's range (the slider stops at the joint's range), and a
`sim.set_targets` your program makes after the launch (the sliders do not follow it). `python
examples/viewer/03_joint_sliders.py` shows the Franka Panda's joints.

## Dragging
`alloy.viewer.launch(sim, drag=True)` (or `--drag`) lets you grab a body and pull it:
- browser viewer: keep "Enable Picking" ticked, click a body, drag the handle that appears; releasing the handle lets
  go (checked with a mouse in a headless browser);
- desktop window: hold the right mouse button on a body and move the mouse; releasing lets go (not clicked on a real
  display);
- any backend, from a script: `drag = alloy.viewer.MouseDrag.of(view)`, then
  `drag.grab(view, "block", world=0, target_xyz=(0, 0, 0.8))` and `drag.release()`: the same pull as the mouse's.

The pull is a spring and damper at the grabbed point, `F = m ω² (target − point) − 2 m ω v` with ω = 2π × 2 Hz and m
the grabbed body's own mass, held on the body with `sim.apply_wrench` (the force at its centre of mass and the torque
of the offset) and recomputed every control period from its pose; letting go removes it. A free body therefore hangs
`g / ω²` = 6.2 cm below a still target (tested: 0.6379 m for a target at 0.7 m). The force is capped at 50 times the
body's weight, and at 200 N on a robot link. Free bodies (tested), robot links and the segments of a rod on the chain
route (both not tested; the spring uses only the link's own mass, so an arm held by stiff servos may barely move) can
be grabbed; static and kinematic objects, and cloth, liquid and soft bodies, cannot (an info line, the grab is let
go). While nothing is grabbed the tool calls nothing on the sim: the run is bit for bit the one without it (tested).
The tool sets the grabbed body's wrench in every world (zero in the others), so do not hold your own
`sim.apply_wrench` on that body at the same time. Not covered: not run on a GPU; dragging cloth, liquid or soft-body
points, a grab that holds a body still (it is a spring, not a weld), and pushing with a torque only. `view.step(n)`
re-applies the pull every period. `python examples/viewer/04_drag.py` drags boxes on a table.

## Backends
`alloy.viewer.backends()` lists them as `{name: {"installed": bool, "install": "pip ..."}}`.

| `backend=` | Needs | Gives |
|---|---|---|
| `"viser"` | `viser>=1.1.1` | browser tab at `view.url`; options `port=8080`, `host=` |
| `"gl"` | `pyglet`, `imgui_bundle`, a display and an OpenGL driver | desktop window |
| `"rerun"` | `rerun-sdk` | a recording `alloy_run.rrd` (`path=` changes it); no server, no window. Look at it with `rerun alloy_run.rrd` on a machine with a display. `port=9090` also serves Rerun's web viewer there (and its data port 9876) |
| `"usd"` | `usd-core` | time-sampled USD file; options `path=`, `frames=` |
| `"null"` | nothing | no output (tests) |

## Not covered yet
- Reload, step back, video recording, and a "Reset" button. Dragging and joint sliders: see their sections for what
  they do not cover.
- Contacts of cloth, soft bodies, liquid, sand, rods and precision-fit pairs: not drawn, not in `contacts()`.
- Drawn (checked on one L4 GPU, 2 worlds): rigid shapes, robot meshes, kinematic fixtures, cloth and soft bodies
  (as meshes), rods (as a chain of capsules) and liquid (as spheres, one per particle; in a scene with liquid the
  vertices of a cloth get spheres too). Each world's cloth and liquid are drawn with that world's shapes, and hidden
  with them. Liquid inside an opaque container shows only from above. Sand, pastes, paper and metal sheets were not
  checked. Floors and objects use the viewer's own colours, not the looks `sim.render()` draws.
- `sim.render()` (`docs/guide/camera.md`) is for still images; `alloy.export.FrameExporter`
  (`docs/guide/frame-export.md`) hands numpy frames to a viewer you write. Use this page to look at a run live.

## Troubleshooting
- `FeatureUnavailableError`: run the pip line in its `fix`.
- Port in use: viser takes the next free port; read `view.url` (or pick one: `launch(sim, port=8781)`, `--port 8781`).
- Blank page or no connection on a remote machine: forward the port (`ssh -L 8080:localhost:8080 host`) and open
  `localhost`, not the machine's name.
- `"gl"` fails to open a window: it needs a display and OpenGL; on a cloud VM or over ssh use `"viser"`.
- No browser at hand (a remote machine): `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080` from a
  second shell prints 200 while the viewer serves; `--backend null` runs the same loop with no output.
- The run is slow with the viewer: show fewer worlds, or call `view.sync()` every N steps.
