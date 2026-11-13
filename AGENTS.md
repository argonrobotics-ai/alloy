# AGENTS.md: Alloy for AI agents

Alloy is a GPU physics engine for robotics, built on Newton, MuJoCo Warp and Warp: rigid bodies and robots step in
MuJoCo Warp (MuJoCo C on the CPU), cloth and soft bodies in Newton's VBD, sand and pastes in Newton's MPM, liquids in
Alloy's own DFSPH. One scene can hold rigid robots (sub-millimetre contact for declared precision fits such as pegs
and threads), cloth, liquids and other materials, stepped together in parallel worlds on one GPU: thousands for free
rigid bodies, 512 to 4,096 for a robot arm, tens to hundreds with cloth or liquid (`docs/guide/worlds.md` has the measured
rows). This file is the whole quick start: read it once, then write code. Every snippet runs as shown. More:
`llms.txt` (index), `docs/guide/` (one page per topic), `examples/` (runnable). Known engine limits, each with its
size and workaround: one file per limit in `docs/known_gaps/` (`docs/guide/known_limits.md`); read the ones for your
area before you trust a result there.

**Use `import alloy` (the public API below).** Do not drive Newton or Alloy's internal solver classes directly: you
lose routing, coupling, health checks and names, and the result is no longer an Alloy simulation.

## 1. Install and first run

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11 or 3.12
pip install alloy-engine
python -c "import alloy; print(alloy.__version__)"     # the import name is alloy
```
- Install with pip, inside a virtual environment: a stock Ubuntu 24.04 refuses pip outside one (PEP 668,
  "externally-managed-environment"), and its `python3 -m venv` fails until `sudo apt install python3-venv`.
- Alloy needs Python 3.11 or 3.12: on an older one (a stock Ubuntu 22.04 has 3.10) pip answers "from versions:
  none". `uv venv -p 3.12 --seed && source .venv/bin/activate` makes a 3.12 environment with pip on any machine with uv.
- The docs and runnable examples install with the package, in its `_data` folder: `python -m alloy.docs` prints both
  folders.
- NVIDIA GPUs: the pinned Warp (`warp-lang` 1.18.0) needs an R580-series or newer NVIDIA driver and a Turing
  (`sm_75`) or newer GPU (Warp's install notes); without a driver Alloy runs on the CPU.

On a fresh Ubuntu machine with an NVIDIA GPU (stock cloud images ship without a driver):

```bash
sudo apt install ubuntu-drivers-common         # stock cloud images lack the ubuntu-drivers tool
sudo ubuntu-drivers install && sudo reboot    # the NVIDIA driver; or start from a CUDA image instead
nvidia-smi                                    # lists the GPU; its "Driver Version" must be 580 or newer
pip install "alloy-engine[envs]"              # in your virtual environment; the envs extra brings PyTorch
```

If `ubuntu-drivers` picked a driver older than 580, `ubuntu-drivers list` shows the newer ones to install.

- Optional extras: `import` (mesh tools: `collision="decompose"`, `alloy.create`, the Hub object library) `mesh` (soft
  bodies from meshes) and `usd` (reading USD files; writing them needs nothing extra).
  Add them with pip: `pip install "alloy-engine[import,mesh]"`.
- No NVIDIA GPU? It runs on the CPU. `alloy.build(scene)` picks `cuda:0` when there is one, else `cpu`; pass
  `device="cpu"` to force it. On the CPU, rigid bodies and robots step in MuJoCo C (float64; with a C compiler
  installed a robot arm runs about real time on one core: that step is a C extension compiled on first use, so without
  a compiler, as on a stock Ubuntu 24.04 until `sudo apt install gcc` (macOS: `xcode-select --install`), Alloy warns
  once and robot scenes run 1.2 to 1.9x slower on the numpy path (up to about 4x with noslip and line contacts), and
  that path was not part of the release validation runs, so install a compiler first; rigid piles are not affected);
  scenes with cloth, liquid, soft bodies, precision fits or kinematic objects are much slower and use one core (no
  setting adds threads): liquid runs 25,000-40,000 particle-steps/s by CPU (about 40,000 on an AMD EPYC 7B13,
  25,000-27,000 on a 2.8 GHz Xeon cloud VM (GCP n2-standard-16): 500 particles 10-15 s per simulated second, 14,000
  5-7.5 min; with a free body floating in it or beside it 10-21x more), and cloth beside a robot arm about 10-45 s per
  10 ms control step, set by the lightest robot body near the cloth (heavier pads, fewer force exchanges:
  `docs/guide/deformables.md`, Cost). For many worlds, `alloy.build(scene, device="cpu", processes="auto")` steps them
  in one process per core (`processes=4` for four; `sim.close()` or a `with` block stops them): 64 worlds on 16 cores
  ran 13-20x faster for cloth, liquid and paste, 6.5x for a robot arm, each world as in one process (one exception:
  `docs/guide/install.md`). The CPU figures in this file are from an AMD EPYC 7B13 unless a line names another
  machine. Or use a GPU. CPU and GPU runs give the same physics model, not the same bits; contact-rich outcomes such
  as a grasp can differ between the CPU (MuJoCo C) and GPU (MuJoCo Warp) routes (one measured case: a Panda placed a
  pepper mill in 8 of 8 worlds on a GPU and 0 of 8 on the CPU, health OK), so check a grasp on the device you train
  on. `ALLOY_CPU_ROUTE=warp` keeps the CPU on the GPU's code path.
- The first run of a new kind of scene compiles kernels once: about 45 s on a CPU (41 s on an 8-vCPU Linux machine,
  44 s on an Apple-silicon Mac) and 3 to 4 minutes on an NVIDIA GPU (190 to 240 s on L4s); a robot environment can
  take up to about 8 minutes (a first `PickCube-v0` build took 7.4 min on an L4). The result is cached, so later
  runs start in seconds. Those figures are for the quick start below; cloth and liquid scenes compile longer
  (about 10 min for cloth on a GPU: the `ClothFold-v0` environment's first step alone compiled for 9 to 10 min on two
  fresh L4s). The cache is `~/.cache/warp`. Nothing shortens the first compile: run one short build and step as a
  warm-up job before a long run, and never time it or give up on it (section 6).
- Conventions: SI units (m, kg, s, N), z up, gravity -9.81 m/s². Sizes are **full** edge lengths. Quaternion keys
  name their order (`quat_xyzw`, `quat_wxyz`); orientation can also be `rpy=(roll, pitch, yaw)` in radians.

## 2. Quick start

```python
import alloy

scene = alloy.Scene(dt=0.01)                         # dt: control period (one sim.step()); physics substeps inside
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)         # top at z = 0.4
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)         # full edges (m), kg
sim = alloy.build(scene)                             # GPU if present, else CPU
traj = sim.run(1.0, record=["block"])                # 1 s; frame 0 is the start state
print(traj["block.pos"].shape)                       # (frames, worlds, 3)
print(sim.state(world=0)["block.pos"], sim.check()["ok"])
traj.world(0).save("block.npz")                      # t, block.pos, block.quat_wxyz, ... as numpy arrays
```

Other ways in: `alloy.load("robot.xml")` (MJCF / URDF / USD, placed on a ground plane), `alloy.load("scene.yaml")`
(a scene card), `alloy.load("<library scene>")` (`alloy.scenes()` lists them), `alloy.Scene.load(...)` for an
editable copy. A `Scene` round-trips through `scene.to_json()` / `alloy.Scene.from_dict(...)` / `scene.save(dir)`.

## 3. Names, state and trajectories

- Everything has a name: objects by the name you gave, robot parts as `"<robot>/<body or joint>"` (a unique joint
  name also works bare), objects from a scene card by card name (`"puck"`, then `"puck#1"` for a second copy). `sim.names()` lists them: `{"bodies": [...], "joints": [...], "particles": [...],
  "static": [...]}`. Static objects have no state.
- Keys: `"t"`, `"<body>.pos"`, `"<body>.quat_xyzw"`, `"<body>.quat_wxyz"`, `"<body>.vel"`, `"<joint>.q"`,
  `"<cloth, liquid or rod>.points"`, `"<rod>.twist"`, `"<rod>.plastic_strain"` (a yielding wire); a scanned library
  object also has `"<name>.body_pos"` (section 4).
- `sim.state(names=None, world=None)`: current values with a leading world axis `(worlds, ...)`, or one world's values
  with `world=w`. `sim.t` is the simulated time, `sim.dt` the control period, `sim.worlds` the world count.
- `sim.run(seconds, record=[...], every=1, targets=None)` returns an `alloy.Trajectory` (a dict of numpy arrays,
  `(frames, worlds, ...)`, plus `traj.meta` with `wall_s`, `physics_steps` and a health summary). `traj.world(w)`
  drops the world axis; `traj.save(path, **extra)` writes an `.npz`. Consecutive `run` calls continue the same sim.
- `sim.step(n)` advances `n` control periods without recording; `sim.reset()` returns every world to its start
  (`sim.t` keeps counting from the build; a running `sim.drive` continues through a masked reset, the per-world reset
  a `VectorEnv` makes, while `sim.reset()` removes it).

## 4. Building scenes

- Rigid shapes: `add_box(name, size=(x, y, z))`, `add_sphere(name, radius)`, `add_cylinder(name, radius, height)`,
  `add_capsule(name, radius, height)`,
  `add_mesh(name, path | vertices=, faces=, collision="convex" | "decompose" | "sdf" | "mesh")` (faces wind
  counter-clockwise seen from outside; `"convex"` collides as the hull; `"decompose"` splits a concave mesh into
  convex parts at build, cached, and needs the `import` extra (section 1), else the build raises
  `FeatureUnavailableError` naming the command; `"sdf"` and `"mesh"` collide as the true concave shape, on Newton's
  contacts). A body that moves on `sdf` or `mesh` contacts settles within 0.25 mm of its rest height but keeps a
  vibration of up to 0.3 mm at 2-25 mm/s (it is never fully at rest), for bodies of about 40 g to 35 kg; lighter or
  heavier ones make the world INVALID (`sim.check()`), so use `convex` or `decompose` for parts that move unless you
  need the exact concave shape (`docs/known_gaps/rigid-sdf-mesh-contact-limits.md`). Mesh files are OBJ or STL.
- Containers and holes, static and exact (built from boxes): `add_open_box(name, inner=(x, y, depth), wall=)`,
  `add_open_cylinder(name, inner_radius, depth, wall=)`, `add_block_with_hole(name, size, hole=(x, y), depth)`;
  `pos` is the centre of the bottom face.
- Shared keywords: `pos` (or a `(worlds, 3)` array: one start per world, for anything that moves: free or kinematic
  objects and fixtures, cloth, liquid, free-base robots), one orientation keyword, `material` (a name from
  `alloy.materials()`), `mass` (kg) or `density` (kg/m³), `friction` (Coulomb μ), `restitution`, `static=True` for
  fixed things (tables, walls, bins: one pose shared by every world; fixtures are static already and accept it;
  `kinematic=True` holds still too and can start per world). Per-world random values: `scene.randomize(name, field,
  lo, hi)`, fields `x`, `y`, `z` (the start coordinate itself, in m: it replaces that coordinate of `pos` rather than
  offsetting it, so give a range around it; for a robot it replaces a coordinate of the robot's `pos`, which moves
  the whole file: section 5), `density`, `friction`, `youngs` or the card path
  `material.surface.mu_dynamic`; `mass` and `restitution` are not: randomize `density`, set `restitution=` for all.
  On MuJoCo contacts (rigid bodies and robots) a restitution of 0 still rebounds at about 0.1-0.15 of the impact
  speed, and every restitution reads a little higher than set (0.5 gives about 0.55); a fix is planned for a later
  release (`docs/guide/scenes.md`, Materials and friction).
- Materials and friction: `scene.add_material("my_rubber", density=1100, friction=1.2)`;
  `scene.set_friction("a", "b", 0.9)` for one pair of objects. Contact uses one friction coefficient per pair of
  materials. There is no separate static friction: a body starts to slide at the same coefficient it slides with. For
  a pair with a pair card it is the card's `mu_dynamic` (aluminium on steel: 0.47, not its static 0.61), or the card's
  `mu_static` when the card says `single_mu: static` (one shipped card, aluminium on aluminium: 1.05, so it also
  slides at 1.05, not at the kinetic 0.40). Without a pair card it is the harmonic mean of the two materials'
  `mu_dynamic`. `friction=` and `scene.set_friction` set that one number. A coefficient above 1 tips a cube before it
  slides, so check such a pair with a flat block. Every library material value records where it comes from
  (`measured`, `catalog`, `fit`, or `model`: an estimate), with a note and an uncertainty range; about half are
  estimates, so read the card (`library/materials/<name>.yaml`) before relying on a value.
- Soft materials (rubbers, foams, silicone) can make an object a soft body (`sim.routes()` says `soft_vbd`; the build
  warns), which steps much slower, especially on the CPU. Pass `rigid=True` to keep it rigid with that material. A
  soft body from a mesh: `add_mesh(name, "part.obj", material="silicone_40a")` (or `rigid=False`) fills a closed mesh
  with tetrahedra at build; it needs the `mesh` extra (section 1). A 6 cm cube from a mesh steps 0.1 s in about 11 s
  on the CPU, the same cube from `add_box` in about 4 s; on one L4, a box resting on a soft pad runs about 0.3x real
  time over 64 worlds.
- Cutting: a soft object whose card has a `topology:` block is cut by a kinematic knife whose card has a `blade:`
  edge, moved with `sim.drive` (cards only: the scene API has no cutting option, and no robot-held knife example is
  published). A flat face crushes instead of cutting. `examples/topology/cut_anywhere.py`, `docs/guide/cutting.md`.
- Library objects: `scene.add_object(name, "<object card>", pos=...)` places a library object: one of the 118 that
  ship with Alloy, or one of about 5,400 object cards in the object library on the Hugging Face Hub (YCB, Google
  Scanned Objects, Objaverse, Lightwheel's kitchen objects as packaged by RoboCasa, and parts of Gym-Anything objects;
  counts in `docs/guide/fetch.md`). Every mass and friction carries a source tag: 122 of the YCB and Google Scanned
  objects use a published or measured weight; about 95 % of the Hub's masses are labelled estimates (a typical mass
  for the category, or mesh volume times a density), and friction is the named library material's, an
  engineering-judgement value for two thirds of the objects, never measured per object.
  `from alloy.api.catalog import objects; objects()` lists the ones on disk, `alloy fetch --list` the Hub's;
  `alloy fetch <name>` downloads one and needs the `import` extra (section 10). To find objects by words ("a mug"),
  use the scene builder (section 10). `pos` is where the object stands: a scanned object's bottom centre goes there
  (`anchor="auto"`; also `"center"`, `"base"`, `"origin"`), Alloy's own objects by their card frame (a cup's base).
  What you set is what you read: a scan's `"<name>.pos"` (state, `record=`, `set_pose`) is that point, turning with
  it; `"<name>.body_pos"` is its raw body frame, which can be cm away.
- Moving things without a robot: `kinematic=True` on a shape or fixture (a tilting tray, a moving wall, a pusher),
  then `sim.drive(name, velocity=(vx, vy, vz), angular_velocity=(wx, wy, wz), pivot=point, duration=s)`,
  `sim.set_pose(name, pos=...)` (teleport, also for free bodies), `sim.stop_drives()`. A kinematic fixture is one
  body: `"<name>.pos"` (also `"<name>_floor.pos"`) is its floor's frame; its other parts have no state of their own.
- Cables, wires and cords: `scene.add_rod(name, length=, radius=, pos=, direction=, clamp="start")` (or a polyline
  `points=[...]`) bends, twists and stretches as its material says (default `cable_usb_composite`), rolls, and can be
  gripped and twisted by a robot. `"<name>.points"` is its centre line, `"<name>.twist"` the twist per joint (rad);
  `sim.apply_wrench(name, force=, torque=, at="end")` holds a load on it (or on any body) until changed. Without a
  clamp a rod is a fast MuJoCo chain (`segments=` does not change that). A clamped rod, or one long or soft enough to
  need more than 40 segments, steps on `rod_vbd`, which beside a robot costs 83-95 s per simulated second on an AMD
  EPYC 7B13 and 139-156 s on a 2.8 GHz Xeon cloud VM (`docs/guide/rods.md`, Routes); `clamp="start", route="chain"`
  keeps a clamped rod on the chain, about 60x faster beside a robot on the CPU, 13-21x on a GPU (`docs/guide/rods.md`,
  Robots). A chain beside a robot follows a twisting grip and, on a GPU, holds the twist; on the CPU it creeps back
  while held (about 0.14 %/s; `docs/known_gaps/rod-chain-grip-slips.md`): there, clamp a cable that must hold a twist
  for long. A wire of a material that yields (`copper_c110` and other metal cards: `plastic.model: j2`) also steps on
  `rod_vbd` and keeps a bend or twist past yield; `"<name>.plastic_strain"` per joint.
- `sim.settle(seconds)` runs with no commands and makes the settled state the one `sim.reset()` returns to (piles,
  drapes and soft parts start a few mm off rest).
- `scene.names()` and `scene.summary()` show what is in the scene before you build it.

## 5. Robots and joint targets

```python
import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))    # or a path: add_robot("arm", "robot.urdf")
sim = alloy.build(scene)
print(sim.names()["joints"])                         # ['arm/shoulder', 'shoulder'] (unique names also bare)
traj = sim.run(1.0, record=["arm/shoulder"], targets={"shoulder": (0.0, 0.8)})   # ramp 0 -> 0.8 rad over 1 s
sim.set_targets({"shoulder": 0.8})                   # hold; targets stay until changed
sim.step(50)
print(sim.state(world=0)["arm/shoulder.q"])
```
- Robots come from MJCF, URDF or USD files (`add_robot(name, path)`) or an MJCF string (`mjcf=`). The base is welded
  to the world unless the file gives it a joint. A free base (`<freejoint/>` on the root body, e.g. a carton with
  flaps) can start per world (`pos=(worlds, 3)`, `scene.randomize("carton", "x", lo, hi)`) and move with
  `sim.set_pose("carton", pos=...)`, links following. A robot's `pos` moves the whole file in every form (tuple,
  per-world rows, drawn coordinates): the root body ends there plus where the file places it (a base at
  `pos="0 0 0.3"` in its file starts 0.3 m above `pos`); `sim.set_pose` puts the root body itself at `pos`.
  Robot shapes make contacts from 5 mm apart (`contact_gap=`).
- Shipped robots (MuJoCo Menagerie and makers' files, licences beside them): `from alloy.cards.library import
  default_root`, then `scene.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")`. Arms:
  `emb_e1_fr3` (Franka FR3), `emb_e2_panda` (Franka Panda), `emb_e3_ur10e_2f85` (UR10e + Robotiq 2F-85),
  `emb_e4_fr3_bimanual` (two FR3), `emb_e5_fr3_duo` (mobile base, two FR3); also `emb_e6_g1`, `emb_e7_h1_2`
  (humanoids), `emb_e8_go2` (quadruped; Go2 joint angles are measured from Menagerie's stand: add thigh 0.9,
  calf -1.8 rad to get Menagerie's angles) and `emb_e9_leap` (a hand). An arm's file is a whole scene: its own floor
  (add no ground), a table, and a free box, cylinder and mug (`"panda/box"`, ...). `docs/guide/robots.md`.
- MJCF reads angles (joint `range`, `euler`) in **degrees** unless it says `<compiler angle="radian"/>`; targets are
  always radians. A hinge whose range is `-2 2` without that line can move only ±2°.
- Position servos (MJCF `<position>`, and `<general>` with an affine bias as in MuJoCo Menagerie files) follow
  `sim.set_targets({...})` (metres or radians; `kind="velocity"` for velocity actuators). A value applies to every
  world; a `(worlds,)` array sets one target per world. A joint without an actuator raises, naming the MJCF line to
  add. Raw actuator inputs by name: `sim.set_controls({"arm/actuator1": v})`, names in `sim.names()["actuators"]`.
- Targets outside a joint's range stop at the limit (or at the servo's `ctrlrange`, when the file gives a wider
  one: a gripper that squeezes by aiming past its closed stop): read the ranges in the robot file first.
- Every moving MJCF body needs mass and inertia: a body with a joint but no geom and no `<inertial>` (a slide stage
  carrying a link) fails the build with MuJoCo's bare `ValueError: Error: mass and inertia of moving bodies must be
  larger than mjMINVAL`. Give each such body an `<inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>` (its
  real mass if known). With `processes="auto"` the error arrives wrapped in a worker's `SimulationError`; the same
  call with `processes=None` shows it in full.
- One joint per MJCF body: two slide joints on one body load as one 2-coordinate joint named `x_y` that no target or
  action can drive. Nest a second body for the second joint (`docs/guide/robots.md`).
- MJCF welds and connects: a `<weld>` / `<connect>` that sets no `solref` / `solimp` is stiffened at build with the
  rule of scene attachments, sized for the rigid step (30 kN stretches a 25 kg pair 0.015 mm, not MuJoCo's 70 mm);
  rows that set their own are kept, `alloy.build(scene, welds="mujoco")` keeps MuJoCo's own. A weld's `anchor` is a
  point of its second body (`<weld body1="b"/>` anchors at the world origin).
  `docs/known_gaps/weld-stretch-under-load.md`.
- A joint spring (MJCF `stiffness`, `springref`) that bends for good: `scene.set_joint_yield("clip/arm", torque=0.02)`
  (`force=` in N for a slide, `hardening=` optional); `"<joint>.rest"` reads where it rests now. MuJoCo routes only,
  not beside cloth, liquid or soft bodies (`docs/guide/robots.md`).
- Per-world robot randomization (opt-in): `scene.randomize("go2", "mass_scale", 0.8, 1.2)` and friction, `kp_scale`,
  `kd_scale`, centre of mass, armature, gravity, drawn once or at every reset; a free-base robot's start
  (`start_x`, `start_yaw`, `when="reset"`) and timed pushes, `scene.add_push("go2/base", force=(80, 80, 0))` (both
  checked on the CPU only); uniform draws only, and not yet joint damping or limits, actuator delay or noise, or
  velocity kicks: `docs/guide/randomize_robots.md`.
- Move with ramps (`targets={"joint": (start, end)}` in `run`, or small steps): a large target jump makes a stiff
  servo kick hard enough to throw a held object.
- Grasping: friction comes from the materials (or `friction=` / `set_friction`); give fingers enough actuator force
  (`kp`) to squeeze, and check the object really moves with the gripper (read its `.pos`, not the fingers'). There is
  no attach or weld call (`docs/known_gaps/grasp-no-attach-or-weld.md`). Thin cloth (2 mm) slips out of a pinch, and a
  cloth edge left over a support's side hangs down it (cloth bends like its fabric, about 35 % too soft along a mesh
  axis: `docs/known_gaps/cloth-bending-free-strip-soft.md`), so aim at where it hangs; no robot grasp that carries a
  cotton sheet is verified yet (`docs/known_gaps/cloth-robot-grasp-carry.md`). To carry a container rigidly (a pour),
  make it a body of the hand in the robot's MJCF (`docs/guide/robots.md`, Grasping).

## 6. Many worlds, randomization, speed

```python
import time
import numpy as np
import alloy

W = 64
scene = alloy.Scene(worlds=W, dt=0.01)
scene.add_ground()
xs = np.stack([np.linspace(-0.2, 0.2, W), np.zeros(W), np.full(W, 0.3)], axis=1)
scene.add_sphere("ball", 0.03, pos=xs, material="steel_carbon")   # one start per world
scene.randomize("ball", "friction", 0.6, 1.0)                     # drawn per world, seeded by build(seed=)
sim = alloy.build(scene, seed=0)
sim.step(1)                                          # warm-up: kernel compile and graph capture happen here
s0, t0 = sim.physics_steps, time.perf_counter()
sim.step(100)
sim.state(["ball"])                                  # reading state waits for the GPU to finish
wall = time.perf_counter() - t0
print(f"{W * (sim.physics_steps - s0) / wall:.0f} world-steps/s")
print(sim.state(["ball"])["ball.pos"][:4, 0])        # x of the first 4 worlds
```
- `alloy.estimate(scene, worlds, seconds)` predicts step time, wall time, memory and first-run compile time before you
  build. Its wall time is fitted to warm runs (robot + water, bins, an arm, cloth, pours at 1-64 worlds, CPU and L4):
  within 1.3x on those rows, further off elsewhere: a check of 8 other CPU scenes on another machine measured
  0.54-1.25x the estimate (it over-predicted a pour and a 64-world arm by up to 1.9x). On an L4, warm, section 7's
  sheet over a stand stepped 1.1-1.4x slower than estimated up to 64 worlds, 2.4x at 256 and 4.2x at 1,024; a sheet
  flat on a table and water at rest in a tub 1.8-2.7x at every count (a refit with many-world rows is planned).
  Right after a build or `sim.reset()` a GPU run first captures CUDA graphs (its first ~5 control periods, again
  when a liquid's step changes): a 1 s run then takes up to 2.7x the estimate (3.8x for 1,024 worlds of that
  sheet). GPU memory (`memory_bytes`) is fitted to 38 L4 rows (arms, a bin, cloth, liquid; 1-16,384 worlds): within
  1.2x there, with about 0.4 GB before the first world. Many worlds of scanned meshes are not among the rows. `measure=True` times a short real run instead; it does not measure memory
  (`memory_bytes` is None), so before a large run build a smaller world count and watch the GPU's memory
  (`nvidia-smi`).
- Same `seed` on the CPU: the same worlds, bit for bit, at every profile but `"fast"`. On a GPU that holds only until
  bodies touch in more than one place: free flight, an arm touching nothing, and one sphere or flat box on the ground
  repeat, but piles, tumbling boxes, meshes, robots in contact and cloth that folds or piles do not, at any profile.
  Differences start at 1e-7 m and grow where contact is chaotic (a pick-and-place at 64 worlds landed 62, then 61), so
  on a GPU compare with a tolerance and judge by success rate (`docs/known_gaps/determinism-gpu-rigid.md`). On a GPU
  every world of a scene with rigid bodies, robots or cloth lists the notice DET_DOWNGRADED and stays OK. Cloth on a
  GPU: a sheet that touches itself (a fold, a pile, an unconverged `cloth_accuracy="fast"` drape) or touches two
  objects at one point can make identical worlds and repeat runs differ, at every `cloth_accuracy` (the build logs
  determinism NONE). On the CPU every cloth setting repeats bit for bit, and identical worlds match, also once a sheet
  touches itself (not checked for a cloth that touches itself in a robot's grip).
- Profiles: `alloy.build(scene, profile=...)`: `"standard"` (default), `"precision"`, `"fast"`, `"debug"`.
- Robot scenes step at 1.25 ms (4 physics steps per 5 ms control period); that is their cost. On one L4, as shipped
  and physics only (re-checked on 35e29b44f): 2,048 Pandas with a table and a cube ran at 0.57x real time per world
  (1,168 simulated s per wall s in all), 4,096 Go2 at 0.28x, a LEAP hand with a cube at 0.13x, 512 G1 at 0.17x. Plain
  MuJoCo Warp is faster on robots: over 17 robot tasks on one RTX PRO 6000, each engine at its best batch, Alloy as
  shipped ran at 0.30x its rate (same robot files and gains; not otherwise like for like: Alloy takes 4 steps per 5 ms
  where MuJoCo Warp's preset there takes 2, and ran flat ground for rough-terrain tasks: `docs/guide/worlds.md`). Two
  opt-ins: `alloy.build(scene, contact_capacity=64)`, 1.8x with no measured quality change; and with it
  `scene.rigid_step = 0.0025`, 3.2x in all in hold tests, but not for an arm that grips (0.47 % of pick-and-place
  episodes glitched at 2.5 ms, 14 % at 5 ms; `"auto"` takes the largest step the robot's caps allow).
  `docs/guide/worlds.md`, `docs/known_gaps/robot-fixed-step.md`.
- Build options for speed (one process only): `rerun_worlds=n` (opt-in) re-runs only the worlds that must repeat a
  period, in a side simulation of n worlds (`worlds // 16` as a rule of thumb, not a measured best), beside cloth or
  soft bodies; `mpm_coupling` for robots in sand or paste: `"auto"` (the default) runs one force pass when only a
  robot starts near the material, 1.9x over `"standard"` on one L4, and `"fast"` (2.5 ms exchanges, faster still; a
  body landing on the material rests about 3 mm deeper); `mpm_sweeps="partial"` (opt-in, pastes about 1.3x faster).
  `docs/guide/worlds.md`.
- Rigid step: a scene of only free rigid bodies (no robots, kinematic objects, cloth, liquid, soft bodies, precision
  fits or plug-ins) steps them at the largest stable step, from 1.25 ms (at `dt=0.01`) up to `dt`. It never makes
  contacts softer, so light parts stay at 1.25 ms. Up to 8 free bodies step on rk4; more step on implicitfast with the
  friction pull capped, on MuJoCo Warp (any GPU, or `ALLOY_CPU_ROUTE=warp`; the CPU's default MuJoCo C route keeps
  rk4): resting stacks of 27 1 m cubes (10 kg/m³) ran at 5 ms, 766x real time over 3,072 worlds on one L4, none
  falling. For scale, the Newton MuJoCo solver bundled with Alloy, used plain with one 10 ms step, ran them at about
  1,800x with softer contacts (rest error 1.00 mm against 0.19 mm). A body spinning in free flight there turns to
  first order (`docs/known_gaps/rigid-clutter-rotation.md`; `profile="precision"` keeps rk4). `sim.routes()` names the
  integrator; it and `alloy.estimate(...)["rigid_step"]` give the step and what bound it. `scene.rigid_step = "fixed"`
  keeps the fixed step (1.25 ms at `dt=0.01`); a number in seconds pins a step (the build warns when it breaks a
  stability cap). `scene.rigid_step = "fast"` (or `profile="fast"` when `rigid_step` was never set) is an opt-in fast
  mode: contacts may soften to a rest sag of 1 % of a body's thinnest member and free bodies step on implicitfast (on
  MuJoCo Warp; rk4 on the CPU's default route), so 1 m cubes step at 10 ms and the 3,072 stacks ran 1,973x to 2,135x
  real time in three runs, about the plain solver's speed, none falling. Its cost: on MuJoCo Warp (a GPU, or
  `ALLOY_CPU_ROUTE=warp`), where the friction pull is capped, a 30 cm cube sliding at 5 m/s rocks under 1° and lifts
  1 mm; on the CPU's default route the fast mode keeps rk4, and that cube rocks 6° and hops 8 cm (2° and 1 cm at
  2 m/s); under `profile="fast"` a failing world freezes at its last checked state (section 8). The build warns, and
  `alloy.estimate(...)["rigid_step"]["fast"]` shows it beside the scene's own step (`docs/guide/worlds.md`).

## 7. Cloth, liquid and heat

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)
scene.add_cloth("sheet", size=(0.5, 0.5), pos=(0, 0, 0.5), resolution=20, mass=0.1)   # flat, 20 x 20 vertices
sim = alloy.build(scene)
print({name: r["route"] for name, r in sim.routes().items()})      # which solver each object went to
traj = sim.run(1.0, record=["sheet"], every=10).world(0)           # sheet.points: (frames, vertices, 3)
traj.save("sheet.npz", faces=sim.faces("sheet"))                   # faces: (F, 3) indices into the points
```
- `add_cloth(name, size=(x, y), pos, resolution, mass, thickness=0.002, material="cotton_woven", self_contact=True)`:
  flat in its local x-y plane, centred at `pos`. Sheets thinner than 2 mm need a smaller `dt`. There is no cloth from
  a mesh: to cover a shape, drape a sheet sized to it from a little above (then `sim.settle`). `self_contact` (on by
  default) makes a sheet collide with itself, so a fold rests as two layers; it costs about 1.4x per step on the GPU
  (2.1x with `cloth_accuracy="fast"`), so pass `False` for a sheet that never folds (`docs/guide/deformables.md`). Two
  cloths always collide with each other, but sheets that fall onto each other can pass through: 12 cotton sheets
  released together 5 cm apart ended with layers crossed (gaps down to -10.5 mm) and health not INVALID; sheets
  resting in a stack keep their order (`docs/guide/deformables.md`).
- Cloth accuracy: cloth runs 40 accelerated VBD iterations per step by default (Newton's `SolverVBD` with an adaptive
  start and Chebyshev acceleration): a drape at 0.43-0.52 % strain, against 13 % at that solver's default 10.
  `alloy.Scene(cloth_accuracy="super")` runs 60, about 1.4x the cloth cost (about 1.5x on the CPU), for a little less
  strain (0.30 %). On the CPU the default's 40 iterations cost about 4x VBD's own 10, which `cloth_accuracy="fast"`
  runs (DC 0.19 at 13 % strain, which health does not flag: only where the cloth's shape is not scored). On a GPU a
  "fast" drape does not repeat (its sheet keeps touching itself: identical worlds and repeat runs differ), and neither
  may any cloth that folds or piles (section 6). `alloy.estimate(scene, ...)["cloth"]` shows each setting.
- Paper and plastic film: `add_cloth(..., material="paper_80gsm")` (also `paper_kraft`, `paper_board`, `film_hdpe`,
  `film_ldpe`, `film_lldpe`) makes a film sheet (`sim.routes()`: `cloth_film`) that keeps a fold the way the material
  does: paper and HDPE stay folded, LDPE and LLDPE mostly spring back. A paste material (`dough_wheat`, `mayonnaise`,
  ...) raises `SceneError`: give it a solid shape, `add_box(name, size=..., material="dough_wheat")`.
- Metal sheets that dent: `add_cloth(..., material="aluminium_6061_t6", thickness=0.0005)` (or another metal with a
  yield law: `steel_tinplate`, `steel_carbon`, `stainless_304`, `brass_c360`, `copper_c110`, ...) makes a sheet that
  springs back below its yield and keeps a dent past it (`sim.routes()`: `cloth_film`). `"<sheet>.plastic_strain"`
  (worlds, vertices; in `sim.state()` and `record=`) is the largest plastic strain in the sheet at each vertex, 0
  until it yields; a small or thick one stays a sheet too. Limits: a metal sheet sags under its own weight far more
  than a real one (mm, not µm), and that sag alone can yield it (steel, stainless and copper read 0.001-0.007 before
  any press, and every metal more on a finer mesh; aluminium and brass read 0 on 5 mm cells), so read `plastic_strain`
  before the press and count only what the press adds; across the mesh's diagonals it bends up to 1.5x too stiff below
  yield and carries up to 1.9x the moment past it; rest and press it on rounded parts, since on a box's sharp edges a
  released sheet creeps sideways (`docs/guide/deformables.md`, Metal sheets; `docs/known_gaps/film-metal-sheets.md`).
- To fill a container, pour: `add_pour(name, volume_m3, pos=nozzle, rate=2e-4, speed=0.5)` streams straight down
  from `pos`; `"<name>.active"` in the state marks the particles poured so far. A block of liquid dropped from above
  splashes several percent over the rim (real flat-impact jets); a stream much less (3 % of a 6 ml pour from 12 cm,
  with health OK: `docs/known_gaps/liquid-pour-over-the-rim.md`).
- Pouring from a held container (a kinematic `add_open_box` tilted with `sim.drive(..., pivot=lip)`): the stream
  leaves the lip at about sqrt(g h) and lands about v sqrt(2 dz / g) beyond it, so place the target a few cm past
  the lip, not under it, and tilt slowly. `examples/api/06_tilt_pour.py`.
- `add_liquid(name, size=(x, y, z), pos, material="water")`: liquid at rest filling that box (already in a container,
  or a puddle); contain it in `add_open_box` / `add_open_cylinder`. Match the container: in a round cup,
  `add_liquid(..., shape="cylinder")` with `size=(diameter, diameter, height)`; a box in a round cup splashes out. `sim.liquid_volume(name)` is the simulated
  volume (whole particles: a few percent under the box; a pour's within half a particle); `sim.particle_volume(name)`
  per particle.
- Liquid accuracy: particles sit 4 mm apart by default. `alloy.Scene(liquid_accuracy="super")` puts them 2 mm apart,
  for finer streams and narrower openings, at about 16x the work (8x the particles, 2x the steps). Measured, the cost
  depends on the device and world count: 11-20x on CPU pours; on one L4, 3.5-3.8x for one world (the GPU has room to
  spare) and 36-49x at 64-256 worlds, in two measurements (a pour ran 18-19 simulated s per wall s at default, 0.4-0.5
  at super). `alloy.estimate(scene, ...)["liquid"]` shows both settings, and the CLI takes `--liquid-accuracy super`.
  - A build warns once when the setting is too coarse, naming the item and the fix: the opening of an `add_open_box`,
    `add_open_cylinder` or `add_block_with_hole` (or a card's passage) under 32 mm at default or 24 mm at super, or a
    pour stream under 2 particles across (8 mm at default, 4 mm at super). An `add_mesh` container is not checked:
    keep its opening at those widths yourself.
  - Containers narrower than 24 mm are unreliable at both settings (`docs/known_gaps/liquid-narrow-containers.md`).
  - Saved cards (`scene.save()`) do not keep the setting, Scene JSON does: on a scene loaded from cards, set
    `scene.liquid_accuracy = "super"` again.
- A kinematic container holds its liquid, still or tilted with `sim.drive`, beside free rigid bodies too: a tilt pour
  beside a free ball lands about the same share in the cup as with the ball static (94 %). On the CPU that scene steps
  10-12x slower (10.5x and 12.3x on two machines) than with every other object static, so make objects that never move
  `static=True`. Cloth and liquid with no free rigid object or robot meet kinematic objects too (a jug of water, a
  cloth on a moving tray), about 2x slower on the CPU than with those objects static.
- Cloth and liquid each interact with rigid objects and robots. Cloth and liquid together build beside static objects,
  and beside free rigid bodies or a robot that weigh at least twice the cloth and soft bodies in all; lighter ones
  raise `BuildError` with the mass to use (`docs/known_gaps/liquid-cloth-light-rigid.md`). `sim.routes()` tells you
  which solver each object uses and why. A stream poured onto a fabric sheet with self-contact on (the default) makes
  the world INVALID, mostly within 0.1 s of landing (0.39 s on a full block top; CPU rows); give a sheet that takes a
  pour `self_contact=False` (it stayed OK in the one scene measured) or use a film sheet (`film_ldpe`). Cloth takes up
  no water (`docs/known_gaps/liquid-on-cloth.md`). Cost: cheap while cloth and liquid do not touch, very slow once
  they do: on one L4 a 10 cm cotton sheet on 0.8 l of water steps at about 50 s per 10 ms control step from first
  touch (about 80 minutes per simulated second in 0.1.0 and 0.1.1, up to about 2.7x that after 0.1.1 while the sheet
  stirs the water; no setting helps), a film-route sheet (paper, metal foil) far faster
  (`docs/guide/deformables.md`, Liquid, Cost with cloth).
- Heat (temperatures in kelvin): `add_liquid(..., temperature=293.15)` (also `add_pour`) makes a liquid take heat;
  `scene.set_heater("pan_floor", temperature=353.15)` (or `power=` in W) makes a static box, such as an open box's
  floor, a heater, and `scene.add_heater(name, size, pos=, temperature=)` adds one; `scene.set_heat(time_scale=60)`
  runs heat 60x faster than motion. Read `sim.heat("water")["mean_K"]`, `sim.temperature("water")`, or
  `"water.temperature"` in `run(record=...)`. Heated liquid does not convect: `scene.set_heat(buoyancy=True)` (water
  only) gives hot water its first push, then the flow stalls (`docs/known_gaps/heat-no-buoyancy.md`). Rigid bodies
  take no heat. `examples/api/07_heat_water.py`; more in `docs/guide/heat.md`.

## 8. Precision contact, health and checks

- Sub-millimetre fits (pegs, connectors, bolts) use precision pairs declared on object cards; see the library scenes
  `demo_peg_d12` and `demo_nut_on_bolt`: `sim = alloy.load("demo_peg_d12", worlds=2, profile="precision")`.
  The precision profile checks every step and raises if a world becomes INVALID.
- `add_block_with_hole` makes a precise-fit scene (fine rigid step, stiff contacts): a dropped peg lands about 0.3 mm
  deep (0.27 mm measured; a test bounds it at 0.5 mm). It costs 5x more rigid steps at `dt=0.01` (0.25 ms against
  1.25 ms); pass `precise=False` for speed when sub-mm depth does not matter.
- `sim.health()`: per world `OK` / `DEGRADED` / `INVALID`, the flags that fired and a fix for each. Flags include
  NONFINITE, PEN_WARN (> 1 mm), PEN_FAIL (past 1 cm or half a thin part for 0.1 s, or past twice that at four checks
  in a row), SPEED_FAIL (> 20 m/s), ENERGY_JUMP, JOINT_FAIL, CONTACT_OVERFLOW (more contacts than the buffer holds:
  results are not trustworthy; the console says so once, with the fix: `alloy.build(scene, contact_capacity=n)`, n per
  world on average, default 8 per collision shape and at least 64; n also sizes the constraint rows, so ROW_OVERFLOW
  has the same fix: a larger n; sizing both to a robot scene is an opt-in speed setting, section 6), UNCONVERGED and
  CLOTH_STRETCH (a cloth over 50 % past its rest size is DEGRADED; over 500 % it blew up: INVALID, named in
  `sim.check()`), TUNNEL (a body passed through a static part: INVALID) and FAST_CONTACT (DEGRADED).
- An INVALID world is frozen at its last good state (under `profile="fast"`, its last checked state: up to 9 steps
  before the failure); the other worlds go on. A frozen world looks calm: always read `sim.check()["ok"]` or
  `traj.meta["health"]` before trusting a result.
- A fast body can pass through a thin static part (a 2 cm steel ball through a 1 cm plate from 9 m/s at the default
  1.25 ms step). How far it may travel per rigid step depends on the thinner part: coins and pins 2-6 mm thick passed
  a 5 mm pane at 3.8 mm per step, a pin of radius 1.5 mm at 1.4 mm. Health flags a pass (TUNNEL); the build warning
  and FAST_CONTACT fire only above 5 mm per step. It is not prevented: `docs/known_gaps/rigid-tunnelling.md`.
- `sim.check()`: `{"ok": bool, "problems": [{"world", "what", "fix", ...}], ...}`; physics checks on the run so far.
  `ok` is False for an INVALID world or a stopped run, and for a world that gained more than 10 % of the scene's
  energy scale with nothing adding it (ENERGY_JUMP as DEGRADED: a body that never comes to rest) or that started with
  an overlap of more than 1 cm (START_PEN). Any other DEGRADED world (PEN_WARN; CLOTH_STRETCH) passes it. Notices
  are listed in a world's flags but leave it OK: FAR_FROM_ORIGIN (an object more than 4 m from the origin; INVALID
  past 16 m under the precision profile), DET_DOWNGRADED (every GPU run with rigid bodies) and FALLBACK_USED. A flag that came and went is in `problems` with level "warn"
  (`seen_flags` in `sim.health()` has them all); one still on is in `sim.health()` flags only.

## 9. Errors

Every API error says what went wrong and what to change: `e.message`, `e.fix`, `e.field`, `e.to_dict()` (JSON).
Classes: `alloy.SceneError` (bad argument or card), `alloy.BuildError` (could not build: routing, capacity, device),
`alloy.SimulationError` (stepping failed, invalid world), `alloy.UnknownNameError` (lists the valid names),
`alloy.FeatureUnavailableError` (not in this install).

| Symptom | Fix |
|---|---|
| `UnknownNameError` on a state key | print `sim.names()`; robot parts are `"<robot>/<body>"` |
| Object sinks into or jumps off another | smaller `dt`; check sizes are full edges; containers via `add_open_box` |
| A fast small object passes through a thin wall or plate (TUNNEL) | step rigid bodies finer (`scene.rigid_step = 0.0003125` in a rigid-only scene, or `Scene(dt=0.0003)` with robots), slow the body, or make the parts thicker; no travel per step is safe for every part (a pin of radius 1.5 mm passed a 5 mm pane at 1.4 mm per step), so measure your own part at your speed and read `sim.check()` |
| Object falls through a mesh's opening / sits on top of a cup | a mesh collides as its hull: use `add_open_box` etc. |
| A run never finishes on the CPU | `sim.routes()`: a soft material made it a soft body (`rigid=True`), or too many worlds |
| A joint does not move | target outside its range, or no actuator on it (the error names the line to add) |
| "more contacts than its buffer holds" (CONTACT_OVERFLOW) | `alloy.build(scene, contact_capacity=n)` with the n it names |
| Liquid splashes over a container's rim | pour it with `add_pour` instead of dropping an `add_liquid` block |
| "liquid ... is too coarse here" warning | do what it names: `alloy.Scene(liquid_accuracy="super")` (11-20x the cost on a CPU, up to about 50x on a GPU batch), a wider container, or a pour with a higher `rate` or lower `speed` |
| Grasped object slips | ramp targets, raise finger `kp`, raise friction on the contact pair |
| PEN_FAIL / INVALID worlds | smaller `dt`; for tight fits use precision cards and `profile="precision"` |
| First run takes minutes | kernel compile, cached afterwards; time only warm runs |

## 10. Camera, checks, files, command line, Gym, scene builder, variants, fetching objects, plug-ins

`import alloy` does not load the modules of this section: import each before you use it (`import alloy.io`,
`import alloy.envs`, `import alloy.compose`, `import alloy.create`, `import alloy.plugins`, `import alloy.check`,
`import alloy.viewer`, `import alloy.il`, `import alloy.sysid`, `import alloy.isaaclab`), or `alloy.envs.make(...)`
raises `AttributeError`.

- Images: `sim.render(width=320, height=240, path="look.png")` (world 0, auto-framed, uint8 `(H, W, 3)`; scanned
  library objects show their photo textures, with Pillow from the `import` extra);
  `sim.render(alloy.Camera("front", look_at=(eye, target), width=128, height=128))` renders every world;
  `cam.render_torch(sim)` gives a torch tensor on the GPU for training. Depth and ids: `sim.render(cam,
  output="depth")` (metres, `inf` = no hit) or `"segmentation"`, or a tuple `("rgb", "depth", "segmentation")` for a
  dict (the default fast camera only; not `quality="full"`). Also `"instance_id"` (one id per shape), `"semantic"`
  (your classes, after `alloy.camera.aov.set_semantics`) and `"flow"` (pixels moved since the camera's last flow
  render; NaN on cloth and liquid).
  Look at a new scene once before a long run. `docs/guide/camera.md`.
- Frame export, for a viewer you write (Alloy's own viewer, next bullet, pauses and steps a run;
  `examples/export/rerun_live.py` logs frames to Rerun): `from alloy.export import FrameExporter`; `exp =
  FrameExporter(sim, worlds=[0, 5])` (up to 8 worlds), then `exp.capture()` after each step, in the stepping thread
  (it never waits), and `exp.poll()` from any thread: None, or one `Frame` per picked world (`body_pos`,
  `body_quat_xyzw`, cloth `meshes`, a fixed subset of liquid `particles`, `rods`; shapes and faces in `frame.topology`
  when they change). Stepping is the same, bit for bit, with it on. `frame_export(sim, [0, 3])` is the one-shot form.
  `docs/guide/frame-export.md`.
- Live viewer: `import alloy.viewer; view = alloy.viewer.launch(sim); print(view.url); view.run()` (a browser tab by
  default, `pip install "viser>=1.1.1"`; also a desktop window, Rerun, a USD file: `alloy.viewer.backends()`): pause,
  single steps, `worlds=[0, 3]` to pick worlds, contact points and force arrows of rigid bodies and robots;
  `alloy.viewer.force_on(sim, "block")` is the net contact force on one object (N) and
  `alloy.viewer.contacts(sim, world=0)` every contact as numpy arrays (`force` acts on `body_b`);
  `python -m alloy.viewer <scene>` from a shell. It only reads the sim, unless you opt in (both off by default):
  `joints=True` adds a slider per position-driven joint (it calls `sim.set_targets`; tested on the CPU on a one-hinge
  arm), `drag=True` lets the mouse pull a body with a spring (it calls `sim.apply_wrench`; tested on the CPU on free
  boxes, by script and with a mouse in a headless browser; dragging robot links not tested). Not yet: reload, step
  back, video, dragging cloth, liquid or soft bodies, contacts of cloth, liquid or soft bodies.
  `docs/guide/viewer.md`.
- Check a scene before using it: `import alloy.check; alloy.check("scene.yaml", worlds=4, seconds=1).findings`
  (each finding ends in the change to make; `expect_rest=False` when things should move). `docs/guide/check.md`.
- Files: robots load directly (`add_robot`); `alloy.io.import_file(path, out_dir)` turns MJCF / URDF / USD into
  cards; `alloy.io.export_scene(scene, out_dir)` writes MJCF, URDF and USD. `docs/guide/import_export.md`.
- No Python: `alloy run '<scene JSON>' --seconds 1 --out t.npz`, `alloy list scenes`, `alloy check <scene>`; JSON in
  and out. `alloy mcp` serves the same as MCP tools (`claude mcp add alloy -- alloy mcp`). `docs/guide/cli.md`.
- Gym: `alloy.envs.make("GripperLift-v0", num_envs=256)` or `VectorEnv(scene, Task(...))`: the shape of Gymnasium's
  vector API (`gym.make` does not work, `gym.make_vec` does after `alloy.envs.register_gymnasium()`). Install the
  `envs` extra for them: `pip install "alloy-engine[envs]"` (PyTorch for torch
  observations on the GPU and for training, `huggingface_hub` for the ready-made envs' library objects, Gymnasium for
  its spaces and `gym.make_vec`; without PyTorch they return NumPy arrays). Training example:
  `examples/envs/04_train_ppo.py`. `docs/guide/envs.md`. Limits to know before you train (workarounds in
  `docs/guide/envs.md` and `docs/guide/rl.md`): a robot with no `<actuator>` ignores actions without an error (check
  `env.sim.names()["actuators"]`); a rod's `.points` is not in the task state (read its link bodies,
  `"<rod>/<i>.pos"`); `VectorEnv(scene, ...)` cannot pass `alloy.build` options such as `contact_capacity`, so build
  first, `VectorEnv(alloy.build(scene, worlds=n, seed=s, contact_capacity=64), task)`, and set worlds, device and seed
  in that build (the env's own are then dropped without a message); and no shipped trainer learns from camera images
  (on a camera env the rsl_rl command trains a state MLP and does not say so). With rsl_rl, Stable-Baselines3 or skrl
  (each a separate `pip install`): `from alloy.envs.wrappers import wrap; wrap(env, "rsl_rl")`, or one command,
  `python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl` (PPO with one fixed setting per library, on small
  tasks of ours; on `GripperLift-v0` runs with the same settings ended anywhere from 0 % to 100 %, so train more
  than one seed, rows in the guide; `PickCube-v0`, a Franka picking a cube, has its own
  settings, a starting point (results vary between runs; watch the success rate the trainer prints):
  `docs/guide/rl.md`, Arm pick);
  `Task(obs={"policy": [...], "critic": [...]})` returns one
  observation per group (a critic that sees more than the policy). Opt-in first versions, tested on the CPU:
  hand-motion actions for arms (`alloy.envs.diffik.EEActionSpec`, differential inverse kinematics; one body, no
  collision avoidance; like delta joint actions, its targets stay within a `window` of the measured joints, so steer
  it closed loop: an open-loop 10 cm command moves the hand about 4 cm), observation noise / delay / history and
  action delay (`Task(layers=[...])`,
  `alloy.envs.realism`; white noise, whole control steps), and PD / DC-motor actuator models on motorized robots
  (`alloy.envs.actuators`; no learned actuators). None is validated on a real robot or on a legged robot.
  `docs/guide/rl.md`. Legged robots: `from alloy.envs.loco import go2_velocity` (the Go2 following velocity commands
  on flat ground) and `go2_rough` (the same on a heightfield of rough, wavy and block tiles with a level curriculum:
  the policy walks blind, with no height scan, and there are no stairs or gaps; one training run measured, at a mean
  level of 1.56 of 4); train with `python -m alloy.envs.loco.train` (`--terrain rough`); no actuator model, not run on
  a real robot: `docs/guide/locomotion.md`.
- Imitation learning: `from alloy.il import Recorder, HDF5Writer`; `Recorder(env, [HDF5Writer("demos.hdf5")])`
  records a `VectorEnv`'s episodes as you step it (`rec.reset`, `rec.step`) into HDF5 in the robomimic layout
  (the `il` extra, or `pip install h5py`) or a LeRobotDataset v3.0 (`LeRobotWriter`; the `lerobot` extra, Python 3.12,
  or `pip install 'lerobot[dataset]'`); `replay` re-runs
  recordings, `evaluate` gives a policy's success rate with an interval, `KeyboardTeleop` drives action columns from
  the terminal, `generate(env, demonstrator(env), writers, episodes)` saves scripted demos from every world (scripts
  ship for 3 of the 9 ready-made envs; no trajectory augmentation or motion planning). It records, replays and
  evaluates; it trains no policy. Teleoperation drives the env's action columns (joints, or the hand on an `EEActionSpec` env),
  with no live view; no cloth or liquid replay.
  `docs/guide/imitation.md`.
- System identification (optional; `import alloy.sysid`): a recording of your real robot (time, commanded joint
  targets, measured joint positions; CSV, HDF5 or a LeRobot dataset) becomes fitted joint parameters:
  `python -m alloy.sysid fit --robot panda --data take.csv --out my_panda.sysid.json` (`excite` writes a trajectory to
  play on the robot, `demo` a synthetic recording to try it); then `scene.add_robot("panda", path,
  sysid="my_panda.sysid.json")`. Fitted per joint: servo stiffness, kd + damping, dry friction, action delay. Not
  fitted: link masses, backlash, flexibility, contact or payload. Checked against simulated robots with hidden
  parameters only, never on hardware. `docs/guide/sysid.md`.
- Isaac Lab 3.0 (optional; `docs/guide/isaaclab.md`): with Isaac Lab installed, `pip install alloy-engine` in the
  same environment, then `physics=alloy` on a task that offers `physics=newton_mjwarp`
  (`isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=alloy`) runs the unmodified task on
  Alloy's rigid route at the task's own step (`rigid_step="alloy"` opts into Alloy's finer robot step) with Alloy's
  per-world health checks; `alloy.isaaclab.alloy_physics(cfg)` does the same in Python. Trained on one GPU (L4):
  Isaac-Cartpole-Direct, Isaac-Velocity-Flat-AnymalD, Isaac-Velocity-Flat-H1, Isaac-Reach-Franka; 18 rigid tasks
  step and reset. Isaac Lab `main`: steps on the CPU only so far. Not yet: Isaac Lab's other Newton solvers, cloth,
  soft bodies or liquid in its scenes.
- A scene from a sentence: `alloy.compose.compose("a mug and two cubes on a table", variations=2).save(dir)`.
  `docs/guide/compose.md`.
- Scene variants: `alloy.vary(scene, 8, pos=0.01, yaw_deg=10, seed=0, out="variants")` returns 8 edited scenes: free
  objects moved within ±1 cm and ±10°, each look given another CC0 texture of its family (wood to wood); the same seed
  gives the same scenes. `alloy vary scene.json --n 8 --pos 0.01 --yaw 10 --out variants` replies with the files and
  what each variant moved and swapped. `docs/guide/variants.md`.
- Objects from the Hugging Face library: `alloy fetch ycb_025_mug` downloads cards, meshes and textures into
  `~/.cache/alloy/imports/` (`alloy fetch --list` shows what there is). `$ALLOY_CACHE` moves that cache, and the
  texture cache with it (`$ALLOY_CACHE/render`; default `~/.cache/alloy-render`, filled by `python -m alloy.looks
  --fetch-textures`). It needs the `import` extra (section 1; it brings `huggingface_hub`); the library is public (no
  login needed). A fetched object then works like any other in `scene.add_object`. With the extra, a build also
  fetches a Hub object you name that is not on disk yet; without it, `add_object` raises `SceneError` for any object
  not on disk, its fix naming the extra (`pip install 'alloy-engine[import]'`)
  and `alloy fetch <name>`. `docs/guide/fetch.md`.
- Your own object (needs the `import` extra, section 1): `alloy.create.create(mesh="cup.stl" | photo= | text=,
  out="my_objects")` makes a checked card;
  `alloy.Scene(roots=["my_objects"])`, then `scene.add_object(name, card)`. `docs/guide/create.md`.
- Your own physics (forces, sensors, objects, solvers): `python -m alloy.plugins new my_force --kind force`, then
  `python -m alloy.plugins check my_force`; run one with `alloy.build(scene, plugins=[alloy.plugins.create("magnet",
  ...)])` (`docs/guide/plugins.md`); guide in `alloy/plugins/GUIDE.md`. A force plug-in's work counts as
  energy from outside in `sim.health()`, as a held `sim.apply_wrench`'s does, so neither raises ENERGY_JUMP.

## 11. Going deeper (engine layers)

The API builds YAML cards and runs Alloy's own layers; you can use them directly when the API lacks something.
- Cards (`schema: alloy.<kind>/1`): `material`, `pair`, `object`, `scene`. Hand-written card quantities need a
  source and a note: `density: {value: 1050, source: measured, note: "weighed 12.3 g / 11.7 cm3"}`.
- `alloy.cards.library.CardLibrary(roots=[...])` reads `materials/ pairs/ objects/ scenes/`;
  `alloy.scene.assembler.SceneAssembler(lib).assemble(scene_card, device)` -> `BuiltScene`;
  `alloy.integration.wiring.build_simulation(built, profile, seed=)` -> the engine simulation (`sim.engine` in the
  API). Runnable: `examples/basics/09_engine_layers.py`; more in `docs/guide/cards.md`.
- Package map: `alloy/api` public API; `alloy/cards` card types; `library/` shipped materials, objects, scenes,
  robots; `alloy/route` object -> solver; `alloy/scene` assembly; `alloy/integration` stepping, profiles;
  `alloy/precision` health and determinism; `alloy/contact` contact layer; `alloy/solvers` cloth (VBD), liquid
  (SPH), and others; `alloy/_vendor` Newton and MuJoCo Warp (section 12). `alloy/testing` is test support, not a
  public API.
- Tests: `python -m pytest tests/<package> -q` (`-m "not gpu"` without a GPU).

## 12. Built on

Alloy's physics core is built on Newton and MuJoCo Warp (both Apache-2.0), on Warp. Rigid bodies and robots step in
MuJoCo Warp through Newton's `SolverMuJoCo` (in MuJoCo C on the CPU), cloth and soft bodies in Newton's `SolverVBD`,
sand and pastes in Newton's implicit MPM solver, liquids in Alloy's own DFSPH solver, and elastic rods in Alloy's rod
solve on `SolverVBD`. Newton and MuJoCo Warp ship inside Alloy as `alloy/_vendor/newton` (with our fixes) and
`alloy/_vendor/mujoco_warp` (files unchanged; some functions replaced at run time), with their own names, licences and
copyright headers; `alloy/_vendor/VENDORED.md` lists the upstream commit each starts from and every change we made.
- Do not `pip install newton` or `mujoco-warp` for Alloy. A copy you install for other work can be imported and
  stepped in the same process as Alloy, in either order: Alloy's copies compile their Warp kernels under names of
  their own. That holds only when your copy accepts Alloy's exact pins, `mujoco==3.17.0` and `warp-lang==1.18.0`
  (pyproject.toml), which installing Alloy puts in the environment; a package that needs another MuJoCo or Warp
  version belongs in its own virtual environment. Objects from the two do not mix (pass Alloy's models only to Alloy's
  solvers).
- Engine-layer code imports Alloy's copies: `from alloy._vendor import newton` (or `mujoco_warp`), and submodules
  by the full path (`from alloy._vendor.newton.solvers import SolverXPBD`). Alloy's engine does the same; the
  cross-engine benchmark scripts in `bench/xbench/` keep a plain `import newton` so they can run on any engine
  (`VENDORED.md`).
- To run an unchanged script that does `import newton` on Alloy's copy, call `alloy._vendor.alias_top_level_names()`
  first. It is opt-in (the engine never calls it; the benchmark's engine shim does) and refuses when another
  `newton` is already imported in the process.
- Warp (`warp-lang==1.18.0`) and MuJoCo (`mujoco==3.17.0`) come from PyPI at those exact versions. Alloy replaces
  one Warp function at run time when a scene has MPM material: fem's Nanogrid cell lookup picks a particle's cell
  in integers, not float32, so worlds packed far along the grid keep their precision
  (`alloy/integration/nanogrid_lookup.py`, guarded by the pinned Warp version).
