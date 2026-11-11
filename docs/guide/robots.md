# Robots

## Adding a robot

- `scene.add_robot(name, "robot.xml")` loads an MJCF, URDF (`.urdf`) or USD (`.usd`, `.usda`) file;
  `scene.add_robot(name, mjcf="<mujoco>...</mujoco>")` takes an MJCF string. `pos=` and an orientation keyword
  place its base.
- The base is welded to the world unless the file gives the base a joint (for example a free joint or a slide).
- Every moving MJCF body needs mass and inertia: a body with a joint but no geom and no `<inertial>` (a slide stage
  carrying a link) fails the build with MuJoCo's bare `ValueError: Error: mass and inertia of moving bodies must be
  larger than mjMINVAL`. Give each such body an `<inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>` (its
  real mass if known). With `processes="auto"` the error arrives wrapped in a worker's `SimulationError`; the same
  call with `processes=None` shows it in full.
- One joint per MJCF body. Two slide joints on one body (`x` and `y` on a pusher's tip) load as one 2-coordinate joint
  named `x_y`: the names `x` and `y` are gone (`no joint named 'pusher/x'`), `sim.set_targets` cannot drive it (a
  2-value target is refused), and in an environment `jitter_joint` refuses it and `ActionSpec` finds no such joints.
  Workaround: nest a second body for the second joint, and give the carrier body an `<inertial>` as above; both names
  and both servos are then there (checked on the CPU).
- Some robot files bring their own floor, table or lights (MuJoCo Menagerie `scene.xml` files do). Do not add a
  second ground under them; `sim.check()` reports the overlap if you do.
- `alloy.load("robot.xml")` is the shortcut for a robot alone on a ground plane; the robot is named after the file
  (`scene.xml` gives `"scene/..."` names), so use `add_robot` when you want to choose the name.
- Shipped robots: see the next section.
- Contacts: a robot's shapes start making contacts 5 mm apart, like every scene object (an MJCF geom's own `gap`
  attribute is kept). `add_robot(..., contact_gap=0.1)` gives the importer's old default; it makes contacts between
  every pair of shapes within 0.2 m and can overflow the GPU contact buffer in packed scenes.

## Shipped robots

The library carries robot files from MuJoCo Menagerie and the makers' own descriptions, each with its licence and
source beside it. Load one by path:

```python
import alloy
from alloy.cards.library import default_root

scene = alloy.Scene(dt=0.01)
scene.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")   # brings its own floor: no ground
sim = alloy.build(scene)
print([j for j in sim.names()["joints"] if j.startswith("panda/joint")])   # panda/joint1 ... panda/joint7
print(sim.names()["actuators"][:4], sim.check()["ok"])                    # position servos: actuator1 ... hand_finger
```

| Folder | Robot | Joints / actuators |
|---|---|---|
| `emb_e1_fr3` | Franka FR3 arm + Franka Hand | 16 / 8 |
| `emb_e2_panda` | Franka Panda arm + Franka Hand | 15 / 8 |
| `emb_e3_ur10e_2f85` | Universal Robots UR10e + Robotiq 2F-85 gripper | 27 / 7 |
| `emb_e4_fr3_bimanual` | two FR3 arms with hands | 28 / 16 |
| `emb_e5_fr3_duo` | mobile base, spine and two FR3 arms with hands | 34 / 21 |
| `emb_e6_g1` | Unitree G1 humanoid with Dex3 hands | 45 / 43 |
| `emb_e7_h1_2` | Unitree H1-2 humanoid with Inspire hands | 52 / 39 |
| `emb_e8_go2` | Unitree Go2 quadruped | 14 / 12 |
| `emb_e9_leap` | LEAP Hand (right) | 18 / 16 |

- Each `scene.xml` is a whole scene. The arm files hold the robot on a table with a free box, cylinder and mug
  (`"panda/box"`, `"panda/cylinder"`, `"panda/mug"`) and their own floor; the others hold the robot and a floor.
  Add no `add_ground()` under them, and add your own objects beside them.
- Joint counts include the free objects' joints and the fingers. The actuators are position servos: drive them with
  `sim.set_targets` by joint name, or `sim.set_controls` by actuator name.
- Folders ending in `_route` and `_stiff` are contact-tuned variants of the same robots (stiffer contact
  `solref`); use the plain folder unless you need them.
- Each robot starts at joint angle 0 (MuJoCo's `qpos0`), not at a keyframe. The Go2 starts standing, held by
  targets 0: Go2 joint angles are measured from Menagerie's stand: add thigh 0.9, calf -1.8 rad to get
  Menagerie's angles. Its joint ranges in the file are shifted the same way (its `source.json`).
- The humanoids (`emb_e6_g1`, `emb_e7_h1_2`) start clean but cannot stand still at targets 0 without a
  balance controller: in 1 s the G1 slides about 35 mm, and the H1-2 tips and slides about 11 cm.
- `examples/basics/02_robot_joint_targets.py` moves the Panda's joints to targets.

## One start per world

A robot whose root body has a free joint (MJCF `<freejoint/>`: a carton with flaps, a loose gripper, a mobile
base) can start somewhere else in every world, and be moved while running; its links follow with their joint angles:

```python
import numpy as np
import alloy

CARTON = """<mujoco><worldbody><body name="box"><freejoint/><geom type="box" size="0.05 0.04 0.03" mass="0.1"/>
  <body name="flap" pos="0.05 0 0.03"><joint name="hinge" type="hinge" axis="0 1 0" range="-1.6 0.2"/>
    <geom type="box" size="0.03 0.04 0.002" pos="0.03 0 0" mass="0.01"/></body></body></worldbody></mujoco>"""

starts = np.array([[0.0, 0.25 * w, 0.031] for w in range(4)])     # one base position per world
scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_robot("carton", mjcf=CARTON, pos=starts)
scene.randomize("carton", "x", -0.02, 0.02)                   # or draw x / y / z per world (seeded by build(seed=))
sim = alloy.build(scene, seed=1)
sim.set_pose("carton", pos=starts + (0, 0, 0.1), quat_wxyz=(1, 0, 0, 0))   # teleport every world's carton
print(sim.state(["carton/flap"])["carton/flap.pos"])          # the flap came along
```

`sim.reset()` returns to the drawn starts. A robot welded to the world has one base pose in every world (the build
says so): move the objects around it per world instead. Runnable: `examples/api/08_robot_per_world.py`.

A per-world `pos` array puts the root body itself at each row, and so does a `pos` tuple once `scene.randomize`
draws the robot's `x`, `y` or `z`. A plain `pos` tuple moves the whole file instead: the root body ends at `pos` plus
where the file places it. For a root body the file puts above its origin (`<body pos="0 0 0.3">`), give that height
in the rows or the tuple (z = 0.3 there, plus any lift), or the robot starts in the ground (START_PEN, PEN_FAIL).

## Names

Robot bodies and joints are named `"<robot>/<body or joint name in the file>"`; a joint name that is unique in the
scene also works bare. `sim.names()["joints"]` and `sim.names()["bodies"]` list them after the build.
State keys: `"<robot>/<body>.pos"`, `.quat_xyzw`, `.quat_wxyz`, `.vel`, and `"<robot>/<joint>.q"`.

## Driving joints

- Position servos follow position targets: MJCF `<position>` actuators and `<general>` actuators with an affine bias
  (how MuJoCo Menagerie robots define theirs). `sim.set_targets({"arm/elbow": 0.5})` (radians for hinges, metres
  for slides). Targets stay until you change them, in every world; pass a `(worlds,)` array for one target per
  world. `sim.set_targets({...}, kind="velocity")` drives `<velocity>` actuators. A joint without an actuator
  raises an error that names the MJCF line to add.
- Raw actuator inputs (torque motors, grippers driven by one actuator): `sim.set_controls({"arm/actuator1": v})`;
  `sim.names()["actuators"]` lists them (an unnamed MJCF actuator is named after its joint).
- A target outside the joint's `range` stops at the limit, each world's target on its own, and the first one warns.
  A servo whose `ctrlrange` is wider than its joint's range keeps that range (a gripper that squeezes by aiming past
  its closed stop); `sim.set_controls` is raw and is not held. Check the ranges in the robot file.
- `sim.run(seconds, targets={"elbow": (0.0, 0.5)})` ramps a target linearly over the run; `{"elbow": 0.5}` holds a
  value. Ramps are the safe default: a large jump makes a stiff servo kick, which can throw a held object.
- Gains, damping, joint limits and actuator ranges come from the robot file. Tune them there.
- Read joint positions with `sim.state(["arm/elbow"])["arm/elbow.q"]`, shape `(worlds,)`.
- Give bodies and joints different names in the robot file: when a body and a joint share a name, the name reads
  the body.

## Springs that yield

A joint spring (MJCF `stiffness` and `springref` on a hinge or slide) is elastic at any load unless you give it a
yield. `scene.set_joint_yield("clip/arm", torque=0.02)` (N·m for a hinge; `force=` in N for a slide) makes it take a
permanent set: past the yield its rest angle moves with the joint, and once released (from `springref` 0, loaded
one way, no hardening) the joint rests at the peak angle minus `torque / stiffness`. `hardening=` (N·m/rad or N/m,
default 0) raises the yield by that much per unit of set; a reversed load yields at the same raised level. Read
where the spring rests now as `sim.state(["clip/arm"])["clip/arm.rest"]` and the accumulated set (it keeps growing
under reversed loads) as `"clip/arm.plastic_strain"`. `reset` puts the spring back. It runs on the MuJoCo routes
(GPU and CPU); a scene that also has cloth, liquid or a soft body refuses it for now
(`docs/known_gaps/plasticity-metal.md`).

## Grasping

- The grip is friction: set it with the object's `material` or `friction=`, or `scene.set_friction(a, b, mu)` for
  the finger-object pair. Robot geometry takes its friction from the robot file.
- Fingers need enough actuator force to squeeze (MJCF `kp` times the target error); close the fingers onto the
  object, then ramp the lift.
- Judge a grasp by the object, not the fingers: record the object's `.pos` and check it follows the gripper and
  stays between the fingers while held.
- Friction is the only grasp: there is no call that attaches or welds an object to a gripper
  (`docs/known_gaps/grasp-no-attach-or-weld.md`). Thin things are hard to pinch. In trials, the default 2 mm cloth
  slipped out of a fingertip pinch (0 of 64 worlds carried an edge). A cloth bends like its fabric at any thickness
  (cotton's bending length is about 1.3 cm), so an edge left over a support's side hangs down it: a 3 cm cotton
  overhang, 2 or 4 mm thick, settles 2 mm out from the side and 3 cm down. Plates aimed where a stiff edge would stick
  out close on the table; aim at where the cloth hangs. No robot grasp that carries a cotton sheet is verified yet
  (`docs/known_gaps/cloth-robot-grasp-carry.md`).
- To carry something rigidly, make it part of the robot: put its geoms in a `<body>` under the hand body of the
  robot's MJCF, with no joint between them. It then moves exactly with the hand. It also pours from a held
  container (`deformables.md`): in a trial the cup built into the hand poured 96-99 % into a second container in
  all 64 worlds. The cost: the object cannot be put down.
- A `<weld>` or `<connect>` in your MJCF that sets no `solref` / `solimp` is stiffened at build (about 4,600x
  MuJoCo's soft spring of 2.3 mm per kN on a 9 kg pair, at the 1.25 ms step); `alloy.build(scene, welds="mujoco")`
  keeps MuJoCo's row. It is still a spring, about 2.3e8 / W N/m at that step (W: the pair's inverse weights), so a
  heavy body on a light carrier is better made a child body (`docs/known_gaps/weld-stretch-under-load.md`).

```python
import alloy

GRIPPER = """<mujoco><worldbody><body name="palm">
  <joint name="lift" type="slide" axis="0 0 1" range="-0.1 0.4" damping="30"/>
  <geom type="box" size="0.03 0.05 0.01" mass="0.2"/>
  <body name="finger_l" pos="0 0.045 -0.035">
    <joint name="close_l" type="slide" axis="0 -1 0" range="0 0.04" damping="4"/>
    <geom type="box" size="0.01 0.005 0.02" mass="0.02"/></body>
  <body name="finger_r" pos="0 -0.045 -0.035">
    <joint name="close_r" type="slide" axis="0 1 0" range="0 0.04" damping="4"/>
    <geom type="box" size="0.01 0.005 0.02" mass="0.02"/></body>
  </body></worldbody>
  <actuator><position joint="lift" kp="2000" kv="100"/>
    <position joint="close_l" kp="300" kv="8"/><position joint="close_r" kp="300" kv="8"/></actuator></mujoco>"""

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_robot("hand", mjcf=GRIPPER, pos=(0, 0, 0.3))
sim = alloy.build(scene)
print(sim.names()["joints"])                    # full names, then unique bare names: "lift", "close_l", ...
sim.run(0.5, targets={"close_l": (0.0, 0.03), "close_r": (0.0, 0.03)})   # close the fingers
sim.run(0.5, targets={"lift": (0.0, 0.1)})                               # raise the palm 10 cm
q = sim.state(["hand/close_l", "hand/lift"], world=0)
print({k: round(float(v), 4) for k, v in q.items() if k.endswith(".q")})
print(sim.state(["hand/finger_l"], world=0)["hand/finger_l.pos"])        # a body: .pos, .quat_*, .vel
```
