# Randomizing robots per world

`scene.randomize` also draws a different value for every world of a robot's masses, centre of mass, friction, servo
gains and armature, and of gravity. It is off by default: a scene with no such rows runs exactly as before. It is
checked on the CPU (a law per field, on both CPU routes) and on one L4 (mass, servo stiffness and gravity, at build
and after resets); for a multi-joint arm the scales are tested from 0.8 to 1.2, and lower ones gave wrong results with
health OK (What is NOT covered). The runnable version of the first section is `examples/api/09_randomize_robot.py`.

Two reset-time events are on this page too, also off by default: a random start position and heading for a free-base
robot at every reset, and timed pushes on a body (`scene.add_push`). They are checked on the CPU only (a law each, on
both CPU routes); they have not been run on a GPU yet, and their printed outputs below are from one CPU run (an
AMD EPYC 7B13). Runnable: `examples/api/10_randomize_robot_pushes.py`.

## A first scene

```python
import alloy
from alloy.cards.library import default_root

scene = alloy.Scene(worlds=8, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")   # the file brings its own floor
scene.randomize("go2", "mass_scale", 0.8, 1.2)                    # every link's mass and inertia x one draw per world
scene.randomize("go2", "friction", 0.4, 1.2)                      # sliding friction of the robot's geoms, floor too
scene.randomize("go2", "kp_scale", 0.8, 1.2)                      # servo stiffness of every actuated joint
scene.randomize("go2/base", "mass_add", 0.0, 3.0, when="reset")   # kg on the base, drawn again at every reset
sim = alloy.build(scene, seed=0)
sim.run(1.0)
print(sim.check()["ok"])
```

Building the Go2 prints hundreds of lines: `NewtonWarning` notes about geom margins, each with the source line that
raised it (`add_geoms(child)`), and on a machine without a GPU Warp's CUDA-driver notice. They are harmless. The
first run compiles kernels: measured once on fresh machines, 43 s on a 14-core cloud VM (2.8 GHz Xeon, GCP
n2-standard-16) and 3 min 7 s on one L4; later runs took 10-14 s on that CPU and 14-18 s on the L4. The printed
numbers on this page are from a CPU run; one L4 run printed the same four digits; on a GPU, runs with robots in
contact do not repeat bit for bit, so your last digit can differ.

## The fields

`scene.randomize(target, field, lo, hi, when="build")`. One value is drawn per row and world, uniformly in `[lo, hi]`.

| field | target | meaning |
|---|---|---|
| `mass_scale` | robot or one link | factor on the link's mass and inertia (`lo` > 0) |
| `mass_add` | one link (a robot name: its root link) | kg added to the link; its inertia scales with its mass |
| `com_x`, `com_y`, `com_z` | one link (a robot name: its root link) | m added to the centre of mass, in the link's frame |
| `friction` | robot or one link | sliding friction coefficient of its geoms (absolute, `lo` >= 0) |
| `kp_scale` | robot or one joint | factor on the position servo's stiffness (`lo` > 0) |
| `kd_scale` | robot or one joint | factor on the servo's damping (a velocity servo's gain) (`lo` > 0) |
| `armature_scale` | robot or one joint | factor on the joint's armature (`lo` > 0) |
| `armature` | robot or one joint | the joint's armature, kg m^2 (a slide: kg) (absolute, `lo` >= 0) |
| `x`, `y`, `z` | `"gravity"` | that component of gravity, m/s^2 (absolute) |
| `start_x`, `start_y`, `start_z` | a free-base robot | m added to its start position at every reset (`when="reset"` only) |
| `start_yaw` | a free-base robot | rad added to its start heading, about the vertical, at every reset (`when="reset"` only) |

Target names:

- The robot's name (`"go2"`) is every link, joint and geom its file made. `"go2/FL_calf"` is one link and
  `"go2/FL_calf_joint"` one joint, with the names `sim.names()` lists. `"gravity"` is the scene's gravity (unless an
  object is named `"gravity"`).
- A robot-wide row moves all its links together by one factor. For independent draws, add one row per link or joint
  (an example is below).

How rows combine:

- Rows combine from the file's nominal values, in any order, and never compound: mass = nominal x the product of the
  `mass_scale` draws + the sum of the `mass_add` draws; inertia = nominal x mass / nominal mass; centre of mass =
  nominal + the offsets; `kp` and `kd` = nominal x the factors. Friction, absolute armature and gravity components are
  set to the drawn value; with two rows on the same item the last one wins.
- Servos: `kp_scale` and `kd_scale` act on MJCF `<position>` and `<velocity>` actuators and on `<general>` actuators
  that are joint servos (an affine bias, as in the MuJoCo Menagerie files such as the shipped Go2). A joint with no
  such actuator is skipped by a robot-wide row; a row that names it raises `BuildError`.

## When values are drawn, and what is repeatable

- `when="build"` (the default): drawn once at build; `sim.reset()` keeps the values.
- `when="reset"`: drawn again each time a world resets, for the worlds that reset. The redraw is host-side work on
  every reset. Measured once on one L4, the Go2 in 1,024 worlds: `sim.reset()` took 1 ms with no reset rows, 27 ms
  with one (`mass_add`) and 68 ms with eight; a reset of 64 of the worlds took 19 ms and 24 ms. Stepping ran at the
  same speed with and without rows (about 170,000 world-steps/s). Larger world counts were not measured.
- The steps right after such a reset cost more too: the redraw changes the model, so on a GPU the step after the
  reset runs without its captured graph and the one after that captures it again. One L4, 1,024 worlds of a
  two-slide pusher and a cube, a reset of 32 worlds then one step: 23 ms + 26 ms with a `kp_scale` reset row,
  1.0 ms + 7.6 ms with the same row drawn at build; an outside audit measured the second step at about 165 ms
  against 9 ms, so the reset and its two steps take about 210 ms in all, against about 18 ms. In a PPO loop where
  some world resets almost every step, that audit measured 1.7 to 2.1 s per iteration with the reset row against
  1.0 s with it drawn at build. On the CPU's default route (MuJoCo C) the step after such a reset rebuilds the MuJoCo
  model of every world, reset or not: that 1,024-world step took 3.9 s against 3.3 s on the 8 vCPUs of a GCP
  g2-standard-8 VM. Rows drawn at build cost about nothing at a reset: the same reset took 1.0 ms against 0.96 ms
  with no row, and the GPU kept its graphs. Draw a value at build unless it must change from one episode to the
  next.
- A draw depends only on the seed, the world number, the row's name and that world's reset count. The same seed
  gives the same values on the CPU and the GPU (compared on one L4) and for any number of worlds: world 3 of a
  4-world build is world 3 of a 64-world build. Another `seed` gives other draws.
- `alloy.envs` vector envs reset worlds through the same engine reset, so `when="reset"` rows redraw for the worlds
  that reset (masked resets and autoresets included). The draws use the seed the env's simulation was built with;
  `env.reset(seed=...)` does not change them.

```python
import alloy
from alloy.cards.library import default_root

def build(worlds):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
    scene.randomize("go2", "mass_scale", 0.8, 1.2)
    return alloy.build(scene, seed=0)

small, big = build(4), build(64)
print(small.randomized()["go2.mass_scale"][3] == big.randomized()["go2.mass_scale"][3])
```
```text
True
```

## Check that each world got its value

`sim.randomized()` returns `{"<target>.<field>": (worlds,) array}`: what every world holds now. Read something
physical beside it. Here the same Go2 with a payload on its base: the servos stand lower the heavier it is (the
example prints a table with four fields):

```python
import numpy as np
import alloy
from alloy.cards.library import default_root

scene = alloy.Scene(worlds=8, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
scene.randomize("go2/base", "mass_add", 0.0, 6.0)
sim = alloy.build(scene, seed=0)
sim.run(1.0)
payload = sim.randomized()["go2/base.mass_add"]
height = sim.state(["go2/base"])["go2/base.pos"][:, 2]
print(np.round(payload, 2))
print(np.round(height, 4))
print("height falls as payload grows:", bool(np.corrcoef(payload, height)[0, 1] < 0), sim.check()["ok"])
```
```text
[2.38 5.77 2.88 5.82 0.28 5.99 4.42 2.27]
[0.259  0.2509 0.2578 0.2508 0.2635 0.2503 0.2542 0.2592]
height falls as payload grows: True True
```

## A random start at every reset

A robot with a free base (the Go2; any file whose root body alone has a `<freejoint/>`) can start each episode at
its own place and heading. The four fields are offsets from the robot's start pose, they take `when="reset"` only,
and they are 0 until the first reset:

```python
import numpy as np
import alloy
from alloy.cards.library import default_root

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
scene.randomize("go2", "start_x", -1.0, 1.0, when="reset")       # m, added to the start position
scene.randomize("go2", "start_yaw", -3.14, 3.14, when="reset")   # rad, about the vertical through the base
sim = alloy.build(scene, seed=0)
sim.reset()                                                      # the first draw: before it every world is at the start
state = sim.state(["go2/base"])
quat = state["go2/base.quat_wxyz"]
print(sim.randomized()["go2.start_x"].round(2), state["go2/base.pos"][:, 0].round(2))
print(sim.randomized()["go2.start_yaw"].round(2), (2 * np.arctan2(quat[:, 3], quat[:, 0])).round(2))
sim.run(1.0)
print(sim.check()["ok"])
```
```text
[-0.92 -0.09  0.03 -0.03] [-0.92 -0.09  0.03 -0.03]
[ 2.83 -1.27 -2.25  1.83] [ 2.83 -1.27 -2.25  1.83]
True
```

- The base moves and every link moves with it; joint angles and the zero start velocity are kept. The yaw turns the
  robot about the vertical through its base.
- Nothing checks where the robot lands: keep the range on floor that is clear, and `start_z` at or above 0 (a
  negative one starts the feet inside the floor). The shipped robots' floors are flat, so x, y and yaw are safe there.
- A start drawn once at build is the fields `x`, `y`, `z` (see [Many worlds and speed](worlds.md)).
- After `sim.settle()`, a reset returns to the settled pose and then applies the offsets.

## Timed pushes

`scene.add_push(body, force, every=(4.0, 8.0), duration=0.1)` pushes one body in every world. Each world waits a
time drawn uniformly in `every` (seconds), then a force drawn uniformly in +-`force` per world axis (N) is held on the
body's centre of mass for `duration` seconds, and the world waits again. Every world has its own clock and its own
forces. `sim.pushes()` returns the force held now, `{"go2/base": (worlds, 3)}`, zeros while a world waits:

```python
import alloy
from alloy.cards.library import default_root

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
scene.add_push("go2/base", force=(30.0, 30.0, 0.0), every=(0.5, 1.5), duration=0.1)
sim = alloy.build(scene, seed=0)
for _ in range(3):
    sim.step(50)                                  # 0.5 s
    print(sim.t, sim.pushes()["go2/base"].round(1).tolist())
print(sim.check()["ok"])
```
```text
0.5 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
1.0 [[-11.4, -12.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
1.5 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
True
```

- A push is a force held for a time, not a jump in velocity: the body gains force x duration / mass of speed if
  nothing resists (a test measures this on a body sliding freely). The Go2 here only stands on its servos, with no
  policy balancing it, so the example keeps the pushes at 30 N.
- Times are whole control periods (`scene.dt`): a wait or a duration is rounded to them, at least one.
- A world that resets drops its push and starts a new wait. The pushes use the build's seed: the same seed gives the
  same times and forces for any number of worlds on the CPU (the only place it was checked; on a GPU the motion
  after a push does not repeat bit for bit: [Many worlds and speed](worlds.md), Determinism).
- A push's work counts as energy from outside, as a held `sim.apply_wrench`'s does, so it does not raise
  ENERGY_JUMP in `sim.health()` (a test pushes a freely sliding body and reads no flag).
- One push per body; `body` is any name in `sim.names()["bodies"]` (a robot link or a free object).
- The pushes run inside `sim.step`, so `sim.run` and `alloy.envs` environments built on the scene get them. The
  locomotion environment has its own (`Pushes` in [Legged locomotion](locomotion.md)) and its `scene_randomize`
  option takes build-time rows only, so the events of this page are for your own scene and `VectorEnv`.
- `sim.apply_wrench` on a pushed body is overwritten when its next push starts or ends.

## More rows: your own variation

Independent draws per leg, one joint's gain, centre of mass, armature and gravity in one scene (everything here is
absolute or a factor around 1, so it also shows the key names in `sim.randomized()`):

```python
import alloy
from alloy.cards.library import default_root

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
for leg in ("FL", "FR", "RL", "RR"):
    scene.randomize(f"go2/{leg}_calf", "mass_scale", 0.9, 1.3)       # four independent draws
scene.randomize("go2/FL_hip_joint", "kp_scale", 0.8, 1.2)            # one joint
scene.randomize("go2", "kd_scale", 0.8, 1.2)
scene.randomize("go2", "armature_scale", 0.5, 1.5)
scene.randomize("go2/base", "com_x", -0.03, 0.03)                    # m, in the base frame
scene.randomize("gravity", "z", -10.3, -9.3)                         # also "x", "y"
sim = alloy.build(scene, seed=1)
sim.run(1.0)
print(sorted(sim.randomized()), sim.check()["ok"])
print(sim.randomized()["gravity.z"].round(2))
```
```text
['go2.armature_scale', 'go2.kd_scale', 'go2/FL_calf.mass_scale', 'go2/FL_hip_joint.kp_scale', 'go2/FR_calf.mass_scale', 'go2/RL_calf.mass_scale', 'go2/RR_calf.mass_scale', 'go2/base.com_x', 'gravity.z'] True
[-10.09  -9.97  -9.82 -10.  ]
```

## What is NOT covered

- The tested range for an arm's scales is 0.8 to 1.2 (`mass_scale`, `kp_scale`, `kd_scale`), the range of this
  page's examples. Below it the Franka Panda's static response is wrong while health reads OK (an outside audit,
  three re-tests, both CPU routes; the cause is not found). `kp_scale` 0.5 gave a joint-2 sag of 0.0099 rad where the
  same gains written into the robot file give 0.02025, and 0.6 gave 0.0092 against 0.0169; 0.7, 1.0 and 2.0 agree
  with the file. Worlds with `mass_scale` under 0.65 sagged 15 to 21 % less than their drawn mass predicts; from 0.7
  up they match within 1 %. A single-hinge arm and the Go2 do not show it (the Go2 matched its edited file to 5
  digits from 0.3 to 2.0). Nothing refuses a lower scale: keep an arm's scales within 0.8 to 1.2, or check a drawn
  world against the same value written into the file. Above the range: `kp_scale` 1.5 to 2.5 on the Panda made worlds
  INVALID (health flags those), and a `mass_scale` near 2 runs into the file's own joint force limit.
- Not randomized yet: joint damping and friction loss, joint limits, actuator delay, noise and force limits, link
  shapes or lengths, restitution, torsional and rolling friction, and observation noise. Distributions are uniform
  only.
- Events: the only ones are the random start and the timed push above. No velocity kicks, no random joint angles
  or start velocity at reset (in an `alloy.envs` environment, `jitter_joint` does joint angles:
  [Gym environments](envs.md)), no start roll or pitch, no torque pushes, and no event that changes a
  model value on a timer (a `when="reset"` row is the way to change one). A start is not checked against other
  objects. Pushes and random starts are not checked on a GPU yet.
- Pushes: `alloy.build(..., processes=N)` and `rerun_worlds=n` raise `BuildError` with `scene.add_push`.
- Friction: a contact uses the larger of its two geoms' friction (MuJoCo's rule). A robot-wide friction row sets the
  robot file's own floor too, so foot-floor contacts get the drawn value. A row on one link, or a robot standing on
  `scene.add_ground()` or another object, only matters where the draw is the larger of the two. A pair set with
  `scene.set_friction(a, b, mu)` keeps its value.
- Mass: inertia scales with the link's mass (the shape of the mass distribution is kept). A centre-of-mass offset
  moves the centre of mass without changing the inertia about it.
- Gravity can be randomized only in scenes of rigid objects and robots; a scene with cloth, liquid, soft bodies, sand
  or paste raises `BuildError`.
- Step sizes and contact settings are chosen from the file's nominal values, not from the drawn ones. Keep ranges
  moderate (the ranges above, 0.8-1.2 on factors and a few kg of payload on a 15 kg robot, are a sane start) and read
  `sim.check()["ok"]`.
- On the CPU's default route (MuJoCo C), a `<position dampratio=...>` servo keeps the damping computed from the
  nominal masses when link masses are drawn.
- `alloy.build(..., processes=N)` (CPU worker processes): the draws apply in every world as in one process, but
  `sim.randomized()` raises `FeatureUnavailableError` there. `alloy.build(..., rerun_worlds=n)` raises `BuildError`
  with these rows.
- `scene.save()` cards do not keep these rows; `scene.to_json()` and `alloy.Scene.from_dict` do.
- Objects (not robots) keep their own fields (`density`, `friction`, `youngs`, `x`, `y`, `z`), drawn at build only:
  see [Many worlds and speed](worlds.md).

## Errors

| symptom | fix |
|---|---|
| `SceneError: robot 'go2': cannot randomize 'mass'; use mass_scale (a factor) or mass_add (kg)` | use a field from the table; the message names the one people usually mean |
| `UnknownNameError: randomize: 'go2/nolink' is not a link of robot 'go2'` | the message lists the valid names (`sim.names()` too) |
| `BuildError: randomize: 'go2/joint0' has no joint servo for kp_scale to scale ...` | name a joint that has a servo actuator, or add one in the robot file |
| `SceneError: 'go2': kp_scale is a factor and must stay above 0, got lo=0.0` | pass a range around 1, such as `(0.8, 1.2)` |
| `SceneError: randomize: when='never' is not known` | `when="build"` or `when="reset"` |
| `SceneError: 'go2': friction cannot be negative, got lo=-1.0` | pass `lo >= 0` |
| `BuildError: a randomized link mass can reach zero or less ...` | raise the low end of the `mass_add` range, or name a link that has mass |
| `SceneError: gravity: cannot randomize 'w'` | `'x'`, `'y'` or `'z'` |
| `SceneError: 'go2': start_yaw is drawn at every reset, not at build` | pass `when="reset"` |
| `BuildError: robot 'arm' has no free base, so its start cannot be randomized` | use a robot whose root body has a `<freejoint/>` |
| `SceneError: push 'go2/base': force must be 3 values >= 0, one of them above 0 ...` | the largest force per axis in N, such as `force=(80, 80, 0)` |
| `BuildError: push: 'go2/nolink' is not a body that moves in the built scene` | the message lists the bodies |
| `BuildError: gravity cannot be randomized in a scene with cloth, liquid, soft bodies, sand or paste ...` | randomize gravity in scenes of rigid objects and robots only |

## See also

- [Many worlds and speed](worlds.md): per-world starts, `randomize` for objects, seeds and determinism.
- [Robots](robots.md): loading robots, joint names, targets.
- [Gym environments](envs.md): vector envs and resets.
