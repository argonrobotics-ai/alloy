# Many worlds and speed

`alloy.Scene(worlds=N)` (or `alloy.build(scene, worlds=N)`) builds N independent copies of the scene that step
together on one device. Every state array has the world axis first: `sim.state()["cube.pos"]` is `(N, 3)`,
a trajectory is `(frames, N, ...)`.

## How many worlds

How many worlds pay off depends on what is in the scene. Measured on one L4 through this API, warm (kernels compiled,
CUDA graphs captured); the rate is simulated seconds per wall second, summed over the worlds:

| scene | worlds | simulated s per wall s, all worlds |
|---|---|---|
| free rigid bodies: resting stacks of 27 cubes (The rigid step) | 3,072 | 766 |
| three library objects dropped into a bin | 512 | 313 |
| a cable as a MuJoCo chain, held by a robot's two pads (`rods.md`) | 1,024 / 4,096 | 197 / 192 |
| the shipped Franka Panda scene: the arm, a table and three free objects | 64 | 74 |
| the same Panda scene | 1,024 | 6: its CUDA graph did not fit in the L4's memory (Estimating before you build) |
| a robot carrying 40 ml of water | 256 | 8.3 |
| 200 ml of water poured into a cup | 64 | 7.1 |
| a 0.6 m cloth sheet dropped onto a table, default accuracy | 64 | 4.1 |
| a box resting on a soft pad (`scenes.md`) | 64 | 0.28 to 0.30 |
| a robot's finger in sand, `mpm_coupling="auto"` (Build options for speed) | 64 | 0.09 |

So thousands of worlds pay off for free rigid bodies and cables. Robot scenes run at 512 to 4,096 worlds as shipped
(Robot scenes, below): `GripperLift-v0` and `PandaReach-v0` train at 512 and 1,024 (`envs.md`), while the shipped
Panda scene with its table and three free objects was too large at 1,024. Scenes with cloth, liquid, soft bodies or
sand have been run at 64 to 256 worlds, where the whole batch steps at 0.1 to 8 times real time. Time your own scene
at a small world count first (`alloy.estimate(scene, worlds, seconds, measure=True)`).

### Robot scenes: what they cost, and the two opt-ins

A scene with a robot steps its rigid bodies at 1.25 ms: 4 physics steps per 5 ms control period, 8 per 10 ms. That
step, with its stiff contacts, is the cost of a robot scene. Measured on one L4, physics only, each shipped robot
holding noisy joint targets at a 5 ms control period, at the largest batch that ran best (one run each; the LEAP
hand's rows, here and in the opt-in table below, are from a later run on its current joint values, its as-shipped
row on another L4 of the same shape; the arms' opt-in rows are from a later run on their current firm contacts,
where their as-shipped speed came within 3 % of this table):

| scene, as shipped | worlds | simulated s per wall s, all worlds | per world | GPU MiB per world |
|---|---|---|---|---|
| Panda, a table and a 5 cm cube | 2,048 | 1,168 | 0.57 | 4.4 |
| Panda alone | 2,048 | 1,322 | 0.65 | 4.2 |
| UR10e + 2F-85 | 4,096 | 1,532 | 0.37 | 1.9 |
| Go2 standing | 4,096 | 1,134 | 0.28 | 1.3 |
| LEAP hand holding a cube | 2,048 | 276 | 0.13 | 3.7 |
| G1 with hands, standing | 512 | 88 | 0.17 | 19.7 |

Re-checked on main 35e29b44f: on one L4, run by run in turn against the tree before the arms' firm contacts and
the LEAP hand's new joint values, the arms' rows moved under 1 % as shipped and with each opt-in (two runs of a row
0.5 % apart), and the LEAP hand's as-shipped row read 287 there (273 before; L4s of this shape differ by a few %).

Plain MuJoCo Warp is faster on robots. Measured on one RTX PRO 6000 (96 GB), physics only, a 5 ms control step, each
engine at its best measured batch (Alloy 16,384 to 65,536 worlds, MuJoCo Warp 16,384 to 262,144), one run: over 17
robot tasks Alloy as shipped ran at 0.30 times MuJoCo Warp's rate (geometric mean; from 0.11 on a cart-pole to 0.76 on
a quadruped task). Two of the tasks have rough terrain on MuJoCo Warp's side and flat ground on Alloy's: that
quadruped task (0.76) and a biped's (0.60). The tasks: four arm tasks (reach, lift); an arm with a four-finger hand,
and that hand alone, moving a cube (three tasks); two quadrupeds and three bipeds or humanoids walking (six tasks); a
cart-pole, an ant and two humanoid models.

Both sides load the same robot files, joint gains and start poses. The rest is not like for like. MuJoCo Warp (3.17)
ran inside the task framework's own loop with that framework's solver preset for each task: 2 solver steps per 5 ms, 1
for the cart-pole and the ant. Alloy ran with its own contact and solver settings and 4 steps per 5 ms, with a PD
drive on every joint and no effort limits, on flat ground where a task has rough terrain, and with one box where a
task spawns objects of several shapes. The ratio of every task, Alloy's rate and both batches are in
`bench/scorecard/row1/rows.jsonl`, and its `README.md` says how each side was run. The Alloy side's script is
`bench/scorecard/mixed/single_task/alloy_hold.py`; the MuJoCo Warp side ran in a harness that is not in this
repository, so only Alloy's side can be re-run from it.

Two settings buy speed. Both are opt-in: a scene that sets neither steps as above.

| setting | speed, six scenes above (geometric mean) | what it cost in these hold tests (1,024 worlds, 4 s; G1 512) |
|---|---|---|
| `alloy.build(scene, contact_capacity=64)` (256 for the G1) | 1.77x; part of it is the larger batch that then fits (0.5 MiB per world, G1 2.2) | no measured change: hold error, contact depth and held objects as shipped |
| the same and `scene.rigid_step = 0.0025` | 3.18x | the build warns that the arms' 2.5 ms contacts act as 5 ms (0.25x as stiff as their files ask); measured: no glitch, every object held; hold error up on the UR10e (0.045 against 0.020 rad); the LEAP hand's contact 0.06 mm deep against 0.02 |
| the same and `scene.rigid_step = 0.005` | 5.6x | the build warns that the arms' contacts act as 10 ms (0.06x as stiff as their files ask); not quality-neutral: Panda hold error 0.151 rad against 0.016, the LEAP hand's contact 0.27 mm deep against 0.02, G1 contacts 4.5 mm deep against 1.5 |

A pinned step is not for an arm that grips: in pick-and-place, 0.47 % of episodes had a penetration glitch at 2.5 ms
and 14 % at 5 ms, none at 1.25 ms. `scene.rigid_step = "auto"` takes the largest step the robot's stability caps
allow instead, never under 1.25 ms ([robot-fixed-step.md](../known_gaps/robot-fixed-step.md)). `contact_capacity` is
under Build options for speed, below.

## Different worlds

- One start per world: pass `pos=` as an `(N, 3)` array to the `add_*` of anything that moves: free and kinematic
  objects, kinematic fixtures (`add_open_box(..., kinematic=True)`: each part follows), cloth, liquids (the
  particles shift per world) and free-base robots. `scene.randomize(name, "x" | "y" | "z", lo, hi)` works for the
  same objects. Static objects and fixtures, static library objects and `add_pour` nozzles are the same in every
  world and refuse an array when added (`SceneError`); pass `kinematic=True` to hold one still and start it per world.
- Random values per world: `scene.randomize(name, field, lo, hi)`, fields `x`, `y`, `z` (the start coordinate itself:
  it replaces that coordinate of `pos`, not an offset, so `pos=(0.3, 0, 0.1)` with `randomize(name, "x", 0.25, 0.35)`
  spreads x around 0.3), `density`, `friction`, `youngs`, or a card path such as `material.surface.mu_dynamic`. Draws
  are uniform and seeded by `alloy.build(scene, seed=...)`: the same seed gives the same worlds, another seed other
  draws, and `sim.reset()` returns to the same draws.
- Robots and gravity take their own randomized fields (mass, friction, servo gains, ...), drawn at build or again
  at each reset (`when="reset"`):
  [Randomizing robots](randomize_robots.md).
- One joint target per world: `sim.set_targets({"arm/elbow": np.array([...])})` with N values.
- A world that fails (INVALID health) is frozen at its last good state (under `profile="fast"`, its last checked
  state: The rigid step); the other worlds go on.

## Determinism

On the CPU the `standard` and `precision` profiles repeat bit for bit: same scene, same `seed`, same results. On a GPU
no profile does once bodies are in contact at more than one pair (rigid piles, tumbling boxes, robots touching things,
and cloth as below); see `docs/known_gaps/determinism-gpu-rigid.md` for which scenes repeat and by how much the others
differ. There, compare runs with a tolerance and judge tasks by success rate; every GPU world of a scene with rigid
bodies, robots or cloth lists the notice DET_DOWNGRADED in `sim.health()` and stays OK (`health.md`). `profile="fast"`
gives up the repeat on the CPU too. Cloth on a GPU runs without Warp's ordered mode (the build logs determinism NONE),
so a sheet that touches itself (a fold, a pile, an unconverged `cloth_accuracy="fast"` drape) or touches two objects
at one point can make identical worlds and repeat runs differ, at every `cloth_accuracy`. On the CPU every cloth
setting repeats bit for bit, and identical worlds match, also once a sheet touches itself (not checked for a cloth
that touches itself in a robot's grip; `deformables.md`, Accuracy).

## Timing

GPU launches are asynchronous and the first steps compile kernels, so:
1. step once to warm up (`sim.step(1)`);
2. time `sim.step(n)`, then read any state (reading waits for the GPU) before stopping the clock;
3. world-steps per second = `worlds * (physics steps taken) / wall seconds`, with `sim.physics_steps` counting
   physics steps per world (several per control period).

`traj.meta` of every `sim.run` has `wall_s` and `physics_steps` for that run.

## Estimating before you build

`alloy.estimate(scene, worlds, seconds)` returns JSON-ready keys: `physics_steps_per_world`, `step_time_s`,
`wall_time_s`, `world_steps_per_s`, `real_time_factor`, `memory_bytes`, `compile_s_first_run`, `counts`, `solvers`
and `basis`. `measure=True` times a short real run instead of using the model. With a liquid the model also gives
`liquid`: for each setting (`"default"`, `"super"`) its spacing, particles, step and wall time, memory and measured
accuracy (`deformables.md`, Accuracy).

The wall time is of warm stepping, fitted to measured runs: a robot carrying water, bins, an arm, cloth at each
setting and a pour, at 1, 4 and 64 worlds (also 256 and 512 on the L4), on the CPU and on one L4. On those 55 rows it
is within 1.3x (`bench/results/estimate_calib/rows.json`, checked by `tests/api/test_estimate_calibration.py`); a
scene unlike them can be further off: no row has `liquid_accuracy="super"`, a soft body, sand, or cloth and liquid
beside kinematic objects (about 2x the same scene with those objects static, on the CPU), and a check of 8 other CPU
scenes on another machine (with other jobs running) measured 0.54-1.25x the estimate: it over-predicted a pour, a bin
of 12 objects and a Panda arm at 4 and 64 worlds by 1.3-1.9x. Past 64 worlds, and for cloth and liquid scenes unlike
the rows, an audit on an L4 measured slower warm steps than estimated: `AGENTS.md` section 7's sheet over a stand
1.1x at 1 world, 1.4x at 64, 2.4x at 256 and 4.2x at 1,024 (its self-contact grows with the pleats); the same sheet
flat on a wide table without self-contact 2.1-2.7x and 4 mm water at rest in a tub 1.8-2.5x, at 1 to 1,024 worlds;
a two-hinge arm 1.1x at 1,024 worlds and 0.64x at 16,384. A refit with many-world rows is planned. Three things
the estimate does not include:
- Graph capture. Right after a build or `sim.reset()`, a GPU run first captures CUDA graphs, over its first ~5
  control periods and again when a new graph is needed (a liquid's step changing mid-pour). A 1 s run right after a
  reset took up to 2.7x the estimate (the robot carrying water); longer runs share that cost out.
- A graph too big for the GPU's memory. Alloy then steps without one, much slower: the Panda arm at 1,024
  worlds on an L4 took 1.65 s per control period. Try fewer worlds.
- Memory for scanned meshes at many worlds: not among the memory rows below (the estimate before them read about
  14x low there).

`memory_bytes` on a GPU is the device memory the run needs to capture its CUDA graphs: what `nvidia-smi` shows for the
process once it steps. It is fitted to 20 rows on one L4 (`bench/results/estimate_calib/memory_rows.json`: a one-hinge
arm, a ball, the bin, the Panda scene, cloth, a pour and a robot carrying water, at 1 to 16,384 worlds) and is within
1.2x of each, and of 18 rows an audit measured on another L4 (section 7's sheet over a stand to 4,096 worlds, the
sheet flat, water in a tub, a two-hinge arm). 16,384 worlds of that sheet and 4,096 of the tub did not fit the L4's 23
GB; the estimate says 26 and 25 GB. About 0.4 GB is there before the first world (the CUDA context, kernels and the
first graph; 0.9 GB with self-colliding cloth, soft bodies or a robot in liquid, 0.65 GB with other cloth). Each world
then adds MuJoCo's constraint rows and contact slots as the build sizes them: 57 kB for the one-hinge arm (16,384
worlds: 1.3 GB), 0.7 MB for the bin and 11 MB for the Panda scene, whose 206 shapes get 9,888 rows a world
(`contact_capacity` and `constraint_rows` shrink that for the build; the estimate assumes the defaults). Where the
graph is too big and Alloy steps without one, the run holds less afterwards (the bin at 16,384 worlds: 11.4 GB at its
peak, 2.3 GB after). On the CPU `memory_bytes` is a rough count of the arrays, not fitted.

Check the wall time with `measure=True`. It does not measure memory (`memory_bytes` is None): before a large run,
build a smaller world count and watch the GPU's memory (`nvidia-smi`).

```python
import alloy

scene = alloy.Scene(worlds=256, dt=0.01)
scene.add_ground()
scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.2), mass=0.1)
scene.randomize("cube", "x", -0.1, 0.1)
est = alloy.estimate(scene, seconds=2.0)
print({k: est[k] for k in ("wall_time_s", "memory_bytes", "compile_s_first_run")})
```

## Profiles

`alloy.build(scene, profile=...)`:
- `"standard"` (default): health checks, bitwise repeat runs on the CPU, CUDA graphs on the GPU;
- `"precision"`: for sub-millimetre fits; checks every step and raises `SimulationError` when a world goes INVALID;
- `"fast"`: throughput first, lighter checks, no repeat guarantee, and the fast rigid mode (The rigid step);
- `"debug"`: no CUDA graphs, easier to step through.

## Build options for speed

Four more `alloy.build(...)` options, each in one process only (`processes=` refuses them):
- `contact_capacity=n` and `constraint_rows=m` (opt-in): MuJoCo's contact and constraint-row buffers per world. By
  default a robot of hundreds of collision shapes gets thousands of rows where it uses tens, and MuJoCo Warp (the
  GPU's route) runs its solver over the whole buffer. On one L4 at 2,048 worlds a Panda stepped 1.53x faster with
  `contact_capacity=64, constraint_rows=384`, a LEAP hand 2.1x and a G1 (512 worlds, 256 and 768) 1.7x, with the
  same contacts and solver iterations. `contact_capacity=n` alone sizes the rows for 2 n contacts from the robot
  file. Leave about twice what a world uses: with too few, the worlds read CONTACT_OVERFLOW or ROW_OVERFLOW and
  are frozen ([health](health.md)). The CPU's default route sizes its own rows: no gain there.
- `rerun_worlds=n` (opt-in; 0, the default, is off): in a scene with several solvers (a robot or free body beside
  cloth or a soft body), the engine sometimes re-runs a control period for a few worlds (a contact force jumped, or a
  world needed a finer exchange step). With `rerun_worlds=n` those worlds re-run alone in a second simulation of n
  worlds, built with this one, instead of every world re-running; the other worlds keep their first run, and the
  re-run ones match a whole re-run to rounding. Take `worlds // 16` as a rule of thumb (memory for n more worlds);
  `sim.report()["fine_reruns"]` counts the re-runs. It helps on the GPU. On the CPU's MuJoCo C route, beside a liquid
  or an MPM material, or when more worlds re-run than it holds, a re-run runs as before. Not with user plug-ins.
- `mpm_coupling`: how a robot or a body coupled to sand, soil or paste (MPM) trades forces with it. `"standard"`: two
  passes per exchange at 1.25 ms. The default, `"auto"`, runs one pass when the scene has a servo-driven robot and
  every body that starts near the material is that robot's link or welded, else standard (a free body that reaches the
  material later also runs one pass, resting within 0.5 mm of standard). `"fast"` runs one pass at 2.5 ms exchanges
  for every body: a free body landing on the material rests about 3 mm deeper, and one resting on it bobs about 15 µm
  before it settles. A robot in sand on one L4, seconds per 10 ms period: 0.365 standard, 0.188 auto, 0.104 fast for
  one world; 13.2, 7.0 and 4.3 over 64 worlds (auto's finger within 0.025 mm of standard). `sim.routes()` names the
  setting on the material; `ALLOY_MPM_COUPLING` sets it for a whole run, worker processes included.
- `mpm_sweeps="partial"` (opt-in; `"full"` is the default): pastes (dough, mayonnaise, ...) and other MPM materials
  re-sweep only the grid blocks still moving, with the same stop test: 256 dough worlds on one L4 ran about 1.3x
  faster, within 0.01 mm.

## The rigid step

Rigid bodies step several times per control period `dt`, at a fixed step of 1/600 s snapped down to `dt / 2^j`
(1.25 ms at `dt=0.01`) unless the scene allows a larger one.
- A scene of only free rigid bodies (no robots, kinematic objects, cloth, liquid, soft bodies, precision fits or
  plug-ins) steps them at the largest step that stays stable for its sizes, start drops, loads and materials:
  `dt / 2^j`, from 1.25 ms up to `dt`. Contacts never get softer than at 1.25 ms, so light parts keep 1.25 ms and
  heavy ones go faster. On the CPU, a 5 cm steel cube keeps 1.25 ms, a 10 cm one gets 2.5 ms and a 30 cm one 10 ms.
- The integrator: up to 8 free bodies step on MuJoCo's rk4 (not in a scene with an `add_mesh` shape on
  `collision="sdf"` or `"mesh"`: `scenes.md`, Meshes). More than 8 (none raised by the inertia floor) step on
  implicitfast with rk4's contact stiffness and a cap on the friction pull, which keeps a pushed or sliding body from
  rocking; when every body's inertia is the same about each axis within 3 % (cubes, spheres), they take a larger step
  with every contact as stiff as at the fixed step. That runs on MuJoCo Warp: on any GPU, and on the CPU with
  `ALLOY_CPU_ROUTE=warp`. The CPU's default route, MuJoCo C, has no cap yet and keeps rk4 for every free-body scene.
  Measured on one L4: 3,072 randomized resting stacks of 27 1 m cubes (10 kg/m³) stepped at 5 ms and ran 766x real
  time (774x and 759x in two runs; 8.9x for one world; on the CPU's default route they step at 2.5 ms); none fell,
  `sim.check()` passed, and the cubes rested within 0.19 mm of their ideal height (median; max 2.23 mm). For scale,
  the Newton MuJoCo solver bundled with Alloy, used plain with its default settings and one 10 ms step per period (a
  step we chose; its contacts are softer), ran the same stacks at about 1,800x (1,776x to 1,835x in five runs; 15.5x
  for one world), resting 1.00 mm from their ideal height (median; max 3.25 mm): Alloy's default is 0.42x as fast here
  and rests closer. What implicitfast gives up: a body spinning in free flight turns to first order, about 0.6 % of
  its angular momentum over a 2 s toss at 10 rad/s (`docs/known_gaps/rigid-clutter-rotation.md`);
  `profile="precision"` keeps rk4.
- `alloy.estimate(scene, ...)["rigid_step"]` gives the step (`dt_s`), the fixed step (`fixed_s`) and the cap that
  bound it (`reason`); `sim.routes()` names the integrator, and a step other than the fixed one, in each body's
  reasons.
- `scene.rigid_step = "auto"` is the default for the scenes above (a robot scene that never sets `rigid_step` keeps
  the fixed step; set `"auto"` or a number on it to opt in: Robot scenes, under How many worlds); `"fixed"` keeps the
  fixed step for every scene; a number in seconds pins a step (snapped down to `dt / 2^j`; the build warns, naming
  each stability cap it breaks). It is a scene attribute, not an `alloy.Scene(...)` keyword, and Scene JSON keeps it.
- The fast rigid mode (opt-in): `scene.rigid_step = "fast"`, or `alloy.build(scene, profile="fast")` on a scene whose
  `rigid_step` was never set (a value you set, `"auto"` included, wins over the profile). It steps the scenes `"auto"`
  steps (free rigid bodies only) at the largest step whose predicted rest sag stays within 1 % of each body's thinnest
  member, so contacts can be softer than at the fixed step; free bodies step on implicitfast (8 or fewer too, when
  each is close to isotropic) on MuJoCo Warp; on the CPU's default route they stay on rk4; and box-to-box faces keep 4
  contact points on the GPU instead of up to 8. Measured on the CPU (one cube, or stacks of 27): 1 m cubes of 10 kg/m³
  and 30 cm cubes step at 10 ms, 10 cm cubes at 5-10 ms, 5 cm cubes at 5 ms and 2 cm steel cubes at 2.5 ms; flat or
  long parts keep 1.25 ms. On one L4 the 3,072 stacks above stepped at 10 ms and ran 1,973x to 2,135x real time (three
  runs: 2,093x, 2,135x, 1,973x), about the plain solver's speed; none fell, `sim.check()` passed, and the cubes rested
  0.86 mm from their ideal height (median; max 0.87 mm). The next bullet is its cost.
- What the fast mode gives up: a resting body can sink up to 1 % of its thinnest member. With the friction pull capped
  (MuJoCo Warp: a GPU, or `ALLOY_CPU_ROUTE=warp`) a 30 cm cube sliding at 5 m/s rocks under 1° and lifts 1 mm. On the
  CPU's default MuJoCo C route, which has no cap, the fast mode keeps rk4: that cube rocks 6° and hops 8 cm (2° and
  1 cm at 2 m/s). Under `profile="fast"` (health checks every 10 steps) a world that goes INVALID freezes at its last
  checked state: up to 9 steps before the failure at one step per period (the fast mode's 10 ms at `dt=0.01`), up to
  lcm(10, steps per period) steps at other counts that share no factor with 10, and at the start of its failing
  period, as under the other profiles, at counts that do (2, 4, 8). The build warns with this;
  `alloy.estimate(scene, ...)["rigid_step"]["fast"]` gives the fast mode's step, wall time and `gives_up` beside the
  scene's own step. Keep `"auto"` (contacts as stiff as at the fixed step) where that sag matters, and on the CPU's
  default route for anything pushed, slid or carried.
- The precision and debug profiles keep the fixed step.
