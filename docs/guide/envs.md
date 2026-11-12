# Gym / PyTorch environments

`alloy.envs.VectorEnv` turns a scene into a vectorized environment where every world is one environment and all of
them step at once. It has the shape of Gymnasium's vector API (`reset`, `step` with `terminated` and `truncated`,
spaces, the three autoreset modes); [With Gymnasium and RL libraries](#with-gymnasium-and-rl-libraries) says what
that covers and what needs glue. Observations and actions are torch tensors on the simulation device when PyTorch is
installed (else NumPy).

```python
from alloy.envs import make

env = make("GripperLift-v0", num_envs=4)              # a ready-made task; GPU if there is one
obs, info = env.reset(seed=0)
print(env.obs_names()[:4], env.action_names())
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(env.xp.to_numpy(reward))
```

- Your own task: [below](#your-own-task). Camera observations: `examples/envs/03_camera_obs.py`. Training with a
  compact PPO: `examples/envs/04_train_ppo.py` ([what it trains](#training)). Training with rsl_rl,
  Stable-Baselines3 or skrl, and observation groups for a privileged critic: [rl.md](rl.md). Alloy provides
  environments and adapters, not its own RL library.
- Finished worlds reset by themselves; `info["_episode"]` marks them and `info["episode"]["r"]` holds their returns.

## With Gymnasium and RL libraries

The envs run without Gymnasium. What works with it today, and what does not:

- Gymnasium comes with the `envs` extra (or `pip install gymnasium`, 1.0 or newer). With it, `VectorEnv` subclasses
  `gymnasium.vector.VectorEnv` and its spaces are Gymnasium's; without it, the same methods with stand-in spaces.
- `gym.make` does not build these envs: there is no single-env form, and nothing is registered on import. After
  `alloy.envs.register_gymnasium()`, `gym.make_vec("alloy/GripperLift-v0", num_envs=64,
  vectorization_mode="vector_entry_point")` builds one (tested with `backend="numpy"`).
- Gymnasium's vector wrappers work on NumPy arrays and expect its next-step autoreset. The envs default to torch
  tensors and same-step autoreset, so build with `backend="numpy", autoreset="next_step"` for them. Only `make_vec`,
  `reset` and `step` are tested with Gymnasium here; the wrappers are not.
- RL libraries: `alloy.envs.wrappers` adapts the env to rsl_rl, Stable-Baselines3 and skrl, and
  `python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl` runs PPO training with one command (each
  library is a separate install; versions tested and what each reached: [rl.md](rl.md)). Any
  other library needs a short adapter to its own env class, written by you. `examples/envs/04_train_ppo.py` trains on
  the env directly.

## What to install

`pip install 'alloy-engine[envs]'` adds what the envs need beyond the
default install:
- PyTorch, for torch observations and the PPO example (`examples/envs/04_train_ppo.py`). Without it the envs return
  NumPy arrays, and `backend="torch"` names the command.
- huggingface_hub, for the library objects of `PickPlace-v0` (a YCB object) and `DrawerOpen-v0` (a Lightwheel cabinet
  from the RoboCasa pack) on the Hugging Face Hub: the first build fetches them. Without it, `make` says so and names
  the command.

## Ready-made environments

`make(name, num_envs=..., device=..., seed=..., **options)`; an option goes to the scene, the task or the env by name
(`camera=True` adds 64x64 images to the observation, `obj=` picks PickPlace's object, `backend="numpy"`).

| name | task |
|---|---|
| `GripperLift-v0`, `GripperLiftCamera-v0` | a two-finger gripper lifts a cube 8 cm and holds it (the second: images in the observation) |
| `PandaReach-v0` | a Franka Panda arm moves its hand to a random target |
| `PickCube-v0` | a Franka Panda arm picks a 4 cm cube off its table and carries it to a random point in the air ([rl.md](rl.md), Arm pick) |
| `PickPlace-v0` | a gantry gripper puts a YCB object in a bin (`obj=`) |
| `PegInsert-v0` | insert a 10 mm peg 2 cm into a 10.5 mm hole |
| `DrawerOpen-v0` | pull the drawer of a Lightwheel cabinet (from the RoboCasa pack) 15 cm open |
| `Pour-v0` | move a bowl under a stream of water and catch 75 % of it |
| `ClothFold-v0` | experimental: fold a towel along its diagonal (slow: a GPU) |

`examples/envs/05_pick_place.py` to `09_cloth_fold.py` solve the library tasks with scripted policies.

Camera images show the scene as `sim.render()` and `alloy.Camera` draw it: library looks, scanned objects' photo
textures (the YCB object of `PickPlace-v0`), a grey ground, and cloth, liquid and soft bodies from the state. A scene
variant's looks (`vary=True`, the library tasks' default: `alloy.vary` textures and tints) are drawn by the fast
camera, as each look's colour and its colour map tiled over the shape, where `Camera.render(sim)` and `sim.render()`
draw them with the full-material tracer: a policy sees the same surfaces, lit and tone-mapped more simply than a video
of the sim. The looks cost nothing measurable: on one L4 at 512 worlds, 64 px images at 4 samples, `PickPlace-v0`
rendered 15,100 images/s with them and 14,400 with the palette colours drawn before; `GripperLiftCamera-v0`, with no
looks, 23,800 either way (median of 3 runs each). The full-material tracer would draw 740 images/s there.

- The images changed in this release: release candidates up to rc8 drew the engine's palette colours (the ground and
  the first object one blue) and no looks, so a policy trained on their images sees new colours; retrain it.
- `vary=True` draws one variant per environment: every world of it shows the same looks (the seed picks which).
- A fresh install has no texture images: a varied look whose texture is not on disk is drawn in a flat colour of its
  family (a brown floor, grey plastic, ...), with one warning. `python -m alloy.looks --fetch-textures` downloads the
  images once; the policy images then show the textures.

In `PickPlace-v0` and `DrawerOpen-v0` the gantry gripper's position servos can push a finger into the bin's 6 mm
wall, the floor or the cabinet past the health limit (10 mm, or half the thinner part): health marks the world
INVALID, `info["invalid"]` is True and the episode ends. In one check on an L4, uniform random actions sent 16 of
256 PickPlace worlds INVALID within 5 s; learned policies lost up to 7 % of their episodes this way (0-17 of 256 on
PickPlace, 0-8 of 256 on DrawerOpen) and the shipped scripted demonstrator, with action noise, 3 of 1,280.
`PegInsert-v0` uses the same gripper and was not counted. Such an episode ends truncated
(`Task(on_invalid="terminate")` terminates it; the rsl_rl, SB3 and skrl wrappers treat it as terminated either way,
so no value is bootstrapped from a frozen world). On that step its reward (except the task's `invalid_penalty`, a
penalty you set; default 0), its reward terms, its success and its last observation read 0, so a policy gains
nothing from a frozen world. The zeroed observation is the episode's last one: `info["final_obs"]` under the default
`autoreset="same_step"` (the observation `step` returns is already the next episode's first), the returned one under
`"next_step"` or `"disabled"`. A reset clears the world's record in `sim.health()` and `sim.check()`; the env keeps
the count: `info["episode"]["invalid"]` marks the episodes that ended frozen, and `env.invalid_episodes` counts them
per world since the env was built: once per frozen episode, and `env.reset()` does not clear it. With
`autoreset="disabled"` a frozen world stays frozen, so it counts once per step until you reset it. Each line the
one-command trainer prints gives `invalid_episodes`, the count since the previous line.

`ClothFold-v0` has two grasps, and results depend on which one you use:
- `grasp="friction"`, the default: the fingers hold the towel by friction alone, as a real gripper does. The
  scripted demo folds 25-27 of 64 worlds in three runs on one L4. A friction pinch on cloth creeps slowly under load
  and lets go before its friction limit (`docs/known_gaps/cloth-robot-grasp-carry.md`).
- `make("ClothFold-v0", grasp="attach")`, opt-in: a shortcut, not a friction grasp. While the fingers are closed on
  the towel, the towel between them is held to a finger until they open, as cloth benchmarks grasp
  (`alloy.envs.cloth_attach`). It is one-way: the gripper does not feel the towel's weight. The same scripted demo
  folds 37 of 64 worlds in this mode (one run on one L4). Compare a result in this mode only with other attach
  results.

## Your own task

`VectorEnv(scene, Task(...), worlds=N)` over any `alloy.Scene`. A `Task` names the observations, the actions, the
reward terms and the episode's end; its terms are checks (`alloy.envs.checks`), each a function of every world's
state:

```python
import alloy
from alloy.envs import Task, VectorEnv, checks as C
from alloy.envs.robots import gantry_actions, gantry_gripper

scene = alloy.Scene(dt=0.02)
scene.add_ground()
scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.025), mass=0.1)
scene.add_robot("gripper", mjcf=gantry_gripper(), pos=(0.0, 0.0, 0.2))      # x, y, z slides, yaw, two fingers

lifted = C.lifted("cube", by=0.05)                                          # bool per world
task = Task(
    obs=["gripper", "cube"],                                # the robot's joints, the cube's pose and velocity
    actions=gantry_actions("gripper", yaw=False),           # x, y, z and close, as deltas in [-1, 1]
    reward={"reach": (-1.0, C.distance("gripper/palm", "cube")),   # name: (weight, check), summed every step
            "lift": (10.0, lifted),
            "upright": (1.0, 1.0 - C.tilt("cube") / 90.0)},         # checks do arithmetic and comparisons
    success=lifted & C.upright("cube"), max_steps=100)
env = VectorEnv(scene, task, worlds=4)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(env.obs_names()[:3], env.action_names(), sorted(info["terms"]))
```

- `Task` also takes `failure=` (a check that ends the episode as failed), `cameras=` (`alloy.camera.Camera`s), `goals=`
  and `randomize=` (per-episode random starts: `alloy.envs.randomize`). `examples/envs/02_custom_task.py` runs one.
- `actions=ActionSpec(joints, mode="delta" | "absolute")` drives any robot's servos; with no `actions`, every servo.
  In delta mode an action of 1 moves a joint's target `delta` (m or rad) in that step, starting from the last target
  kept within `window` of the joint's measured position (default 5 x `delta`), so a joint held by a wall, a table or
  a part does not run its target on and push harder every step. A slow servo, one that lags its target by more than
  the window at full speed, moves slower under full actions than with no window: the pusher of
  `examples/envs/02_custom_task.py` (kp 400) reaches its goal in 91 steps, 47 without one. Trajectories change even
  where outcomes do not: the window binds in 86 of the GripperLift demonstrator's 100 steps, which still succeeds in
  every world. `window=float("inf")` turns it off (targets then add up from the last target, unbounded).

Limits of your own task, each with its workaround:

- A robot with no servos ignores actions, silently. A robot file with no `<actuator>` section builds, and `ActionSpec`
  over its joints steps with no error or warning while the actions move nothing (30 delta actions left a slide at 0;
  outside an environment `sim.set_targets` on such a joint raises). Workaround: before training, print
  `env.sim.names()["actuators"]` and check it covers every joint you drive; add a `<position joint="..." kp="..."/>`
  line for each one that is missing.
- A rod's centre line is not in the task state: `"<rod>.points"` is a key of `sim.state()` only. Workaround: the rod's
  link bodies, `"<rod>/0.pos"` and up ([rods.md](rods.md), State).
- Two slide joints on one MJCF body merge into one joint named `x_y` that actions cannot drive: use one body per joint
  ([robots.md](robots.md), Adding a robot).
- `VectorEnv(scene, task, ...)` builds the simulation itself with `worlds`, `device`, `seed` and `profile` only: it
  cannot pass `contact_capacity`, `constraint_rows` (the robot speed opt-in of [worlds.md](worlds.md)) or any other
  `alloy.build` option, and neither can `alloy.envs.make`. Workaround: build first and hand over the simulation,
  `sim = alloy.build(scene, worlds=1024, seed=0, contact_capacity=64, constraint_rows=384)`, then
  `VectorEnv(sim, task)`. The world count, the device and the seed of `scene.randomize` are then the build's:
  `worlds=`, `device=` and `seed=` given to `VectorEnv` beside a built simulation are dropped without a message (a
  built 3-world simulation with `worlds=8` gives 3 environments), so set them in `alloy.build`.

### Checks

Every check in `alloy.envs.checks` gives one value per world, read at the step it is called on. Where a check takes a
point, pass a body name (its origin), `"<body>.com"` (its centre of mass), a goal (`"target"`) or a fixed `(x, y, z)`.

| Check | Per world |
|---|---|
| `height(body)` | the body's height z (m) |
| `distance(a, b)` | the distance between two points (m) |
| `speed(body)` | the body's linear speed (m/s) |
| `joint(name)` | a one-axis joint's value (m or rad) |
| `tilt(body)` | the angle between the body's z axis and world up (degrees) |
| `rise(body)` | how far the body rose since the episode started (m; negative if it fell) |
| `fraction_inside(particles, lo, hi)` | the share of a liquid's or cloth's particles in the box `lo`-`hi` (0 to 1) |
| `action_size()` | the last action squared and summed over its entries (a smoothness penalty) |
| `near(a, b, tol)` | bool: the two points are at most `tol` m apart |
| `lifted(body, by)` | bool: the body rose at least `by` m since the episode started |
| `upright(body, max_tilt_deg=10.0)` | bool: `tilt(body)` is at most `max_tilt_deg` |
| `at_rest(body, max_speed=0.01)` | bool: the body moves at most `max_speed` m/s |
| `inside(body, lo, hi)` | bool: the body's point inside the box `lo`-`hi` (corners in m) |
| `between(obj, a, b, max_offset=0.03, max_gap=0.15)` | bool: `obj` between fingers `a` and `b` (at most `max_offset` m off their line, fingers under `max_gap` m apart) |
| `all_of(*checks)` | bool: every check holds |
| `any_of(*checks)` | bool: at least one check holds |
| `custom(name, fn)` | your own: `fn(state)` returns one value per world, e.g. `lambda s: s["cube.pos"][:, 0]` |

Checks combine with `+ - * /`, comparisons (`<`, `<=`, `>`, `>=`), `&`, `|` and `~`; a bool check counts 1 or 0 in
arithmetic. A `reward` term is summed every step; `success` and `failure` are read every step too.

### When an episode counts as a success

`info["success"]` is the `success` check at this step, per world. `info["episode"]["success"]` (with
`info["_episode"]` marking the worlds whose episode just ended) is that check at the **last step** of the episode,
not at any step before it: it says whether the task holds when the episode ends.

- With `end_on_success=True` (the default), an episode ends at the first step its `success` check holds, so its
  `info["episode"]["success"]` is True.
- With `end_on_success=False`, an episode runs to `max_steps`: a cube that was lifted and then dropped ends with False.
- An episode ended by `failure=` or `max_steps` reports the check at that step (usually False); one ended by an
  INVALID world reports False.
- A scene whose worlds share one episode clock (`env.together`) ends every episode at `max_steps`; success is still
  read at that step. The task-port envs below report their port's verdict, judged when the run ends.

To count a success at any step, keep your own record of `info["success"]`:

```python
import numpy as np

import alloy
from alloy.envs import Task, VectorEnv, checks as C
from alloy.envs.robots import gantry_actions, gantry_gripper

scene = alloy.Scene(dt=0.02)
scene.add_ground()
scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.3, 0.0, 0.3), mass=0.1)     # starts in the air and falls
scene.add_robot("gripper", mjcf=gantry_gripper(), pos=(0.0, 0.0, 0.2))
task = Task(obs=["cube"], actions=gantry_actions("gripper", yaw=False),
            success=C.height("cube") > 0.1,                 # holds for the first 0.2 s only
            end_on_success=False, max_steps=25)             # so every episode runs 25 steps
env = VectorEnv(scene, task, worlds=2, backend="numpy")
obs, info = env.reset(seed=0)
ever = np.zeros(env.num_envs, bool)
for _ in range(task.max_steps):
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, np.float32))
    ever |= info["success"]                                 # held at this step
print(info["_episode"], info["episode"]["success"], ever)   # [ True  True] [False False] [ True  True]
```

## Training

`examples/envs/04_train_ppo.py` is a compact PPO to show the envs learn. These are small tasks of ours, not a standard
benchmark, and each number is one run of the script as shipped (seed 0) on one L4: `GripperLift-v0` with 512 worlds
reaches about 97 % success by iteration 100 and 99 % after 200 (4.5 min with the first-run compile); `PandaReach-v0`
with 1,024 worlds about 97 % in 4 min; `DrawerOpen-v0` with 256 worlds opens the drawer in about 27 % of episodes
after 80 iterations (2.6 min). It does not learn `PickPlace-v0`, and the env steps several times slower once the
fingers touch the object. Picking up an object and carrying it to a bin is a
long, sparse task that needs reward shaping, a curriculum or demonstrations and many more samples: bring your own
trainer for it. The scripted policy in `05_pick_place.py` puts the object in the bin in most worlds, not in all. On
the CPU: 4 of 4, 8 of 8 and 64 of 64 worlds (CPU runs repeat bit for bit). On one L4: 4 of 4 and 8 of 8 worlds, 60
to 62 of 64 worlds in four runs (94-97 %) and 250 of 256 in one; runs with the same seed differ there, because
contacts on a GPU do not repeat bit for bit.


## Ports of Gym-Anything for Robotics tasks (unofficial)

Of Gym-Anything's 200 tasks, none has been run and scored by Gym-Anything's own verifier on Alloy. 28 have a scripted
Alloy port with our own checks: Alloy's re-implementation of the task's success gates, judging k of n of them (5 ports
judge all n, each with 1 to 6 gates by another measure than Gym-Anything's; 23 judge part), with Alloy's scripted tools
on a fixed start, not the benchmark's robot. For 132 of the 200 we read every physics class the task needs as supported
in Alloy (67 partly, 1 not). That is a reading of physics coverage from commits and tests, not a run of the task: of 20
such tasks the audit sampled, none exists in Alloy as Gym-Anything scores it.

`make("PaintRollStrip-v0")` and 27 more envs are Alloy's own ports of tasks from Gym-Anything for Robotics (its
RoboWorld benchmark; `gym-anything/robotics` at commit `febb260`, MIT). They are unofficial: its authors did not make,
review or endorse them, and no number from these envs is a RoboWorld result. The 5 ports that judge every success
gate of their task are `PaintLoadRoller-v0`, `ConcretePatchFlush-v0`, `ConcreteToolSealant-v0`,
`ConcretePatchDrywall-v0` and `GiftOpenFlaps-v0` (each judges 1 to 6 of its gates by another measure than the task's);
the other 23 judge a part of them.

- **A port is not the task.** Each port rebuilds a task's scene on Alloy, often smaller (a crop of the wall or paper,
  one rack bay, a coarser grid), with Alloy's own scripted expert and another actor than the task's mobile robot or
  humanoid. Only 4 of the 28 drive a robot arm, a fixed FR3 (`PaintRollStrip-v0`, `PaintLoadRoller-v0`,
  `PaintBrushStroke-v0`, `PaintSprayStencil-v0`). In the other 24 the policy moves a tool, a scripted hand's targets, a
  gantry's servos, a tool carriage or the truck's controls directly.
- **The criteria are Alloy's re-implementation.** Each port re-implements the task's success checks in Alloy's own
  code, on Alloy's own simulation. No port was run through Gym-Anything's verifier. The tables under
  [What each port judges](#what-each-port-judges) list, per env, the task's criteria the port judges as the task's
  code does, those it judges by another measure, and those it does not judge.
- **A success is the port's.** `info["success"]` is the port's verdict: every check the port judges holds. It is not
  a pass of the Gym-Anything task. The reward is 0 on every step and the port's score at the end (1 for a pass).
- **What was run.** Each port's own scripted episode was run through its env. In 27 of the 28 the port's own checks
  held at the end on our machines, which is not a pass of the Gym-Anything task; four of those episodes ran a reduced
  scene: a window of the pad or wall (`ConcretePadEdge-v0`, `ConcreteToolJoints-v0`, `ConcreteToolSealant-v0`) or
  10 mm cells (`ConcretePatchDrywall-v0`). In a second set of runs with default options, the checks held in 25 of
  the 28, `RinseGlassware-v0` and `ConcreteToolJoints-v0` gave no verdict in the time allowed and
  `ConcretePadEdge-v0` was not run ([What one episode costs](#what-one-episode-costs)). Both sets ran before the
  ports' verdicts were reworked to count judged criteria only, and were not run again through the envs. Since the
  rework, on a GPU, the reheat-porridge port's own run passed its verdict (3 of 5 criteria judged) in 64 of 64
  worlds, and the stir-sauce port's (not an env; 4 of 6 judged) in 64 of 64.

Each env runs its port's own scene and script one control period per `step`; your action replaces the script's command
for that period, and `info["expert_action"]` is the script's (stepping with it reproduces the port). A hand's or a
kinematic tool's position may be set up to 3 m either side of where its body starts (the scenes sit metres from the
origin), so the expert's action lies inside the action space and a wrapper may clip to it. Observations are
the proprio of what the policy moves and, with `camera=(width, height)`, an image framed on the scene (paint, concrete
paste, rinse water and soil films, the gift bag's paper and the porridge are drawn). Run one:
`examples/envs/10_task_ports.py`. Each family's module in `alloy/envs/ports/` (`painting.py`, `rinsing.py`,
`concrete.py`, `forklift.py`, `gifts_cooking.py`) lists its envs, actions and options. An episode is the port's whole
run and can take hours: [What one episode costs](#what-one-episode-costs) gives each env's measured time, and names
the three whose default has not been seen to finish.

| family | envs | the policy moves |
|---|---|---|
| painting (8) | `PaintRollStrip-v0`, `PaintLoadRoller-v0`, `PaintBrushStroke-v0`, `PaintSprayStencil-v0`, `PaintSwatchCard-v0`, `PaintClearCoat-v0`, `PaintCutInTrim-v0`, `PaintBehindRadiator-v0` | an FR3's joint targets, or the brush or can (pose, velocity, valve) |
| surface rinsing (7) | `RinseSponge-v0`, `RinseDrainWash-v0`, `RinseTomatoes-v0`, `RinseFlushFloor-v0`, `RinseGlassware-v0`, `RinsePlates-v0`, `RinseGarbageCan-v0` | a hand's targets on the held items, levers, faucet, hose valve |
| concrete finishing (5) | `ConcretePadEdge-v0`, `ConcretePatchFlush-v0`, `ConcreteToolJoints-v0`, `ConcreteToolSealant-v0`, `ConcretePatchDrywall-v0` | the tool carriage's targets, or the knives |
| forklift (5) | `ForkliftStartAndPark-v0`, `ForkliftRampDownLoaded-v0`, `ForkliftPutAwayRack-v0`, `ForkliftLoadTrailer-v0`, `ForkliftLoadBales-v0` | the truck's controls |
| gift bags (2), cooking (1) | `GiftOpenFlaps-v0`, `GiftBagCandle-v0`, `CookReheatPorridge-v0` | a gantry's servos; the spoon and the hob |

### What one episode costs

One episode on the port's own script, one world, default options unless the row names one, first-run compiles
included. L4: one NVIDIA L4, from the 0.1.0 release-candidate wheel. CPU: 16 or 32 cores, one run or two per number;
the 16-core runs from that wheel, the 32-core runs from a source tree. All of these ran before the ports' verdicts
were reworked. The envs do not end early: a policy's episode takes as long as the script's. "No verdict" means the
run was stopped before the episode ended.

Three defaults have not been seen to reach a verdict from the released package, on any machine:
`RinseGlassware-v0`, `ConcreteToolJoints-v0` and `ConcretePadEdge-v0`. Each has a shorter option (below): one glass
(`glasses=1`) or a `window`. That is a piece of the scene, not the task, and its row says what its verdict was.

| env | steps | one L4 | CPU |
|---|---|---|---|
| `PaintRollStrip-v0` | 18,258 | not run | 6,306-8,460 s (1.8-2.4 h); `size="small"`: 8,533 steps, 393 s |
| `PaintLoadRoller-v0` | 18,258 | not run | 6,228-8,438 s (1.7-2.3 h); `size="small"`: 8,533 steps, 373 s |
| `PaintBrushStroke-v0` | 1,443 | not run | 361-504 s |
| `PaintSprayStencil-v0` | 4,966 | not run | 3,477-4,717 s (1.0-1.3 h) |
| `PaintSwatchCard-v0` | 4,712 | 165 s | 189-287 s |
| `PaintClearCoat-v0` | 2,207 | 80 s | 1,102-2,038 s; `size="small"`: 953 steps, 62 s |
| `PaintCutInTrim-v0` | 7,103 | 60 s | 1,262-2,367 s |
| `PaintBehindRadiator-v0` | 91,896 | 626 s | 12,620 s (3.5 h) |
| `RinseSponge-v0` | 1,298 | 391-462 s | 292-349 s |
| `RinseDrainWash-v0` | 2,480 | 57 s | 46-76 s |
| `RinseTomatoes-v0` | 2,211 | 258 s | 887-1,510 s |
| `RinseFlushFloor-v0` | 16,990 | 649-879 s | 7,020 s (2.0 h) |
| `RinseGlassware-v0` | 23,200 in the last run that finished, on an older build; this build's count has not been measured | no verdict on this build: stopped after 2,700 s (1 world) and after 2.8 h (4 worlds). The last run seen to finish was the port's own on an older build, not through the env: 4 worlds in 2,950 s (49 min), 0.13 s per step | no verdict: about 2 s per step on 16 cores (about 1,800 steps in 3,600 s; 23,200 steps at that rate is about 13 h); stopped after 4 h on 16 cores and after 10.5 h on 32. One glass, not the six-glass task: `glasses=1, coarse=True` (the port's coarse test scene) runs 3,060 steps. From the released package the expert's own action passes the port's verdict (4 of 8 criteria judged) in 137-194 s (three runs: two on a 16-core cloud machine, one on another host). With every action clipped to that package's action space it failed 4 runs of 4 in 121-160 s (`soap_free`, `sanitized`, `placed`, `intact_and_not_dropped`): the hand's position box was 3 m about the origin and the glass station sits beyond x = 3 m. The box now sits about where each glass starts, so a clip keeps the expert's action; a clipped run on this build has not been timed. `glasses=1` alone: 3,950 steps, 1,072 s from a source tree |
| `RinsePlates-v0` | 10,600 | 590 s | 12,049 s (3.3 h) |
| `RinseGarbageCan-v0` | 7,519 | 405-662 s | 944-1,379 s |
| `ConcretePadEdge-v0` | not known for the default | not run | the default (the whole pad) has never been run, on any machine. `window=(0.37, 0.37)`, a corner of the pad: 1,053 steps, 1,283 s and 1,742 s (1.2-1.7 s per step) through the env from a source tree; the port's own run of that corner 637 s and 852 s |
| `ConcretePatchFlush-v0` | 4,028 | 612 s | 2,206 s; `cell=0.001` (the task's grid) was not run: about 53 times the cost per step |
| `ConcreteToolJoints-v0` | 46,271 | no verdict through the env: stopped after 2,700 s, on the port's older program. The port's own run of the whole wall on its current program, not through the env and not from the released package: 1 world, 11,267 s (3.1 h), 0.24 s per step | no verdict: stopped after 4 h on 16 cores (the older program); no whole-wall run has finished on a CPU. `window=(3.86, 4.06, 0.783, 1.04)`, a 0.2 x 0.26 m piece of the wall: through the env from a source tree, 3,685 steps in 2,360 s on the older program and 3,466 steps in 3,115 s (0.9 s per step) on the current one |
| `ConcreteToolSealant-v0` | 15,364 | 2,488 s (41 min) | the default was not run; `window=(1.155, 1.185)`: 212 steps, 28 s |
| `ConcretePatchDrywall-v0` | 910 | 76 s | 12,642 s (3.5 h); `cell=0.01`: 27 s |
| `ForkliftStartAndPark-v0` | 867 | 213 s | 37-51 s |
| `ForkliftRampDownLoaded-v0` | 2,513 | 301 s | 114-122 s |
| `ForkliftPutAwayRack-v0` | 2,824 | 321 s | 96-126 s |
| `ForkliftLoadTrailer-v0` | 2,963 | 355 s | 150-176 s |
| `ForkliftLoadBales-v0` | 9,187 | 1,077 s | 337-440 s |
| `GiftOpenFlaps-v0` | 2,650 | 40-163 s | 91-132 s |
| `GiftBagCandle-v0` | 1,100 | 942-1,285 s | 12,071 s (3.4 h) on 32 cores; no verdict after 4 h on 16 |
| `CookReheatPorridge-v0` | 6,000 | 1,285 s; a second start was stopped at 2,400 s | not run |

The four FR3 painting envs were timed from a source tree (they did not build from that wheel; they do now), and the
times predate the engine's last update. For a quick look use `ForkliftStartAndPark-v0`, `RinseDrainWash-v0`,
`GiftOpenFlaps-v0` or a row with an option.

### What each port judges

Read from the port code and Gym-Anything's task code at the commit above; names are Gym-Anything's.

- **as the task does**: the same quantity and threshold as the task's code, computed from Alloy's simulation.
- **by another measure**: the port's verdict needs a check of that name, but it reads another quantity, a smaller
  region, coarser cells or fewer samples; the note says which.
- **not judged**: the port has no such check, only reports the value, reads only a part of what the task reads (the
  note says which part), or the check cannot pass or fail on the simulation (the thing is static or missing, or the
  check reads a command, not a state). A pass says nothing about these.

Where a port's verdict reports its own count, the row quotes it ("judged k of n"); `info["verdict"]` holds the port's
result, with that count and the reason for each criterion it does not judge. Success counts the judged criteria only.

**Painting.** The task has a mobile FR3; the ports use a fixed FR3 (the first four) or a free-moving brush or can
(the last four), on a smaller chart, and all but behind-radiator read the film after a settle of 10 s or 60 s where
the task settles until the film sets (up to 30 min). No port tracks marks on wet paint. All but the load-roller port
also need their paint books to close, which the task does not ask.

| env (task) | as the task does | by another measure | not judged |
|---|---|---|---|
| `PaintRollStrip-v0` (roll-strip; judged 4 of 7) | `hide`, `no_laps` | `no_runs` (paint front travel over a 60 s settle, the task set's older rule), `uniform` (other patch sampling) | `protected` (only paint spilled over the chart's bottom edge is read: the port has no ceiling, trim, outlet or radiator), `roller_in_tray` (the position is read, not the release: the roller is never let go), `untouched` |
| `PaintLoadRoller-v0` (load-roller; judged 4 of 4) | `load`, `spread`, `held_above_tray` | `no_drip_10s` (allows one drop) | |
| `PaintBrushStroke-v0` (brush-stroke; judged 1 of 3) | | `within_line` (inside a 0.24 x 0.10 m crop of the paper only) | `coverage` (the film's thickness is read; with one paint the pigment share cannot fail), `untouched` |
| `PaintSprayStencil-v0` (spray-stencil; judged 4 of 5) | `coat`, `sharp_edges`, `no_runs` | `overspray` (the port's own count of paint outside the box) | `untouched` |
| `PaintSwatchCard-v0` (swatch-card; judged 2 of 3) | | `coverage` (total film thickness, not the blue film's: looser by up to 5 %), `within_line` (inside an 81 mm crop only; brush drips added) | `untouched` |
| `PaintClearCoat-v0` (clear-coat; judged 4 of 5) | `coat`, `even` | `no_runs` (also fails a film at the port's depth cap), `overspray` (the port's own count) | `untouched` |
| `PaintCutInTrim-v0` (cut-in-trim; judged 4 of 5) | `hide`, `no_laps` | `no_runs` (the port's own flowed-paint measure, stricter), `protected` (only the trim and plate inside each cropped chart) | `untouched` |
| `PaintBehindRadiator-v0` (behind-radiator; judged 3 of 4) | `hide` | `no_runs` (stricter, as above), `protected` (two drip-catch charts and the film lost; looser for a thin smear in a sink) | `no_marks` |

Scenes the paint verdicts state: roll-strip paints a 0.6 x 1.0 m strip (the task: 0.6 x 2.23 m) over a uniform first
coat (the task: patchy by 15 %); brush-stroke and swatch-card use an acrylic of 0.65 solids by volume (the task: 0.45,
so the same wet coat leaves 1.44 times the dry film against the same 5 µm and 15 µm limits); spray-stencil has one
fixed crate pose (the task moves it with the seed); every port has one start (the seed changes nothing).

**Surface rinsing.** The task has a two-arm mobile FR3; the ports use a scripted hand on the held items (no arm) or a
free-moving hose valve. In every port `no_simulation_fault` is the port's own check (its water and soil books close,
the state is finite), counted as judged. No port judges the hands half of `water_off_and_hands_free` (the scripted
hand has no finger pads; flush-floor has no hand), so that criterion is not judged anywhere: the water half is read
from the outlet's flow, and an open outlet or a hand still gripping fails the port.

| env (task) | as the task does | by another measure | not judged |
|---|---|---|---|
| `RinseSponge-v0` (rinse-sponge) | `no_suds`, `wrung_out`, `sponge_on_ledge`, `intact_and_not_dropped` | | `water_off_and_hands_free` (the faucet's flow is read, not the task's spray valve) |
| `RinseDrainWash-v0` (drain-wash-compartment; judged 4 of 7) | `drained`, `waste_closed`, `others_kept` | | `floor_dry` (only bowl overflow is read, not floor films), `water_off_and_hands_free`, `intact_and_not_dropped` (the port has no item) |
| `RinseTomatoes-v0` (rinse-tomatoes; judged 7 of 8) | `none_lost`, `placed`, `drained` | `visibly_clean` (a tomato's total dust over 1 cm2, stricter than its worst 1 cm patch), `contained` (the soil film only, read stricter; the port's own floor-water count), `intact_and_not_dropped` (tomatoes and colander only) | `water_off_and_hands_free` |
| `RinseFlushFloor-v0` (flush-floor; judged 4 of 6) | `valve_parked`, `contained` | `floor_flushed` (per 10 mm cell) | `water_off_and_hands_free`, `intact_and_not_dropped` |
| `RinseGlassware-v0` (rinse-glassware; judged 4 of 8) | `drained`, `placed` | `sanitized` (the port's own contact clock, which runs on through the settle) | `soap_free` (mass and gauge are read; there is no foam), `contained` (floor water is read; the drainboard is not), `intact_and_not_dropped` (only released glasses' speed and height), `water_off_and_hands_free` |
| `RinsePlates-v0` (pre-rinse-plates; judged 3 of 7) | `valve_parked`, `placed` | | `visibly_clean` (the plate's centre disc is not charted), `contained` (floor water is read; the clean table is not), `intact_and_not_dropped` (only released plates' speed and height), `water_off_and_hands_free` |
| `RinseGarbageCan-v0` (rinse-garbage-can) | `valve_parked`, `contained` | `emptied_upright` (a stricter free-water count) | `soap_free` and `visibly_clean` (30 mm cells; the can's underside is not charted; there is no foam), `water_off_and_hands_free`, `intact_and_not_dropped` |

**Concrete finishing.** The task has a mobile FR3 holding the tools; the ports move the tools on a tool carriage with
no robot (drywall: two free-moving knives). The joint and sealant walls lie flat.

| env (task) | as the task does | by another measure | not judged |
|---|---|---|---|
| `ConcretePadEdge-v0` (pad-edge; judged 5 of 8) | `radius_profile`, `lip_passed`, `no_gouge`, `field_untouched`, `edge_not_worked_wet` | | `tools_back` (only the edger of the task's three tools is read, within 2 cm of its rest point), `forms_undisturbed` (the forms are not bodies), `arms_clear` (no robot) |
| `ConcretePatchFlush-v0` (patch-flush; judged 7 of 7) | `flush_mean`, `float_texture`, `finished_in_time` | `no_lip`, `no_hard_lip`, `against_the_cut` (5 mm cells by default; `cell=0.001` gives the task's 1 mm), `no_feather_edge` (the same, and paste emptied into the port's bucket does not count as off the patch) | |
| `ConcreteToolJoints-v0` (tool-joints; judged 4 of 5) | `concave`, `compacted`, `tooled_thumbprint_hard`, `no_void` | | `smear` (the smear on the faces and the mortar that leaves the trowel or the chart are read and can fail the port; a pass is not counted: with the wall laid flat a lifted trowel cannot drop the gob the task's blade would) |
| `ConcreteToolSealant-v0` (tool-sealant; judged 5 of 5) | `concave_and_thick`, `wall_contact`, `no_void`, `tooled_before_skin` | `smear` (off the wall, only gobs the tool drops count) | |
| `ConcretePatchDrywall-v0` (patch-drywall; judged 5 of 5) | `holes_filled`, `flush`, `over_mesh`, `mesh_covered` | `no_stray_compound` (compound off the board by the port's own tally from the knives) | |

`float_texture` already holds at the start of the patch-flush port (the start texture sits on the band's edge). The
drywall port also needs its compound books to close.

`ConcreteToolJoints-v0` under random actions on a GPU: in one check on an L4 (one world, seed 0, 300 random
actions inside the action space) the observation held NaN or inf values from step 39 on, and the env raised
nothing and kept stepping; on the CPU it stayed finite. The checked path is the scripted expert's
(`expert_action`), which stayed finite over its 5,998 steps in the same check. Test observations for non-finite
values before training on this env.

**Forklift.** The task has a humanoid in the seat working the controls; in the ports the action is the truck's
controls and nobody is in the seat, so `operator_seated` and `head_clear_of_guard` are not judged in any port. Every
port reads its checks once per 20 ms control period, where the task latches them every physics step: the checks
marked "sampled" use the task's thresholds on those samples. `no_strike` counts only the objects the port builds, and
its impulse takes the largest contact of each sample. `time_limit` is judged from whether the port's scripted plan
finished before the limit. Starts are fixed (the task draws them per seed).

| env (task) | as the task does | by another measure | not judged |
|---|---|---|---|
| `ForkliftStartAndPark-v0` (start-and-park; judged 10 of 16) | `in_parking_bay`, `forks_lowered`, `forks_tilted_to_floor` | sampled: `no_strike`, `no_wheel_lift`, `roll_under_6deg`, `no_dropped_load` (the scene has no load), `speed_limit`, `stationary_2s`; `time_limit` | `operator_seated`, `head_clear_of_guard`, and `parking_brake_set`, `direction_neutral`, `engine_off`, `key_off` (they read the last commanded control, not a lever's or the engine's state: a wrong command fails the port, a right one is not counted) |
| `ForkliftRampDownLoaded-v0` (ramp-down-loaded; judged 12 of 14) | `in_drop_zone`, `resting_on_floor`, `backed_out`, `forks_lowered` | sampled: `no_strike`, `no_wheel_lift`, `roll_under_6deg`, `no_dropped_load`, `speed_limit`, `stationary`, `slow_on_ramp`; `time_limit` | `operator_seated`, `head_clear_of_guard` |
| `ForkliftPutAwayRack-v0` (put-away-rack; judged 13 of 15) | `on_both_beams`, `at_beam_level`, `front_overhang`, `side_clearance`, `level`, `forks_withdrawn_lowered` | sampled: `no_strike`, `no_wheel_lift`, `roll_under_6deg`, `no_dropped_load`, `speed_limit`; `stored_loads_undisturbed` (the port stores one load; one fixed rack slot); `time_limit` | `operator_seated`, `head_clear_of_guard` |
| `ForkliftLoadTrailer-v0` (load-trailer, restrained trailer only; judged 12 of 14) | `in_trailer_on_floor`, `gap_to_pallets_ahead`, `gap_to_side_wall`, `level`, `truck_back_on_dock` | sampled: `no_strike`, `no_wheel_lift`, `roll_under_6deg`, `no_dropped_load`, `speed_limit`, `trailer_movement`; `time_limit` | `operator_seated`, `head_clear_of_guard` |
| `ForkliftLoadBales-v0` (load-bales; judged 9 of 11) | `two_bales_at_nose`, `truck_back_on_dock` | sampled: `no_strike`, `no_wheel_lift`, `roll_under_6deg`, `no_dropped_load`, `speed_limit`, `trailer_movement`; `time_limit` | `operator_seated`, `head_clear_of_guard` |

**Gift bags and cooking.** The gift tasks have two FR3 arms, the ports a gantry with a hook or two fingers. Both judge
the task's hands-off window by another measure: at no control period of the last 3 s is a tool box (the hook, or the
two fingers) within 5 mm of the carton, bag or other parts. The task reads a contact force above 1e-3 N instead, so a
tool within 5 mm that bears no load fails the port, and a graze between two vertices of the bag's paper can be
missed. The porridge task has an FR3 with a spoon and a hob knob; the port has a free-moving spoon and the
hob's power, on a smaller scene: 240 mL in an 80 x 75 mm pot (the task: a 152 mm pan), a hob of 800-1,200 W (the
task: 580 W at seed 0), a start at 4-20 °C (the task: 5 °C), and a 120 s limit of the port's own.

| env (task) | as the task does | by another measure | not judged |
|---|---|---|---|
| `GiftOpenFlaps-v0` (open-flaps; judged 8 of 8) | `carton_in_place`, `carton_upright` | the four `flap_..._open` checks (read every 0.01 s control period; the task reads every 4 ms step), `contents_in_place` (height bounds 0-203 mm; the task: 8-211 mm), `hands_off_window` (no tool within 5 mm, not a contact force) | |
| `GiftBagCandle-v0` (bag-candle; judged 10 of 11) | `standing`, `on_mat`, `bottom_squared`, `base_at_rest`, `candle_in_bag`, `candle_upright` | `mouth_open` (rim gap along the depth axis), `board_in_bag` (other sample points), `candle_on_board` (a vertical gap), `hands_off_window` (no tool within 5 mm, not a contact force) | `bag_not_torn` (the paper has no crack measure) |
| `CookReheatPorridge-v0` (reheat-porridge; judged 3 of 5) | | `hot_through` (four 60 mL portions, not 39 mm cells; within the port's 120 s), `not_scorched` (the floor area under particles that boiled dry, at most 2 cm2), `contained` (at most 1 % of the particles outside the pot at the end) | `vessel_in_place` (the pot is static), `appliance_off` (the script turns the hob off; the scene has no knob) |
