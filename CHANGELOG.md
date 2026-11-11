# Changelog

Changes to Alloy's public API: `alloy.api`, re-exported at the top level (`alloy.Scene`, `alloy.build`, ...).
Versions follow [semantic versioning](https://semver.org): from 0.1.0, a minor version may add to the public API and
a patch version only fixes it; anything that would break code written against it waits for the next major version
(or, before 1.0, the next minor version) and is announced here first as a deprecation. Modules outside `alloy.api`
are engine internals with no stability promise.

## [0.1.0] - unreleased

### Fixed
- With `scene.randomize` on an object (friction, density, start `x` / `y` / `z`, ...), resetting some worlds (one
  world's episode ending in a `VectorEnv`) now resets only those worlds. Before, every world's objects went back
  to their start while their episodes went on, with no error or warning. Such a reset also no longer writes the drawn
  values again, so it costs about what a reset without the rows does and a GPU run keeps its captured step graphs:
  on one L4, 1,024 worlds, a reset of 32 of them and the step after it took 1.0 ms + 7.7 ms, against 27-37 ms + 26 ms
  before (on an AMD EPYC 7B13 CPU a density row's reset took 2.2 s before; now 13 ms, as without the row). Rows
  drawn at every reset (`when="reset"`) still cost more: `docs/guide/randomize_robots.md`.
- `alloy.envs`: a world the engine froze (INVALID) earns nothing: on its last step its reward (except
  `invalid_penalty`), reward terms, success and last observation read 0, and the rsl_rl, SB3 and skrl wrappers
  never bootstrap its value; `info["episode"]["invalid"]` and `env.invalid_episodes` count frozen episodes, the
  latter since the env was built. `docs/guide/envs.md`, Ready-made environments.
- The rsl_rl wrapper no longer resets every env with seed 0; the env's own seed stands.
- `ObsNoise` works on a flat observation list (it raised `EnvError`). `docs/guide/rl.md`.
- Gym camera observations (`camera=True`, `Task(cameras=...)`) show the scene as `sim.render()` and `alloy.Camera`
  draw it: library looks, scanned objects' photo textures and a grey ground. Before, they drew the engine's palette
  colours (the ground and the first object the same blue) and left out every look, so `alloy.vary` textures never
  reached a policy. A scene variant's looks (`vary=True`) are drawn by the fast camera as each look's colour and
  colour map, where `Camera.render(sim)` uses the full-material tracer; on one L4 at 512 worlds that costs nothing
  measurable (`PickPlace-v0`: 15,100 64 px images/s, 14,400 before). `docs/guide/envs.md`.
- Since 0.1.0rc2: `sim.stop_drives()`, `sim.set_pose()` and `sim.reset()` bring a kinematic object to rest in both of
  the engine's states. Before, a stopped or reset tool kept its old speed or pose every second exchange, and MPM
  material under it (sand, paste, dough) kept being pushed.
- Since 0.1.0rc2: two `add_pour` calls in one world each pour their own volume. Before, they split their particles
  evenly: a 20 ml and a 60 ml pour delivered 38 ml each.
- Since 0.1.0rc2: MPM (sand, paste) at many GPU worlds no longer lets grains into or through moving and free bodies.
  Warp's packed sparse-grid cell lookup lost float32 precision as a world's offset grew, so higher-numbered worlds
  behaved worse (a blade let sand through in 57 of 64 worlds); Alloy now picks the cell in integers. Validated up to 64
  worlds on one L4; one world is unchanged bit for bit.
- Since 0.1.0rc2: `alloy.envs` delta and hand-motion actions (`ActionSpec(mode="delta")`, `EEActionSpec`) start each
  step from the last target pulled back toward the joint's measured position (within `window`: 5 x `delta` for joint
  deltas; 0.03 rad for an arm under `EEActionSpec`, set for the shipped Franka Panda's gains, the arm's lead
  shrinking as one vector so the hand keeps its direction), so a joint held by a table, a part or a stop no longer
  runs its target on: after a 3 s press on the table, an up command lifts the Panda's hand 1 cm in 4 steps (12-23
  before). Costs: the servos now set the top speed, so an open-loop 10 cm hand command over 5 steps moves 4.1 cm and
  a slow servo moves slower (the custom-task example's pusher: 91 steps, 47 before). Trajectories change even where
  outcomes do not: the window binds in 86 of the GripperLift demonstrator's 100 steps, and it still succeeds in every
  world. With the shipped arms' firm contacts it does not change whether a world freezes. `window=float("inf")`
  restores the old behaviour.
  `docs/guide/envs.md`, `docs/guide/rl.md`.

### Added
- `alloy.envs`: `PickCube-v0`, a Franka Panda that picks a 4 cm cube and carries it to a point in the air, with its
  own rsl_rl settings (`train_rsl_rl.RECIPES`). With these settings training does not yet
  learn the pick reliably: some runs learn it and others do not. Treat them as a starting point; tuning continues.
  `docs/guide/rl.md`, Arm pick.
- Scanned library objects (YCB, Google Scanned Objects, Objaverse and the other fetched scans) render with their own
  photo texture in both cameras: the quick look (`sim.render()`, `alloy.Camera`) and `quality="full"` / looks. The
  camera reads each scan's `visual.glb` (mesh, UVs, base-colour image); a look that names the object still wins.
  Without Pillow (the `import` extra) an object keeps its flat colour. Scans that simulate as soft bodies show it
  too (fixed at the build pose; thin streaks can show along the texture's seams). Cost: 2-3 % on a scene of scans on
  one L4, none on scenes without them. `docs/guide/camera.md`.
- Opt-in building blocks for RL environments (`docs/guide/rl.md`): hand-motion actions for arms by differential
  inverse kinematics (`alloy.envs.diffik.EEActionSpec`, `panda_reach_ee`), per-world observation noise, observation
  delay, observation history and action delay (`Task(layers=[...])`, `alloy.envs.realism`), and PD / DC-motor
  actuator models computed at every physics step on motorized robots (`alloy.envs.actuators`) (first versions,
  tested on the CPU; limits in the guide).
- `alloy.viewer` (opt-in: `import alloy.viewer`; each backend needs its own optional package): `launch(sim,
  backend="viser" | "gl" | "rerun" | "usd" | "null")` shows a running sim with pause, single steps, world picking
  and contact points and forces of rigid bodies and robots; `alloy.viewer.contacts(sim, world=0)` returns those
  contacts as numpy arrays and `alloy.viewer.force_on(sim, name)` the net contact force on one object;
  `python -m alloy.viewer <scene>`. The viewer only reads the sim, unless you opt in (both off by default; tested
  on the CPU: sliders on a one-hinge arm, dragging on free boxes; dragging robot links not tested):
  `launch(..., joints=True)`
  (`--joints`) adds a slider per position-driven joint that calls `sim.set_targets`, `launch(..., drag=True)`
  (`--drag`) lets the mouse pull a body with a spring through `sim.apply_wrench` (`alloy.viewer.MouseDrag` grabs from
  a script). `docs/guide/viewer.md`.
- Fixes from a user-side audit of the release candidate. `sim.set_targets` (and `run(targets=)`) holds a position
  target inside the joint's range, each world on its own: a target past a stop rested up to 0.32 rad past it, health
  OK. It raises for a robot with no actuators, naming the MJCF line to add (the call returned and the arm fell).
  `sim.set_controls` on an MJCF `<position>` or `<velocity>` actuator moves its joint (it was ignored). `sim.reset()`
  returns to the `sim.settle()` state in a scene with `scene.randomize` too (it went back to the build start). A robot
  joint named like its body keeps its `"<joint>.q"`. `scene.randomize` refuses, when it is called, a card number the
  engine cannot draw per world (`material.surface.restitution`, `material.elastic.poisson`: both failed the build)
  and lists the fields it can. A malformed robot MJCF and `add_liquid` of a solid material raise `SceneError` (a bare
  XML ParseError; one rigid box named like the liquid). A soft body from a mesh without the `mesh` extra raises
  `FeatureUnavailableError` with the install command. `Scene.save` to a relative folder writes a robot path that
  loads. At the world-count limit `alloy.build` raises `BuildError` (field `worlds`) naming what ran out (Warp's
  32-bit array size, or GPU memory) and at how many worlds, instead of a raw Warp error. A step graph that runs out
  of GPU memory steps down without losing the GPU (it failed later with "CUDA error 101: invalid device ordinal").
- Health, from the same audit. Under `profile="precision"`, `sim.step` and `sim.run` raise `alloy.SimulationError`
  in the control period a world becomes INVALID, naming the world, once per world until `sim.reset()` (they
  returned, and only `sim.check()` raised, with an engine error that was not a `SimulationError`); `sim.check()` and
  `sim.report()` read afterwards. Each world row of `sim.health()` has `seen_flags`: every flag raised since the
  build or the last reset, also one that has gone, and `sim.check()` lists those as warnings without changing `ok`.
  A steady energy gain of more than 10 % of the scene's energy scale with nothing adding it is ENERGY_JUMP
  (`energy_gain` in the row; `sim.check()` is not ok): a free mesh body that pumped itself upward read OK.
  PEN_WARN and PEN_FAIL read the contacts of rigid bodies and robots on the MuJoCo routes too (they read none
  there); `peak_penetration_m` keeps the deepest overlap since the build or the last reset. JOINT_FAIL covers a weld
  or connect more than 1 mm or 1 mrad apart, and is measured under `profile="fast"` too. TUNNEL (INVALID) is a point
  inside a moving body (a free body or a robot link) that passed through a static box, plane, sphere, capsule,
  cylinder or box-like convex mesh. A `profile="precision"` build warns once when its runs cannot repeat bit for bit
  on the device (`docs/guide/health.md`).
- `sim.render(cam, output="depth" | "segmentation" | "body_id" | "world_id")` or a tuple such as `("rgb", "depth")`:
  depth in metres and per-pixel body ids from one ray per pixel (no sampling noise); colour images are
  unchanged and a depth-only render traces no colour. `alloy.camera.aov.segment_labels(sim)` names the ids. Fast
  camera only. `docs/guide/camera.md`.
- `sim.render(cam, output="instance_id" | "semantic" | "flow")`, alone or in a tuple: one id per drawn shape
  (`alloy.camera.aov.instance_labels`), classes you name (`set_semantics`), and optical flow in pixels since the
  camera's previous flow render, from the poses of rigid bodies (robot links and kinematic objects included), static
  shapes and the camera (NaN on cloth, soft bodies, films and liquid). Fast camera only.
  `docs/guide/camera.md`.
- `scene.randomize` for robots and gravity, off by default: per-world `mass_scale`, `mass_add`, `com_x/y/z`, `friction`,
  `kp_scale`, `kd_scale`, `armature(_scale)` and gravity `x/y/z`, drawn at build or again at each reset
  (`when="reset"`); `sim.randomized()` reads the draws (`docs/guide/randomize_robots.md`).
- Reset-time events, off by default: `start_x/_y/_z` and `start_yaw` rows move a free-base robot's start at every
  reset, and `scene.add_push(body, force, every, duration)` pushes a body at random times per world (`sim.pushes()`
  reads the force held now). Checked on the CPU only (`docs/guide/randomize_robots.md`).
- `alloy.envs.wrappers`: adapters from `VectorEnv` to rsl_rl, Stable-Baselines3 and skrl (each an optional install),
  one-command PPO training (`python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl`), and observation
  groups, `Task(obs={"policy": [...], "critic": [...]})`, for a critic that sees more than the policy
  (`docs/guide/rl.md`).
- `alloy.envs.loco`: a reference legged environment, the Go2 on flat ground following velocity commands, with
  pushes, gravity / accelerometer / solver foot-force observations and `python -m alloy.envs.loco.train` (rsl_rl
  PPO; a walk learned in the two runs we finished; `docs/guide/locomotion.md`).
- `alloy.envs.loco.go2_rough` (opt-in): the same Go2 velocity task on a heightfield of terrain tiles (rough
  ground, waves, blocks) in 5 levels, with a per-world level curriculum (`docs/guide/locomotion.md`, Rough terrain).
- `alloy.envs.make("ClothFold-v0", grasp="attach")` (opt-in): the cloth vertices between the closed fingers are held
  to them by a spring, a shortcut rather than a friction grasp; the default `"friction"` is unchanged
  (`docs/guide/envs.md`).
- Opt-in, `ALLOY_PROXY_TENSOR=1`: a robot or rigid body's proxy in cloth, soft bodies, rods and film moves with the
  body's own 3x3 inverse mass instead of one scalar, so a body that moves easily along one axis and hard along
  another pushes the cloth as it should. CPU only so far (`docs/known_gaps/coupling-proxy-isotropic-mass.md`).
- Engine layers, opt-in: `FilmConfig(limiter="stress")` holds a film past its stretch cap with a force inside the
  solve: a 15 µm HDPE strip with 50 g hung on it stays 1.03x its cap, where the default projection let it run to
  24-34x (`docs/known_gaps/film-projection-force.md`).
- `alloy.build(..., contact_capacity=n)`: MuJoCo's contact buffer, n contacts per world on average (MuJoCo Warp's
  pool of n x worlds is shared by the worlds; on the CPU's MuJoCo C route, the contacts health reads). Unless
  `constraint_rows` is given it also sizes the constraint rows per world from the model: each contact's rows x 2 n
  plus the scene's joint-limit, equality and friction-loss rows. Default None sizes it from the scene as before (8
  per collision shape, at least 64, 6 rows per contact). Worlds that overflow are flagged CONTACT_OVERFLOW and
  frozen as before, and the console now says so once, naming the first flagged world and the n to pass, instead of
  MuJoCo Warp's warning on every step. One process only for now.
- `alloy.build(..., constraint_rows=m)` (opt-in): MuJoCo's constraint row buffer, rows per world. The default gives
  a robot of hundreds of collision shapes thousands of rows where it uses tens, and MuJoCo Warp runs its solver
  over the whole buffer. Sized to the scene, on one L4 at 2,048 worlds a Panda stepped 1.55x faster
  (`contact_capacity=64, constraint_rows=384`), a LEAP hand 2.2x and a G1 (512 worlds) 1.7x, with the same
  contacts, rows and solver iterations. With too few rows the worlds are ROW_OVERFLOW and frozen. The CPU's MuJoCo
  C route sizes its own rows.
- A scene with a robot can step its rigid bodies coarser than the fixed 1.25 ms (opt-in): `scene.rigid_step =
  seconds` pins a step, and an explicit `scene.rigid_step = "auto"` takes the largest step the robot's caps allow
  (its servos, joint damping, contact time constants, link speed against the contact gap, loose bodies in its
  file). The shipped Go2 gets 2.5 ms, half the physics steps. A pinned step is taken as asked, and the build warns
  once, naming each cap it breaks and the cost. A robot scene that never sets `rigid_step` steps bit for bit as
  before (`docs/known_gaps/robot-fixed-step.md`).
- `alloy.estimate` is fitted to measured runs. Wall time: 55 rows (a robot carrying water, bins, an arm, cloth, a
  pour; 1 to 512 worlds; CPU and one L4), within 1.3x on each; it was up to 18x optimistic. GPU `memory_bytes`: 20
  rows on one L4 (1 to 16,384 worlds), within 1.2x on each, with the 0.4 GB a GPU run holds before its first world;
  it read 4-6x low. The keys are unchanged, and a scene unlike the rows can be further off (`docs/guide/worlds.md`).
- Cloth and soft bodies take their velocity from each step's displacement, with the part float32 positions round
  away kept beside them: a sheet falls at g at every substep and height. Before, a coupled cloth at z 2 m fell at
  0, g or 2 g depending on the substep. Default cloth and soft-body results change bits on the CPU and the GPU.
- A cable on the chain route, twisted by a gripper and held, keeps its twist on MuJoCo Warp (any GPU): all of it
  after 5 s on one L4 with pads that slide on the wrist (87 % before), 99.8 % with pads welded to it (52 % before),
  at 1.00-1.04x the cost. On the CPU's MuJoCo C route it creeps as before.
- `sim.reset()` puts the coupled exchange clock back too, so in a scene with several solvers the run after a reset
  repeats the first one bit for bit on the CPU. The adapted clock stayed before (water ended 3.4 mm apart).
- Cutting: by default a blade cuts only seams aligned with it (`docs/guide/cutting.md`). Over 2,048 random stroke
  cuts of a block on one L4, clean episodes rose from 1,431 to 1,959 and failed ones fell from 615 to 86.
- MJCF weld / connect equalities that keep MuJoCo's default solref / solimp take the stiff rule of scene
  attachments by default, sized for the rigid step (30 kN on a 25 kg pair stretches the weld 0.015 mm instead of
  70 mm; 1,024 and 4,096 worlds, no cost measured); rows that set their own are kept, and
  `alloy.build(scene, welds="mujoco")` keeps every row as written (`docs/known_gaps/weld-stretch-under-load.md`).
- Cloth and soft-body self-contact rows are sized to the scene by default: each row grows between control periods
  before it fills, and a period that dropped pairs re-runs. They held 32 / 64 pairs per vertex / edge, and a pile
  of folded sheets dropped pairs. A row stops at 256 pairs; past it pairs drop and health flags
  SELF_CONTACT_OVERFLOW as before. Opt-in, `ALLOY_SELF_CONTACT_CROSS=report` or `invalid` measures whether dropped
  pairs let surfaces pass through each other (`docs/known_gaps/vbd-self-contact-buffer-overflow.md`).
- A kinematic fixture (`add_open_box`, `add_open_cylinder`, `add_block_with_hole` with `kinematic=True`) is one
  kinematic body: `"<name>"` is its state (the floor's frame; `"<name>_floor"` still works), and `sim.drive(name)`
  moves it. A kinematic cup beside a 2-dof arm is 8 dofs, was 152 (25 bodies). Its wall parts have no state of their
  own (reading one names the fixture). Static fixtures are unchanged.
- Cloth that keeps a crease: a cloth material with KES bending hysteresis (`elastic.bending_hysteresis`, KES 2HB in N)
  holds a fold by fibre friction, M = B K + HB sign(dK/dt), so a released fold keeps the curvature HB / B instead of
  springing back flat. The library card `cotton_single_jersey` (KES-sourced B, 2HB, MIU) has it; other cloth cards
  are unchanged. `docs/guide/deformables.md`.
- `Scene.add_rod(name, points | length=, radius=, direction=, material="cable_usb_composite", segments=,
  clamp="start" | "end" | "both")`: an elastic rod (cable, wire, cord) that bends, twists and stretches as its
  material says, rolls on surfaces and can be gripped and twisted by a robot. A clamped rod steps on the elastic-rod
  solver (`sim.routes()` says `rod_vbd`); state keys `"<rod>.points"` (centre line) and `"<rod>.twist"` (rad per
  joint); the CLI recipe kind `"rod"`. `docs/guide/rods.md`.
- A robot's grip on an elastic rod (`rod_vbd`) holds once the gripper stops: the rod's contacts with the robot keep
  their friction anchor from step to step (sticky contact matching), so a twisted cable no longer runs on or
  untwists (was +1.9 % after 0.7 s, and a hard squeeze untwisted it). `docs/guide/rods.md`, Robots.
- Liquid beside a moving body steps about 1.3x faster: the coupled clock takes the liquid's own stable step (its
  CFL step already carries its margin) instead of half of it, so the liquid runs half as many substeps (2 instead
  of 4 in a tilting jug beside a free ball). The liquid laws hold as before.
- `Scene.add_rod(..., clamp="start", route="chain")`: a clamped rod stays on the chain, its first segment welded
  where it is built, instead of the elastic-rod solver. Beside a robot that is about 60x faster on the CPU (a
  gripped, twisted cable: 1.3 s per simulated second, was 83), and the grip holds as on `rod_vbd` on the CPU; on a
  GPU a held chain stays with its grip too (it crept back about 2 % per second until a stick anchor on the gripped
  links' friction rows; `docs/known_gaps/rod-chain-grip-slips.md`). `route="rod_vbd"` asks for the elastic rod.
  `docs/guide/rods.md`, Robots.
- An elastic rod beside a robot steps 1.7x faster on the CPU: a solver entry of rods only runs 4 VBD iterations
  instead of 10 (the grip holds as before). The cable squeezed by two pads takes 95 s per simulated second, was
  161. `docs/known_gaps/rod-vbd-beside-robot-cost.md`.
- `Sim.apply_wrench(name, force=, torque=, at=)`: hold a force and a torque on a body (a rod's `"start"` or
  `"end"` segment) until changed; `sim.reset()` removes every wrench. Health counts its work as energy from outside,
  so a held push raises no ENERGY_JUMP.
- Wires that keep a bend: a rod whose material has `plastic.model: j2` (`copper_c110` and other metal cards)
  routes to `rod_vbd`, yields per joint through a 220-fibre round section (von Mises, linear hardening) and keeps
  its set; `"<rod>.plastic_strain"` gives each joint's largest equivalent plastic strain. `docs/guide/rods.md`.
- A rod that needs more than 40 segments steps on the elastic-rod solver (`rod_vbd`) instead of failing to build;
  short elastic rods stay on the chain route. `alloy.estimate(...)["routes"]` names each object's route and the
  reasons, as `Sim.routes()` does.
- `alloy.build` picks the rigid step per scene: a rigid-only scene of free bodies steps its rigid bodies at the
  largest stable step (up to the control period), with contacts no softer than before. Up to 8 free bodies step on
  rk4; more step on implicitfast with rk4's gains and the friction pull capped, on MuJoCo Warp (any GPU, or
  `ALLOY_CPU_ROUTE=warp`; the CPU's default MuJoCo C route keeps rk4). Stacks of 27 cubes (1 m, 10 kg/m³, resting;
  3,072 worlds, one L4) step at 5 ms and ran 766x real time, none falling (2.5 ms on the CPU's MuJoCo C route).
  Newton's SolverMuJoCo at the same pinned commit with one 10 ms step ran them at about 1,800x (five runs) with
  softer contacts (rest error 1.00 mm against 0.19 mm): the default is about 2.4x slower there
  (`docs/known_gaps/rigid-free-body-solve-cost.md`). `Scene.rigid_step`: "auto" (default), "fixed" (1/600 s snapped
  down to the dt grid, the step used before: 1.25 ms at dt 0.01), a step in seconds, or "fast" (also `profile="fast"`
  when `rigid_step` was never set), an opt-in fast mode: contacts may soften to a rest sag of 1 % of a body's thinnest
  member, and free bodies step on implicitfast on MuJoCo Warp (rk4 on the CPU's default MuJoCo C route), so 1 m cubes
  step at 10 ms and the same stacks ran 1,973x to 2,135x in three runs, about the same as the plain solver. Its cost:
  a 30 cm cube sliding at 5 m/s rocks under 1° and lifts 1 mm with the friction pull capped (MuJoCo Warp), 6° and
  8 cm on the CPU's MuJoCo C route (2° and 1 cm at 2 m/s); the build warns. `sim.routes()` and
  `alloy.estimate(...)["rigid_step"]` report the step and what bound it. Robots, kinematic objects, deformables,
  precision fits, plug-ins and the precision / debug profiles keep their step bit for bit.
- `alloy.Scene(cloth_accuracy="default" | "super" | "fast")`: cloth solves close to convergence by default, with 40
  accelerated solver iterations per step (adaptive start, Chebyshev) instead of the cloth solver's own 10: a draped
  sheet stretches 0.36 % instead of 12 % and hangs at drape coefficient 0.728 instead of 0.19 (converged 0.74). It
  costs time: a 64-world drape on one GPU ran 2.7x slower than at 10 iterations, and on the CPU each cloth step runs
  4x the iterations. "super" runs 60 iterations, about 1.4x the default's cloth cost, for 0.34 % and 0.732. "fast"
  runs the cloth solver's own 10, about 0.4x the default's cloth cost (0.25x on the CPU), for runs where the cloth's
  shape does not matter: far from converged (12 %, 0.19), and under CLOTH_STRETCH's warning, so health reads OK. Soft
  bodies in the same scene step with the same setting; a scene without cloth is unchanged. Kept in Scene JSON;
  `alloy.estimate` shows every setting under `"cloth"` and charges cloth per solver iteration (`["vbd"]`: its
  particles per world, iterations and self-contact).
- Health flag CLOTH_STRETCH: a cloth stretched more than 50 % past its rest shape makes its world DEGRADED for the
  rest of the run (`sim.check()` stays ok); past 500 % the sheet blew up, so the world is INVALID and `sim.check()`
  names the cloth with a fix. `sim.health()` rows carry `cloth_stretch` (`percent`, `cloth`, `level`), the run's
  peak. Before, a sheet that blew up but stayed finite and slow read OK.
- `Scene.set_joint_yield(joint, torque= | force=, hardening=0)`: a hinge's or slide's spring (MJCF `stiffness`,
  `springref`) takes a permanent set past a yield torque (N m) or force (N), with optional linear hardening;
  `"<joint>.rest"` (where the spring rests now) and `"<joint>.plastic_strain"` (accumulated yield) in `sim.state()`
  and `run(record=)`; kept in Scene JSON. MuJoCo routes only; a scene with cloth, liquid or a soft body refuses it.
- `Scene.add_cloth` with a paper or plastic film material (`paper_80gsm`, `paper_kraft`, `paper_board`, `film_hdpe`,
  `film_ldpe`, `film_lldpe`) builds a film sheet (`cloth_film`); it failed to build before (the sheet was taken for a
  paste). A paste material (dough, mayonnaise, ...) raises `SceneError` at `add_cloth`, naming `add_box` as the fix.
- Metal sheets that dent: `Scene.add_cloth(..., material=<a metal with a yield law>)` (`aluminium_6061_t6`,
  `steel_tinplate`, `stainless_304`, ...) steps on the film route (`cloth_film`), whose hinges bend as the metal's
  section (pure bending within 0.3 % of the closed form) and keep a permanent bend past its yield;
  `"<sheet>.plastic_strain"` (worlds, vertices), the largest plastic strain in the sheet at each vertex, in
  `sim.state()` and `run(record=)`. Other cloth and metal objects keep their routes.
- Coupled scenes (solvers that exchange forces, such as cloth beside a robot): a world's first escalation (its
  energy or residual check tripped) now re-runs only that period, one step finer, and the exchange clock every world
  shares is unchanged afterwards. Before, one world's trip slowed every world for the rest of the run. A repeat trip
  within the watchdog's window, or a period that stays escalated, still takes the lasting remedy.
  `sim.health()["refined_periods"]` counts each world's re-run periods since the build (None without coupling; joined
  over `processes=` workers). Engine layers: `alloy.coupling.types.WatchdogPolicy(per_period_ladder=False)` restores
  the old behaviour.
- Coupling on the GPU (MuJoCo Warp route): a robot body that cannot translate at its centre of mass, such as a link
  on a hinge through its centre of mass, now couples to cloth, liquid and soft bodies with its own mass, as on the
  CPU route. Before, MuJoCo Warp's inverse-weight fallback gave it a proxy hundreds of times too light (a 0.3 kg
  wrist read as 1.6 g).
- `alloy.Scene(liquid_accuracy="default" | "super")`: liquid particles 4 mm apart by default, 2 mm with "super"
  (about 16x the cost); kept in Scene JSON, not in saved cards; `alloy.estimate` shows both settings; the CLI and MCP
  take `--liquid-accuracy`. A build warns once, naming the fix, when a container's opening (under 32 mm at default,
  24 mm at super) or a pour stream (under 2 particles) is too narrow for the setting. The router no longer refines
  the particle size silently for a narrow container card's opening.
- `alloy.Scene(worlds=, dt=, roots=[...])`: build a scene in Python; `roots` are your own card folders, such as
  cards made by `alloy.create` or `alloy.io`. Rigid boxes, spheres, cylinders, capsules and meshes (`add_mesh`, from a
  file or from vertex/face arrays). Cloth (`add_cloth`), liquid (`add_liquid`), robots from MJCF / URDF / USD or an
  MJCF string (`add_robot`), and library objects (`add_object`). Per-object mass, density, friction and restitution,
  per-world start positions, per-world randomization seeded by `alloy.build(seed=)` (randomizing friction caps the
  dynamic coefficient at the lowest static draw), and pair friction (`set_friction`). `rigid=True` keeps an object
  rigid: a soft material can otherwise make it a soft body, which is much slower. The build warns when that
  happens. Scenes round-trip
  through JSON (`to_dict` / `from_dict`) and save to engine cards (`save`).
- `alloy.load(source)` loads and builds in one call, returning an `alloy.Sim`. `source` is a library scene, a
  scene card, a Scene JSON file or a robot file (MJCF / URDF / USD). The robot stands on a ground plane, or on
  its own floor when the MJCF has one, as MuJoCo Menagerie `scene.xml` files do.
  `alloy.Scene.load(source)` returns the same scene as an editable `Scene` instead.
- `alloy.build(scene, worlds, device, seed, profile=)` and `Scene.build(...)` return an `alloy.Sim`. It runs on the
  GPU when there is one, else on the CPU.
- `alloy.Sim`:
  - `step(n)`, `run(seconds, record=[...], targets=...)` (returns an `alloy.Trajectory`; a `(start, end)` target
    ramps over the run), and `state()` by name;
  - `set_targets()` for joint position or velocity targets. It drives MJCF `<position>` / `<velocity>` actuators,
    and `<general>` servos with an affine bias, as MuJoCo Menagerie robots use. A joint with no actuator raises
    an error that says what to add;
  - `set_controls()` writes raw MuJoCo `ctrl` per actuator name (`sim.names()["actuators"]`);
  - `reset()`, `faces()`, `particle_volume()`, `routes()`, `health()`, `check()` and `render()` (renders through
    `alloy.camera` when it is installed).
- `alloy.estimate(scene, worlds, seconds)`: predicts memory, step time, wall time and first-run compile time.
  `measure=True` times a short real run instead.
- `alloy.Camera` (from `alloy.camera`) is re-exported, and `sim.render()` returns an image of a world.
- `alloy.materials()` and `alloy.scenes()` list what the card library holds. The library includes every imported
  library (`<alloy-assets>/imports/<source>/`: YCB, Google Scanned Objects, RoboCasa, Gym-Anything, ...), so
  `Scene.add_object(name, card)` and `alloy.load` find all of their objects. An imported object's meshes are fetched
  from `$ALLOY_ASSET_STORE` the first time a scene that uses it is built.
- `Scene.add_liquid(..., shape="cylinder")`: liquid at rest in a round container, an upright cylinder of diameter
  `size[0]` (= `size[1]`) and height `size[2]`. A box of liquid in a round cup is not at rest (its corners stick out
  of the cup), collapses and jets over the rim; a cylinder stays put.
- Precise fits by default (fresh-agent round 2, T3): `add_block_with_hole(..., precise=True)` makes its scene a
  precise-fit scene: rigid bodies step at 0.25 ms or finer (`sim.dt` is unchanged; the engine runs several
  periods per `step()`) and plastics, wood and metals get a machined part's contact stiffness, 2e7 N/m. At the
  default 1-1.25 ms step the contact stiffness is capped by the step, and a 10 mm peg landing on the hole floor at
  0.8 m/s sank 1.01 mm (both T3 misses: 1.013 mm against a 1 mm limit); precise, 0.27 mm (CPU). `precise=False`
  keeps the default step (4x fewer rigid steps). `alloy.estimate` counts the extra steps.
- Scene-card pours: `alloy.load` / `Scene.load` turn a card's `emitters` into `Scene.add_pour`
  streams named `<material>_pour_<k>` (the engine does not run card emitters, so a card pour built with no liquid).
  An emitter must be a finite downward stream (velocity `[0, 0, -v]`, `stop_s`); card sinks are dropped (the
  escape box retires liquid that leaves the scene).
- Liquids beside rigid bodies: `"<liquid>.active"` and `retired_particles` read the liquid solver's
  live flags when the coupler steps it on its own model view (a scene with a free rigid body read 0 poured);
  a pour in a scene with cloth and rigid bodies (the liquid then follows one way) builds. A kinematic container
  holds liquid beside free rigid bodies, still or tilted with `sim.drive`: a tilt pour lands about the same share in
  the cup as with every other object static, and a driven object stays on schedule when the engine re-runs a control
  period. With cloth and liquid only (no rigid solver) kinematic objects are carried too: before, the liquid fell
  through a kinematic container and the cloth through a kinematic tray.
- `add_mesh(..., collision="decompose")` builds (it always failed: the engine wants a convex-parts file beside the
  mesh, and the API never wrote one). The build splits the mesh into at most 32 convex parts with CoACD, the
  importers' settings, once per mesh (cached in `$ALLOY_CACHE_DIR/hulls`, default `~/.cache/alloy`), or copies a
  `<stem>.hulls.obj` shipped beside the mesh file. CoACD and trimesh come with `pip install "alloy-engine[import]"`;
  without them the build raises `FeatureUnavailableError` naming that command and the alternatives (`"convex"`,
  `"sdf"`). A ball dropped into an L-shaped block's inner corner now falls past it (the hull filled the corner).
- Heat (on the heat v2 plug-in, `alloy.plugins.thermal`): `add_liquid(..., temperature=K)`,
  `add_pour(..., temperature=)`, `add_object(..., temperature=)` or `Scene.set_temperature(name, K,
  heat_material=)` start a liquid or particle object at a temperature and let it take heat;
  `Scene.add_heater(name, size, pos=, temperature=K | power=W)` and `Scene.set_heater("pan_floor", ...)` make a
  static box a heater; `Scene.set_heat(time_scale=, ambient=, convection=, ...)` sets the rest. `alloy.build` runs
  the plug-in inside the `Sim`, so `run`, `record`, `state` and `reset` keep working: `"<name>.temperature"` (K per particle) is in `sim.state()` and recordings,
  `sim.temperature(name)`, `sim.heat(name)` (per world mean / min / max K, melted, done, evaporated) and
  `sim.set_heater(name, temperature= | power=)` while running. Kelvin throughout; a value under 150 K is refused
  as a likely °C. The guide's pan of water gives the same numbers as the plug-in's own path (mean 307.548 K after 90 s).
  `examples/api/07_heat_water.py`.
- `Scene.set_heat(buoyancy=True)`: the thermal buoyancy force on water (NIST's water density table at 1 atm). At
  the default liquid accuracy a hot region starts to rise within 10 % of the textbook rate for its shape, but the
  flow then stalls, so it does not give convection yet. It is off by default (heat without it is unchanged, bit for
  bit), and only water has the data: `docs/known_gaps/heat-no-buoyancy.md`.
- `alloy.build(scene, plugins=[...])`: your own `alloy.plugins` plug-ins (or functions of the built scene that return
  one) run inside the `Sim`; `sim.plugins` reads them.
- Robots collide like every scene object: their shapes start making contacts 5 mm apart (an MJCF geom's own `gap`
  is kept). They had the importer's 0.1 m, so each robot shape made contacts with everything within 0.2 m and packed
  robot scenes overflowed MuJoCo Warp's contact buffer (a carton among six boxes: 53 rigid contacts, now 28).
  `add_robot(..., contact_gap=0.1)` keeps the old value; a scene card's robot entry takes `contact_gap` too.
- Free-base robots per world: a robot whose root body has a free joint (MJCF `<freejoint/>`, such as a carton with
  flaps) takes `add_robot(pos=(worlds, 3))`, `Scene.randomize(robot, "x" | "y" | "z", lo, hi)` and, while running,
  `sim.set_pose(robot, pos=, quat_wxyz=)`. Its links follow the base with their joint angles kept, and `reset`
  returns to the drawn starts. A robot welded to the world raises an error naming the fix. `sim.set_pose` also takes
  one orientation per world, and a teleport no longer reads as an ENERGY_JUMP that froze the world (the health
  check's energy baseline restarts at the new pose). `examples/api/08_robot_per_world.py`.
- `alloy.vary(scene, n, pos=, yaw_deg=, textures=, seed=)`: `n` edited copies of a scene for visual and placement
  diversity, the same for the same seed. Free objects' starts move a little (touching ones together, never into an
  overlap), and each look takes another texture of its family plus a small tint; `out=` writes the variants and a
  manifest, `contact_sheet=True` a picture of them all.
- Objects from the object library on the Hugging Face Hub work like library objects on a fresh install:
  `Scene.add_object` finds an object fetched with `alloy fetch <name>` (it said "no object card": the API's catalog
  did not read the fetch cache), accepts an object that is only in the Hub's index, and the build fetches it.
  `alloy.catalog.objects()` lists fetched objects, and an unknown name's fix suggests close names from the Hub.
- `sim.drive` of an object replaces its earlier drive. Drives used to pile up: one per control period made every step
  slower (4 s of a whisk task: 1,600 drives, 3,200 kernel launches per step). The motion is the same, since the
  newest drive always won.
- Per-world starts for everything that moves: `pos=` as a `(worlds, 3)` array (and `Scene.randomize(name, "x" | "y"
  | "z")`) now work for kinematic objects and kinematic fixtures (each part follows), cloth and liquids (their
  particles shift per world), besides free bodies and free-base robots. Static things (static shapes and fixtures,
  static library objects, a pour's nozzle) are refused when added, with the fix `kinematic=True`. Before, they
  failed at build with three different errors (a raw AttributeError among them).
- `add_mesh` reads STL files (binary or ASCII; vertices merged): the build used to refuse the `.stl` that `add_mesh`
  had accepted.
- `add_cloth(..., self_contact=True)`: the sheet collides with itself, so a fold rests as two layers one thickness
  apart. It is on by default; `False` steps a sheet that never folds faster (cost in `docs/guide/deformables.md`).
  Scenes saved earlier (`to_dict` / JSON) keep it off.
- A scene whose only dynamics are plug-ins builds and steps: a surface film on a static plate failed with "scene has
  no dynamic entry to step" and needed a dummy free body. It now flows exactly as it did with the dummy.
- An engine error that carries its own fix (an unsupported solver pair names its nearest supported alternative) now
  gives that fix in `err.fix` instead of the generic line for its kind.
- Fixtures (`add_open_box`, `add_open_cylinder`, `add_block_with_hole`) accept `static=True`, as every `add_*` does
  (they raised TypeError; a fixture is static unless `kinematic=True`).
- `Scene.randomize` checks the field when it is named: a wrong one (`"mass"`) is a `SceneError` that lists the valid
  fields and says what to use instead (`density`), not a `CardError` at build.
- `alloy.build(scene, device="cpu", processes=N | "auto")` splits the worlds over worker processes (Warp's CPU
  kernels use one core each) and returns an `alloy.Sim`-like object with the same calls; `sim.close()` stops the
  workers. A world steps bit for bit as it would in one process while every process keeps the same clock; adaptive
  steps are chosen per process, and the sim warns when the clocks part. 64 worlds on 16 cores, 16 processes against
  1: liquid 14.4x, cloth 12.9x, paste 20x, rigid 6.5x faster. On the CPU's MuJoCo C route, each world's contact
  softness now follows its own randomized density (it took the first world's), so split runs stay bit for bit with
  randomized density too.
- `add_liquid(..., material="mayonnaise")` (any paste) no longer crashes `alloy.build`: the build's liquid-volume
  report skipped nothing and asked a paste (MPM) for its SPH particle size.
- `Scene.add_object(..., anchor=)`: a library object's own geometry goes to `pos`, not its file origin. Scanned
  objects (YCB, Google Scanned Objects, RoboCasa, the Hub) have origins 19-85 mm off the mesh, so objects placed by
  origin landed inside walls and tables. Default `"auto"`: an object from an imported library stands on `pos` (its
  bounding box's bottom centre), Alloy's own and your own cards keep their authored frame (a cup's base), so existing
  scenes are unchanged; `"center"`, `"base"` and `"origin"` choose explicitly. Per-world positions and randomized
  x / y / z ranges follow the anchor. Scenes saved before keep their poses. What you set is what you read: for an
  object placed by its anchor, `"<name>.pos"` (`sim.state()`, `run(record=)`, trajectories) is that anchor point in
  the world, turning with the object, `sim.set_pose(name, pos=)` puts the anchor there (and a turn alone keeps it
  where it is), `sim.drive`'s default pivot is it, and `"<name>.body_pos"` is the body frame's origin. Primitives,
  Alloy's own and your own cards and `anchor="origin"` read as before.
- `alloy.estimate`'s memory for MPM particles (sand, pastes) uses the measured model the build refuses on: about 6 kB
  per particle, 32 kB beside a moving body. It took 400 B (right for liquid and cloth), 15-80x low for MPM on a GPU.
  `counts` has `mpm_particles`.
- `traj.meta["mpm_grid"]`: the grid each MPM solver used in the run (sparse or dense, its cell budget, and whether a
  dense grid fell back to sparse); `[]` without MPM.
- Static `add_open_box`, `add_open_cylinder` and `add_block_with_hole` walls reach down through the floor, so dry sand
  (MPM) no longer leaks out through the seam at a wall's foot. Kinematic fixtures keep walls that stand on the floor.
- API warnings (frozen worlds, soft-body routes, liquid volume) print to stderr as `alloy: ...` when no log handler
  shows them; Alloy's package logger has a NullHandler, which silenced them in most scenes.
- `Scene.add_pour(name, volume, pos, rate=, speed=)` streams liquid from a nozzle (an SPH emitter): the way to fill
  a container. `add_liquid` is for liquid that starts at rest, such as liquid already in a container. A block of
  liquid dropped from above splashes several percent over the rim, because a flat liquid face hitting a flat floor
  makes real pressure jets; the same 200 ml poured stays 100 % in the cup (L4). The state and trajectories have
  `"<liquid>.active"` per particle: False until a pour releases it, and False again once it is retired (it left the
  scene and is frozen where it left). `sim.health()` and `sim.check()` report `retired_particles` per world, and
  check() warns for each world that has any. The energy a pour brings in, minus what retired particles take out, is
  booked as external work after every step (the SPH solver's `export_flux_energy`), so the health monitor's energy
  check stays on in scenes with a pour.
- `examples/api/06_tilt_pour.py`: pouring by tilting a held (kinematic) jug about its lip. The stream lands
  about v·sqrt(2 dz / g) beyond the lip, so the cup goes a few cm past it: 95 % of the water lands in a cup 4 cm
  past the lip, about a third in the same cup under the lip (CPU).
- Kinematic objects: `kinematic=True` on `add_box` / `add_sphere` / ... and on the fixtures. `sim.drive(name,
  velocity=, angular_velocity=, pivot=, duration=)` moves one with a constant twist about a pivot, and a fixture's
  parts move together; for example, tilt a cup about its lip to pour. Its pose and velocity are written every physics
  step, so contacts and liquids feel the moving walls. `sim.set_pose` teleports a free or kinematic body and removes
  its drive, which would write over the pose (a one-world teleport under a running drive raises instead),
  `sim.stop_drives()` stops every drive, and `sim.settle(seconds)` runs undisturbed and makes the result the state
  `reset()` returns to.
- Static fixtures built from boxes: `Scene.add_open_box` (a bin or square cup), `Scene.add_open_cylinder` (a round
  cup) and `Scene.add_block_with_hole` (a socket). MuJoCo collides each box exactly, so a 10 mm peg drops into a
  10.2 mm hole. `alloy.api.meshes` gives the same shapes as watertight meshes for `add_mesh`.
- `alloy.build(..., contacts="auto")` uses Newton's contact pipeline when a mesh has collision "sdf" or "mesh".
  MuJoCo would collide such a mesh as its convex hull.
- Error standard: every public error (`alloy.SceneError`, `BuildError`, `SimulationError`, `UnknownNameError`,
  `FeatureUnavailableError`) carries `message`, `fix` and `field`, and `to_dict()` gives it as JSON. Engine errors
  raised while building or stepping are wrapped with a fix, and the engine error stays as the cause. Each health
  flag comes with what it means and what to change.
- Lazy imports: `import alloy` loads no NumPy, Warp or Newton. Engine set-up that needs Newton runs when Newton is
  first imported.
- The project description (AGENTS.md, llms.txt, the README) names the solver each material steps in: MuJoCo Warp for
  rigid bodies and robots (MuJoCo C on the CPU), Newton's VBD for cloth and soft bodies, Newton's MPM for sand and
  pastes, Alloy's DFSPH for liquids. Sub-millimetre contact is stated for declared precision fits (pegs, threads),
  and the world counts by kind of scene: thousands for free rigid bodies, hundreds for a robot arm, tens to hundreds
  with cloth or liquid, with the measured rows in `docs/guide/worlds.md`.
- `docs/guide/worlds.md`, "Robot scenes: what they cost": the shipped robots' rates on one L4 at 1.25 ms, physics
  only (for example a Panda with a cube 1,168x real time over 2,048 worlds, a G1 88x over 512), and Alloy against
  plain MuJoCo Warp on 17 robot tasks: 0.30x its rate (geometric mean; 0.11-0.76), mostly because Alloy takes 4
  solver steps per 5 ms where the tasks' preset takes 2, with what is not like for like. Two opt-ins buy speed:
  `contact_capacity=64` 1.79x with no measured change in quality, plus `scene.rigid_step = 0.0025` 3.33x with
  deeper contacts and, in pick-and-place, glitches in 0.47 % of episodes (none at 1.25 ms).
- Bundled engines: Newton release-1.8 (`db915e1c`, 2026-11-09; version 1.8.0rc1, with the Steel solver's 1.8
  backports) plus 12 commits and 14 patches of ours, and MuJoCo Warp v3.17.0 (`f34c6dd5`, files unchanged), on
  MuJoCo 3.17.0 and Warp 1.18.0. Before: Newton `ac7f7830` plus 13 commits, MuJoCo Warp `86e2fae0`, MuJoCo 3.16.0.
  `alloy/_vendor/VENDORED.md` lists each commit and patch with its reason and its test, and what moved between the
  pins; `alloy._vendor.VERSIONS` names the versions. What you see: every library scene's `check()` verdict is the one
  it had. On the CPU, cloth and poured liquid keep their state bit for bit, and robot scenes move at the solver's
  convergence level (0 to 0.015 mm: MuJoCo 3.17 scales its stop test by the current inertia). A mesh hull that
  rested 41.6 mm deep on MuJoCo Warp 3.16 rests at 0.05 mm. A seated peg pushed sideways leans 2.3-5.5 % past its
  geometric limit (2.5-4.5 % before), with wall penetration unchanged at 13-20 µm. On a GPU the cloth drape is bit
  for bit the previous pin's (VBD fast math stays off by default). Newton's Steel solver comes in the bundled copy
  (`from alloy._vendor.newton.solvers import SolverSteel`), but no Alloy route uses it: robots and rigid bodies step
  on MuJoCo as before (`sim.routes()`), and stepped directly it lets a robot arm's contacts through far more often
  than the copy before (`docs/known_gaps/rigid-steel-pile-backend.md`).
- Alloy and a `newton` / `mujoco_warp` you installed yourself now run in one process, imported in either order:
  Alloy's bundled copies compile their Warp kernels under names of their own (`alloy_vendor__` in Warp errors, the
  profiler and the kernel cache), so neither copy runs the other's kernels. Objects from the two still do not mix.
- `alloy.export`: frames of up to 8 picked worlds for a live viewer, as numpy arrays: body poses, cloth and soft-body
  vertices, a fixed subset of liquid and MPM particles, rod centre lines, and shapes and faces when the topology
  changes. `FrameExporter(sim, worlds)`: `capture()` between steps never waits, `poll()` from any thread makes no GPU
  call, and stepping with it on equals it off, bit for bit. `frame_export(sim, worlds)` is the one-shot form
  (`docs/guide/frame-export.md`).
- `alloy.il` (opt-in, outside `alloy.api`): `Recorder` records a `VectorEnv`'s episodes to HDF5 in the robomimic
  layout (`HDF5Writer`, needs h5py) or a LeRobotDataset v3.0 (`LeRobotWriter`, needs lerobot); `replay`, `evaluate`,
  and keyboard / SpaceMouse teleoperation of action columns. It trains no policy; joint-space teleoperation only, no
  live view, no cloth or liquid replay. `docs/guide/imitation.md`.
- `alloy.il.generate(env, policy, writers, episodes)` saves scripted demonstrations from every world of a
  `VectorEnv`; `PhaseScript` writes a demonstrator as phases, `demonstrator(env)` gives the shipped one (GripperLift,
  GripperLiftCamera, PickPlace). No trajectory augmentation or motion planning. `docs/guide/imitation.md`.
- `alloy.il`: `EvalResult.invalid` marks and `GenerateResult.invalid` counts the episodes whose world went INVALID
  (the engine froze it) at any step. They are read from `info["invalid"]` at each step, because the env's reset
  clears the health record; such an episode is never a success and `keep="success"` does not save it.
  `docs/guide/imitation.md`.
- `alloy.sysid` (opt-in, outside `alloy.api`): `python -m alloy.sysid fit --robot <name or file> --data <recording>
  --out my_robot.sysid.json` fits a robot's servo stiffness, kd + damping, dry friction and action delay per joint to
  a recording (CSV, HDF5, LeRobot dataset) by replaying it in parallel worlds; `excite`, `report`, `demo`;
  `scene.add_robot(..., sysid=)` applies a file; without `sysid=` nothing is imported and a scene steps as before.
  Its accuracy numbers are on synthetic recordings made with the same model (hidden parameters), not on a real robot;
  link masses, backlash and contact are not fitted. `docs/guide/sysid.md`.
