# P3-2: rival engines on Alloy's own tests (contract)

Brief: `[internal note]`. Owner WP8. Engines: Isaac Lab (PhysX), Genesis, ManiSkill3 (SAPIEN); Alloy beside
on the same VM in the same session. Results: `bench/xbench/OUR_TESTS.md` + `bench/xbench/our_tests_rows.json`,
linked from CROSS_ENGINE.md §10.

## Layout

| path | runs in | what |
|---|---|---|
| `packs/t12/` | (data) | T12 port pack: `scene.xml` (hull OBJs), `t12.json` (compiled facts), from `scene_pack.py` |
| `packs/laws/` | (data) | law port packs, one dir per condition: rigid / rod laws from `law_pack.py`, A15 from `soft_pack.py`, A13 / A14 (SPH) from `fluid_pack.py` |
| `b0.py` | any (numpy) | SB-7 B0 random-legal traces, bitwise equal to what Alloy plays (`tests/xbench/test_ours_b0.py`) |
| `digest.py` | any (numpy) | per-world SB-11 digests of an engine's raw state |
| `score_laws.py` | Alloy venv | engine law npz -> Alloy's own twins (`record` / `scalars` / `judge_analytic`); D1 ulps |
| `fluid_pack.py` (+ `fluid_spec.py`, its pack text) | Alloy venv | SPH laws A13.hydrostatic / A13.archimedes / A14.dam_break: `pack` (`packs/laws/<law>/<cond>/fluid.json` + `particles.npz`: Alloy's float32 particles, tank, box, SPH parameters with meanings, the export per law), `score RUN_DIR` (engine npz -> the twins' own `record` / `scalars`; extra keys `wall_impulse`, `particle_frame`), `alloy RUN_DIR` (Alloy beside, same npz) |
| `score_s1.py`, `sb7_watch.py` | Alloy venv | engine T12 / embodiment npz -> Alloy's `BenchMonitors` + classifier (SB-7 S1), batch service; contacts from Newton's pipeline over the pack's `contact_body_pairs` only (`packs_io.mujoco_shape_pairs`: MuJoCo's filter, which Newton's importer is looser than), with mesh geoms collided as convex hulls as MuJoCo does (`hull_meshes.py`) |
| `score_repeat.py` | any | SB-11 shares (R1 rerun, R2 fresh processes, R4 batch invariance) from repeat JSONs |
| `sdf_pack.py` (+ `sdf_scene.py`, `sdf_geom.py`) | Alloy venv | mesh-SDF laws A8.push / A9.thread_rundown: `pack` -> `packs/laws/<law>/<condition>/` (OBJ in body frames, `cond.json`: masses, joints, drives, friction, dt, Alloy's collision as information, static check; `drive.npz`); `score RUN_DIR` (Alloy's twins on the engine npz: A8 GapMonitor on the peg poses, `--device cuda:0`; A9 needs `drive_tau`, optional `joint_q`); `alloy RUN_DIR` (Alloy beside; A8 needs CUDA) |
| `alloy_t12.py`, `alloy_laws.py`, `alloy_repeat.py` | Alloy venv | Alloy beside: the same episodes / conditions, same file formats |
| `engines/<engine>/` | engine env | the port: loaders, runners, static check, settings notes |

Validation (CPU tests, `tests/xbench/test_ours_score.py`): Alloy's own runs replayed through the npz path give
Alloy's direct law errors exactly and its online S1 monitor values (non-contact rows exactly; penetration from
Newton's pipeline on the replayed poses within 10 % of MuJoCo's native contacts).

Engine code imports numpy, torch and its engine only (no `alloy`), so it runs in the engine's own venv/container.

## Trajectory npz (engine -> scorer)

One file per run. Keys:

- `meta`: JSON string: `engine`, `engine_version`, `versions` (every package that matters), `gpu`, `test`,
  `condition`, `dt` (physics step, s), `substeps`, `record_every` (physics steps per frame), `settings` (every engine
  setting that differs from the engine's defaults, and the recommended-settings source for each), `seed`, `worlds`,
  `row` (`recommended` | `tuned`), `notes`.
- `body_names` [B]: pack body names (MJCF names), the order of the B axis.
- `t` [T] float64: sim time of each frame (frame 0 = initial state).
- `body_q` [T, W, B, 7] float32: body frame origin position (m) and orientation quaternion **x y z w**, world frame.
- `body_qd` [T, W, B, 6] float32: **COM** linear velocity and angular velocity, both world frame (Newton's
  `State.body_qd`). Convert from the engine's convention (link-origin velocity, w-first quaternions) in float64 and
  say how in `meta.notes`.
- optional `particle_q` / `particle_qd` [T, W, P, 3] (soft / liquid laws).
- `raw_*` (SB-11 only): the engine's own state arrays exactly as its API returns them, [W, ...], no conversion.

## Rules (brief "Fairness rules", binding)

1. **Versions** pinned and recorded in `meta.versions`.
2. **Verdict row = the engine's documented recommended settings for the scene type** (solver, iterations, dt,
   contact offsets, ...), each with a source (doc URL or `repo@sha:file:line`). A `tuned` row may go beside, listing
   every change. Alloy is not changed; targets do not move.
3. **Port check before any verdict**: static (mass, inertia, joint limits, friction per body pair within 1 %;
   geometry within 0.1 mm), a short sanity trajectory, and a review by a second agent. A failure caused by the port
   is a port bug, never an engine result.
4. **Same checkers**: scores come from the npz through `score_*.py` / `sb7_watch.py`, i.e. Alloy's own code on the
   engine's poses; nothing is re-implemented per engine.
5. **Missing features** -> `not supported`; never forced.
6. **Hardware**: L4 (spot `g2-standard-8`), one VM per engine, `alloy-bench-<engine>-<n>`, label `purpose=alloy`.

## T12 port facts (`packs/t12/t12.json`)

- Geometry: the convex hulls MuJoCo collides (WP13's split pieces, <= 64 vertices each, under every engine's
  convex-hull vertex limit). Load the OBJ as a convex shape **without** re-hulling, decimation or decomposition.
- Inertia: explicit (`mass`, `ipos`, `iquat_wxyz`, `inertia_principal`); never recomputed from geometry.
- Contacts: `contact_body_pairs` lists every body pair MuJoCo collides and its effective sliding friction; pairs not
  listed never collide (MuJoCo's weld-body filter: bodies welded together and a weld body vs its parent's weld body,
  so T12's jointless cameras touch neither link 5 nor the fingers; excludes; contype/conaffinity). Match both.
- Joints: limits, armature, damping, frictionloss as listed; fingers b = -a (`equality`, mimic joint).
- Actuators: affine PD `force = kp (ctrl - q) - kv qdot`, clipped to `force_clamp`: +-28 N m on joints 1-3, +-10 N m
  on joints 4-6, +-100 N on finger_a (scene.xml `actuatorfrcrange` on the joints; the actuators themselves are
  `forcelimited` false, which is what this line wrongly said until XB4-4). The clamp binds in 15/16 B0 episodes
  (peaks 88 N m against 28), so a port without it is not comparable; every T12 row says `actuator_clamp`.
  In every pack `force_clamp` is the composed clamp MuJoCo applies (actuator `forcerange` x gear, then the joint's
  `actuatorfrcrange`; null = none); `tests/xbench/test_ours_packs.py` reads it back from each scene.xml.
- Initial state: `initial.qpos` / `qvel` (MuJoCo order; free joints are `x y z qw qx qy qz`).
- Bully: `action_space` (14 actuators, 50 Hz zero-order hold, rate-limited); episode `(seed, k)` = row `k` of
  `b0.applied_traces(space, 2048, seed)` (Alloy's runner legalises twice; play exactly this); campaign seeds 100000.. (`b0.campaign_seeds`), 2 s horizon.
- Alloy's route: dt 0.625 ms (32 steps per 20 ms tick), SolverMuJoCo.

## Scene packs for P3-2b embodiments (WP13 builds scenes + targets, WP8 builds packs, engine agents run)

WP13 commits the scenes and the Alloy targets; WP8 builds and commits the pack dirs. One pack per embodiment,
`packs/<name>/` (name `emb_e1_fr3`, ...), made by
`scene_pack.py --target <bully target> --scene <scene.xml> --initial keyframe:init --out packs/<name>`
(`--licence` text recorded in the pack). What the builder needs from WP13:

1. **One MJCF scene file** with the robot (model files from the pinned source: mujoco_menagerie commit or
   gym-anything/robotics commit, licence recorded), the fixtures and the objects (arms, bimanual, mobile: table with
   a box, a cylinder and a mug mesh; legged: flat floor and a box obstacle; hand: a cube in the palm). Every body has
   an explicit `<inertial>`. Collision meshes are convex hulls of at most 64 vertices (split bigger hulls the way
   `alloy.tasks.split_scene` does for T12, so every engine's convex-hull vertex limit and the 0.1 mm static check
   hold). Every
   actuated joint has an MJCF position actuator (gain / bias as the model ships them). A keyframe `init` holds the
   start state (legged robots standing).
2. **An Alloy bully target** (`alloy.bully.targets.register_target(name, factory)`) that runs exactly that MJCF on
   the Alloy route, like `task_t12`: action kind `mujoco.ctrl` over all of the MJCF's actuators in order, legal box =
   joint ranges (and inside the torque limits), 50 Hz, a rate limit per tick, `initial` = the keyframe's ctrl,
   `robot_bodies` set, and the route's dt (Simulation's clock rule for 20 ms).
3. **Checks on Alloy's side**: the target builds on CPU and GPU, `alloy_sb7.py --pack packs/<name> --worlds 16
   --seeds 100000-100000` runs clean or its S1 is understood, and the pack's `check_obj_poses` passes (the builder
   refuses a pack whose OBJ placement misses MuJoCo's compiled vertices by more than 1 um).

Everything downstream is keyed by the pack: `b0.py` reads its `action_space`, the S1 scorer rebuilds its
`alloy_target`, and the engine agents' loaders read `<name>.json` + `scene.xml` exactly as for T12.
