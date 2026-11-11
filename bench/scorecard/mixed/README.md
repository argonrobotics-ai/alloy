# Mixed-batch hold benchmark (scorecard rows 1 and 2)

Several robot task types stepped together on one GPU, physics only: one 5 ms env step, small noisy hold-position
targets (the start pose plus uniform noise, 16 pre-drawn sets), no policy, observations, rewards, resets or cameras,
and a device sync after every env step. It follows the mixed-task protocol described in skild.ai/blogs/steel.
Beside every speed number it gives what that post does not: GPU memory, a glitch rate with its 95 % bound, and
whether each contact is held, on the same worlds.

## Run it

```bash
source .venv/bin/activate          # an Alloy install with a GPU (AGENTS.md, section 1)
M5=franka_lift,ur10_reach,go2_flat,g1_flat,hand_cube
# one point: speed, memory and each type's share of the step time
PYTHONPATH=. python bench/scorecard/mixed/mixed.py --engine alloy --tasks $M5 --envs 5120 --mode speed --out out/mix5
# the same worlds, 4 s, scored per task type
PYTHONPATH=. python bench/scorecard/mixed/mixed.py --engine alloy --tasks $M5 --envs 5120 --mode quality --out out/mix5
# the whole row: every engine form, batch doubled until a point fails, one process per point
PYTHONPATH=. python bench/scorecard/mixed/mjw_buffers.py --tasks $M5 --out out/buffers.json
MJW_BUFFERS=out/buffers.json bash bench/scorecard/mixed/sweep.sh out/mix5 mix5 $M5 256 65536
python3 bench/scorecard/mixed/build.py out out/report      # rows.jsonl and tables.md
```

`--envs` is the total, split evenly over the task types. Each point writes `<tag>.json` (a failed point writes its
error, so an out-of-memory batch is a row too). The first run of a scene compiles kernels: time only warm runs
(the harness warms up before its timed blocks).

## What is measured

| Engine (`--engine`) | What it is |
|---|---|
| `alloy` | Alloy as shipped: robots step at 1.25 ms, 4 physics steps per 5 ms env step |
| `alloy_5ms` | a probe, not a supported setting: one 5 ms physics step, to show the cost per solver step |
| `newton_mjw` | MuJoCo Warp through Newton's `SolverMuJoCo` alone on the same robot files (no Alloy routing, contact settings or health checks), `--substeps` steps per env step, CUDA graphs. By default it runs the Newton copy pinned inside Alloy, which carries Alloy's patches: `--newton-module newton` measures a stock Newton install instead. Every row records which ran |

Neither Alloy nor Newton's MuJoCo solver steps different robots in different worlds of one model, so a mixed batch
takes one of two forms, and every row says which:

- `--form multi`: one simulation per task type, all in one process on one GPU, stepped in turn, one sync per env step.
- `--form one_scene` (Alloy): every world holds all the robots on a grid; a world counts as one env per robot.

Speed is total envs x steps / wall time, the median of 3 blocks of 100 env steps. Memory is what `nvidia-smi`
reports for the GPU right after the timed blocks. Quality (`single_task/pxs_score.py`) is read from each engine's
own state over 800 env steps: S1 = a non-finite state, a body faster than 20 m/s, a body origin more than 1 cm under
the floor, or the cube more than 1 cm into its support; held = the resting cube moves under 5 mm, a standing robot
keeps 70 % of its settled base height, the in-hand cube stays within 10 cm of its start in the hand's frame.
Alloy rows also give Alloy's own health check.

## Fairness notes

- **MuJoCo Warp buffers.** Newton sizes MuJoCo Warp's constraint and contact buffers from the start state. A robot
  that makes more contacts later overflows them: constraints are dropped and a warning is printed every step, which
  also slows the step. `mjw_buffers.py` measures what each task needs; pass its file (`--buffers` or
  `MJW_BUFFERS`) so those rows are valid. Rows without it are labelled `mjw_*`, with it `mjws_*`.
- **Frozen worlds are not a speed.** Alloy freezes a world that fails its health check (INVALID) at its last good
  state, and the other worlds go on. Frozen worlds cost little and look calm to the outside quality rule, so a speed
  point where more than 5 % of any type's worlds are frozen is reported as a failed point, and an Alloy row's glitch
  count is at least its INVALID count.
- **Step counts differ and are shown.** Alloy takes 4 solver steps per env step; the MuJoCo Warp rows are given at
  4, 2 and 1.
- **Best batch.** Each engine is reported at its best measured batch; the sweep table shows every batch and the
  first one that failed.

## Task types

`single_task/tasks.py` holds six tasks built from the robots that ship with Alloy (MuJoCo Menagerie models in
`library/robots/`): Panda lift and reach, UR10e reach, Go2 and G1 standing, a LEAP hand holding a cube.
`extra_tasks.py` adds four more library embodiments (FR3, two FR3, a mobile base with two FR3, H1-2). Each file is
re-zeroed at its `init` keyframe and held there. H1-2's keyframe is not a stable standing pose under a plain hold:
read its S1 column, not its held column.

A type can also be an Isaac Lab task's own robot file: `--tasks usd:SPEC.json,...` (Alloy engines, `--form multi`;
the prepared robot files go in `<out>/models/`). `single_task/usd_task.py` builds that scene. Those files are not
part of this repository: it reads them from your own Isaac Lab install (the asset cache Isaac Lab fills when it runs
the task) and a task description you dump from it; its docstring names the fields. It needs `usd-core` and
`newton-usd-schemas`. It carries the task's drive gains, default pose, joint armature (also on multi-axis joints)
and self-collision switch. It does not carry effort limits, per-body gravity switches (Isaac Lab turns gravity off
for some hand and arm robots) or explicit actuator models: say so beside any row built from those files.

## Files

| File | What |
|---|---|
| `mixed.py` | one point: build the groups, step them together, speed + memory or quality |
| `sweep.sh` | the row: every engine form, batch sweep, one process per point, resumable |
| `mjw_buffers.py` | the buffer sizes MuJoCo Warp needs per task |
| `build.py` | `rows.jsonl` and `tables.md` from the point files |
| `extra_tasks.py` | four more task types |
| `single_task/` | the single-task hold harness the mixes import: tasks, scorer, Alloy and Newton steppers |
