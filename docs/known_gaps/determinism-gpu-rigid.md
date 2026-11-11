# GPU: rigid bodies and robots in contact do not repeat bit for bit

## What is wrong
On a GPU, two runs of the same scene with the same `seed` give the same bits only while no body takes more than
one contact pair: free flight, a robot arm touching nothing, one sphere or one flat box on the ground. As soon as
bodies pile, tumble onto an edge, or a robot touches its table or an object, repeat runs differ, and worlds that
start identical differ from each other. This holds at every profile: `"standard"` and `"precision"` ask for
repeatable runs but get none on a GPU (the build logs "effective determinism NONE (requested RUN_TO_RUN) ...
Warp deterministic lowering is off"). On the CPU the same scenes repeat bit for bit.

The differences start at float32 roundoff and stay there in a resting scene; where contact is chaotic (a pile of
spheres, a grasp) they grow until worlds land in different places.

`sim.health()` on a GPU lists the notice DET_DOWNGRADED in every world of a rigid, robot or cloth run, at every
profile, also in the scenes that do repeat. A notice leaves the world OK (until 2026-11-10 it made every such world
DEGRADED at `"standard"` and `"precision"`, and `"fast"` did not list it).

A build under `profile="precision"` on a GPU warns once that its runs will not repeat, with the reason; the other
profiles say it through the notice and the build log only.

## Number
One L4 (driver 580), 4 worlds, seed 3, `dt=0.01`, 2 s, largest difference in position (m) or quaternion between
two runs. Fresh builds in one process, two processes, and a run repeated after `sim.reset()` give the same picture;
main b9441c38 and the Newton 1.8 / MuJoCo Warp 3.17 tree agree (on that tree the two-process check ran for the
tilted box only, and the Panda row at `"fast"` stopped with a CUDA graph error: no number; GAP-E3, 2026-11-10).

| Scene | standard | precision | fast |
|---|---|---|---|
| Sphere in free fall | same bits | same bits | same bits |
| One sphere dropped on the ground | same bits | same bits | same bits |
| One box resting flat on the ground | same bits | same bits | same bits |
| Robot arm, one joint ramp, touching nothing | same bits | same bits | same bits |
| One tilted box dropped on the ground | 4e-7 | 2e-8 to 5e-7 | 8e-7 |
| 10 cubes of 5 cm dropped into a pile | 5e-6 to 4e-5 | 5e-5 | 4e-6 to 9e-6 |
| A scanned mesh object on the ground | 1e-4 | 9e-5 | 1e-4 |
| Capsule and cylinder dropped on a table | 1e-3 to 2e-2 | 1e-3 to 2e-2 | 1e-3 to 2e-2 |
| Panda arm with its table, box and mug | 1e-2 | 1e-2 | 1e-2 |
| 10 spheres dropped into a pile | 0.4 to 1.9 | 0.4 to 1.9 | 0.4 to 1.9 |

- On that machine's CPU, the six scenes run there (free fall, sphere, flat box, tilted box, cube pile, sphere
  pile): the same bits in every profile, health OK. The other four were not run on the CPU.
- A task result moves with it: the pick-and-place example (`05_pick_place`) at 64 worlds landed 62, then 61 of 64 with the same
  seed (0.969, then 0.953).
- `"precision"` does not buy repeat runs: the cube pile differs by 5e-5 there, and a warm 2 s run cost 2.2 s
  against 0.99 s at `"standard"`.

## Why
MuJoCo Warp builds its contact and constraint lists with atomic counters, so on a GPU the order of the rows depends
on thread timing; the solver's sums then round differently from run to run. One contact pair has one order. Warp's
ordered lowering, which would fix the order, is off for this route on CUDA.

## Workaround
- Need the same bits (a regression test, a replay): run on the CPU (`alloy.build(scene, device="cpu")`; many
  worlds with `processes="auto"`).
- On a GPU, compare runs with a tolerance, and judge a task by its success rate over worlds and seeds, not by one
  run: expect a few worlds in a hundred to change sides between two runs of a contact-rich task.

## Planned fix
- Repeat runs: Alloy's ordered MuJoCo Warp route (sorted contact rows, fixed-order sums) repeats bit for bit on a
  GPU at 1,024 worlds (`tests/precision/test_mjw_route_gpu.py`), but it is an engine-layer option, off by default
  and not reachable from `alloy.build`; its cost and coverage on these ten scenes are not measured yet.
