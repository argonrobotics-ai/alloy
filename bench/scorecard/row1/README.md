# Scorecard rows 1 and 2: robot speed in Skild's protocol, and quality on the same tasks

`rows.jsonl` here (row 1) and `../row2/rows.jsonl` (row 2) are the published form of one measurement campaign:
17 robot tasks (16 of the 20 in Skild's Steel post, skild.ai/blogs/steel, plus Go2 flat), one spot NVIDIA RTX PRO
6000 Blackwell (96 GB) on GCP (g4-standard-48), Alloy main `ca9e66eec`, 2026-11-10. Each line is an
`alloy.scorecard.row/1` record (fields: `alloy/bench/scorecard/types.py`); each Alloy record names its command.

**Row 1: Alloy as shipped runs at 0.30x MuJoCo Warp's speed (geometric mean over 17 tasks; 0.11x on Cartpole to
0.76x on ANYmal-D rough). A loss.** One run per task, each engine at its best measured batch. Each task's record
gives the ratio, Alloy's env-steps/s and best batch, and the solver steps per env step of each engine.

**Row 2:** Alloy's glitches (`glitch_rate`, k of n worlds with its 95 % Wilson interval) and contact held
(`held_rate`; `extra.kind`: `inhand` a cube in the Allegro hand, `stand` a legged robot standing with no policy,
`rest` objects at rest, `free` a lifted object), 1,024 worlds, 4 s of the same loop, in a run of its own (not the
best-batch speed run). Glitches: 0 of 1,024 on 16 of the 17 tasks. On the Allegro cube task 77 of 1,024 (7.5 %; 6.1
to 9.3 % at 95 %): the cube dropped from the hand falls with no ground below (Alloy's scene of that task has no
ground) until Alloy's own health check flags the world INVALID, and an INVALID world counts as a glitch here.
Cube held 4 s in the Allegro hand: 885 of 1,024.
These row 2 counts were measured on Alloy main `ca9e66eec`, before this release's penetration flag. On the release
candidate (rc2-1110, one L4, no policy, 4 s) that flag marks fallen two-legged robots INVALID: H1 976 of 1,024,
Humanoid 419, Humanoid-Direct 426, Cassie 14. Their links sink 2 to 3 cm into the floor after the fall, over the
flag's 1 cm; they sank as deep before, unflagged. Robots that stand are not flagged: the other 13 tasks read 0, and
the Allegro cube 77 as above.

## Protocol (Skild's)
Physics stepping only: per env step, a small noisy hold target on every actuator, one 5 ms env step, a device sync.
No policy, observations, rewards, resets or cameras. Env-steps/s = worlds x steps / the median wall time of three
blocks of 100 steps, after a warm-up.

## How each side was run
- **MuJoCo Warp:** a third-party robot-learning framework's robot tasks, each at its own MuJoCo Warp preset there:
  2 solver steps per 5 ms env step (1 for Ant and Cartpole). That harness is not in this repository; its own numbers
  are not published, only the ratio.
- **Alloy:** `bench/scorecard/mixed/single_task/alloy_hold.py` (`--mode speed` and `--mode quality`, `--step
  default`), loading each task's own robot file (USD) through `alloy.Scene.add_robot`, with the task's joint gains,
  start pose, self-collision switch and armature. The task specs it reads (`usd:SPEC.json`) are not in this
  repository; the six Menagerie stand-in tasks in `single_task/tasks.py` run from the repository alone.

## Not like for like (read before quoting)
- Alloy as shipped (its default) takes 4 solver steps of 1.25 ms per 5 ms env step; MuJoCo Warp takes 2 (1 for Ant
  and Cartpole). This is most of the gap. These rows were measured at that default. A coarser robot step is an
  opt-in (`scene.rigid_step = 0.0025`, or `"auto"`: `docs/guide/worlds.md`), not measured in these rows; scorecard
  row 1b measures what it buys. The default does not change in this release.
- Alloy uses its own contact and solver settings, a PD drive on every joint, no effort limits, gravity on every body
  (the original tasks switch it off for the Allegro hand and the reach OSC robot), flat ground for the rough-terrain
  tasks, one box (6 cm) where the original task spawns objects of several shapes, and no ground under fixed-base
  robots.
- Alloy's best batch is 16,384 to 65,536 worlds. As shipped, legged robots
  in Alloy slow down sharply at 32,768 worlds (ANYmal-D flat 50x slower than at 16,384; not traced yet).
- Standing with no policy: Alloy's G1 stands in all 1,024 worlds, its H1 in 37, its Humanoid and Cassie in 0. Alloy
  holds the pose with a PD drive where the original tasks use their own actuator models, so those speed rows time
  different states.
- Not measured: Skild's two camera tasks of the Kuka-Allegro hand (no camera in this row), H1 / G1 rough and the
  mixed batches (`bench/scorecard/mixed/`).

Built from the campaign's own rows by the scorecard maintainers; the numbers are unchanged from those rows.
