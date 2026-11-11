# An elastic rod beside a robot runs far slower than the chain on the CPU

## What is wrong
A rod on the elastic-rod route (`rod_vbd`) that a robot can touch steps in its own solver. The robot steps in
MuJoCo, and the two exchange forces many times per control step. On the CPU that makes a rod-and-robot scene much
slower than the same rod on the chain route, which steps inside MuJoCo with the robot.

## Number
CPU, one core, warm, a 0.4 m USB cable (r 3 mm, 20 segments), `alloy.Scene` + `alloy.build`. Measured with a
development API-speed script (not shipped).

| scene | chain | rod_vbd |
|---|---|---|
| the cable dropped on the ground | 0.90 s per simulated second | 2.49 s |
| the cable's end squeezed by two 0.2 kg pads on a MuJoCo wrist | 1.00 s | 94.9 s (95x) |

The clamped gripper rig of `tests/api/test_rods_grip.py` runs 82.9 s per simulated second. A rod-only solver entry
runs 4 VBD iterations; at 10, on the same machine, the three rows read 4.49, 161.2 and 137.1 s (before a cached
attribute lookup in the coupled model view, the squeezed cable took 216 s).

On one L4 GPU (MuJoCo Warp), in simulated s per wall s summed over worlds, at 4 iterations. Both scenes were
measured in one run on one tree (2026-11-09; development scripts, not shipped); every row passed `sim.check()`.

The squeezed cable (the CPU table's second row):

| worlds | 1 | 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|---|
| chain | 1.2 | 35 | 105 | 168 | 172 |
| rod_vbd | 0.062 | 2.6 | 6.2 | 17.6 | not measured |
| chain / rod_vbd | 20x | 13x | 17x | 10x | |

The clamped gripper rig of `tests/api/test_rods_grip.py`, the chain by `clamp="start", route="chain"`:

| worlds | 64 | 1,024 |
|---|---|---|
| chain | 55 | 225 |
| rod_vbd | 2.6 | 17.7 |
| chain / rod_vbd | 21x | 13x |

So on a GPU `rod_vbd` beside a robot runs 10-21x slower than the chain. An earlier run of the squeezed cable read
22-55x: its `rod_vbd` rows were at 10 iterations (half the speed), and its chain rows read 61 and 160 at 64 and 256
worlds, on an earlier tree.

## Why
The exchange step is the largest dt / 2^j no longer than 0.5 sqrt(m / k), where m is the lightest robot body that
could touch the rod and k the contact stiffness (2e6 N/m). Two 0.2 kg pads give 158 us, so j = 6: 64 exchanges per
10 ms control step, each one a MuJoCo step and two coupling passes of a 4-iteration VBD solve of the rod and the
robot bodies that touch it. The chain needs none of this.
- On the CPU the time is in the solver's launches, not the physics: an exchange at 10 iterations was about 380 small
  kernel launches, about 90 us each, and MuJoCo takes 3 % of the wall.
- One coupling pass instead of two, or 2 VBD iterations instead of 4, is faster but loses the grip: the cable
  stops following the wrist (`tests/api/test_rods_grip.py`).

## Workaround
- Short elastic rods next to a robot route to the chain by default; keep them there unless the scene needs the
  elastic rod (more than 40 segments, both ends clamped, a yielding wire).
- A rod clamped at one end can stay on the chain: `add_rod(..., clamp="start", route="chain")`. The gripper rig
  then runs 1.3 s per simulated second on the CPU instead of 83 s, and holds its twist as the rod does (on the CPU
  a held chain creeps back about 0.14 % per second; `docs/guide/rods.md`, Robots).
- Give pads and links near the rod realistic masses: the clock grows with sqrt(m).
- Run elastic-rod-and-robot scenes on a GPU, many worlds at once: on one L4 the squeezed cable ran 17.6 simulated
  s per wall s over 1,024 worlds (the table above; `docs/guide/rods.md`, Routes).

## Planned fix
- The per-period exchange ladder (a world re-runs only the period that needs a finer step) does not change these
  rows: none of their periods refined.
- A cheaper exchange. Done: a cached attribute lookup in the coupled model view (216 -> 161 s, bit for bit the same
  result) and 4 VBD iterations for a rod-only entry (161 -> 95 s; the grip law holds); a clamped short rod on the
  chain as an opt-in (Workaround). Next: fewer launches per exchange, or a coupling that holds a stiff pad contact
  at a coarser step.
- `rod_vbd` beside a robot at 4,096 worlds was not run in this pass.
- The default stays the chain for short rods until this gap closes.
- GPU free rods were named the next candidate to move to `rod_vbd`. Through `alloy.build` on an L4 the chain is
  faster at 1,024 worlds (178 against 148 simulated s per wall s) and `rod_vbd` at 4,096 (303 against 180;
  `docs/guide/rods.md`, Routes).
