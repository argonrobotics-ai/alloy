# A precision fit pushed sideways leans 2 to 6 % past its clearance, by push direction

## What is wrong
A peg seated in a bore (`mode: sdf`, the default) and pushed sideways at its top leans 2 to 6 % further than the
clearance allows, and how much depends on the direction of the push in the world frame. The same peg, bore and force
read 1.02 times the limit in one direction and 1.05 or more in another, on Newton 1.8 and before it. A rigid peg
would stop at the limit in every direction. The law for this fit is 1.05 times the limit; a push along a world axis
on Newton 1.8 reads 1.048 to 1.055 on the MuJoCo Warp path, on and over it. The wall penetration is 13-20 µm of the
50 µm allowed.

## Number
A 12 mm peg in a 12.2 mm bore, 20 mm deep (0.1 mm radial clearance; allowed lean 0.575 deg), pair stiffness
2e6 N/m. It is pushed 5 N sideways at its top for 0.5 s at a 0.3125 ms step, in 16 worlds with small random tilts and
offsets. GPU (one L4), MuJoCo Warp path, 2026-11-09. Lean / allowed, worst world / mean over the worlds:

| push direction, from +x | Newton before 1.8 (main 47157c433) | Newton 1.8 (bump 43ae205ba) |
|---|---|---|
| 0° | 1.026 / 1.024 | 1.051 / 1.044 |
| 7.5° | 1.044 / 1.040 | 1.044 / 1.029 |
| 15° | 1.040 / 1.035 | 1.029 / 1.024 |
| 22.5° | 1.025 / 1.019 | 1.023 / 1.020 |
| 30° | 1.021 / 1.017 | 1.033 / 1.020 |
| 45° | 1.045 / 1.041 | 1.040 / 1.038 |
| 60° | 1.029 / 1.025 | 1.032 / 1.025 |
| 90° | 1.026 / 1.023 | 1.048 / 1.040 |
| 135° | 1.044 / 1.040 | 1.040 / 1.037 |
| 180° | 1.026 / 1.025 | 1.051 / 1.043 |
| 270° | 1.027 / 1.025 | 1.048 / 1.039 |

Repeats of the 0° push, 2026-11-09 and -10. `tests/contact/test_lean_gpu.py` makes this push: its law is 1.05, an
expected failure on Newton 1.8's MuJoCo Warp path (7 of the 15 runs read under 1.05), with a hard guard at 1.08.
It also holds the pushes at 45° and 135° to 1.05 (1.040 and 1.039-1.040, 3 runs each); the push at 7.5° read
1.048, 1.046 and 1.052 in 3 more runs and is an expected failure of the same kind.

| Newton | path | runs | worst world | mean over the worlds | wall penetration |
|---|---|---|---|---|---|
| before 1.8 | MuJoCo Warp | 6 | 1.026 (every run) | 1.024-1.025 | 15-16 µm |
| before 1.8 | Newton | 4 | 1.026 (every run) | 1.026 | 14 µm |
| 1.8 | MuJoCo Warp | 15 | 1.048-1.055 | 1.041-1.046 | 13-20 µm |
| 1.8 | Newton | 11 | 1.040-1.044 | 1.037-1.038 | 10-14 µm |

The lean holds for another 0.5 s of push (1.052 after 0.5 s and after 1.0 s). On the MuJoCo Warp path a world can
lean 0.01 further for about 8 steps when the row of its bottom patch is re-picked (seen once in two runs read at
every step). A solver tolerance of 1e-8 or 300 iterations changes none of these numbers.

## Why
The lean past the limit is the penetration of the peg's two wall patches divided by the 19.9 mm between them. The
contact layer gives each patch (the rows of one pair whose normals fall in the same 27° cone) the pair's stiffness
and shares it equally among the rows of the patch that touch. With one touching row the top patch sinks 3.8 µm and
the bottom one 1.3 µm: 1.026. Newton's contact reduction keeps, for each of its own normal bins, the deepest contact
and the outermost ones. When the push points near the edge of a bin, the contacts kept at that edge sit on the
bore's rim 18-22° from the deepest one and still touch, 0.4-3 µm deep. Each takes an equal share of the stiffness,
so the deepest row keeps a half or a third of it and sinks to 6.2-8.6 µm: 1.035-1.055.

Counted per world at the end of the push: before Newton 1.8 at 180°, all 16 worlds have one touching top row (3.8
µm) and read 1.023-1.026; on Newton 1.8 at 180° they have two or three (8.3, 3.0 and 0.4 µm in one world) and read
1.036-1.051; on Newton 1.8 at 7.5°, 13 worlds have one (1.023-1.033) and 3 have two (1.038-1.044).

Newton 1.8 turned the reduction's normal bins so that every world axis sits 14° from a bin edge. That moved a push
along a world axis from the middle of a bin to near an edge. The spread itself is older: before 1.8 it shows at
7.5°, 45° and 135°.

## Workaround
- Allow 6 % more lean than the clearance gives for a side-loaded fit: 11 µm of summed wall penetration over 20 mm
  of engagement at 5 N.
- Where the lean matters more than that, turn the scene so the side load points 22.5° off a world axis in the
  plane across the fit's axis (1.023 on Newton 1.8).

## Planned fix
Share the patch stiffness by depth instead of equally, so a row that barely touches takes almost none. Planned
after the release candidate; it turns the expected failure in `tests/contact/test_lean_gpu.py` back into a pass. A
first trial (2026-11-09) reads 1.022-1.036 in the six directions tried on the MuJoCo Warp path and 1.032 at 0° on
Newton's, but is not ready: on the MuJoCo Warp path the barely touching rows then switch between loaded and
speculative from step to step, and in 1 to 3 of 16 worlds the lean jumps to 1.05-1.10 for about 20 steps, several
times in 0.4 s. The share law has to keep such rows from switching first.
