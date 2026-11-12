# A two-pad pinch does not lift plies it presses hard

## What is wrong
Two pads that land on a stack of fabric plies on a table, press, close and lift carry the stack up when the press is
light. Pressed at 100 N, the closing pads gather nothing: they rise and the plies stay on the table. The world is
valid throughout (no health flag), so a script that only checks health reads a grasp that did not happen.

Before 2026-11-10 the sewing note stated a 96 % carry at 100 N. That row was not a pinch: the pads had slipped off the
mesh's vertex row and stood on the table, and the lift drew up cloth that lay above their bottoms.

## Number
The rig of `alloy.tasks.sew_ply_pinch` (CPU): two 16 x 6.4 mm pads, 0.2 kg each, on 0.61 mm muslin plies (10 mm
cells) on a table; press, close at 10 N per pad, lift 10 mm. Each pad is centred on a vertex row; every row below is
`pressed` (each ply has a vertex under each pad through press and close, no pad reaches the table, no constraint row
dropped). Spot n2 VM, main ec7a47cb2 + the rig; carry is the rise of the particles between the faces over the pads'
9.94 mm.

| plies | press | carry | pad bottom over the table |
|---|---|---|---|
| 2 | 10 N | 9.88 mm, 99 % | 0.76 mm |
| 2 | 30 N | 8.81 mm, 89 % | 0.62 mm |
| 2 | 100 N | -0.93 mm (on the table) | 0.31 mm |
| 3 | 30 N | 8.02 mm, 81 % | 0.87 mm |
| 3 | 100 N | -0.80 mm (on the table) | 0.44 mm |

- Two plies at 100 N with the pads opened 16 mm instead of 13.6 mm (still on the row): -0.90 mm.
- With a 1 mm self-contact radius on the cloth (the sewing proof's setting) the 100 N pinch lifts two plies: 9.81 mm,
  99 %. One row.
- Between 30 and 100 N only one press was run, with Newton's linear contact penalty: 50 N, 3.92 mm (39 %).
- Under 100 N the two plies' 1.22 mm stack is pressed to 0.31 mm under the pads.

## Why
Not established. What the rows show:
- The loss comes with the press, not with the ply count: two and three plies both lift at 30 N and both stay at 100 N.
- Each pad bears on one vertex of each ply (a 6.4 mm pad on a 10 mm grid), so the whole press goes through that
  vertex, and the stack under the pad is squeezed to a quarter of its free thickness (two plies 0.31 of 1.22 mm,
  three 0.44 of 1.83 mm).
- With a 1 mm self-contact radius the pads sit higher under the same press (0.42 mm), and the pinch lifts the plies.
A finer mesh under the pads has not been tried.

## Workaround
- Press lightly before closing: 10 to 30 N lifts two plies (99 %, 89 %).
- Check the grasp, not only health: `PinchResult.pressed` in this rig; in your own scene, that the cloth between the
  fingers rises with them.
- Mesh the cloth so that a finger pad covers vertices in its interior (the rig's pads cover one row; a pad on a row's
  edge gives carries that move by tens of points with the landing spot).

## Planned fix
After the release candidate, with the cloth-grasp work (a law for lifting a ply stack off a flat support): the same
pinch on a finer mesh, and the press swept between 30 and 100 N.
