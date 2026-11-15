# Pushing one block out of a stack drags the blocks above it

## What is wrong
When a block is pushed out from under other blocks, the blocks resting on it are dragged along, more than real
friction allows, and the stack shakes while the block slides. A stack at rest is still (0.001 mm/s; see the
CHANGELOG entry "objects resting beside a robot or a moving fixture no longer jitter").

## Number
A block tower: 18 levels of three 12.6 g alder blocks (7.5 x 2.5 x 1.5 cm), each level laid across the one below. A
kinematic steel rod pushes the middle block of level 7 out at 5 cm/s. CPU, MuJoCo Warp route, one world. Development probe, not shipped.

| run | top of the tower moves | blocks moved over 5 mm |
|---|---|---|
| now (after the resting fix) | 23.6 mm | 38 of 53 |
| before the resting fix | 32.0 mm | 41 |
| middle block 0.4 mm thinner (loose), before the fix | 3.1 mm | 0 |

- Before the rod touches, the tower is still. From first touch to the end of the push, blocks across the tower bob
  at 1.5-2 mm/s, 10 levels away included, and the contacts between levels use 50-90 % of their friction.
- How far the tower moves depends on how far the block is pushed, not on how fast. Over the same 1.6 cm of push the
  top moved 9.3 mm at 1.25 cm/s and 8.2 mm at 5 cm/s.
- The tower moves without breaking: health stays OK.

## Why
MuJoCo's own soft friction lets a pushed, sliding block drag the blocks resting on it. Where a block slides under
another, MuJoCo shifts weight onto that sliding contact and lets its friction act like a damper rather than a
constant sliding force. The blocks beside it then carry less weight, their friction gives way first, and the blocks
on top move with the pushed one. Real friction would hold them: the sides carry twice the drag.

This is MuJoCo's contact model, not Alloy's additions. The same tower in plain MuJoCo C (double precision, none of
Alloy's contact code) is dragged 29 mm. A plain MuJoCo C rig of six blocks drags the top three 13 mm during a 63 mm
push at 5 cm/s, and 44.5 mm at 1.25 cm/s: the slower the push, the more the drag. The solver's iteration count,
the step size, the rod's own contact and MuJoCo's noslip pass make little or no difference.

## Workaround
- Push a loose block, as players do: a block 0.4 mm thinner than its neighbours carries no load, and pushing it out
  moved the top 3.1 mm with no other block moved over 5 mm.
- More load above helps: in the rig, ten more levels on top cut the drag from 13 mm to 1 mm.
- Check the tower's other blocks (their `.pos`) after a push rather than assuming the tower stood.

## Planned fix
Being measured: shorter friction time constants hold the top blocks in the rig (1.8 mm at 5 ms) but have a
known cost elsewhere. No change until those rows are in.
