# A rod folded tighter than one segment passes through itself

## What is wrong
A rod on the elastic-rod route (`rod_vbd`) collides with itself: every pair of segments that are not neighbours
makes contacts. When the rod folds back on itself (a hairpin, a coil) more tightly than one segment's length, the
segments at the fold overlap. They go deep, the world fails its health check (PEN_FAIL, ENERGY_JUMP), and it is frozen
at its last good state.

## Number
CPU, `alloy.Scene` + `alloy.build`. A USB cable (`cable_usb_composite`, r 3 mm), laid flat on the ground as a
hairpin: two 0.15 m legs 30 mm apart, joined by a half circle. A held 0.05 N push on one leg's end closes the legs
(a development script and its logs, not shipped):

| segments (length) | closest legs, centre to centre (6 mm = touching) | `sim.check()` |
|---|---|---|
| 40 (8.7 mm, 1.45 diameters) | 5.0-5.4 mm | fails at 0.55 s: a 5.7 mm deep contact, an energy jump |
| 80 (4.3 mm, 0.72 diameters) | 6.7-7.7 mm, legs touching and holding | ok |

## Why
Two causes, not yet separated:
- The fold at the hairpin's end closes to about the rod's own radius. A segment longer than that fold cannot bend
  around it, so the segments on either side of it overlap.
- Contacts between two segments of the same rod are solved Jacobi: each treats the other as held for that
  iteration. This is stated in the rod solver's own notes. It is weaker than the contact against a fixed or rigid
  body.

## Workaround
- Where a rod folds or coils, give it segments shorter than about its diameter (`Scene.add_rod(segments=)`).
- Read `sim.check()` after a run in which a rod touches itself.

## Planned fix
- Self contact solved inside the chain's own block solve, so a fold is not Jacobi. It is not scheduled.
- Coils and knots get a law of their own (contact force and the shape they settle to) before anyone relies on them.
