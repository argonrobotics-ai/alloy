# Granular MPM: slopes rest at the card's friction angle; a collapsed heap's flank reads lower, a free end slumps

Status: the slope part is closed (2026-11-09). Sand rests at its card's friction angle within 1-2 deg; the
first reading, that it "stands well below the friction angle", came from a law whose layer had a free end. What is
left: the flank a collapse leaves is not the repose angle (26-27 deg for this sand), a squat collapse's runout still
grows with resolution at 16 cells, and one angle serves for both starting and stopping. Pair friction to an MPM
object (`set_friction`) was ignored and is fixed (2026-11-08).

## What

- **Slopes (closed).** A layer of soil_sand_dry (34.5 deg) held between end walls rests at 33 deg and flows at 36. A
  free end of the same layer relaxes to a wedge whose surface is at 33-34 deg to the horizontal.
- **A free end slumps, as it should.** A layer on a plate tilted below the friction angle cannot keep a free end: the
  end relaxes to a wedge at the friction angle, so it slumps back about H / (2 tan(phi - beta)) from the toe (H the
  depth, beta the plate's slope). For a 2 cm layer of this sand that is 4.5 cm at 22 deg, 6.7 cm at 26 and 12.7 cm at
  30. On a short plate the whole layer moves. The first law for this page read that as flow below the friction angle.
- **A collapsed heap's flank is not the repose angle.** A squat block whose sides collapse runs out past the wedge
  it would rest in, so its flank ends under the friction angle: 26-27 deg for this 34.5 deg sand, the same at 4, 8
  and 16 cells in plane strain. To read a repose angle, use a walled layer or a free end's wedge (above).
- **Runout of a squat collapse (open).** That block's runout is 0.68, 0.83 and 0.93 of its height at 4, 8 and 16
  cells: still rising. In 3D the 4-cell block also reads a lower flank (22 deg) than the 8-cell one (27 deg).
- **One angle (open).** Real sand starts to avalanche a few degrees above the angle it stops at (glass beads: 5-8 deg,
  GDR MiDi 2004). Alloy's sand starts and stops at the card's angle.
- **Friction to MPM (fixed).** `scene.set_friction(a, b, mu)` used to write rigid contact pairs only, so a pair set
  between an MPM object and a table or tool was ignored, without a warning. Since 2026-11-08 the build writes every
  sourced pair (from `set_friction` or a library pair card) into the MPM collider (`alloy.integration.mpm_pairs`). One
  limit remains: one MPM solver gives a shape one friction for all its materials, so a shape that two MPM materials
  pair with differently is refused at build, naming both.

## Number

All rows: soil_sand_dry (34.5 deg), 5 mm cells, plane strain between frictionless side walls, a plate with friction
1.2, CPU, 2026-11-09 (laws in `tests/integration/test_mpm_cone.py`).

Layer 2 cm deep between frictionless end walls (a tilting box). Top of the middle third, travel and speed:

| slope | box 16 cm, after 1.0 s | box 64 cm, after 0.3 s |
|---|---|---|
| 26 deg | 0.02 mm, at rest | |
| 30 deg | 0.05 mm, at rest | 0.08 mm, at rest |
| 33 deg | 0.37 mm, 0.08 mm/s | 0.52 mm, 0.35 mm/s, slowing |
| 36 deg | 1.9 mm, 1.8 mm/s, speeding up | 6.9 mm, 22 mm/s |
| 38 / 39 deg | 8.7 mm, 8.8 mm/s (39) | 12.8 mm, 85 mm/s (38, at 0.2 s) |

- The short box holds longer above the angle because its toe wall carries a layer only 4 cells deep.
- A layer 4 cm deep (8 cells) in the 16 cm box is at rest at 30 and 33 deg too (0.12 and 0.18 mm at 0.9 s).

The same layer, 16 cm long, with both ends free, after 1.5 s:

| slope | top of the middle third | wedge surface, to the horizontal | top dropped back from the toe (estimate) |
|---|---|---|---|
| 22 deg | 1.0 mm, at rest | 32.8 deg | 5.7 cm (4.5) |
| 26 deg | 8.1 mm, at rest | 33.2 deg | 9.2 cm (6.7); 8.2 cm at 0.7 s |
| 30 deg | 60 mm, 1 mm/s, stopping | 33.8 deg | the whole layer (12.7) |

Squat block in plane strain, height over half-length 0.5, on a table with friction 0.8, both ends free, at rest
(the flank is a linear fit to the surface between 15 % and 85 % of the top; a wedge at 34.5 deg would run out 0.73 H):

| block height | cells up the flank | flank | runout / height | top left flat at |
|---|---|---|---|---|
| 2 cm | 4 | 26.6 deg | 0.68 | 0.99 H |
| 4 cm | 8 | 27.2 deg | 0.83 | 0.99 H |
| 8 cm | 16 | 26.3 deg | 0.93 | 1.00 H |

- By height band the 8 cm block's flank is 20 deg in its lowest third and 25-26 deg above.

The same in 3D, height over half-width 0.44, settled from 0.2 s (2026-11-08):

| block | cells up the flank | flank | edge runout |
|---|---|---|---|
| 8 x 8 x 2 cm, table mu 0.4 (`set_friction(..., 0.8)`, before the fix) | 4 | 21.5 deg | 1.83 cm |
| 8 x 8 x 2 cm, table `friction=0.8` | 4 | 22.3 deg | 1.36 cm |
| 16 x 16 x 4 cm, table `friction=0.8` | 8 | 27.2 deg | 3.34 cm |

- A squat block's flank also depends on how far the block spreads. With table friction 0.1 instead of 0.8, it runs
  out 8.9 cm and the flank reads 8 deg.
- A tall column's runout matches the dry-collapse law (Lube et al. 2004) within 1-4 %.

## Why

- The yield rule is Newton's Drucker-Prager cone, sqrt(J2) <= mu p, and Alloy passes the card's tan(phi) as mu. In a
  flow that keeps its volume the stress lines up with the strain rate, so the shear on the slip plane reaches mu
  times the normal stress and a slope rests at tan(beta) = mu: the card's angle.
- A wedge that has just stopped reads 1-2 deg under the angle because a moving toe overshoots.
- The first diagnosis (2026-11-08) was wrong. Its layer had free ends, and the stresses read behind the retreating
  toe (along the slope 0.6-0.9 of the one normal to it) were taken for a fault of the cone. With end walls the same
  layer rests at 30 deg, where that law saw 52 mm of travel.
- The collapsed heap: a collapse is not a slow relaxation. The falling side arrives with speed and runs out past the
  static wedge, so the flank it leaves is under the repose angle; the free end on a tilted plate, which moves less,
  ends 1-2 deg under it. The flank's angle does not change with resolution; the runout does, and is not converged.
- The friction gap (fixed): pair friction lived only in the rigid contact table. Nothing passed it to the MPM
  collider until the build learned to (2026-11-08).

## Workaround

- Contain a slope that must stand: a wall or a lip at its lower end, or leave room for the end to slump (the
  estimate above).
- A heap whose shape after a collapse matters: 16 or more cells up the block, and expect the runout to be a few
  percent short still. Do not read a repose angle from it.
- `ALLOY_MPM_CONE=mc` is not a workaround. It was offered here on 2026-11-08 for slopes; it sets Mohr-Coulomb's
  compression-cone match, mu = 6 sin(phi) / (sqrt(3) (3 - sin(phi))), which is the cone of a 38.9 deg sand for a
  34.5 deg card. The walled layer then still rests at 36-40 deg, and a collapsing column stops 31 % short of the
  dry-collapse runout (CARD-9, Lube et al. 2004; 2 % by default). It stays as an opt-in for comparing with codes
  that use that match (`sim.routes()` names it).
- Friction against MPM material: `set_friction(sand, table, 0.8)` and `add_box(..., friction=0.8)` on the shape
  now step the sand alike. With two MPM materials on one shape, give both the same pair friction.

## Planned fix

- Slopes: none needed. The laws are the tilting box (rests at 33 deg, flows at 36) and the free-end wedge (31.5-35.5
  deg, slump-back within 1.5x the estimate).
- Squat collapses: the runout's convergence past 16 cells is not measured (32 cells is 200,000 particles on the
  CPU). No change to the solver is planned for it.
- Start and stop angles: not planned.
- Pair friction to MPM: done (2026-11-08).
