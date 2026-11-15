# A body spinning fast on one off-axis contact point can gain energy

## What is wrong
A body spinning fast on one off-axis contact point can gain energy. MuJoCo's soft friction pushes a contact point
that whirls with the spin along its slip instead of against it.

## Number
- Measured: a 373 g brass top, 5 cm across, spun to 250 rad/s on one tip vertex 0.3 mm off its axis. In 4.5 s it
  gained 2 rad/s and 23 mJ, and it leaned from 15 to 37 deg, where tip friction should slow it.
- At the contact, friction did +14.6 mW of work, and the normal contact dissipated 5.5 mW.
- Measured on the CPU (MuJoCo C, rk4). The same happened:
  - at a 4x finer step, under Euler at 19.2 kHz, and on the CPU Warp route;
  - with impratio 1 or 300, mu 0.36 or 1.0, a pyramidal cone, or noslip;
  - in plain MuJoCo with that hull.

## Why
The tip vertex sat 0.3 mm off the spin axis, so the contact point whirled round the axis with the spin, and the soft
friction force followed it. Since the hull fix, a convex mesh keeps its shape to within 0.1 mm, up to 2,048 hull
vertices. A rounded tip stays rounded, and the same top slows as it should: spin 248.7 -> 247.5 rad/s, -34 mJ, and
it rises a little. A body whose actual shape spins on a sharp point off its axis can still gain energy.

## Workaround
Round such a point, for example with a small sphere or capsule shape at the tip, or keep its spin low. Read
`sim.check()` and the body's energy before trusting a long spin.

## Planned fix
None planned for a sharp off-axis point: plain MuJoCo does the same with that shape. The hull fix removes the case
where the mesh's own hull made the sharp point.
