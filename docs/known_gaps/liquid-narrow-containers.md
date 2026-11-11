# Liquid in narrow containers and thin streams needs finer particles than the default

## What
- Liquid runs at 4 mm particle spacing by default (`liquid_accuracy="default"`). `"super"` gives 2 mm spacing, at
  about 16x the cost.
- An opening narrower than 8 particles across is too coarse. That covers a cup, a bottle mouth, a hole or a funnel neck
  the liquid sits in or must pass through: under 32 mm at default, 16 mm at super.
  - Liquid poured into it partly misses.
  - Liquid at rest in it can start INVALID: particles jet off the walls in the first few hundredths of a second, and
    the health monitor freezes the world.
- Containers narrower than 24 mm are unreliable at both settings.
- A pour whose nozzle is under 2 particles across is a column of single drops, not a stream (under 8 mm at default),
  and most of it scatters.
- The build warns once about each case. The warning names the item, its width in particles, and the fix:
  `liquid_accuracy="super"`, a wider opening, or a pour with a higher `rate` or lower `speed`. It never refines
  silently. It checks `add_open_box`, `add_open_cylinder`, `add_block_with_hole` and cards with a `passage_width`;
  an `add_mesh` container is not checked.

## Number
CPU probes through `alloy.Scene`, main 14275571e (development probe scripts, not shipped):
- **Pour (half the cup's volume, the stream half its width), share kept in the cup:**
  - 4 mm: 95.6 % at 24 mm (6 across); 99.7 % at 32, 48 and 70 mm (8, 12 and 18 across).
  - 2 mm: 97.6-99.8 %.
- **Water at rest, 40 mm deep:**
  - 4 mm: cups of 12, 16 and 24 mm (3, 4 and 6 across) went INVALID with ENERGY_JUMP within 0.04 s. 32 and 48 mm were
    healthy.
  - 2 mm: 8 and 16 mm went INVALID. 12 mm stayed healthy but lost 68 particles over the rim.
- **Stream into a 70 mm cup, share that lands:** 36 % at 1 particle across (either spacing); 99.0-100 % at 2-5
  across.

## Why
- SPH resolves the pressure near a wall over a kernel of 2 particle spacings each side.
- With fewer than about 8 particles across, the kernels of opposite walls overlap. The pressure solve then has
  almost no bulk to balance, so the first steps push particles off the walls.
- A one-particle stream has no neighbours to hold it together.
- The 24 mm floor at super is measured, not explained. It may come from the cup's walls: `add_open_cylinder` builds
  them from 24 flat 4 mm slabs.

## Workaround
- Build with `liquid_accuracy="super"` when the warning says it is enough (openings of 24-32 mm, streams of 4-8 mm).
- Otherwise widen the container, or widen the pour's stream (`rate` up, `speed` down).

## Planned fix
None scheduled. A third, finer setting would cost about 16x more again.
