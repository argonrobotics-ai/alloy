# A thin elastic rod can fall through a soft body between its vertices

## What is wrong
- An elastic rod (`add_rod`, route `rod_vbd`) dropped on a soft body (a soft material such as `silicone_40a`, route
  `soft_vbd`) can sink through it to whatever is below. The soft body does not move.
- It happens when the rod is thinner than the gaps between the soft body's vertices. A rod that lands between two
  rows of vertices falls straight through. One that lands on a row rests there for a moment, rolls off sideways, then
  falls through.
- Nothing flags it. `sim.check()["ok"]` is True and `sim.health()` shows no penetration, because the rod and the
  soft body never touch. The build does not warn.

## Number
- **Scene.** A 0.2 x 0.2 x 0.04 m `silicone_40a` pad lies on the ground. A copper wire (`copper_c110`, radius
  1.5 mm, 0.16 m long, along y) is dropped from 30 mm above the pad's top. CPU, one world, `dt` 0.01 s, a development
  probe script (not shipped).
- **The pad's mesh.** Its vertex rows are 12.5 mm apart (16 x 16 x 3 cells, 1,156 vertices). Each vertex touches
  what comes within 2.5 mm of it. So the wire meets a vertex only when its centre line passes within 4 mm of it, and
  the gap between two rows that it meets nothing in is 12.5 - 2 x 4 = 4.5 mm wide.

Pad bottom 4.5 mm above the ground, pad top 42.4 mm once settled. z is the wire's centre line, so 1.5 mm is the
wire on the ground. In both runs `sim.check()["ok"]` is True with no problems listed.

| wire starts over | what happens | lowest z at the end |
|---|---|---|
| a vertex row (x = 0) | rests 4 mm above the top from 0.10 s, slides 3.4 mm aside by 0.40 s, falls | 1.5 mm at 0.50 s |
| a gap between rows (x = 6.25 mm) | never touches the pad, falls straight through | 1.5 mm at 0.15 s |

- **The audit's start.** The audit's pad started with its bottom 1 mm above the ground, inside the vertices' 2.5 mm
  reach. The ground pushed it up at the first steps: its top rose from 41 mm to 77.6 mm by 0.10 s before it fell back.
  `sim.check()` then lists ENERGY_JUMP, and `ok` stays True. The wire still ended on the ground the same way (1.5 mm
  at 0.50 s from over a row, at 0.15 s from a gap), so the jump is not the cause.

## Why
- A soft body touches rigid shapes and elastic rods at its vertices only. Each vertex is a small sphere, here of
  radius 2.5 mm, a fifth of the cell. There is no contact between a rod and the faces of the soft body between the
  vertices.
- A rod meets a vertex only when its centre line passes within the rod's radius plus the vertex radius of it. When
  half the vertex spacing is larger than that (6.25 mm against 1.5 + 2.5 mm here), the rod fits between two rows and
  meets nothing.
- A round rod resting on a single row of vertices is not stable. Any sideways offset tilts the vertices' push away
  from the row, so the rod rolls off into a gap.
- Cloth has the same limit for small rigid bodies:
  [cloth-small-body-passes-through.md](cloth-small-body-passes-through.md).

## Workaround
- If the softness of the body does not matter to the task, make it rigid: `add_box(..., rigid=True)`. The same
  wire on the same pad made rigid rested on its top (centre line 1.50 mm above it, the wire's radius, through 0.60 s;
  `sim.check()["ok"]` True, no problems). An elastic rod beside rigid bodies is verified against a law.
- The chain route is not a way out here. A cable of the same size on it (`cable_usb_composite`, `sim.routes()` says
  `chain`), dropped on the same pad, failed its health check as it landed: PEN_FAIL between 0.05 and 0.10 s, the world
  frozen with the cable still 9 mm above the pad, `sim.check()["ok"]` False.
- Size check. A soft box of volume V gets cells about (6 V / 5,000)^(1/3) on a side (12.4 mm for this pad), with
  vertices of a fifth of a cell. A rod whose radius plus that vertex radius is under half a cell can fall between
  the rows.
- Read the rod's lowest z and the soft body's top after a short run before you trust a rod on a soft body.

## Planned fix
- Contact between rod segments and the faces of a soft body, so that a rod between vertices still meets the surface.
  Until then, a build warning when a rod is thinner than the soft body's gaps. After the 0.1.0 release.
