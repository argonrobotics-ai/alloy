# A fast small body passes through a thin static part

## What is wrong
Rigid contact has no swept test. A body that travels about its own size in one rigid step meets a thin part already
overlapping it, and the contact cannot stop it: it comes out of the far side. A 2 cm steel ball dropped on a static
1 cm plate lands from 4 m (8.9 m/s) and passes through from 5 m (9.9 m/s). Below that speed the ball lands, but
dips into the plate first, by more than the travel of one step. Thin moving parts pass at much less travel: coins
and pins 2-6 mm thick went through a 5 mm pane at 3.8 mm per step, and a pin of radius 1.5 mm at 1.4 mm per step, so
no single travel per step is safe for every part (second table).

Since this page was written the build warns and health flags it (below); the pass-through itself is not fixed.

## Number
Steel ball on a static plate 0.3 m above the ground, `Scene(dt=0.01)` (rigid step 1.25 ms), CPU, MuJoCo C route,
main ba2713ec0. The ball's speed is swept in 0.25-1 m/s steps at two or three start offsets within one step's
travel; "passes from" is the travel per rigid step (speed x step) at the first speed that passes. One run per
case (the CPU repeats bit for bit). Probe and rows: development notes, not shipped.

| ball radius | plate | passes from (speed at 1.25 ms) | travel per step there |
|---|---|---|---|
| 10 mm | 2 mm | 6.5 m/s | 8.1 mm |
| 10 mm | 5 mm | 7.5 m/s | 9.4 mm |
| 10 mm | 10 mm | 9.0 m/s (8.5-9.5 over start offsets) | 10.6-11.9 mm |
| 10 mm | 20 mm | 11.5 m/s | 14.4 mm |
| 10 mm | 40 mm | 18.0-18.5 m/s | 22.5-23.1 mm |
| 5 mm | 10 mm | 6.0-6.5 m/s | 7.5-8.1 mm |
| 25 mm | 10 mm | 13.0 m/s | 16.2 mm |

- Above the first passing speed the ball passed in 207 of 207 cases started with a speed and in all but 1 of 81
  drops.
- The rigid step: the same travel per step (10.6-11.2 mm, 10 mm ball and plate) at 2.5, 1.25 and 0.625 ms, so the
  passing speed doubles when the step halves (4.5, 9.0, 17.5 m/s); at 0.3125 ms it is 25 m/s (7.5-8.1 mm).
  `Scene(dt=...)` of 5, 10 or 20 ms does not change the rigid step (1.25 ms in all); `scene.rigid_step` does, and
  so does a `dt` of 1 ms or less.
- A 5 cm cube (125 g) is not monotone: on the 10 mm plate it first passes at 15.5-18 m/s (19-23 mm per step),
  lands again between 16 and 18 m/s at one start offset, and passes at every speed from 18.5 m/s.
- A landing ball still dips: at 5 / 7 / 8.5 m/s its lowest point goes 8.4 / 11.1 / 14.0 mm below the plate's top
  face (the plate is 10 mm thick) before it is pushed back out. On the ground plane alone it never passes, and dips
  1.2 to 1.9 x the travel per step, by where in a step it meets the plane: 9 mm at 6 m/s, 16 mm at 10 m/s,
  31 mm at 19 m/s when started with that speed; 25.6 and 26.8 mm in drops from 6 and 8 m (10.8 and 12.5 m/s).
- `profile="precision"` and the CPU's MuJoCo Warp route (`ALLOY_CPU_ROUTE=warp`, the GPU's code; 1 m/s grid) give
  the same 9 m/s. One L4, 1,024 worlds: the same drops pass (5 m and up) and land (4 m).
- The contact gap (5 mm; a robot's `contact_gap=`) does not change any of this: gaps of 5, 10, 20 and 40 mm gave
  the same rows to the last digit.

Thin parts onto a thin pane: a static glass pane 5 mm thick, parts dropped from 1.0 m (4.4-4.6 m/s) or 7.7 m
(12.3-12.4 m/s) with `scene.rigid_step` pinned (0.3125 ms in the last column), 32 worlds per cell, one L4 (MuJoCo
Warp), alloy 0.1.0 (rc6-1110); each cell counts the worlds that read TUNNEL. Probes: development
probes (probe3.py and its log), not shipped.

| moving part (thinnest size) | 3.8 mm per step (12.3 m/s, 0.3125 ms) | 1.9 mm (12.3 m/s, 0.156 ms) | 1.4 mm (4.5 m/s) |
|---|---|---|---|
| ball, radius 5 mm (10 mm) | 0 of 32 | | |
| cube 10 mm | 0 of 32 (7 INVALID, no TUNNEL) | | |
| ball, radius 3 mm (6 mm) | 7 of 32 | | |
| capsule, radius 3 mm, lying flat (6 mm) | 32 of 32 | 0 of 32 | 0 of 32 |
| coin 4 mm thick, radius 12 mm, flat | 32 of 32 | 0 of 32 | 0 of 32 |
| coin 2 mm thick, flat (tilted 12 degrees) | 32 of 32 (32 of 32) | 0 of 32 (16 INVALID, no TUNNEL) | 0 of 32 |
| capsule, radius 1.5 mm, flat (upright) | 32 of 32 (32 of 32) | 32 of 32 | 19 of 32 |

- At the default 1.25 ms step (5.7 mm per step at 4.5 m/s) all four thin parts passed in 32 of 32 worlds, with
  FAST_CONTACT seen in every world; at the pinned steps FAST_CONTACT never showed and the build did not warn.
- The sample scene (a 4 mm coin tilted 12 degrees, a radius 3 mm capsule lying across and a radius 5 mm ball on the
  same pane) held from 4.4-4.85 m/s at 0.3125 ms (1.4-1.5 mm per step): no TUNNEL and no INVALID world in 2,048
  (a development sample, not shipped).
- A smaller pinned step did not do worse when re-run: the sample scene (run.py, 256 worlds, 12.2-12.4 m/s) read
  TUNNEL in 256 of 256 worlds at 0.3125 ms (3.8 mm per step) and was OK in 256 of 256 at 0.156 and 0.078 ms on one L4
  (main 3beb7251c; the 0.078 ms row also on rc6-1110), and in 8 of 8 at 0.078 ms on the CPU (MuJoCo C and the Warp
  path; development notes, not shipped). So on this scene 1.9 mm per step, about half the 4 mm coin, held.

## Why
MuJoCo's contacts are soft and are found from where the bodies are at each step, with no test of the path between
steps. A contact pushes a body out along the shortest way; once the body's centre is past the middle of a thin
part, that way is the far side. The 5 mm contact gap only lists the pair early: no force acts until the surfaces
overlap (measured: gaps of 5 to 40 mm change nothing).

## What Alloy does about it
- Build warning: when a body's start drop makes it travel more than 5 mm per rigid step onto a static part thinner
  than twice that travel, `alloy.build` warns once, naming the body, the part's thickness, the speed, the travel per
  step and three fixes with numbers (a rigid step, a lower drop, a thicker part).
  `alloy.estimate(...)["rigid_step"]["warnings"]` has the same text. Both bounds are below every passing case in the
  first table (the lowest pass is 7.5 mm per step; every pass has twice its travel above the plate's thickness), but
  not below the thin-part cases: parts 2-6 mm thick passed at 1.4-3.8 mm per step with no warning and no
  `FAST_CONTACT`, and only `TUNNEL` caught them. It reads start drops only: a body thrown, pushed or accelerated
  during the run gets no build warning, and neither does a drop onto the ground or a thick part.
- Health: `FAST_CONTACT` (DEGRADED while it happens) when a body moves more than 5 mm toward a static shape in one
  rigid step, the shape is thinner along the approach than twice that travel, and the body ends within 5 mm of
  it; `TUNNEL` (INVALID: the world is frozen, `sim.check()` names the body and the part, `profile="precision"`
  raises at the check) when a point inside a body enters a static shape and leaves through the opposite side (or, from
  2 mm deep, through any side but the one it entered by), or the whole body ends inside one (a ball about 2 cm or more under the ground plane). `docs/guide/health.md`. On one
  L4 the check cost about 4 % of a 1,024-world run (3 runs each way, inside their spread).
- `FAST_CONTACT` shows for 64 rigid steps after the approach (80 ms at 1.25 ms) and then clears: read
  `sim.health()` during the run, since a read at the end shows no trace of it. Under `profile="fast"` (checks
  every 10 exchanges) a scene with more than 6 rigid steps per exchange can miss it. `TUNNEL` stays.

## Not covered
- The flags follow points inside each moving body's shapes (the centre and, for a capsule, cylinder or long box,
  two points on its long axis) against static boxes, planes, spheres, capsules, cylinders and static convex meshes
  that fill their bounding box (a plate or block given as a mesh). A robot link that swings through a static
  plate reads `TUNNEL`. A part of a body that passes between those points (a corner of a wide box clipping a thin
  static edge) is not seen.
- The points sit inside the moving body (a box's axis points half its thinnest edge from its faces), so a thin
  moving part (a 0.8 mm card, a 2 mm plate) that leaves a static part by another side is caught only once its
  surface is about half its thickness plus 2 mm inside: past PEN_WARN (1 mm) anyway.
- A static box counts as passed only through its thickness: a point that slides along one of its faces while a little
  inside it (a thin latch tab under a 1.4 mm port roof, in by the front face and out by the back) is not a pass, and
  neither is a point that moves inside a thin wall along its plane (the penetration flags see how deep it is).
- Not judged, by name; each of these passes with no flag today:
  - a ball through a thin free (not static) plate resting on supports: two moving bodies;
  - two balls closing at 20-30 m/s that swap sides: two moving bodies;
  - a kinematic paddle driven at 16-20 m/s through a resting ball: a kinematic object;
  - a static mesh that is not a filled box (a concave mesh, a decomposed part, a thin shell).
- A fast body that lands near a part's edge can be pushed out sideways by the contact (45 mm past a plate's edge in
  one of 16 worlds of a 15 m/s drop) and fall on: it went round the part, not through it, so no `TUNNEL`; the build
  warns for such a start drop.
- A cable dropped across the edge of a plate can end lying partly under it without passing through it (it is
  dragged round the edge by its hanging half): that reads no `TUNNEL`, rightly.
- The flags watch simulations made by `alloy.build` / `alloy.load`. The bench harness's direct stepping
  (`alloy.bully` DirectSim on the MuJoCo C route) is not watched.
- Fast bodies still pass through; the run is flagged, not corrected.

## Workaround
No travel per step is safe for every part: the bound depends on the thinner of the two parts (moving or static), its
shape and how it meets the other (a 10 mm ball held at 3.8 mm per step on a 5 mm pane; a capsule of radius 1.5 mm
passed in 19 of 32 worlds at 1.4 mm). To lower the risk, step rigid bodies finer with `scene.rigid_step` (a rigid-only
scene; at 0.3125 ms the 4-12 m ball drops above all land) or `alloy.Scene(dt=0.0003)` in a scene with robots, slow the
body, or make the parts thicker. Then measure your own part at your speed: run it in many worlds and read
`sim.check()`, where `TUNNEL` (INVALID) names each world a body passed in. The build warning and `FAST_CONTACT` fire
only above 5 mm per step, so they do not cover thin parts.

## Fix
Not small: making contacts act before the surfaces meet (a contact margin of about the travel per step) stops the
pass-through in a probe (a 10 / 20 / 40 mm margin moved the passing travel to 18.8 / 26-27.5 / 39-41 mm) but
bodies then rest that margin above their support. The planned fix is the rigid step following the speed (sub-steps
while a body's travel per step is above the cap), with its own laws. The alternative on the contact side, also
not small: a speculative normal row on listed pairs that are still apart, which stops only the approach that
would overlap within the step and is idle at rest.
