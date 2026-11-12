# Free-body stacks step 2.4x slower by default than stock Newton MuJoCo; the opt-in fast mode softens contacts

Status: open. Since 2026-11-09 (RIGID-DEFAULT) the default steps free-body clutter on implicitfast with a capped
friction pull, at a step that keeps contacts as stiff as at the fixed step. The opt-in rigid fast mode
(`profile="fast"`, or `scene.rigid_step = "fast"`) also softens contacts (a resting body sinks up to 1 % of its
thinnest member); in one run it stepped these stacks at about the stock solver's speed. It states what it gives up.

## What

A scene of resting free rigid bodies (box stacks, piles) steps through the public API by default slower than the same
bodies in a stock Newton `SolverMuJoCo` with its default settings and one 10 ms step per period.
- **The default on MuJoCo Warp** (a GPU, or the CPU with `ALLOY_CPU_ROUTE=warp`): an entry of more than 8 free bodies
  steps on implicitfast with Alloy's own contact gains, at the largest step that keeps contacts as stiff as at the
  fixed 1.25 ms step: 5 ms for the 27-cube stacks, 2 steps per 10 ms period. The friction rows' pull is capped at 20
  m/s², so a pushed or sliding box no longer hops and rocks ([rigid-sliding-box-hops.md](rigid-sliding-box-hops.md)).
  Stock solves once per period, with softer contacts.
- **Still on rk4:** an entry of 8 or fewer free bodies (free flight stays accurate), and every entry on the CPU's
  MuJoCo C route (the default for `device="cpu"`), which has no friction cap
  ([rigid-clutter-rotation.md](rigid-clutter-rotation.md) names each route's integrator).
- **The fast mode** steps such scenes at the largest step whose predicted rest sag stays within 1 % of each body's
  thinnest member (10 ms here), with the same cap (on MuJoCo C: on rk4, no cap), and keeps 4 contact points per box
  face instead of up to 8.

## Number

One L4, 3,072 worlds of 27 resting 1 m cubes (10 kg/m³, a 3 x 3 x 3 stack, the benchmark's jitter), 3.29 s simulated,
main 04f81f0ff of 2026-11-09, back to back; real time summed over the worlds (simulated world-seconds per
wall-second):

| | default (`profile="standard"`) | `profile="fast"` | stock Newton `SolverMuJoCo` |
|---|---|---|---|
| integrator, step | implicitfast + cap, 5 ms (2 per period) | implicitfast + cap, 10 ms | 10 ms, 1 per period |
| 3,072 worlds | 766x (774x / 759x; 0.42x stock) | 2,093x (one run; about stock's speed) | 1,816x (1,835x / 1,796x) |
| 1 world | 8.9x | 17.7x (another L4, 2026-11-09) | 15.5x |
| settle error, median / worst | 0.19 / 2.23 mm | 0.86 / 0.87 mm | 1.00 / 3.26 mm |

All 3,072 stacks stand in every row. Before RIGID-DEFAULT the default stepped these cubes on rk4 at 2.5 ms: 92-96x.
Without the cap the default runs 756x, worst settle 1.65 mm: the cap costs nothing in speed.

## Why

- **The default step:** at the 10 ms period these cubes' contacts would be 0.25x as stiff as at the fixed step (rest
  sag 1.17 mm against 0.295 mm under three cubes' load, the build's own check), so the default takes 5 ms: two contact
  solves per period against stock's one, each about 1.2x stock's.
- **rk4 where it stays:** with 8 or fewer free bodies, implicitfast's first-order spin is a visible error in free
  flight; on the MuJoCo C route the friction cap is missing, and under implicitfast a sliding box rocks more and more
  there: a 30 cm cube sliding at 5 m/s toppled within 2 s. So every entry on that route stays on rk4, the fast
  mode's too (that cube in fast mode: 6 deg and an 8 cm hop on MuJoCo C; under 1 deg and 1 mm on MuJoCo Warp; the
  rows are in development notes, under the cube pushed and sliding at 5 m/s per CPU route; not shipped).

## Workaround

- `alloy.build(scene, profile="fast")` on a scene of free rigid bodies (or `scene.rigid_step = "fast"` with any
  profile that allows it). It costs: contacts softer than at the fixed step (a resting body sinks up to 1 % of its
  thinnest member); on the CPU's MuJoCo C route it keeps rk4, and a tall body sliding at m/s rocks and hops (a 30 cm
  cube at 5 m/s: 6 deg, 8 cm; under 1 deg and 1 mm on MuJoCo Warp); under the fast profile a world latched INVALID
  freezes at its last saved checked state, up to 9 steps before the failure at one step per period. The build warns
  once, and `sim.routes()` and
  `alloy.estimate(...)["rigid_step"]["fast"]` say so.
- On the CPU, `ALLOY_CPU_ROUTE=warp` takes MuJoCo Warp's route (implicitfast + cap for clutter).

## Planned fix

After launch: the friction cap on the MuJoCo C route, so a CPU build takes the same route as a GPU; then
implicitfast with the cap for entries of 8 or fewer free bodies, behind the free-flight laws.
