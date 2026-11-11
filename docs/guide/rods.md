# Rods: cables, wires and cords

`scene.add_rod` adds an elastic rod: a round section along a curve that bends, twists and stretches as its material
says, rolls on surfaces, and can be gripped and twisted by a robot. It is a chain of rigid segments joined by elastic
(Cosserat) joints; Alloy picks the segment count. Add it to the same `Scene` as rigid parts and robots.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_rod("cable", length=0.4, radius=0.003, pos=(0, 0, 0.6), direction=(0, 0, -1), segments=20,
              clamp="start")                                 # hangs straight down from a clamp at z = 0.6
sim = alloy.build(scene)
print(sim.routes()["cable"]["route"])                        # rod_vbd: the elastic-rod solver
sim.apply_wrench("cable", torque=(0.0, 0.0, 1e-4), at="end")   # world z, held until changed
sim.run(2.0)
st = sim.state(["cable"], world=0)
print(st["cable.points"][-1])                                # free end, about (0, 0, 0.2)
print(st["cable.twist"].sum())                               # -0.106 rad = -T L / (G J): the cable points down
print(sim.check()["ok"])
```

## Adding a rod

`scene.add_rod(name, points=None, *, radius, length=None, pos=(0, 0, 0), direction=(1, 0, 0),
material="cable_usb_composite", segments=None, clamp=None, mass=None, friction=None, route=None)`
- the shape: `points`, a polyline relative to `pos` (at least two points), or a straight `length` along `direction`
  from `pos`. `radius` is the round section's radius (m).
- `material`: its E and nu set the stiffness: bending E I and twisting G J, with I = pi r⁴ / 4, J = pi r⁴ / 2 and
  G = E / (2 (1 + nu)). Cable cards: `cable_usb_composite` (default), `cable_harness_composite`,
  `pvc_cable_jacket`; any card from `alloy.materials()` works.
- `segments` (at least 3): rigid segments along the rod. By default there are enough that the rod sags under its
  own weight within 5 % of a continuous rod. More segments resolve tighter bends and cost more.
- `clamp="start" | "end" | "both"` holds that end segment where it is built, in every world.
- `route`: None (the router chooses, Routes below), `"rod_vbd"`, or `"chain"`. With `clamp="start"`, `route="chain"`
  keeps a clamped rod on the chain, its first segment welded where it is built: beside a robot about 60x faster on the
  CPU, measured on one machine (Robots, below). The chain holds its first end only: to clamp the other end, pass the
  points in reverse.
- `mass` (kg, total) or the material's density; `friction` overrides the material's.

## Routes

`sim.routes()[name]` (and `alloy.estimate(scene)["routes"][name]`) says which solver steps the rod and why:
- `chain`, the default for a short elastic rod: a chain of capsules in the rigid solver (MuJoCo), with bend and twist
  spring joints, when 40 segments or fewer are enough. It steps with a robot in one solver, with no exchange.
- `rod_vbd`, the elastic-rod solver, wherever the chain cannot do the job:
  - a rod that needs more than 40 segments (long or soft);
  - a clamped rod, unless it asks for `route="chain"` with `clamp="start"`;
  - a wire whose material yields (`plastic.model: j2`, as `copper_c110`): it keeps a bend (Wires that keep a
    bend, below).
  It bends, twists and stretches with any number of segments.
- The same rule holds on every device, so a scene routes the same way on the CPU and the GPU.

What each costs, measured through `alloy.build` with a 0.4 m USB cable of 20 segments:

| scene, device | `chain` | `rod_vbd` |
|---|---|---|
| on the ground, CPU, one core: wall s per simulated s | 0.9 | 2.5 |
| beside a robot (two 0.2 kg pads), CPU: wall s per simulated s | 1.0 | 83-95, by scene (Robots, below), on an AMD EPYC 7B13; 139-156 on a 2.8 GHz Xeon cloud VM |
| on the ground, L4 GPU: simulated s per wall s, over 1 / 64 / 256 / 1,024 / 4,096 worlds | 1.5 / 69 / 150 / 178 / 180 | 0.4 / 19 / 46 / 148 / 303 |
| beside a robot (two 0.2 kg pads), L4 GPU: simulated s per wall s, over 1 / 64 / 256 / 1,024 / 4,096 worlds | 1.2 / 35 / 105 / 168 / 172 | 0.06 / 2.6 / 6.2 / 17.6 / not measured |

- On a GPU the chain stops gaining from about 1,024 worlds. On the ground, `rod_vbd` is 1.7x faster than the chain
  at 4,096 worlds and 1.2x slower at 1,024. The ground rows at 1-256 worlds were measured on an earlier version.
- Beside a robot `rod_vbd` exchanges forces with it on a fine clock, and on a GPU it runs 10-20x slower than the
  chain in this scene, 13-21x with the cable clamped (Robots, below;
  [rod-vbd-beside-robot-cost.md](../known_gaps/rod-vbd-beside-robot-cost.md)). The robot rows are one L4 run.

## State

- `"<name>.points"`: the centre line from start to end, `(worlds, segments + 1, 3)`, on both routes.
- In a `VectorEnv` (`envs.md`) the task state has no `"<name>.points"` and no `"<name>.twist"`: a check or an
  observation that reads them raises `EnvError: no state key`. Workaround, for a rod on the chain route: read its link
  bodies, `"<name>/0.pos"` to `"<name>/<segments - 1>.pos"` (each link's centre: a 12-segment rod has 12 links and 13
  centre-line points). Not checked for a rod on `rod_vbd`.
- `"<name>.twist"`: the twist of each joint between neighbouring segments, `(worlds, segments - 1)`, in radians,
  right-handed about the rod's own direction (start to end), on both routes. Its sum is the end-to-end twist.
- `sim.apply_wrench(name, force=(fx, fy, fz), torque=(tx, ty, tz), at="start" | "end")` holds a force (N) and a
  torque (N m), in world coordinates, on the end segment until changed; zeros remove it, `sim.reset()` removes
  every wrench, and a `(worlds, 3)` array sets one per world. It works on any body, not only rods.

## Accuracy (measured, CPU)

| check | result |
|---|---|
| end torque held: end twist / (T L / (G J)) | 1.0000 (error under 1e-5) |
| clamped rod sagging under its weight: tip drop vs the exact discrete chain | within 0.08 % |
| chain hanging between two points: sag vs the catenary | within 0.16 % |
| rope over a fixed drum: the hold / slip load ratio vs the capstan law (friction 0.2, 0.4) | 1.00, 1.00 |
| rolling down a 10° / 20° slope: acceleration vs the rolling body's | 0.996 / 0.991 |
| a copper wire bent past yield (1.5 M_y) and let go: curvature held, and kept, vs the closed form | +0.66 %; within 0.7 % of the bend |

## Wires that keep a bend

A rod whose material yields (`plastic.model: j2` on its card: `copper_c110` and the library's other metal cards)
bends and twists elastically up to its yield moment, then keeps a set. Each joint's fibres (a section of 220 fibres in tension
and shear, von Mises with linear hardening) yield on their own, so a wire wound round a peg keeps the shape of the
peg where it yielded and springs back where it did not. `"<name>.plastic_strain"` (worlds, segments - 1) gives each
joint's largest equivalent plastic strain.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_rod("wire", length=0.2, radius=0.001, pos=(0, 0, 0.5), direction=(0, 0, -1), material="copper_c110",
              segments=20, clamp="start")                    # hangs straight down from a clamp at z = 0.5
sim = alloy.build(scene)
print(sim.routes()["wire"]["route"])                         # rod_vbd: the only route that keeps a set
moment = 0.08                                                # N m: 1.5x this wire's yield moment of 0.054
for k in range(200):                                         # ramp it in over 2 s, about the world x axis
    sim.apply_wrench("wire", torque=(moment * (k + 1) / 200, 0.0, 0.0), at="end")
    sim.step(1)
for k in range(100):                                         # and out again over 1 s
    sim.apply_wrench("wire", torque=(moment * (1.0 - (k + 1) / 100), 0.0, 0.0), at="end")
    sim.step(1)
sim.step(50)
print(sim.state(["wire"], world=0)["wire.plastic_strain"].max())    # about 5e-4: it yielded
print(sim.state(["wire"], world=0)["wire.points"][-1])       # y about 5 mm off the vertical: a kept bend
```

- Measured on the CPU: a clamped copper wire (0.2 m, R 1 mm, 20 segments) under a held end moment of 1.5 M_y holds
  the closed-form curvature of a round, hardening section to +0.66 %. After release, the curvature left is within
  0.7 % of the bend of the closed-form set (`tests/api/test_rods_plastic.py`). Twist yields the same way: at 0.94
  of the fully plastic torque the twist is within 0.74 %.
- Load slowly: a load that overshoots also yields. At the card's 2 % damping a 0.2 s ramp left 5.6 % more bend than
  a slow one.
- Close to the fully plastic moment the set settles slowly and less exactly
  ([rod-plastic-near-full-yield.md](../known_gaps/rod-plastic-near-full-yield.md)). A bend reversed past yield has
  no Bauschinger effect ([plasticity-metal.md](../known_gaps/plasticity-metal.md)).

## Robots: gripping and twisting

A robot's fingers grip a rod through friction (there is no attach call), and a wrist that turns twists it. Measured
on the CPU, on `rod_vbd`: a clamped 0.4 m USB cable (r 3 mm, 20 segments), its free end between two 0.2 kg pads on a
wrist that rolls 0.3 rad about the cable's axis over 0.5 s and then holds (`tests/api/test_rods_grip.py`):
- the free span twists uniformly: every joint within 0.1 % of the mean;
- the wrist carries the cable's torque G J θ / L: MuJoCo's torque on the roll joint is 0.98 of it with pads fixed to
  the wrist (0.92-0.99 per sample), and 0.98 with pads on sliding joints when the cable's end is on the wrist's axis.
  A cable end that sags below the axis before the pads close is also bent as the wrist turns, and the wrist then
  carries that too (1.1-1.3 here);
- the gripped end turns with the wrist and stays with it once the wrist stops: within 0.01 % with pads fixed to the
  wrist (0.7 s hold), 0.05 % with sliding pads at 4.2 N over a 2 s hold, and 0.01 % at 21 N (0.7 s). The rod's
  contacts with a robot keep their friction anchor from step to step, so friction holds it. Read the rod's own
  `"<name>.twist"` when the angle matters, not the wrist's.

On the `chain` route (a short rod without a clamp, beside a robot) the cable follows the twisting grip: its end turns
1.000 of the wrist's roll on the CPU (MuJoCo C) and on the GPU (MuJoCo Warp). Held, it stays with the grip on the GPU
(the same to four decimals after 5 s, in every world of 1,024). On the CPU it creeps back about 0.14 % per second.
[rod-chain-grip-slips.md](../known_gaps/rod-chain-grip-slips.md) has the rows, what the GPU's hold does not cover (a
contact that slid and is still loaded past half its friction limit) and its off switch, `ALLOY_STICK_ANCHOR=0`. For a
cable that must hold a twist for long on the CPU, use a clamp (`clamp="start"` or `"end"`), which routes it to
`rod_vbd`.

A clamped cable can stay on the chain instead: `add_rod(..., clamp="start", route="chain")`. The same rig there
(`tests/api/test_rods_chain_clamp.py`) runs 1.3 s per simulated second on the CPU, against 83 s on `rod_vbd` (an AMD
EPYC 7B13):
- the gripped end turns with the wrist: 1.0000 when the ramp ends, and -0.03 % after a 0.7 s hold (MuJoCo C);
- the free span twists uniformly, every joint within 0.11 % of the mean;
- the wrist carries the cable's torque G J θ / L within 0.3 % (0.997) with pads fixed to the wrist. With sliding
  pads it reads 0.89: the pads let the sagging end sit about 4 mm below the roll axis and carry its weight there.
  The cable's own twist torque at the pads is right (1.002), and with gravity off the wrist reads 1.000.

On a GPU (MuJoCo Warp) the clamped chain runs 55 / 225 simulated s per wall s at 64 / 1,024 worlds, against 2.6 / 17.7
for `rod_vbd` (21x and 13x). With sliding pads (a squeezing grip) it follows the wrist to 1.000 and holds: 1.0000
after 5 s. Gripped rigidly (pads fixed to the wrist) it follows 0.997 and holds: 0.9976 after 5 s; the wrist then
reads 1.04 of the cable's torque (MuJoCo Warp on the CPU; 0.90 with sliding pads, as on MuJoCo C). Until 2026-11-10 a
held chain crept back there, 1.7 % in 0.7 s with sliding pads and 10 % gripped rigidly. `route="chain"` now serves a
held grip on a GPU too; keep the default `rod_vbd` for the elastic rod's other features (both ends clamped, more than
40 segments, a yielding wire).

Cost: the rod and the robot exchange forces on a clock set by the lightest robot body that could touch the rod, 0.5
sqrt(m / k) with k the contact stiffness (about 2e6 N/m), as for cloth ([deformables.md](deformables.md), Cost). Two
0.2 kg pads mean 64 exchanges per 10 ms control step. An AMD EPYC 7B13 takes 95 s per simulated second with the cable
free and squeezed, and 83 s with it clamped and twisted (139-156 s on a 2.8 GHz Xeon cloud VM). A solver entry of rods
only runs 4 VBD iterations (an entry shared with cloth or a soft body keeps the cloth's count); at 10 the same rows
took 161 and 137 s. Give pads realistic masses.

## Cost

- Through `alloy.build`: the table under Routes.
- The elastic-rod solver alone on an L4 (no contacts with other objects, no health checks; the shipped route runs
  slower, as Routes shows), timed warm with CUDA graphs, world-steps per second:

  | worlds | 1 | 64 | 256 | 1,024 | 4,096 |
  |---|---|---|---|---|---|
  | clamped rod, 30 segments, no contacts, 1.25 ms step | 2,500 | 113,000 | 274,000 | 909,000 | 1,950,000 |
  | rope over a drum, 34 segments, contacts, 0.5 ms step | 580 | 25,900 | 61,000 | 195,000 | 379,000 |

- On the CPU, one core, `rod_vbd` alone: 2.5 s per simulated second for the cable on the ground (4.5 s at 10
  iterations; the hanging cable in the example above took 6 s at 10).
- The first build on a GPU compiles the rod kernels for about 5 minutes; they are cached afterwards.
- The first build, on a machine, of a cable card that routes to a link chain also runs the router's self-test of that
  cable (a cantilever at its link count) before the first step, and caches the result in
  `$ALLOY_CACHE_DIR/rodcheck.json` (default `~/.cache/alloy`), so later builds there skip it. It runs on the CPU, also
  in a GPU build. For fine cables it is long: a scene with two fine cords (32 and 40 links) waits about 45 min before
  the first step, at any world count (2,714 s, measured once on an 8-vCPU cloud VM with one L4).
  `python -m alloy.core.warm` does not fill it for cables of your own: it warms the shipped library's cables only.

## Limits

- One start for every world: `pos` takes no `(worlds, 3)` array, and `scene.randomize` refuses a rod's `x`, `y`
  and `z`.
- `scene.save()` writes cards without the clamp (the build warns); Scene JSON keeps it.
- A clamp holds its end where it was built; it does not move. To move an end, grip it with a robot.
- Only `rod_vbd` rods yield; a rod on the chain route is always elastic (a yielding material routes to `rod_vbd`).
- A rod collides with itself (all but neighbouring segments), but only if the segments are shorter than its
  tightest fold. A USB cable (r 3 mm) pushed into a hairpin stopped with its legs touching (6.7-7.7 mm apart, 6 mm =
  touching) at 80 segments (4.3 mm each). At 40 segments (8.7 mm, longer than the fold) the fold passed through
  itself and the world failed its health check. Where a rod folds or coils, keep segments under about its diameter
  (`segments=`), and read `sim.check()`. Coils and knots are not verified yet.
- A chain rod and an elastic rod in one scene step together, but where they touch the chain rod sinks into the
  elastic one by up to its radius, and on a wire of 1.5 mm radius the world fails its health check (PEN_FAIL at
  0.29 s). Put rods that touch on one route (`docs/known_gaps/rod-chain-on-elastic-rod-sinks.md`).
- Other materials. A chain rod steps with the rigid bodies, beside anything they step beside. An elastic rod
  (`rod_vbd`) builds and steps beside rigid bodies and robots, a chain rod, cloth, a soft body and a liquid; of
  these only rigid bodies and robots are verified against a law. On a soft body, a rod thinner than the gaps
  between the soft body's vertices falls through it with nothing flagged: a copper wire of 1.5 mm radius went
  through a silicone pad of 12.5 mm cells (`docs/known_gaps/rod-on-soft-body-falls-through.md`). Beside sand,
  dough or another MPM material, and dropped on a paper or metal sheet of the film route, the build raises
  `BuildError` naming the pair and the way out (rigid grains, or the rod on the chain route).
- Twist dynamics and the rotation of light bodies: [rod-twist-dynamics.md](../known_gaps/rod-twist-dynamics.md),
  [vbd-rigid-first-order-rotation.md](../known_gaps/vbd-rigid-first-order-rotation.md).
