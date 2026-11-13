# How do I simulate a cable or wire in Python?

Use `scene.add_rod` in the Alloy physics engine, installed with `pip install alloy-engine`. A rod is a chain of
segments that bends, twists and stretches like the material you name, and you read its centre line back as numpy
points. This page clamps one end of a cable and lets the rest sag under its own weight.

## 1. Clamp a cable and let it sag

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_rod("cable", length=0.4, radius=0.003, pos=(0, 0, 0.5), direction=(1, 0, 0), segments=16,
              clamp="start", route="chain")                    # held horizontal at z = 0.5, then released to sag
sim = alloy.build(scene)
print(sim.routes()["cable"]["route"])
```

```text
chain
```

`clamp="start"` welds the first segment where it is built. `route="chain"` keeps a clamped rod on the fast chain of
rigid segments instead of the slower elastic-rod solver.

## 2. Run it and read the centre line

```python
sim.run(2.0)
points = sim.state(["cable"], world=0)["cable.points"]
print(points.shape, points[-1][[0, 2]].round(2))               # the free end's x and z
print(sim.check()["ok"])
```

```text
(17, 3) [0.35 0.31]
True
```

`"cable.points"` has one row per segment end, 17 for 16 segments. The free end hangs at x = 0.35 and z = 0.31 m, below
the clamp at 0.5. A rod that yields, such as a `copper_c110` wire, keeps a bend; it steps on the slower solver
([Rods guide](../guides/cables.md)).

One measured number: on an AMD EPYC 7B13 CPU, one core, a 0.4 m cable on the ground costs 0.9 s of wall time per
simulated second on the chain route and 2.5 s on the elastic-rod solver.

## Keep in mind

- A clamped cable steps on the slower solver unless you pass `route="chain"`. Beside a robot on the CPU that solver
  costs 83-95 s per simulated second on an AMD EPYC 7B13 ([details](../../known_gaps/rod-vbd-beside-robot-cost.md)).
- A chain held by a robot on the CPU slowly loses its twist, about 0.14 % per second
  ([details](../../known_gaps/rod-chain-grip-slips.md)). Clamp a cable that must hold a twist for long.
- A cable that needs more than 40 segments (long or soft) also uses the slower solver.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Grip a cable with a robot: [Robots](../guides/robots.md).
- The short cable guide: [Cables](../guides/cables.md).
- Read the health checks: [World health](../guides/health.md).
- Related: [deformable object manipulation](../answers/deformable-object-manipulation.md),
  [squeeze a soft object](squeeze-soft-object.md).
