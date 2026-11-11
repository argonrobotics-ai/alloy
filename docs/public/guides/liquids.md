# Liquids

Fill a container with water using `scene.add_liquid`, run it, and check how much liquid stays inside.

## 1. Put water in a cup

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("cup", inner=(0.04, 0.04, 0.05), wall=0.01, pos=(0, 0, 0))     # pos: centre of the bottom
scene.add_liquid("water", size=(0.032, 0.032, 0.032), pos=(0, 0, 0.03))
sim = alloy.build(scene)
print(sim.routes()["water"]["route"])
```

```text
liquid_sph
```

The cup is static, 4 cm wide and 5 cm deep inside. The water is a 3.2 cm cube that starts at rest in it.

## 2. Run it and measure

```python
sim.run(0.5)
pos = sim.state(["water"], world=0)["water.points"]
inside = (abs(pos[:, 0]) < 0.02) & (abs(pos[:, 1]) < 0.02) & (pos[:, 2] < 0.05)
print(len(pos), round(sim.liquid_volume("water") * 1e6), inside.mean().round(2))
```

```text
512 33 1.0
```

The first number is the particle count and the second the simulated volume in millilitres. The third is the share of
particles inside the cup walls: 1.0 means none spilled.

## 3. Check the run

```python
print(sim.check()["ok"])
```

```text
True
```

To fill a container from above, pour instead of dropping a block:
`scene.add_pour("stream", 2e-5, pos=(0, 0, 0.12))` streams 20 ml straight down from that point.

## Limits

- Containers narrower than 24 mm are unreliable ([details](../../known_gaps/liquid-narrow-containers.md)).
- A block of liquid dropped from above splashes over the rim; a poured stream much less, with `scene.add_pour`
  ([details](../../known_gaps/liquid-pour-over-the-rim.md)).
- Liquid touching cloth is very slow, and cloth takes up no water ([details](../../known_gaps/liquid-on-cloth.md)).
- Heated liquid does not convect ([details](../../known_gaps/heat-no-buoyancy.md)).
- Liquid is slow on the CPU, so keep the particle count small while you try things out.

## Next

- Drape a sheet instead: [Cloth](cloth.md).
- Put a robot beside the cup: [Robots](robots.md).
- Read the health checks: [World health](health.md).
