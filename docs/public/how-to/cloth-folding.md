# How do I simulate cloth folding for robot learning in Python?

The Alloy physics engine, installed with `pip install alloy-engine`, simulates cloth that folds and rests as two
layers (`self_contact=True`, the default), in many parallel worlds beside a robot. It ships `ClothFold-v0`, a
ready-made environment in which a robot folds a towel. This page bends a sheet over a block on the CPU.

## 1. Fold a sheet over a block

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("bar", size=(0.12, 0.5, 0.1), pos=(0, 0, 0.15), static=True)           # top at z = 0.2
scene.add_cloth("sheet", size=(0.5, 0.5), pos=(0, 0, 0.3), resolution=10, mass=0.1)
sim = alloy.build(scene)
print(sim.routes()["sheet"]["route"])
```

```text
cloth_vbd
```

The sheet is flat, 10 cm above a low block, with 11 x 11 vertices. It runs on the cloth solver.

## 2. Run it and measure the two layers

```python
sim.run(1.0)
points = sim.state(["sheet"], world=0)["sheet.points"]
left, right = points[points[:, 0] < 0], points[points[:, 0] > 0]
print(points.shape, (0.2 - left[:, 2].min()).round(1), (0.2 - right[:, 2].min()).round(1))
print(sim.check()["ok"])
```

```text
(121, 3) 0.2 0.2
True
```

The sheet bends over the block and one half hangs down each side. The two numbers are how far the lowest vertex of
each half hangs below the block top, in metres; both reach the ground.

To fold with a robot, call `alloy.envs.make("ClothFold-v0", num_envs=64)` after `pip install "alloy-engine[envs]"`.
It folds a towel along its diagonal. Its default `grasp="friction"` holds the towel by friction alone;
`grasp="attach"` is a shortcut that holds the towel to the fingers, not a friction grasp.

One measured number: on one L4 GPU, 64 worlds of a 12 s cloth fold beside a robot took 47 min (self contact off,
at 10 iterations as in "fast").

## Keep in mind

- A friction grasp that carries a cotton sheet is not verified
  ([details](../../known_gaps/cloth-robot-grasp-carry.md)).
- Cloth bends about 35 % too soft along a mesh axis
  ([details](../../known_gaps/cloth-bending-free-strip-soft.md)).
- Sheets that fall onto each other can pass through
  ([details](../../known_gaps/cloth-falling-sheets-pass-through.md)).
- `ClothFold-v0` is experimental and slow, and needs a GPU. Cloth beside a robot is slow on a CPU: time a short
  run first.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- The short cloth guide: [Cloth](../guides/cloth.md).
- Reinforcement learning environments: [Reinforcement learning](../guides/rl.md).
- Robots beside cloth: [Robots](../guides/robots.md).
