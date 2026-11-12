# Alloy vs MuJoCo Playground: which should I use?

Use the Alloy physics engine when the task has more than rigid bodies: pouring a liquid or moving sand, or starting
from a ready-made cloth-folding task (`ClothFold-v0`), in many parallel worlds with a camera in each. It steps
robots in MuJoCo Warp, one of Playground's backends, and installs with `pip install alloy-engine`.

## What Alloy adds

- Tasks with liquid and cloth. `Pour-v0` and `ClothFold-v0` are two of the nine ready-made environments, and
  `alloy.envs.make` builds each in as many worlds as you ask for. Playground's engine models fluid as drag on
  bodies; we found no liquid, sand or paste solver in it.
- The same robot solver. Alloy steps rigid bodies and robots in MuJoCo Warp, and an MJCF robot loads directly.
- Health in every world. `sim.health()` grades each world `OK`, `DEGRADED` or `INVALID` and names a fix for every
  flag, so a physics failure is not counted as a policy failure. Playground has overflow and bad-acceleration
  warnings.
- Cutting of soft objects, heat in liquids, and a different step size for each solver in one scene.
- It is made for coding agents: the docs ship inside the package (`python -m alloy.docs`), every error carries its
  fix, and `alloy mcp` serves the engine as MCP tools ([agents](../agents.md)). It also records demonstrations
  ([Imitation learning](../guides/imitation.md)).

## A pour in two worlds

```python
import alloy

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
scene.add_open_box("cup", inner=(0.05, 0.05, 0.05), wall=0.01, pos=(0, 0, 0))
scene.add_pour("stream", 2e-5, pos=(0, 0, 0.1))                 # 20 ml, straight down from 10 cm
sim = alloy.build(scene, device="cpu")
sim.run(1.0)
points = sim.state(["stream"])["stream.points"]
in_cup = (abs(points[..., 0]) < 0.025) & (abs(points[..., 1]) < 0.025) & (points[..., 2] < 0.06)
print(len(points), in_cup.mean(axis=1).round(2))
print(sim.check()["ok"], [w["severity"] for w in sim.health()["worlds"]])
```

```text
2 [0.99 0.99]
True ['OK', 'OK']
```

Two worlds each poured 20 ml of water into a cup, stepped together; the first line is the world count and the share
of the water inside the cup in each world, and each world got its own health grade. `Pour-v0` wraps a pour like
this as a ready-made environment (`pip install "alloy-engine[envs]"`). It ran on a CPU with no GPU.

## Table: Alloy and MuJoCo Playground

One rule for every engine: "yes" means the feature is in the engine's public release and documented; "not
documented" means our survey has no public source for it. A note in brackets says where a "yes" holds. The other
engines' cells were read from their public code and docs on 2026-11-09, not run by us. The scorecard file,
`bench/scorecard/row6/capabilities.jsonl` in Alloy's source repository, has 26 rows and also grades how complete
each feature is; this table shows the 22 rows Alloy's public package documents.

| Capability | Alloy | MuJoCo Playground |
|---|---|---|
| Liquids | yes | no |
| Particle fluids | yes | no |
| Sand and grains | yes | no |
| MPM (pastes, snow-like materials) | yes | no |
| Cutting | yes | no |
| Heat | yes | not documented |
| Soft bodies with liquid | yes | no |
| Different step sizes per solver | yes | no |
| Same result on a re-run | yes (CPU) | not documented |
| Rigid bodies with liquid | yes | yes |
| Cloth | yes | yes |
| Soft bodies (volumetric FEM) | yes | yes |
| Rigid bodies with cloth | yes | yes |
| Rigid bodies with soft bodies | yes | yes |
| Tactile sensing | yes | yes |
| Thin shells (paper, film, sheet metal) | yes | yes |
| Plasticity (dents, permanent bends) | yes | yes |
| Cables and rods | yes | yes |
| Per-world health checks | yes | yes |
| Many worlds on one GPU | yes | yes |
| Batched camera rendering | yes | yes |
| Double precision | yes (CPU) | yes (jax_enable_x64) |

## When MuJoCo Playground is the better fit

- You want a large set of ready-made locomotion, manipulation and dexterous-hand environments with tuned training
  recipes.
- You want to start from published sim-to-real results for those tasks.
- Your task is rigid-only and you want the fastest training on MJX or JAX.

## Keep in mind

Alloy's limits are listed on one page, each with its size and a workaround: [Keep in mind](../keep-in-mind.md).

## Next

- [Reinforcement learning](../guides/rl.md), [Robots](../guides/robots.md), [Which simulator?](index.md)
