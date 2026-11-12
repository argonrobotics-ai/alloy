# Alloy vs Genesis: which should I use?

Use the Alloy physics engine when you want robots stepped by MuJoCo's solver beside liquids, cloth and sand, with a
health grade for every world and docs written for coding agents. Both engines mix materials in one scene; Alloy also
documents cutting. Install it with `pip install alloy-engine`.

## What Alloy adds

- Robots on MuJoCo. Alloy steps rigid bodies and robots in MuJoCo Warp on a GPU and MuJoCo C on the CPU, so an MJCF
  robot runs in the solver it was written for, and loads directly.
- A health grade for every world. `sim.health()` marks each world `OK`, `DEGRADED` or `INVALID` and names a fix for
  every flag: non-finite values, penetration, speed, energy jumps, contact overflow. Genesis collects NaN error
  flags per environment.
- Cables as elastic rods that bend, twist and stretch as their material says ([Cables](../guides/cables.md)).
  Genesis builds a cable as a chain of jointed rigid links.
- Cutting of soft objects with a knife, in the docs the package installs. We found no public source for cutting in
  Genesis.
- It is made for coding agents: the docs ship inside the package (`python -m alloy.docs`), every error carries its
  fix, and `alloy mcp` serves the engine as MCP tools ([agents](../agents.md)). It ships ready to use: nine
  ready-made environments, nine robots (arms, a hand, a quadruped, two humanoids), 90 materials and a library of
  5,423 objects ([What ships](../what-ships.md)).

## Water in two worlds, each with a health grade

```python
import alloy

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
scene.add_open_box("cup", inner=(0.04, 0.04, 0.05), wall=0.005, pos=(0, 0, 0))
scene.add_liquid("water", size=(0.032, 0.032, 0.03), pos=(0, 0, 0.02), material="water")
sim = alloy.build(scene, device="cpu")
print(sim.routes()["water"]["route"])
sim.run(0.2)
print(sim.state(["water"])["water.points"].shape, round(sim.liquid_volume("water") * 1e6, 1))
print(sim.check()["ok"], [w["severity"] for w in sim.health()["worlds"]])
```

```text
liquid_sph
(2, 448, 3) 28.7
True ['OK', 'OK']
```

The water went to Alloy's own particle liquid solver (DFSPH): 448 particles and 28.7 ml in each of two worlds,
stepped together, and each world got its own health grade. It ran on a CPU with no GPU.

## Table: Alloy and Genesis

One rule for every engine: "yes" means the feature is in the engine's public release and documented; "not
documented" means our survey has no public source for it. A note in brackets says where a "yes" holds. The other
engines' cells were read from their public code and docs on 2026-11-09, not run by us. The scorecard file,
`bench/scorecard/row6/capabilities.jsonl` in Alloy's source repository, has 26 rows and also grades how complete
each feature is; this table shows the 22 rows Alloy's public package documents.

| Capability | Alloy | Genesis |
|---|---|---|
| Cutting | yes | not documented |
| Heat | yes | not documented |
| Liquids | yes | yes |
| Particle fluids | yes | yes |
| Sand and grains | yes | yes |
| MPM (pastes, snow-like materials) | yes | yes |
| Rigid bodies with liquid | yes | yes |
| Soft bodies with liquid | yes | yes |
| Different step sizes per solver | yes | yes |
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
| Same result on a re-run | yes (CPU) | yes (opt-in) |
| Double precision | yes (CPU) | yes |

## When Genesis is the better fit

- You already build on Genesis's API, renderer and tooling, and its solvers cover your scene.
- You want double precision on a GPU: Genesis has a 64-bit option.

## Keep in mind

Alloy's limits are listed on one page, each with its size and a workaround: [Keep in mind](../keep-in-mind.md).

## Next

- [Liquids](../guides/liquids.md), [World health](../guides/health.md), [Which simulator?](index.md)
