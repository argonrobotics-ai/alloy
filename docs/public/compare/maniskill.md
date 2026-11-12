# Alloy vs ManiSkill: which should I use?

Use the Alloy physics engine when your manipulation task needs physics that ManiSkill3 does not model: cloth,
liquids and soft bodies, stepped in the same scene as the robot, in many parallel worlds behind a Gym-style API,
with a health grade for every world. Install it with `pip install alloy-engine`.

## What Alloy adds

- Deformable and flowing materials. Cloth, soft bodies, liquids, sand and pastes step in the same scene as the
  robot, each in a solver suited to it. ManiSkill3 is rigid-body; its soft-body tasks are in the older ManiSkill2.
- A Gym-style API over all of it. `alloy.envs.make("Pour-v0", num_envs=256)` has the shape of Gymnasium's vector
  API, with wrappers for rsl_rl, Stable-Baselines3 and skrl ([Reinforcement learning](../guides/rl.md)).
- Tactile images from gel fingertip sensors in every world (the package's tactile examples). ManiSkill3 reads
  contact impulses on a hand's touch pads.
- Health in every world. `sim.health()` grades each world `OK`, `DEGRADED` or `INVALID` and names a fix for every
  flag. We found no NaN or blow-up detection in ManiSkill3's physics core.
- It is made for coding agents: the docs ship inside the package (`python -m alloy.docs`), every error carries its
  fix, and `alloy mcp` serves the engine as MCP tools ([agents](../agents.md)). It ships ready to use: nine
  ready-made environments, nine robots (arms, a hand, a quadruped, two humanoids), 90 materials and a library of
  5,423 objects ([What ships](../what-ships.md)).

## A soft body in Alloy

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("pad", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.03), material="silicone_40a")
sim = alloy.build(scene, device="cpu")
print({n: r["route"] for n, r in sim.routes().items() if n == "pad"})
sim.run(0.1)
print(sim.check()["ok"], round(float(sim.state(["pad"])["pad.points"][0, :, 2].min()), 3))
```

```text
{'pad': 'soft_vbd'}
True 0.003
```

Naming a soft material is enough: the silicone box became a soft body on the VBD solver, and its lowest vertex rests
at 0.003 m after 0.1 s. The world passed its health check. It ran on a CPU with no GPU.

## Table: Alloy and ManiSkill

One rule for every engine: "yes" means the feature is in the engine's public release and documented; "not
documented" means our survey has no public source for it. A note in brackets says where a "yes" holds. The other
engines' cells were read from their public code and docs on 2026-11-09, not run by us. The scorecard file,
`bench/scorecard/row6/capabilities.jsonl` in Alloy's source repository, has 26 rows and also grades how complete
each feature is; this table shows the 22 rows Alloy's public package documents.

| Capability | Alloy | ManiSkill3 / SAPIEN 3 |
|---|---|---|
| Liquids | yes | no |
| Particle fluids | yes | not documented |
| Sand and grains | yes | not documented |
| MPM (pastes, snow-like materials) | yes | no |
| Cutting | yes | not documented |
| Heat | yes | not documented |
| Rigid bodies with liquid | yes | not documented |
| Soft bodies with liquid | yes | not documented |
| Different step sizes per solver | yes | not documented |
| Cloth | yes | no |
| Soft bodies (volumetric FEM) | yes | no |
| Rigid bodies with cloth | yes | not documented |
| Rigid bodies with soft bodies | yes | no |
| Thin shells (paper, film, sheet metal) | yes | not documented |
| Plasticity (dents, permanent bends) | yes | not documented |
| Per-world health checks | yes | no |
| Double precision | yes (CPU) | no |
| Tactile sensing | yes | yes |
| Cables and rods | yes | yes |
| Many worlds on one GPU | yes | yes |
| Batched camera rendering | yes | yes |
| Same result on a re-run | yes (CPU) | yes (opt-in) |

## When ManiSkill is the better fit

- You want a large ready-made suite of manipulation tasks with demonstrations and baselines.
- Your vision pipeline is built on ManiSkill's GPU-parallel RGBD and segmentation.
- Your tasks are rigid-only and already written for ManiSkill.

## Keep in mind

Alloy's limits are listed on one page, each with its size and a workaround: [Keep in mind](../keep-in-mind.md).

## Next

- [Cloth](../guides/cloth.md), [Liquids](../guides/liquids.md), [Which simulator?](index.md)
