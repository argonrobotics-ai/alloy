# Alloy vs Newton: which should I use?

Use the Alloy physics engine when you want Newton's solvers behind one scene API that picks the solver for each
object: you name a material, and Alloy routes it, couples it to the rest and checks every world. It adds a particle
liquid, tactile images and cutting. Install it with `pip install alloy-engine`.

## What Alloy adds

- One scene API over the solvers. Cloth and soft bodies step in Newton's VBD, sand and pastes in Newton's implicit
  MPM, robots in MuJoCo Warp; Alloy chooses from the material and shape, and `sim.routes()` gives the reasons. You
  do not wire solvers together yourself.
- A particle liquid of its own (DFSPH), which pours, fills containers and floats bodies. Newton's liquid is
  grid-based implicit MPM.
- Tactile images from gel fingertip sensors in every world (the package's tactile examples), and cutting of soft
  objects with a knife. Newton's public sensors and solvers list neither.
- Health in every world. `sim.health()` grades each world `OK`, `DEGRADED` or `INVALID` and names a fix for every
  flag: non-finite values, penetration, speed, energy jumps. Newton warns when a contact estimate is large.
- It is made for coding agents: the docs ship inside the package (`python -m alloy.docs`), every error carries its
  fix, and `alloy mcp` serves the engine as MCP tools ([agents](../agents.md)). It ships ready to use: nine
  ready-made environments, nine robots (arms, a hand, a quadruped, two humanoids), 90 materials and a library of
  5,423 objects ([What ships](../what-ships.md)).

## Sand and a rigid block: Alloy picks the solvers

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("tray", inner=(0.1, 0.1, 0.05), wall=0.01, pos=(0, 0, 0))
scene.add_box("sand", size=(0.06, 0.06, 0.03), pos=(0, 0, 0.03), material="sand_silica_dry")
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0, 0.025), mass=0.5)
sim = alloy.build(scene, device="cpu")
for name in ("sand", "block"):
    print(name, sim.routes()[name]["route"], sim.routes()[name]["solver"])
sim.run(0.1)
print(sim.check()["ok"])
```

```text
sand grain_mpm mpm
block rigid mujoco
True
```

Both objects are `add_box` calls; only the material differs. Alloy sent the sand to Newton's implicit MPM solver and
the block to MuJoCo, and the world passed its health check. It ran on a CPU with no GPU.

## Table: Alloy and Newton

One rule for every engine: "yes" means the feature is in the engine's public release and documented; "not
documented" means our survey has no public source for it. A note in brackets says where a "yes" holds. The other
engines' cells were read from their public code and docs on 2026-11-09, not run by us. The scorecard file,
`bench/scorecard/row6/capabilities.jsonl` in Alloy's source repository, has 26 rows and also grades how complete
each feature is; this table shows the 22 rows Alloy's public package documents.

| Capability | Alloy | Newton |
|---|---|---|
| Particle fluids | yes | no |
| Cutting | yes | no |
| Heat | yes | not documented |
| Tactile sensing | yes | no |
| Liquids | yes | yes |
| Sand and grains | yes | yes |
| MPM (pastes, snow-like materials) | yes | yes |
| Rigid bodies with liquid | yes | yes |
| Soft bodies with liquid | yes | yes |
| Different step sizes per solver | yes | yes |
| Cloth | yes | yes |
| Soft bodies (volumetric FEM) | yes | yes |
| Rigid bodies with cloth | yes | yes |
| Rigid bodies with soft bodies | yes | yes |
| Thin shells (paper, film, sheet metal) | yes | yes |
| Plasticity (dents, permanent bends) | yes | yes |
| Cables and rods | yes | yes |
| Per-world health checks | yes | yes |
| Many worlds on one GPU | yes | yes |
| Batched camera rendering | yes | yes |
| Same result on a re-run | yes (CPU) | yes (opt-in, most solvers) |
| Double precision | yes (CPU) | yes (Kamino linear algebra only) |

## When Newton is the better fit

- You want the solvers themselves, or want to change their internals: Alloy is built on Newton, so go to it
  directly.
- You need differentiable simulation, inverse kinematics, Newton's own sensors and viewers, or its other solvers:
  they are in Newton's API.

## Keep in mind

Alloy's limits are listed on one page, each with its size and a workaround: [Keep in mind](../keep-in-mind.md).

## Next

- [Cloth](../guides/cloth.md), [Liquids](../guides/liquids.md), [Which simulator?](index.md)
