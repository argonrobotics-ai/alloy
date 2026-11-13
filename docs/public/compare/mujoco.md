# Alloy vs MuJoCo and MuJoCo Warp: which should I use?

Use the Alloy physics engine when your robot must handle more than rigid parts: it steps your MJCF robot in MuJoCo
Warp (MuJoCo C on the CPU) and adds liquids, sand and pastes to the same scene, beside cloth and cables, with a
health check for every world. Install it with `pip install alloy-engine`; a robot file loads directly.

## What Alloy adds

- MuJoCo's robots, plus more materials in the same scene. Alloy steps rigid bodies and robots in MuJoCo Warp on a
  GPU and in MuJoCo C on the CPU, and steps liquids, sand, pastes, cloth and soft bodies beside them, each in a
  solver suited to it. `sim.routes()` says which solver each object went to, and why.
- A particle liquid, sand and pastes that push on rigid bodies and robots. MuJoCo and MuJoCo Warp model fluid as
  drag on rigid bodies.
- Health in every world. `sim.health()` grades each world `OK`, `DEGRADED` or `INVALID` and names a fix for every
  flag: non-finite values, penetration, speed, energy jumps, contact overflow. MuJoCo C has warning counters and a
  NaN check on accelerations; MuJoCo Warp prints overflow warnings.
- Cutting of soft objects with a knife, and heat in liquids, both in the docs the package installs.
- It is made for coding agents: the docs ship inside the package (`python -m alloy.docs`), every error carries its
  fix, and `alloy mcp` serves the engine as MCP tools ([agents](../agents.md)). It ships ready to use: nine
  ready-made environments, nine robots (arms, a hand, a quadruped, two humanoids), 90 materials and a library of
  5,423 objects ([What ships](../what-ships.md)).

## A robot arm and water in one scene

```python
import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.1))
scene.add_open_box("tub", inner=(0.08, 0.08, 0.05), wall=0.01, pos=(0.3, 0, 0))
scene.add_liquid("water", size=(0.064, 0.064, 0.024), pos=(0.3, 0, 0.024))
sim = alloy.build(scene, device="cpu")
print({name: sim.routes()[name]["solver"] for name in ("arm", "water")})
sim.run(0.6, targets={"shoulder": (0.0, 0.3)})
print(len(sim.state(["water"], world=0)["water.points"]), sim.state(world=0)["arm/shoulder.q"].round(2))
print(sim.check()["ok"], sim.health()["worlds"][0]["severity"])
```

```text
{'arm': 'mujoco', 'water': 'sph'}
1536 0.28
True OK
```

The MJCF string is the arm; the water is 1,536 particles in a static tub beside it. Alloy sent the arm to MuJoCo and
the water to its own liquid solver, stepped both for 0.6 s while the shoulder servo ramped, and graded the world. It
ran on a CPU with no GPU.

## Table: Alloy and MuJoCo or MuJoCo Warp

One rule for every engine: "yes" means the feature is in the engine's public release and documented; "not
documented" means our survey has no public source for it. A note in brackets says where a "yes" holds. The other
engines' cells were read from their public code and docs on 2026-11-09, not run by us. The scorecard file,
`bench/scorecard/row6/capabilities.jsonl` in Alloy's source repository, has 26 rows and also grades how complete
each feature is; this table shows the 22 rows Alloy's public package documents.

| Capability | Alloy | MuJoCo (C, CPU) | MuJoCo Warp 4.0.0 |
|---|---|---|---|
| Liquids | yes | no | no |
| Particle fluids | yes | no | no |
| Sand and grains | yes | not documented | no |
| MPM (pastes, snow-like materials) | yes | not documented | no |
| Cutting | yes | not documented | no |
| Heat | yes | not documented | not documented |
| Rigid bodies with liquid | yes | not documented | no |
| Soft bodies with liquid | yes | not documented | no |
| Different step sizes per solver | yes | not documented | no |
| Many worlds on one GPU | yes | no | yes |
| Double precision | yes (CPU) | yes (CPU) | no |
| Cloth | yes | yes | yes |
| Soft bodies (volumetric FEM) | yes | yes | yes |
| Rigid bodies with cloth | yes | yes | yes |
| Rigid bodies with soft bodies | yes | yes | yes |
| Tactile sensing | yes | yes | yes |
| Thin shells (paper, film, sheet metal) | yes | yes | yes |
| Plasticity (dents, permanent bends) | yes | yes | yes |
| Cables and rods | yes | yes | yes |
| Per-world health checks | yes | yes | yes |
| Batched camera rendering | yes | yes | yes |
| Same result on a re-run | yes (CPU) | yes (CPU) | yes (opt-in) |

## When MuJoCo or MuJoCo Warp is the better fit

- Your scene is rigid robots only and you want the leanest, fastest stack for rigid robots: use MuJoCo Warp
  directly. Alloy runs on it, so the robot model carries over.
- You want to change solver internals, or use everything an MJCF file can set (keyframes, sensors, its own time step
  and integrator): work in MuJoCo itself.
- You want MuJoCo C alone on the CPU, in double precision, with its viewer and tools.

## Keep in mind

Alloy's limits are listed on one page, each with its size and a workaround: [Keep in mind](../keep-in-mind.md).

## Next

- [Robots](../guides/robots.md), [Liquids](../guides/liquids.md), [Which simulator?](index.md)
- Related: [MuJoCo models with liquids and sand](../answers/mujoco-models-with-liquids-and-sand.md),
  [engines built on MuJoCo Warp or Newton](../answers/built-on-mujoco-warp-or-newton.md).
