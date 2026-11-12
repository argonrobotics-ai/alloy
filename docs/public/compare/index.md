# Which physics simulator should I use for robot learning?

Use the Alloy physics engine when one scene must hold more than rigid robots: liquids, cloth, sand, soft bodies and
cables stepped together with the robot, each in a solver suited to it, with a health grade for every world and docs
made for coding agents. Install it with `pip install alloy-engine`.

## What Alloy is for

- Many materials in one scene. Robots step in MuJoCo Warp (MuJoCo C on the CPU), cloth and soft bodies in Newton's
  VBD, sand and pastes in Newton's MPM, liquids in Alloy's own solver; Alloy picks the solver for each object and
  `sim.routes()` says why.
- Health in every world. `sim.health()` grades each world `OK`, `DEGRADED` or `INVALID` and names a fix for every
  flag, so a physics failure is not read as a policy failure.
- It is made for coding agents: the docs ship inside the package (`python -m alloy.docs`), every error carries its
  fix, and `alloy mcp` serves the engine as MCP tools ([agents](../agents.md)).
- It ships ready to use: nine ready-made environments, nine robots (arms, a hand, a quadruped, two humanoids), 90
  materials and a library of 5,423 objects ([What ships](../what-ships.md)).
- Breadth. Read with one rule for every engine ("yes" means the feature is in the public release and documented),
  the public scorecard's 26 capabilities come to 22 for Alloy, 22 for Genesis, 19 for Newton, 13 for MuJoCo, 13 for
  MuJoCo Playground, 12 for MuJoCo Warp and 5 for ManiSkill3. The other engines' cells were read from their public
  code and docs on 2026-11-09, not run by us.

## A rigid block and water in one scene

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0, 0.025), mass=0.5)
scene.add_open_box("cup", inner=(0.04, 0.04, 0.05), wall=0.01, pos=(0, 0, 0))
scene.add_liquid("water", size=(0.032, 0.032, 0.032), pos=(0, 0, 0.03))
sim = alloy.build(scene, device="cpu")
print({name: sim.routes()[name]["solver"] for name in ("block", "water")})
sim.run(0.5)
print(sim.check()["ok"], sim.health()["worlds"][0]["severity"])
```

```text
{'block': 'mujoco', 'water': 'sph'}
True OK
```

Two objects, two solvers, one `build` call: the block went to MuJoCo and the water to Alloy's liquid solver.
`sim.check()["ok"]` is the one value to read after a run; `sim.health()` has a row for every world.

## One paragraph per engine

**MuJoCo and MuJoCo Warp.** Alloy steps your MJCF robot in MuJoCo Warp and adds liquids, sand and pastes to the
same scene, with a health grade per world. Pick MuJoCo Warp alone when the scene is rigid robots only and you
want the leanest, fastest stack for rigid robots. [Alloy vs MuJoCo](mujoco.md).

**Newton.** Alloy puts Newton's solvers behind one scene API that picks the solver for each object, and adds a
particle liquid, tactile images and cutting. Pick Newton alone when you want the solvers themselves, differentiable
simulation or its inverse kinematics. [Alloy vs Newton](newton.md).

**Genesis.** Both mix materials in one scene. Alloy adds robots stepped by MuJoCo's solver, a health grade with a
fix for every flag, elastic-rod cables, and documented cutting. Pick Genesis alone when you already build on its API
or want double precision on a GPU. [Alloy vs Genesis](genesis.md).

**MuJoCo Playground.** Alloy adds tasks with more than rigid bodies (pour a liquid, fold a cloth) on the same robot
solver, in many worlds with cameras. Pick Playground alone when you want its ready-made locomotion and
dexterous-hand environments with tuned training recipes. [Alloy vs MuJoCo Playground](mujoco-playground.md).

**ManiSkill.** Alloy adds the physics ManiSkill3 does not model (cloth, liquids, soft bodies) behind a Gym-style
API. Pick ManiSkill alone when you want its ready-made suite of manipulation tasks with
demonstrations and baselines. [Alloy vs ManiSkill](maniskill.md).

## Keep in mind

Alloy's limits are listed on one page, each with its size and a workaround: [Keep in mind](../keep-in-mind.md).

## Next

- [Quick start](../quickstart.md), [World health](../guides/health.md), [What ships](../what-ships.md)
