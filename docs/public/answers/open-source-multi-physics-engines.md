# Which open-source engines do multi-physics robot simulation?

The Alloy physics engine, installed with `pip install alloy-engine`, mixes rigid robots, cloth, liquids, sand, soft
bodies and cables in one scene, and grades every world `OK`, `DEGRADED` or `INVALID` with a fix for each flag.
Genesis and Newton also mix materials. This page runs three materials in one scene and reads the health grades.

- Alloy: robots stepped by MuJoCo's solver beside liquids, cloth and sand, health grades with fixes, elastic-rod
  cables and documented cutting ([Alloy vs Genesis](../compare/genesis.md)).
- Genesis: both mix materials in one scene; pick it when you already build on its API or want double precision on a
  GPU ([Alloy vs Genesis](../compare/genesis.md)).
- Newton: the solvers Alloy builds on; pick it when you want the solvers themselves, differentiable simulation or its
  inverse kinematics ([Alloy vs Newton](../compare/newton.md)).

```python
import alloy

scene = alloy.Scene(worlds=2, dt=0.01)
scene.add_ground()
scene.add_open_box("cup", inner=(0.04, 0.04, 0.05), wall=0.005, pos=(0, 0, 0))
scene.add_liquid("water", size=(0.032, 0.032, 0.03), pos=(0, 0, 0.02), material="water")
scene.add_box("pad", size=(0.04, 0.04, 0.04), pos=(0.3, 0, 0.03), material="silicone_40a")
scene.add_cloth("sheet", size=(0.2, 0.2), pos=(-0.3, 0, 0.2), resolution=10, mass=0.05, self_contact=False)
sim = alloy.build(scene, device="cpu")
print({n: sim.routes()[n]["solver"] for n in ("water", "pad", "sheet")})
sim.run(0.1)
print(sim.check()["ok"], [w["severity"] for w in sim.health()["worlds"]])
```

```text
{'water': 'sph', 'pad': 'vbd', 'sheet': 'vbd'}
True ['OK', 'OK']
```

Water, a soft silicone pad and a cloth sheet share one scene in two worlds. Alloy sent the water to its own liquid
solver and the pad and sheet to VBD, and each world got its own health grade. `sim.check()["ok"]` is the one value
to read after a run.

One measured number: on one NVIDIA L4, physics only, warm, in a pre-release build measured on 2026-11-09, with dt
10 ms and one run per point, a 0.6 m cotton sheet on a table gave 0.20 / 0.73 / 4.1 simulated seconds per wall
second, summed over all worlds, at 1 / 4 / 64 worlds, and a 200 ml pour gave 0.53 / 2.0 / 7.1.

## Keep in mind

- Cloth and liquid together need rigid bodies or a robot that weigh at least twice the cloth and soft bodies
  ([details](../../known_gaps/liquid-cloth-light-rigid.md)).
- Cloth takes up no water, and a stream poured onto fabric can make the world INVALID
  ([details](../../known_gaps/liquid-on-cloth.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Engine layers: [Built on MuJoCo Warp or Newton](built-on-mujoco-warp-or-newton.md).
- Robots with cloth and liquids: [Robots, cloth and liquids in one scene](robots-cloth-liquids-one-scene.md).
- Check a world's health: [World health](../guides/health.md)
