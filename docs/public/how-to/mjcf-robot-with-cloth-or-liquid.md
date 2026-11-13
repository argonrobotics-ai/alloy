# How do I load a MuJoCo MJCF robot into a scene with cloth or liquid?

Call `scene.add_robot` with an MJCF string or file in the Alloy physics engine (`pip install alloy-engine`), then add
cloth or liquid to the same scene. The robot steps in MuJoCo, the cloth or water in its own solver, and the two
exchange contact forces. This page builds a one-joint arm beside a cotton sheet, then beside water, on the CPU.

## 1. An arm and a cloth sheet

```python
import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("stand", size=(0.2, 0.2, 0.2), pos=(0.15, 0, 0.1), static=True)
scene.add_robot("arm", mjcf=ARM, pos=(-0.1, 0, 0.35))
scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0.15, 0, 0.3), resolution=10, mass=0.02)
sim = alloy.build(scene)
routes = sim.routes()
print(routes["sheet"]["route"], routes["arm"]["route"])
sim.run(0.2, targets={"shoulder": (0.0, 0.2)})
print(sim.state(world=0)["arm/shoulder.q"].round(2))
print(sim.check()["ok"])
```

```text
cloth_vbd rigid
0.15
True
```

`sim.routes()` says which solver took each object: the sheet goes to the cloth solver, the arm stays rigid. The arm
swings while the sheet settles on the stand. A joint target is a ramp here, from 0 to 0.2 rad over 0.2 s.

## 2. The same arm and water in a tray

```python
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("tray", inner=(0.12, 0.12, 0.06), wall=0.004, pos=(0.15, 0, 0))
scene.add_liquid("water", size=(0.1, 0.1, 0.03), pos=(0.15, 0, 0.008), material="water")
scene.add_robot("arm", mjcf=ARM, pos=(-0.1, 0, 0.2))
sim = alloy.build(scene)
routes = sim.routes()
print(routes["water"]["route"], routes["arm"]["route"])
sim.run(0.2, targets={"shoulder": (0.0, 0.2)})
print(sim.state(world=0)["arm/shoulder.q"].round(2))
print(sim.check()["ok"])
```

```text
liquid_sph rigid
0.15
True
```

The water is particles in Alloy's own solver, and the arm moves beside it. Always read `sim.check()["ok"]`
before you trust a run. Cloth and liquid in one scene with a robot need a robot that weighs at least twice the cloth.

One measured number: physics only, warm, one NVIDIA L4, a pre-release build measured on 2026-11-09, dt 10 ms, one
run per point, simulated seconds per wall second summed over all worlds: a Franka Panda scene 1.9 / 6.9 / 74 at 1 / 4
/ 64 worlds, and a 0.6 m cotton sheet on a table 0.20 / 0.73 / 4.1 at 1 / 4 / 64.

## Keep in mind

- No robot grasp that carries a cotton sheet is verified yet
  ([details](../../known_gaps/cloth-robot-grasp-carry.md)).
- Cloth and liquid beside a robot or free body lighter than twice their mass raise `BuildError`
  ([details](../../known_gaps/liquid-cloth-light-rigid.md)).
- Cloth takes up no water ([details](../../known_gaps/liquid-on-cloth.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Robots, cloth and liquids in one scene: [answer](../answers/robots-cloth-liquids-one-scene.md).
- MuJoCo models with liquids and sand: [answer](../answers/mujoco-models-with-liquids-and-sand.md).
- Robots in Alloy: [Robots](../guides/robots.md).
