# Which simulator can put robots, cloth and liquids in the same scene?

The Alloy physics engine, installed with `pip install alloy-engine`, can: one scene holds a rigid robot, a cloth
sheet and a liquid, and `alloy.build` sends each to a solver suited to it. The robot steps in MuJoCo, the cloth in
Newton's VBD and the liquid in Alloy's own solver. This page builds that scene on a CPU.

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
scene.add_box("stand", size=(0.1, 0.1, 0.1), pos=(-0.3, 0, 0.05), static=True)
scene.add_cloth("sheet", size=(0.2, 0.2), pos=(-0.3, 0, 0.16), resolution=10, mass=0.05, self_contact=False)
scene.add_open_box("tub", inner=(0.08, 0.08, 0.05), wall=0.01, pos=(0.5, 0, 0))
scene.add_liquid("water", size=(0.064, 0.064, 0.024), pos=(0.5, 0, 0.024))
sim = alloy.build(scene, device="cpu")
print({name: sim.routes()[name]["solver"] for name in ("arm", "sheet", "water")})
sim.run(0.1, targets={"shoulder": (0.0, 0.3)})
print(sim.state(world=0)["arm/shoulder.q"].round(2), sim.check()["ok"])
```

```text
{'arm': 'mujoco', 'sheet': 'vbd', 'water': 'sph'}
0.17 True
```

The MJCF string is the arm (1 kg). The 50 g sheet drapes over a static stand on one side and the water sits in a tub
on the other, so the cloth and the liquid never touch. The arm weighs more than twice the cloth, which a scene with
cloth and liquid together needs. All three stepped for 0.1 s while the shoulder servo ramped, and the world passed
its health check. The first build of a new kind of scene compiles kernels for some minutes and is cached after.

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

- Liquids and sand beside MJCF robots: [MuJoCo models with liquids and sand](mujoco-models-with-liquids-and-sand.md).
- Mixed materials: [Open-source multi-physics engines](open-source-multi-physics-engines.md).
- Cloth in depth: [Cloth](../guides/cloth.md)
