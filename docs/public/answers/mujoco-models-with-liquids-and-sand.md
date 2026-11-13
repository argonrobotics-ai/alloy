# Which simulators that read MuJoCo models also support liquids and sand?

The Alloy physics engine, installed with `pip install alloy-engine`, loads an MJCF robot directly, steps it in MuJoCo
Warp (MuJoCo C on the CPU), and adds liquids, sand and pastes to the same scene.
This page puts an MJCF arm beside a tub of water and a tray of sand and prints the solver each object got.

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
scene.add_open_box("tub", inner=(0.08, 0.08, 0.05), wall=0.01, pos=(0.5, 0, 0))
scene.add_liquid("water", size=(0.064, 0.064, 0.024), pos=(0.5, 0, 0.024))
scene.add_open_box("tray", inner=(0.1, 0.1, 0.05), wall=0.01, pos=(0.5, 0.4, 0))
scene.add_box("sand", size=(0.06, 0.06, 0.03), pos=(0.5, 0.4, 0.03), material="sand_silica_dry")
sim = alloy.build(scene, device="cpu")
print({n: sim.routes()[n]["solver"] for n in ("arm", "water", "sand")})
sim.run(0.3, targets={"shoulder": (0.0, 0.3)})
print(sim.state(world=0)["arm/shoulder.q"].round(2), sim.check()["ok"])
```

```text
{'arm': 'mujoco', 'water': 'sph', 'sand': 'mpm'}
0.25 True
```

The MJCF string is the arm. Alloy sent it to MuJoCo, the water to its own liquid solver (DFSPH) and the sand to
Newton's MPM solver, stepped all three for 0.3 s while the shoulder servo ramped, and graded the world. A robot
file such as a Franka loads the same way with `add_robot`.
For rigid robots only, MuJoCo Warp alone is the leaner choice ([Alloy vs MuJoCo](../compare/mujoco.md)).

One measured number: on one NVIDIA L4, physics only, warm, in a pre-release build measured on 2026-11-09, with dt
10 ms and one run per point, a 200 ml pour gave 0.53 / 2.0 / 7.1 simulated seconds per wall second, summed over
all worlds, at 1 / 4 / 64 worlds.

## Keep in mind

- Many worlds of sand or paste are limited by the stress solve ([details](../../known_gaps/mpm-solve-speed.md)).
- A liquid beside sand meets it only at the sand's surface ([details](../../known_gaps/sph-mpm-mixing.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Sand in depth: [Sand and granular materials](sand-and-granular-materials.md).
- A robot with cloth or liquid: [MJCF robot with cloth or liquid](../how-to/mjcf-robot-with-cloth-or-liquid.md).
- Robots guide: [Robots](../guides/robots.md)
