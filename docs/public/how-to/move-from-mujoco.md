# How do I move a MuJoCo scene to Alloy?

Give your MJCF file to `scene.add_robot` or `alloy.load` in the Alloy physics engine
(`pip install alloy-engine`). You do not edit the file, and the robot still steps in MuJoCo (MuJoCo Warp on a GPU).
`sim.step`, `sim.state` and `sim.set_targets` take the place of `mj_step`, `data.qpos` and
`data.ctrl`. Then add water, cloth or sand beside the robot.

## 1. The same MJCF in MuJoCo and in Alloy

```python
import mujoco
import numpy as np
import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.5">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position name="motor" joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

model = mujoco.MjModel.from_xml_string(ARM)          # plain MuJoCo: 500 steps of 2 ms
data = mujoco.MjData(model)
data.ctrl[0] = 0.5
for _ in range(500):
    mujoco.mj_step(model, data)
print(data.qpos.round(2), round(data.time, 2))

scene = alloy.Scene(dt=0.01)                         # Alloy: 100 control periods of 10 ms
scene.add_robot("arm", mjcf=ARM)                     # or a path: add_robot("arm", "robot.xml")
sim = alloy.build(scene, device="cpu")
sim.set_targets({"shoulder": 0.5})
sim.step(100)
print(sim.state(world=0)["arm/shoulder.q"].round(2), round(sim.t, 2))
```

```text
[0.51] 1.0
0.51 1.0
```

Both hold the arm at 0.51 rad after one second: the servo's target is 0.5 and gravity adds the rest. MuJoCo comes
with Alloy (`mujoco==3.17.0`), so both halves run in one environment.

## 2. The calls, side by side

| MuJoCo | Alloy |
|---|---|
| `MjModel.from_xml_path(path)`, `MjData(model)` | `alloy.load(path)`, or `scene.add_robot(name, path)` and a build |
| `mj_step(model, data)` | `sim.step(n)` for n control periods, `sim.run(seconds)` to record |
| `data.qpos`, `data.xpos` by index | `sim.state()["arm/shoulder.q"]`, `["arm/link1.pos"]` by name |
| `data.ctrl[i] = v` | `sim.set_targets({"shoulder": v})`; raw input: `sim.set_controls({"arm/motor": v})` |
| `mj_resetData(model, data)`, `data.time` | `sim.reset()`, `sim.t` |
| one `MjData` per copy | `alloy.Scene(worlds=n)`: every state array starts with a world axis |

`sim.names()` lists every body, joint and actuator name. One `sim.step()` is one control period (`dt`); Alloy takes
the physics steps inside it, 1.25 ms each for a robot.

## 3. Four worlds, and water beside the arm

```python
scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_robot("arm", mjcf=ARM)
scene.add_open_box("tub", inner=(0.08, 0.08, 0.05), wall=0.01, pos=(0.3, 0, 0))
scene.add_liquid("water", size=(0.064, 0.064, 0.024), pos=(0.3, 0, 0.024))
sim = alloy.build(scene, device="cpu")
sim.set_targets({"shoulder": np.linspace(0.0, 0.6, 4)})          # one target per world
sim.run(0.5)
print(sim.state()["arm/shoulder.q"].round(2), sim.check()["ok"])
print({name: sim.routes()[name]["solver"] for name in ("arm", "water")})
```

```text
[0.01 0.21 0.41 0.61] True
{'arm': 'mujoco', 'water': 'sph'}
```

The arm stays in MuJoCo and the water steps in Alloy's liquid solver, in four worlds at once, each with its own
health check. Cloth works the same way: [an MJCF robot with cloth or liquid](mjcf-robot-with-cloth-or-liquid.md).

## 4. What to check in your MJCF

- Angles are degrees unless the file says `<compiler angle="radian"/>`, as in MuJoCo. Alloy's targets are radians.
- One joint per body: two slide joints on one body load as one joint that no target can drive. Nest a second body.
- Every moving body needs mass and inertia: add an `<inertial>` to a body with a joint and no geom.
- A `<weld>` or `<connect>` with no `solref` or `solimp` is made stiffer at build.
  `alloy.build(scene, welds="mujoco")` keeps MuJoCo's own.
- Robot shapes make contacts from 5 mm apart; `add_robot(..., contact_gap=)` changes it.
- A robot scene file that has its own floor needs no `scene.add_ground()`.
- Alloy steps your MJCF with its own settings: the file's `timestep`, `integrator` and `cone` are not used. A
  robot starts at joint angle 0, not at a keyframe, and no call reads the file's `<sensor>` elements
  ([Robots](../guides/robots.md)).

One measured number: physics only, warm, one NVIDIA L4, a pre-release build measured on 2026-11-09, dt 10 ms, one
run per point, simulated seconds per wall second summed over all worlds: a Franka Panda scene 1.9 / 6.9 / 74 at 1 / 4
/ 64 worlds.

## Keep using MuJoCo when

- Your scene is rigid robots only and speed is what counts: plain MuJoCo Warp is faster on robots.
- You need what the file sets for MuJoCo itself: keyframes, sensors, its own time step, integrator and
  friction cone.

## Keep in mind

- Robot scenes step physics at 1.25 ms unless you ask for a coarser step
  ([details](../../known_gaps/robot-fixed-step.md)).
- A weld is a stiff spring, not a rigid joint ([details](../../known_gaps/weld-stretch-under-load.md)).
- Cloth, liquid or soft parts beside a robot run much slower on the CPU than the robot alone.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Alloy and MuJoCo side by side: [compare](../compare/mujoco.md).
- MuJoCo models with liquids and sand: [answer](../answers/mujoco-models-with-liquids-and-sand.md).
- Robots in Alloy: [Robots](../guides/robots.md).
