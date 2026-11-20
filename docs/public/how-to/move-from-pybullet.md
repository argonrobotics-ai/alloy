# How do I move a PyBullet scene to Alloy?

Rebuild the scene call by call in the Alloy physics engine (`pip install alloy-engine`): `loadURDF` and body creation
calls become `scene.add_robot`, `scene.add_box` or `scene.add_sphere`, `alloy.Scene(worlds=)` replaces separate CPU
clients with parallel GPU worlds, and `sim.state()` reads poses and joints by name instead of integer body IDs.

## 1. Four parallel worlds with a box, a ball and a robot

```python
import numpy as np
import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.5">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position name="motor" joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_box("crate", size=(0.1, 0.1, 0.1), pos=(0.5, 0, 0.05), mass=0.5)
scene.add_sphere("ball", 0.04, pos=(0.5, 0.3, 0.04), mass=0.1)
scene.add_robot("arm", mjcf=ARM)                          # or a path to an MJCF, URDF or USD file
sim = alloy.build(scene, device="cpu")                    # drop device= to use a GPU when there is one
sim.set_targets({"shoulder": np.linspace(0.0, 0.6, 4)})   # one target per world, in radians
sim.step(100)
print(sim.state()["arm/shoulder.q"].round(2))
sim.set_pose("crate", pos=(0.5, 0, 0.4))
print(sim.state(["crate"])["crate.pos"][:, 2].round(2), sim.check()["ok"])
```

```text
[0.01 0.21 0.41 0.61]
[0.4 0.4 0.4 0.4] True
```

Sizes in Alloy are full edge lengths (in PyBullet, `halfExtents` are half-lengths), z is up, units are SI, and every
world runs inside one simulation instead of separate `DIRECT` clients.

## 2. The calls, side by side

| PyBullet | Alloy |
|---|---|
| `p.connect(p.DIRECT)` | no client handle; `alloy.build(scene, device="cpu")` |
| `p.setGravity(0, 0, -9.81)` | fixed gravity; per-world draws via `scene.randomize("gravity", "z", lo, hi)` |
| `p.setTimeStep(0.01)` | `alloy.Scene(dt=0.01)` |
| `p.loadURDF("plane.urdf")` | `scene.add_ground()` |
| `p.loadURDF("robot.urdf")` | `scene.add_robot("robot", "robot.urdf")` (no actuators; use MJCF to drive) |
| `p.createMultiBody(...)` with box `halfExtents` | `scene.add_box(name, size=(2*hx, 2*hy, 2*hz), pos=)` |
| `p.stepSimulation()` | `sim.step(n)` or `sim.run(seconds)` |
| `p.getBasePositionAndOrientation(uid)` | `sim.state()["crate.pos"]`, `["crate.quat_xyzw"]` |
| `p.resetBasePositionAndOrientation(u, p, q)` | `sim.set_pose(name, pos=p, quat_wxyz=(w,x,y,z))` (`q` is `xyzw`) |
| `p.setJointMotorControlArray(..., POSITION_CONTROL)` | `sim.set_targets({"joint": q})` (MJCF only; URDF raises) |
| `p.changeDynamics(uid, -1, lateralFriction=...)` | `scene.add_box(..., friction=...)` or `scene.randomize(...)` |
| `p.getCameraImage(w, h, ...)` | `sim.render(width=w, height=h)` or `alloy.Camera` |
| Multiple CPU processes (`DIRECT` clients) | `alloy.Scene(worlds=N)` on one GPU |

You can also place one of liquid (`scene.add_liquid`), cloth (`scene.add_cloth`), sand (`material="sand_silica_dry"`)
or a cable (`scene.add_rod`) beside the robot in the same scene (liquids and sand are not in PyBullet), and
`sim.check()` grades every world.

## Keep using PyBullet when

- Your robots are URDF files you drive as they are: Alloy needs the robot as MJCF with `<actuator>` lines to take
  targets.
- You run on Windows: Alloy runs on Linux, and on the CPU of an Apple-silicon Mac; Windows is not tested.
- A small single-process CPU rigid-body script, an existing PyBullet codebase, or its built-in
  `calculateInverseKinematics`, is all you need.

## Keep in mind

- A URDF robot's joints get no actuator in Alloy, so `sim.set_targets` raises and tells you to convert the robot to
  MJCF with `<actuator>` lines.
- Not every material pair builds in one scene ([details](../../known_gaps/scene-pairs-not-built.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Move from MuJoCo or Genesis: [Move from MuJoCo](move-from-mujoco.md), [Move from Genesis](move-from-genesis.md).
- Parallel cameras: [Camera for many worlds](camera-many-worlds.md).
- Robots and joints: [Robots](../guides/robots.md).
