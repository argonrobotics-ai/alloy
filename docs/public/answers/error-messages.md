# What do Alloy's error messages mean and what should I change?

Every error in the Alloy physics engine (`pip install alloy-engine`) carries `e.message`, which says what went wrong,
and `e.fix`, which says what to change. This page runs four mistakes people make and shows the exact text of each.
For the error classes and two more examples, see [Errors and fixes](../errors.md).

```python
import textwrap
import alloy

def explain(call):
    try:
        call()
    except alloy.AlloyAPIError as e:
        print(textwrap.fill(e.message, width=110))
        print(textwrap.fill(e.fix, width=110))

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05))
```

## `add_cloth: 'dough_wheat' is a paste (it yields and flows), and add_cloth makes sheets that bend and stretch`

A paste material cannot be a sheet (`alloy.SceneError`). Give the paste a solid shape instead.

```python
explain(lambda: scene.add_cloth("rag", size=(0.2, 0.2), pos=(0, 0, 0.3), material="dough_wheat"))
```

```text
add_cloth: 'dough_wheat' is a paste (it yields and flows), and add_cloth makes sheets that bend and stretch
give the paste a solid shape instead, e.g. add_box(name, size=(x, y, depth), material='dough_wheat'); or pick
a sheet material (alloy.materials())
```

## `name 'block' is already in the scene`

Names are unique, because every state key is built from them.

```python
explain(lambda: scene.add_box("block", size=(0.1, 0.1, 0.1)))
```

```text
name 'block' is already in the scene
pick another name (taken: ground, block)
```

## `device 'cuda:0' asked for but no CUDA GPU is visible`

You asked for a GPU on a machine without one. This text appears on a CPU-only machine only.

```python
explain(lambda: alloy.build(scene, device="cuda:0"))
```

```text
device 'cuda:0' asked for but no CUDA GPU is visible
pass device='cpu' (slower) or run on a machine with an NVIDIA GPU and driver
```

## `joint 'shoulder' has no actuator, so a position target cannot move it`

A target needs an actuator on the joint (`alloy.SimulationError`). Add the line the fix names to the MJCF.

```python
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody></mujoco>"""
arm_scene = alloy.Scene(dt=0.01)
arm_scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
explain(lambda: alloy.build(arm_scene).set_targets({"shoulder": 0.5}))
```

```text
joint 'shoulder' has no actuator, so a position target cannot move it
add <position joint="..." kp="..."/> for it to the robot's MJCF <actuator> block
```

## Keep in mind

- A run can go wrong without raising: read `sim.check()` ([World health](../guides/health.md)).
- Health does not catch every overlap between rigid bodies and robot parts
  ([what it misses](../../known_gaps/health-mujoco-penetration.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Read a failed run: [Did my simulation blow up?](did-my-simulation-blow-up.md)
- Connect an agent that reads these messages: [An MCP server for agents](mcp-server-physics-engine.md)
- Every health flag: [World health](../guides/health.md).
