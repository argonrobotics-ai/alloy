# Which pip-installable Python physics engine supports robot manipulation with soft bodies?

The Alloy physics engine (`pip install alloy-engine`) simulates soft bodies beside rigid tools and robots. Name
a soft material such as `silicone_40a` on a box and the build makes it a soft body. This page pushes a silicone
pad along the ground with a kinematic pusher.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("pad", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.026), material="silicone_40a")
scene.add_box("pusher", size=(0.02, 0.06, 0.04), pos=(-0.06, 0, 0.026), kinematic=True)
sim = alloy.build(scene)
print({name: r["route"] for name, r in sim.routes().items()})
start = sim.state(["pad"], world=0)["pad.points"]
sim.drive("pusher", velocity=(0.1, 0, 0), duration=0.4)
sim.run(0.5)
state = sim.state(world=0)
print(state["pusher.pos"].round(2), start.shape)
print(start[:, 0].min().round(3), state["pad.points"][:, 0].min().round(3), state["pad.points"][:, 0].max().round(3))
print(sim.check()["ok"])
```

```text
{'ground': 'static', 'pad': 'soft_vbd', 'pusher': 'static'}
[-0.02  0.    0.03] (729, 3)
-0.025 -0.009 0.042
True
```

`sim.routes()` says the pad took the `soft_vbd` route, the soft-body solver. The pusher moved 4 cm along x in 0.4 s,
and the pad's 729 points followed: its left edge went from x = -0.025 to -0.009 m and its right edge reached 0.042 m.
`sim.drive` moves a kinematic body; read `sim.check()` after the run.

## Keep in mind

- A soft body steps much slower than a rigid one, especially on a CPU: build a small one first.
- A small body resting on a coarse soft solid can sink into one surface particle
  ([details](../../known_gaps/soft-small-body-rests-on-one-particle.md)).
- A body dropped on a soft solid bounces back far more than the material's restitution says
  ([details](../../known_gaps/soft-solid-bounce.md)).
- No call attaches an object to a gripper: friction is the only grasp
  ([details](../../known_gaps/grasp-no-attach-or-weld.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Cloth and cables beside soft bodies: [Deformable object manipulation](deformable-object-manipulation.md)
- Install with pip: [Pip-installable simulators](pip-installable-simulators.md)
- Robots that hold things: [Robots](../guides/robots.md).
