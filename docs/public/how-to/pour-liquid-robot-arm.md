# How do I simulate pouring a liquid with a robot arm?

In the Alloy physics engine (`pip install alloy-engine`) the water is particles that the container carries and
pours. Two methods work: tilt a kinematic container with `sim.drive(..., pivot=lip)`, or make the container part of
the robot's hand in its MJCF and drive the wrist joint. This page shows the second method, on the CPU.

## 1. Build a one-joint arm that carries a container

```python
import numpy as np
import alloy

LIP_Z, INNER, WALL = 0.12, 0.05, 0.004
CUP_X, CUP_R, CUP_DEPTH = 0.04, 0.035, 0.06
OUT = INNER + 2 * WALL


def wall(name, centre, half):
    pos, size = " ".join(map(str, centre)), " ".join(map(str, half))
    return f'<geom name="{name}" type="box" pos="{pos}" size="{size}" mass="0.02"/>'


# the container is a body of the arm's link: five box geoms, its +x top edge (the lip) at the joint
GEOMS = "\n".join([
    wall("floor", (-OUT / 2, 0, -OUT + WALL / 2), (OUT / 2, OUT / 2, WALL / 2)),
    wall("back", (-OUT + WALL / 2, 0, -OUT / 2), (WALL / 2, OUT / 2, OUT / 2)),
    wall("front", (-WALL / 2, 0, -OUT / 2), (WALL / 2, OUT / 2, OUT / 2)),
    wall("left", (-OUT / 2, -OUT / 2 + WALL / 2, -OUT / 2), (OUT / 2, WALL / 2, OUT / 2)),
    wall("right", (-OUT / 2, OUT / 2 - WALL / 2, -OUT / 2), (OUT / 2, WALL / 2, OUT / 2)),
])
ARM = f"""<mujoco><compiler angle="radian"/><worldbody><body name="hand" pos="0 0 0">
  <joint name="wrist" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
  {GEOMS}
  </body></worldbody>
  <actuator><position joint="wrist" kp="40" kv="4"/></actuator></mujoco>"""
```

The hand's link holds five box geoms (a floor and four walls, 4 mm thick, 5 cm inside) with its top edge, the lip, at
the joint. The container cannot be put down, since it is part of the arm.

## 2. Add water and a cup, tilt the wrist, count the water

```python
scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_robot("arm", mjcf=ARM, pos=(0, 0, LIP_Z))
scene.add_liquid("water", size=(0.04, 0.04, 0.016), pos=(-OUT / 2, 0, LIP_Z - OUT + WALL + 0.0085))
scene.add_open_cylinder("cup", inner_radius=CUP_R, depth=CUP_DEPTH, wall=WALL, pos=(CUP_X, 0, 0))
sim = alloy.build(scene)
print(sim.routes()["water"]["route"], len(sim.state(["water"], world=0)["water.points"]))
sim.settle(0.2)
sim.run(2.0, targets={"wrist": (0.0, 1.5)})
sim.step(100)
p = sim.state(["water"], world=0)["water.points"]
in_cup = (np.hypot(p[:, 0] - CUP_X, p[:, 1]) < CUP_R) & (p[:, 2] < WALL + CUP_DEPTH)
past_lip = p[:, 0] > 0
print("over a third in the cup:", bool(in_cup.mean() > 1 / 3))
print("over 0.4 past the lip:", bool(past_lip.mean() > 0.4))
print(sim.check()["ok"])
```

```text
liquid_sph 400
over a third in the cup: True
over 0.4 past the lip: True
True
```

The water is 400 particles. The wrist ramps 1.5 rad (about 86 degrees) over 2 s, and the water is read
0.5 s later. On the CPU 0.52 of it was past the lip and 0.45 in the cup; some missed the cup and lies on the ground,
and the rest is still in the container. The stream leaves the lip slowly and lands a few cm past it, so the cup axis
sits 4 cm past the lip.

One measured number: on an AMD EPYC 7B13 CPU, no GPU, this program took about 165 s warm, including the build.

## Keep in mind

- Liquid is slow on a CPU: use few particles and a short run, or a GPU.
- Containers narrower than 24 mm are unreliable ([details](../../known_gaps/liquid-narrow-containers.md)).
- Tilt slowly and place the cup past the lip, not under it. Drop a block of water from above and some of it splashes
  over the rim ([details](../../known_gaps/liquid-pour-over-the-rim.md)).
- The first method, a kinematic jug tilted with `sim.drive`, needs no robot file but no arm moves it.
- A container held only by finger friction is not verified for pouring.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Liquids in more detail: [Liquids](../guides/liquids.md).
- Joint targets and robots: [Robots](../guides/robots.md).
- Read the health checks: [World health](../guides/health.md).
- Related: [robots, cloth and liquids in one scene](../answers/robots-cloth-liquids-one-scene.md),
  [imitation data with liquids](imitation-data-with-liquids.md).
