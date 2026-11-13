# How do I randomise friction and mass across parallel worlds?

In the Alloy physics engine (`pip install alloy-engine`), `scene.randomize` draws a different value for every world
inside one simulation, seeded by `alloy.build(seed=)`. Friction and density work on any object, and `mass_scale` works
on a robot. Mass itself is not a field: randomise density. For looks, `alloy.vary` swaps textures between scenes.

## 1. Friction and density of free objects

```python
import alloy

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground(friction=1.0)
scene.add_box("light_to_heavy", size=(0.1, 0.1, 0.1), pos=(0, -0.3, 0.05), density=500, friction=0.05)
scene.add_box("slick_to_grippy", size=(0.1, 0.1, 0.1), pos=(0, 0.3, 0.05), density=500)
scene.randomize("light_to_heavy", "density", 300, 900)       # mass is not a field: randomise density
scene.randomize("slick_to_grippy", "friction", 0.1, 0.6)
sim = alloy.build(scene, seed=0)
for name in ["light_to_heavy", "slick_to_grippy"]:
    sim.apply_wrench(name, force=(3.0, 0, 0))                # the same 3 N push in every world
sim.run(1.0)
state = sim.state()
print("density box x:", state["light_to_heavy.pos"][:, 0].round(2))
print("friction box x:", state["slick_to_grippy.pos"][:, 0].round(2))
print(sim.check()["ok"])
```

```text
density box x: [4.46 1.36 1.51 1.45]
friction box x: [0.63 0.   1.48 1.38]
True
```

The same push moves each world's box a different distance. A lighter box goes further, and a grippier one may not
move at all (0.0 m). Contact friction comes from the pair of materials, so the ground is given a high value here.

## 2. Robot mass

```python
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20" kv="2"/></actuator></mujoco>"""

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
scene.randomize("arm", "mass_scale", 0.8, 1.2)               # every link's mass and inertia, one draw per world
sim = alloy.build(scene, seed=0)
sim.set_targets({"shoulder": 0.0})
sim.run(2.0)
print("sag:", sim.state()["arm/shoulder.q"].round(3))
print(sim.check()["ok"])
```

```text
sag: [0.077 0.063 0.061 0.068]
True
```

A soft servo holds the arm at target 0, and each world's arm sags by a different angle (rad) because its mass differs.
Robots also take `friction`, `kp_scale`, `kd_scale`, `armature`, centre of mass and gravity rows
(`docs/guide/randomize_robots.md` in the package); for an arm the scales are tested from 0.8 to 1.2.

## Keep in mind

- On a GPU, runs with bodies in contact do not repeat bit for bit with the same seed
  ([details](../../known_gaps/determinism-gpu-rigid.md)). Compare with a tolerance.
- Per-world values are uniform draws only. Lighting is not a per-world field: `alloy.vary` changes textures and
  tints, not lights.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Robots with cloth or liquid: [answer](../answers/robots-cloth-liquids-one-scene.md).
- GPU-parallel engines for RL: [answer](../answers/gpu-parallel-engines-for-rl.md).
- Every world's health: [World health](../guides/health.md).
