# How do I move a Genesis scene to Alloy?

Rebuild the scene call by call in the Alloy physics engine (`pip install alloy-engine`): each `add_entity` becomes
one `scene.add_...` call, `build(n_envs=)` becomes `alloy.Scene(worlds=)`, and your MJCF or URDF robot loads as it
is and steps in MuJoCo. Liquid, sand and cloth need no solver options: Alloy picks the solver and its steps.

## 1. Rigid bodies, a robot and four worlds

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
scene.add_box("box", size=(0.2, 0.2, 0.2), pos=(0.6, 0, 0.1))
scene.add_sphere("ball", 0.05, pos=(0.6, 0.4, 0.05))
scene.add_robot("arm", mjcf=ARM)                     # or a path to an MJCF, URDF or USD file
sim = alloy.build(scene, device="cpu")               # leave device out to use a GPU when there is one
sim.set_targets({"shoulder": np.linspace(0.0, 0.6, 4)})          # one joint target per world, in radians
sim.step(100)
print(sim.state()["arm/shoulder.q"].round(2))
sim.set_pose("box", pos=(0.6, 0, 0.5))
print(sim.state(["box"])["box.pos"][:, 2].round(2), sim.check()["ok"])
```

```text
[0.01 0.21 0.41 0.61]
[0.5 0.5 0.5 0.5] True
```

Everything is read and set by name, with a leading world axis. The servo gains come from the robot file's
`<actuator>` lines, and `sim.check()` grades every world.

## 2. The calls, side by side

The Genesis column was read from the Genesis World documentation (`latest`, user guide pages Hello Genesis World,
Control your robot, Parallel simulation, Rigid bodies, Beyond rigid bodies, and the API reference) on 2026-11-17,
when the package on PyPI was `genesis-world` 1.4.7. It was read, not tested. The Alloy column runs on this page.

| Genesis | Alloy |
|---|---|
| `gs.init(backend=gs.cpu)` | no init call; `alloy.build(scene, device="cpu")` |
| `gs.Scene(sim_options=gs.options.SimOptions(dt=0.01))` | `alloy.Scene(dt=0.01)` |
| `scene.add_entity(gs.morphs.Plane())` | `scene.add_ground()` |
| `gs.morphs.Box(pos=, size=)`, `gs.morphs.Sphere` | `scene.add_box(name, size=, pos=)`, `add_sphere(name, radius)` |
| `gs.morphs.MJCF(file=)`, `gs.morphs.URDF(file=)` | `scene.add_robot(name, path)` |
| `scene.build(n_envs=4)` | `alloy.Scene(worlds=4)`, then `alloy.build(scene)` |
| `scene.step()` | `sim.step(n)`, or `sim.run(seconds)` to record |
| `entity.get_pos()`, `entity.get_dofs_position()` | `sim.state()["box.pos"]`, `["arm/shoulder.q"]` |
| `entity.set_pos(...)` | `sim.set_pose("box", pos=...)` |
| `entity.control_dofs_position(q, dofs_idx)` | `sim.set_targets({"shoulder": q})` |
| `entity.set_dofs_kp(...)` | gains in the robot file's `<actuator>` lines |
| `material=gs.materials.SPH.Liquid()` | `scene.add_liquid(name, size=, pos=)` |
| `gs.materials.MPM.Sand` | `scene.add_box(name, size=, material="sand_silica_dry")` |
| `gs.materials.PBD.Cloth` on a sheet mesh | `scene.add_cloth(name, size=, pos=, resolution=)` |
| `entity.get_particles_pos()` | `sim.state()["water.points"]` |
| `scene.reset()` | `sim.reset()` |

Genesis' guide sets `substeps` and a domain box (`lower_bound`, `upper_bound`) for its particle solvers. Alloy has
neither setting: it picks each solver's steps from the materials, and `sim.routes()` shows the choice.

## 3. Liquid, sand and cloth

```python
def run(fill):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tub", inner=(0.08, 0.08, 0.05), wall=0.02, pos=(0, 0, 0))
    fill(scene)
    sim = alloy.build(scene, device="cpu")
    sim.run(0.2)
    return sim.routes()["m"]["solver"], sim.state(["m"], world=0)["m.points"].shape, sim.check()["ok"]


print(run(lambda s: s.add_liquid("m", size=(0.064, 0.064, 0.024), pos=(0, 0, 0.034))))
print(run(lambda s: s.add_box("m", size=(0.05, 0.05, 0.03), pos=(0, 0, 0.04), material="sand_silica_dry")))
print(run(lambda s: s.add_cloth("m", size=(0.1, 0.1), pos=(0, 0, 0.08), resolution=20, mass=0.01)))
```

```text
('sph', (1536, 3), True)
('mpm', (4800, 3), True)
('vbd', (441, 3), True)
```

The material decides the solver: water goes to the liquid solver, sand to MPM, a cotton sheet to the cloth solver.
Each is read back as points. The sand build prints a notice that the block became grains, which step slower
than a rigid body (`rigid=True` keeps it rigid). The three run in three scenes here; water with cloth, and each
with a robot, share one scene ([robots, cloth and liquids](../answers/robots-cloth-liquids-one-scene.md)).

One measured number: physics only, warm, one NVIDIA L4, a pre-release build measured on 2026-11-09, dt 10 ms, one
run per point, simulated seconds per wall second summed over all worlds: a 200 ml pour 0.53 / 2.0 / 7.1 at 1 / 4 /
64 worlds, and a 0.6 m cotton sheet on a table 0.20 / 0.73 / 4.1.

## Keep using Genesis when

- You already build on Genesis' API, renderer and tooling, and its solvers cover your scene.
- You want double precision on a GPU: Genesis has a 64-bit option.

## Keep in mind

- Sand or paste beside fabric cloth does not build yet; the error names what to change
  ([details](../../known_gaps/scene-pairs-not-built.md)).
- Liquid beside sand or paste sees it as a solid wall ([details](../../known_gaps/sph-mpm-mixing.md)).
- Cloth, liquid and light free bodies do not build in one scene
  ([details](../../known_gaps/liquid-cloth-light-rigid.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Alloy and Genesis side by side: [compare](../compare/genesis.md).
- Liquids: [Liquids](../guides/liquids.md). Cloth: [Cloth](../guides/cloth.md).
- Read the health checks: [World health](../guides/health.md).
