# Which physics engines are built on MuJoCo Warp or Newton?

The Alloy physics engine, installed with `pip install alloy-engine`, is built on both. Rigid bodies and robots step in
MuJoCo Warp (MuJoCo C on the CPU), cloth and soft bodies in Newton's VBD solver, and sand and pastes in Newton's
implicit MPM solver. Alloy picks the solver for each object, and `sim.routes()` names it. This page prints it.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("tray", inner=(0.1, 0.1, 0.05), wall=0.01, pos=(0, 0, 0))
scene.add_box("sand", size=(0.06, 0.06, 0.03), pos=(0, 0, 0.03), material="sand_silica_dry")
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0, 0.025), mass=0.5)
sim = alloy.build(scene, device="cpu")
for name in ("sand", "block"):
    print(name, sim.routes()[name]["route"], sim.routes()[name]["solver"])
sim.run(0.1)
print(sim.check()["ok"])
```

```text
sand grain_mpm mpm
block rigid mujoco
True
```

Both objects are `add_box` calls; only the material differs. The sand went to Newton's MPM solver and the block to
MuJoCo. A second scene sends a soft pad and a cloth sheet to Newton's VBD solver:

```python
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0, 0.025), mass=0.5)
scene.add_box("pad", size=(0.04, 0.04, 0.04), pos=(-0.3, 0, 0.03), material="silicone_40a")
scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0.4, 0.2), resolution=10, mass=0.05, self_contact=False)
sim = alloy.build(scene, device="cpu")
for name in ("block", "pad", "sheet"):
    print(name, sim.routes()[name]["route"], sim.routes()[name]["solver"])
sim.run(0.1)
print(sim.check()["ok"])
```

```text
block rigid mujoco
pad soft_vbd vbd
sheet cloth_vbd vbd
True
```

Alloy's liquid solver (DFSPH) is its own. Newton and MuJoCo Warp ship inside the package, so one `pip install`
brings both. Newton itself is also an engine you can use
directly: it is the better fit when you want the solvers themselves, differentiable simulation or its inverse
kinematics ([Alloy vs Newton](../compare/newton.md)). MuJoCo Warp alone suits scenes of rigid robots only
([Alloy vs MuJoCo](../compare/mujoco.md)). MuJoCo Playground also runs on MuJoCo Warp, one of its backends
([Alloy vs MuJoCo Playground](../compare/mujoco-playground.md)).

One measured number: a ten-block tower in 1,024 worlds on one NVIDIA L4 ran 5 simulated seconds with 0 of 1,024
worlds with an error (a block moved or sank more than 1 cm, or a non-finite value) at each of three accuracy
settings, in two runs (pre-release builds).

## Keep in mind

- A liquid beside sand or paste meets it only at the paste's surface: the paste is a solid wall to the liquid
  ([details](../../known_gaps/sph-mpm-mixing.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Install with pip: [Pip-installable simulators](pip-installable-simulators.md).
- Mixed materials: [Open-source multi-physics engines](open-source-multi-physics-engines.md).
- Scenes with robots and liquids: [Robots](../guides/robots.md)
