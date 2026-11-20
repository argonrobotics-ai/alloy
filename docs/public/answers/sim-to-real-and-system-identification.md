# How does Alloy support sim-to-real transfer and system identification?

The Alloy physics engine (`pip install alloy-engine`) provides tools for sim-to-real work: per-world domain
randomization (`scene.randomize`), visual texture variation (`alloy.vary`), joint system identification
(`alloy.sysid`) from CSV, HDF5 or LeRobot recordings, and per-world health grades (`sim.health()`). This page
randomizes robot `mass_scale` and pad friction across four worlds beside a soft pad and checks every world.

```python
import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.4">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position name="motor" joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_robot("arm", mjcf=ARM)
scene.add_box("pad", size=(0.05, 0.05, 0.05), pos=(0.25, 0, 0.03), material="silicone_40a")
scene.randomize("arm", "mass_scale", 0.9, 1.1)
scene.randomize("pad", "friction", 0.5, 0.9)
sim = alloy.build(scene, seed=0, device="cpu")
sim.set_targets({"shoulder": 0.4})
sim.run(0.2)
print(sim.routes()["pad"]["route"], sim.check()["ok"])
```

```text
soft_vbd True
```

Each of the four worlds drew its own robot `mass_scale` and pad friction from `seed=0`, stepped the arm in MuJoCo and
the silicone pad in VBD, and passed its health checks.

## Tools for sim-to-real workflows

- **Joint system identification (`alloy.sysid`):** play an excitation trajectory on your robot
  (`python -m alloy.sysid excite`), log commands and measured joint angles as CSV, HDF5 or a LeRobot dataset, and run
  `python -m alloy.sysid fit` to fit per-joint stiffness (`kp`), `kd` + damping, dry friction and action delay in
  parallel worlds, then load the result with `scene.add_robot(..., sysid="my_robot.sysid.json")`.
- **Per-world physics randomization (`scene.randomize`):** draw friction, density, robot `mass_scale`, `kp_scale` and
  `kd_scale` independently per world inside one parallel build.
- **Visual domain randomization (`alloy.vary`):** returns separate scene variants with varied placements and textures,
  each built on its own; `alloy.Camera` renders parallel worlds inside a build.
- **Health grades (`sim.health()`):** flag penetration, speed spikes, energy jumps and non-finite state in every
  world so physics artefacts are caught before deploying a policy on hardware.

## Keep in mind

- `alloy.sysid` is verified on synthetic recordings, not yet validated on real robot hardware; it fits joint tracking
  only, not backlash, contact or payload parameters.
- Joint damping, dry friction and delay cannot be randomized per world yet.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Randomize physics per world: [Randomise friction and mass](../how-to/randomise-parallel-worlds.md).
- Deformable manipulation: [Deformable object manipulation](deformable-object-manipulation.md).
- Measured accuracy and throughput: [Measured numbers](benchmarks-and-accuracy.md).
