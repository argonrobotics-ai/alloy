# Which robot learning frameworks run on MuJoCo Warp?

The Alloy physics engine (`pip install alloy-engine`) runs robots on MuJoCo Warp and couples them in one scene to one
of cloth, soft bodies, sand or liquids, with Gym-style environments, per-world health grades and an MCP server. Other
open-source stacks on MuJoCo Warp include mjlab and MuJoCo Playground. This page runs a robot beside water in four
worlds.

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
scene.add_open_box("cup", inner=(0.04, 0.04, 0.05), wall=0.008, pos=(0.3, 0, 0))
scene.add_liquid("water", size=(0.03, 0.03, 0.025), pos=(0.3, 0, 0.021))
sim = alloy.build(scene, device="cpu")
sim.set_targets({"shoulder": 0.3})
sim.run(0.1)
print({k: sim.routes()[k]["solver"] for k in ("arm", "water")}, sim.check()["ok"])
```

```text
{'arm': 'mujoco', 'water': 'sph'} True
```

The arm steps on MuJoCo (MuJoCo Warp on a GPU, MuJoCo C on a CPU) while the water steps in Alloy's DFSPH liquid
solver inside the same control loop, and every world passed its health check.

## Open-source frameworks on MuJoCo Warp

- **Alloy (`pip install alloy-engine`):** MuJoCo Warp for robots and rigid bodies, coupled to one of Newton VBD (cloth
  or soft bodies), Newton MPM (sand or pastes) or DFSPH (liquids); not every material pair mixes in one scene
  ([material limits](../../known_gaps/scene-pairs-not-built.md)). Ships nine ready-made environments, wrappers for
  `rsl_rl`, `skrl` and Stable-Baselines3, per-world health grades and `alloy mcp`.
- **mjlab:** a lightweight PyTorch manager-based RL framework built directly on MuJoCo Warp with a browser viewer
  (Viser) and velocity and motion-tracking examples.
- **MuJoCo Playground:** a PyTorch suite on MuJoCo Warp covering control, manipulation, humanoid tracking, teleop,
  sysid and sim-to-real, with its earlier MJX environments under `rss2025`
  ([Alloy vs MuJoCo Playground](../compare/mujoco-playground.md)).
- **MuJoCo Warp directly:** a lean stack when your scene holds rigid robots only and needs no other material solvers:
  plain MuJoCo Warp is faster on robots ([Alloy vs MuJoCo](../compare/mujoco.md)).

One measured number: on one NVIDIA L4 with `alloy-engine 0.1.0rc2` and one training run, the shipped Unitree Go2 task
in 4,096 worlds reached 95 % held-out walking success in 4.1 minutes (warm cache) and 97 % in 15.5 minutes.

## Keep in mind

- Installing Alloy pins `mujoco` and `warp-lang` to the versions it builds against; keep another MuJoCo Warp stack in
  its own virtual environment if it expects different pins.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Engine layers: [Built on MuJoCo Warp or Newton](built-on-mujoco-warp-or-newton.md).
- Compare stacks: [Which simulator should I use?](../compare/index.md).
- Measured numbers: [Throughput, rendering and accuracy](benchmarks-and-accuracy.md).
