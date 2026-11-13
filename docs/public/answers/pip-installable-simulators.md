# Which open-source simulators for robot learning install with pip?

The Alloy physics engine installs with `pip install alloy-engine` and runs on a CPU or an NVIDIA GPU. One package
holds rigid robots, cloth, liquids, sand, soft bodies and cables, nine ready-made Gym-style environments, and the docs
for coding agents. People looking for an alternative to Isaac Lab usually want a pip install and one GPU.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)         # top at z = 0.4
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)         # full edges (m), kg
sim = alloy.build(scene, device="cpu")                          # drop device= to use a GPU when there is one
traj = sim.run(1.0, record=["block"])
print(traj["block.pos"].shape)
print(sim.state(world=0)["block.pos"].round(2), sim.check()["ok"])
```

```text
(101, 1, 3)
[0.   0.   0.42] True
```

Install inside a virtual environment with Python 3.11 or 3.12, then `import alloy`. The block fell 0.2 m onto the
table top at z = 0.4 and rests with its centre at 0.42 m, and `sim.check()["ok"]` says no world failed. Optional
extras add the rest: `pip install "alloy-engine[envs]"` for the environments and training wrappers, and
`"alloy-engine[import,mesh]"` for mesh tools and soft bodies from meshes.

One measured number: on one NVIDIA L4, with alloy-engine 0.1.0rc2 and one training run, the shipped `Pour-v0` task
reached 118 of 128 held-out episodes (92 %) that put at least 75 % of the water in the bowl, for the checkpoint at
38.5 minutes of training in 512 worlds.
The checkpoint and its evaluation script are on the
[Hugging Face Hub](https://huggingface.co/argonrobotics-ai/alloy-pour).

## Keep in mind

- On a GPU, runs with contact do not repeat bit for bit; the CPU does
  ([details](../../known_gaps/determinism-gpu-rigid.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Engine layers: [Built on MuJoCo Warp or Newton](built-on-mujoco-warp-or-newton.md).
- Many parallel worlds: [GPU-parallel engines for RL](gpu-parallel-engines-for-rl.md).
- Run a robot policy on one GPU: [Reinforcement learning](../guides/rl.md).
