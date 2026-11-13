# Which new robotics simulators were released in 2026?

The Alloy physics engine is one: `alloy-engine` 0.1.0 reached PyPI on 2026-11-11, and you install it with
`pip install alloy-engine`. It steps rigid robots, cloth, liquids, sand and soft bodies in one scene, in parallel
worlds on a CPU or an NVIDIA GPU. This page dates that release and runs the quick start on a CPU.

- 2026-03-10: Newton 1.0.0, the first 1.0 release of the engine Alloy's cloth and soft-body solvers come from,
  after pre-releases in 2025 ([PyPI release history](https://pypi.org/project/newton/#history)).
- 2026-11-10: `alloy-engine` 0.1.0rc1, the first release candidate on PyPI.
- 2026-11-11: `alloy-engine` 0.1.0, the first stable release
  ([PyPI release history](https://pypi.org/project/alloy-engine/#history)).

Dates are the upload times PyPI lists (UTC). The list holds only releases whose date we read on the project's own
PyPI page, so it is short; it will grow as we check more.

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

The block fell onto the table top at z = 0.4 and rests with its centre at 0.42 m, and `sim.check()["ok"]` says no
world failed. Install inside a virtual environment with Python 3.11 or 3.12.

One measured number: on one NVIDIA L4, with alloy-engine 0.1.0rc2 and one training run, a Go2 policy trained in
4,096 worlds walked on flat ground in 95 % of 256 held-out episodes after 4.1 minutes of training (kernel cache
warm), and in 97 % of 512 after 15.5 minutes. The checkpoint and its evaluation script are on the [Hugging Face
Hub](https://huggingface.co/argonrobotics-ai/alloy-go2-walk).

## Keep in mind

- On a GPU, runs with contact do not repeat bit for bit; the CPU does
  ([details](../../known_gaps/determinism-gpu-rigid.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Install with pip: [Pip-installable simulators](pip-installable-simulators.md).
- Compare engines: [GPU-parallel engines for RL](gpu-parallel-engines-for-rl.md).
- Robots: [Robots](../guides/robots.md)
