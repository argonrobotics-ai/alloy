# What open-source GPU-parallel physics engines are there for reinforcement learning?

The Alloy physics engine, installed with `pip install alloy-engine`, steps many worlds on one GPU, each with a
health grade and Gym-style environments for training. This page lists the engines the compare pages cover and
builds a four-world Alloy scene on a CPU.

- Alloy: robots in MuJoCo Warp, plus cloth, liquids, sand and soft bodies in the same scene, in parallel worlds.
- MuJoCo Warp: many worlds on one GPU for rigid robots, and the solver Alloy steps robots in
  ([Alloy vs MuJoCo](../compare/mujoco.md)).
- Newton: the solvers Alloy builds on, with many worlds on one GPU ([Alloy vs Newton](../compare/newton.md)).
- Genesis: many worlds on one GPU, with mixed materials in one scene ([Alloy vs Genesis](../compare/genesis.md)).
- MuJoCo Playground: ready-made locomotion and dexterous-hand environments
  ([Alloy vs MuJoCo Playground](../compare/mujoco-playground.md)).
- ManiSkill3: a ready-made suite of manipulation tasks ([Alloy vs ManiSkill](../compare/maniskill.md)).

```python
import numpy as np
import alloy

W = 4
scene = alloy.Scene(worlds=W, dt=0.01)
scene.add_ground()
xs = np.stack([np.linspace(-0.2, 0.2, W), np.zeros(W), np.full(W, 0.3)], axis=1)
scene.add_sphere("ball", 0.03, pos=xs, material="steel_carbon")   # one start per world
scene.randomize("ball", "friction", 0.6, 1.0)                     # drawn per world, seeded by build(seed=)
sim = alloy.build(scene, seed=0, device="cpu")                    # drop device= to use a GPU when there is one
sim.step(100)
print(sim.worlds, sim.state(["ball"])["ball.pos"][:, 2].round(2))
print(sim.check()["ok"])
```

```text
4 [0.03 0.03 0.03 0.03]
True
```

Four balls fell from 0.3 m, one per world, each with its own friction, and all rest on the ground with their
centres at 0.03 m. Raise `W` to hundreds or thousands on a GPU; `alloy.estimate` predicts time and memory first.

One measured number: on one NVIDIA L4, with alloy-engine 0.1.0rc2 and one training run, a Go2 policy trained in
4,096 worlds walked on flat ground in 95 % of 256 held-out episodes after 4.1 minutes of training (kernel cache
warm), and in 97 % of 512 after 15.5 minutes. The checkpoint and its evaluation script are on the [Hugging Face
Hub](https://huggingface.co/argonrobotics-ai/alloy-go2-walk).

## Keep in mind

- On a GPU, runs with contact do not repeat bit for bit; the CPU does
  ([details](../../known_gaps/determinism-gpu-rigid.md)).
- Robot scenes step at a fixed 1.25 ms physics step by default ([details](../../known_gaps/robot-fixed-step.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Install with pip: [Pip-installable simulators](pip-installable-simulators.md).
- Mixed materials: [Open-source multi-physics engines](open-source-multi-physics-engines.md).
- Train a policy: [Reinforcement learning](../guides/rl.md)
