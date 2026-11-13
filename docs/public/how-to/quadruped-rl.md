# How do I train a quadruped robot to walk with reinforcement learning?

Use the Unitree Go2 velocity environment in the Alloy physics engine (`pip install alloy-engine`, with its `envs`
extra for training): one command trains a walking policy with PPO. The environment follows velocity commands on
flat ground or rough tiles. This page builds it on the CPU and reads it; training itself needs PyTorch, rsl_rl and
in practice a GPU.

## 1. Build the environment and step it

```python
import numpy as np
from alloy.envs.loco import go2_velocity

env = go2_velocity(num_envs=2, device="cpu")
obs, info = env.reset(seed=0)
print(env.obs_groups, {k: tuple(v.shape) for k, v in obs.items()})
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(np.zeros((2, 12), np.float32))   # action 0 = hold the stand
print(tuple(reward.shape), env.xp.to_numpy(env.state["base.height"]).round(2))
env.close()
```

```text
('policy', 'critic') {'policy': (2, 55), 'critic': (2, 79)}
(2,) [0.27 0.27]
```

The environment returns two observation groups: 55 numbers for the policy (what a robot could measure) and 79 for the
critic (the same plus values only a simulator knows). The action is 12 joint targets; zeros hold the stand, and the
base sits 0.27 m above the ground.

To train, install PyTorch and rsl_rl, then run the trainer on a GPU:

```bash
pip install "alloy-engine[envs]" rsl-rl-lib
python -m alloy.envs.loco.train --num-envs 4096 --iterations 400 --out runs/go2
```

It trains PPO, prints a line every 25 iterations and saves the checkpoint as `runs/go2/model_last.pt`. Add
`--terrain rough` for rough tiles.

One measured number: with 4,096 Go2 worlds standing, the simulation ran at 1,134 simulated seconds per wall second
on one NVIDIA L4 GPU, as shipped and physics only.

## Keep in mind

- The policy walks blind: it has no height scan, and there are no stairs or gaps.
- There is no actuator model, and no policy trained here has been run on a real robot.
- Training in practice needs a GPU. This page's first build compiles kernels, which takes minutes.
- Robot scenes step at 1.25 ms by default, which is their cost ([details](../../known_gaps/robot-fixed-step.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- RL libraries, tasks and wrappers: [Reinforcement learning](../guides/rl.md).
- Record demonstrations instead: [Imitation learning](../guides/imitation.md).
- Look at a run: [Camera](../guides/camera.md).
- Related: [the Gymnasium vector API](gymnasium-vector-api.md),
  [GPU-parallel engines for RL](../answers/gpu-parallel-engines-for-rl.md).
