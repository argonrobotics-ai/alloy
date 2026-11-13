# How do I use a GPU-parallel simulator with the Gymnasium vector API?

The Alloy physics engine (`pip install alloy-engine`) turns a scene into a vector environment where every world is
one environment and all of them step at once, on a GPU when there is one and on the CPU otherwise. `VectorEnv` has
the shape of Gymnasium's vector API: `reset`, `step`, spaces and `terminated` and `truncated`.

## 1. Make an environment, reset and step it

```python
import alloy.envs

env = alloy.envs.make("GripperLift-v0", num_envs=4)
obs, info = env.reset(seed=0)
print("obs", tuple(obs.shape))
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("reward", tuple(reward.shape), "terminated", tuple(terminated.shape))
print(env.num_envs, env.action_names())
```

```text
obs (4, 22)
reward (4,) terminated (4,)
4 ['gripper/lift', 'gripper/close_left', 'gripper/close_right']
```

Each row is one world: 4 worlds, 22 observation values, 3 actions. Observations and actions are torch tensors on the
simulation device when PyTorch is installed (the `envs` extra), else NumPy arrays. Finished worlds reset by
themselves.

## 2. Build the same task through Gymnasium

```python
import gymnasium as gym

alloy.envs.register_gymnasium()
venv = gym.make_vec("alloy/GripperLift-v0", num_envs=4, vectorization_mode="vector_entry_point", backend="numpy")
obs, info = venv.reset(seed=0)
obs, reward, terminated, truncated, info = venv.step(venv.action_space.sample())
print(type(obs).__name__, obs.shape, reward.shape)
print(venv.single_action_space.shape, venv.num_envs)
```

```text
ndarray (4, 22) (4,)
(3,) 4
```

`gym.make` does not build these environments (there is no single-environment form); `gym.make_vec` does, after
`register_gymnasium()`. `backend="numpy"` returns NumPy arrays, which Gymnasium's own tools expect.

One measured number: on one NVIDIA L4, with alloy-engine 0.1.0rc2 and one training run, a Go2 policy trained in
4,096 worlds walked on flat ground in 95 % of 256 held-out episodes after 4.1 minutes of training (kernel cache
warm), and in 97 % of 512 after 15.5 minutes. The checkpoint and its evaluation script are on the [Hugging Face
Hub](https://huggingface.co/argonrobotics-ai/alloy-go2-walk).

## Keep in mind

- Only `make_vec`, `reset` and `step` are tested with Gymnasium; its vector wrappers are not.
- Robot scenes use a fixed physics step of 1.25 ms by default, which sets their cost
  ([details](../../known_gaps/robot-fixed-step.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Training with RL libraries: [Reinforcement learning](../guides/rl.md).
- GPU-parallel engines for RL: [answer](../answers/gpu-parallel-engines-for-rl.md).
- Pip-installable simulators: [answer](../answers/pip-installable-simulators.md).
