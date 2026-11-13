# Train with RL

Alloy's environments follow the shape of Gymnasium's vector API: every world is one environment, stepped together.

```bash
pip install "alloy-engine[envs]"
```

Make a ready-made environment, reset it and take one step with a sampled action.

```python
import alloy
import alloy.envs

env = alloy.envs.make("GripperLift-v0", num_envs=4)
obs, info = env.reset(seed=0)
env.action_space.seed(0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(tuple(obs.shape), env.action_space.shape, tuple(reward.shape))
env.close()
```

```text
(4, 22) (4, 3) (4,)
```

Four worlds give four rows: 22 observation values, 3 action values (lift and two finger closes) and one reward each.
With PyTorch installed the arrays are torch tensors on the simulation's device; without it, NumPy arrays.

Your own task is a scene, a `Task` and a `VectorEnv`. The task names the observations, the actions, the reward terms
and when an episode succeeds:

```python
from alloy.envs import Task, VectorEnv, checks as C
from alloy.envs.robots import gantry_actions, gantry_gripper

scene = alloy.Scene(dt=0.02)
scene.add_ground()
scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.025), mass=0.1)
scene.add_robot("gripper", mjcf=gantry_gripper(), pos=(0.0, 0.0, 0.2))

lifted = C.lifted("cube", by=0.05)
task = Task(
    obs=["gripper", "cube"],
    actions=gantry_actions("gripper", yaw=False),
    reward={"reach": (-1.0, C.distance("gripper/palm", "cube")), "lift": (10.0, lifted)},
    success=lifted & C.upright("cube"), max_steps=100)
env = VectorEnv(scene, task, worlds=4)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(env.action_names(), sorted(info["terms"]))
env.close()
```

```text
['gripper/x', 'gripper/y', 'gripper/z', 'gripper/close'] ['lift', 'reach']
```

The reward is the weighted sum of its terms, and `info["terms"]` holds each term. A world that ends starts its next
episode by itself.

To train, one command runs PPO with a library of your choice (it is not run here):

```bash
pip install rsl-rl-lib
python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl
```

Every line it prints carries the success rate of the episodes that ended lately: watch that number, not the reward.
For a task of your own, start from the closest ready-made environment (`sorted(alloy.envs.REGISTRY)` lists them)
and change it one step at a time, training after each change. A task and reward written from scratch rarely learn at
the first try.

## Legged robots

`from alloy.envs.loco import go2_velocity` makes a Unitree Go2 that follows velocity commands on flat ground. Train it
with `python -m alloy.envs.loco.train`. The policy walks blind, and nothing here was run on a real robot.

## Randomize per world

Draw a different value in every world with `scene.randomize("ball", "friction", 0.6, 1.0)`: it is seeded by
`alloy.build(scene, seed=)`. Other fields such as `density` work the same way.

## Trained policies to download

Three checkpoints are on the Hugging Face Hub, each with an evaluation script and its measured result
(alloy-engine 0.1.0rc2, one training run, one NVIDIA L4):
[Go2 walking](https://huggingface.co/argonrobotics-ai/alloy-go2-walk),
[cable lift](https://huggingface.co/argonrobotics-ai/alloy-cable-lift) and
[pour](https://huggingface.co/argonrobotics-ai/alloy-pour).

## Keep in mind

- No shipped trainer learns from camera images: on a camera environment (`GripperLiftCamera-v0`) the one-command
  trainer trains on the state only and does not say so.
- A robot with no `<actuator>` ignores actions without an error: print `env.sim.names()["actuators"]` first.
- On a GPU [runs do not repeat bit for bit](../../known_gaps/determinism-gpu-rigid.md) once bodies touch, so two
  training runs with the same seed differ.
- A world that goes INVALID is frozen and its episode ends: count `info["invalid"]` before you trust a result
  ([World health](health.md)).
- Training an arm to grasp gives results that vary between runs on a GPU: watch the success rate the trainer
  prints, and treat the shipped settings as a starting point. `PickPlace-v0` does not learn to place: no reward
  term pays the release into the bin.

## Next

- [Evaluate a trained policy on held-out seeds](../how-to/evaluate-checkpoint.md).
- [Imitation learning](imitation.md): record demonstrations and evaluate a policy.
- [Robots](robots.md): load a robot and drive its joints.
- [Keep in mind](../keep-in-mind.md)
