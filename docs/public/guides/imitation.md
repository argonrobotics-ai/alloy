# Imitation learning

`alloy.il` records demonstrations from a Gym-style environment and measures how well a policy does in it.

```bash
pip install "alloy-engine[envs]"
```

A scripted policy squeezes the gripper, then lifts. Record four episodes from four worlds and keep the ones that
succeed. `MemoryWriter` keeps them in a list; it needs no extra package.

```python
import numpy as np
import alloy.envs
from alloy.il import MemoryWriter, Recorder, evaluate

env = alloy.envs.make("GripperLift-v0", num_envs=4)

def policy(obs):
    act = np.tile(np.array([0.0, 1.0, 1.0], np.float32), (env.num_envs, 1))
    act[env.xp.to_numpy(env.state["steps"]) >= 25, 0] = 1.0
    return env.xp.asarray(act)

writer = MemoryWriter()
rec = Recorder(env, [writer], task="lift the cube and hold it", keep="success")
obs, info = rec.reset(seed=0)
while rec.saved < 4:
    obs, reward, terminated, truncated, info = rec.step(policy(obs))
rec.close()
ep = writer.episodes[0]
print(rec.saved, rec.dropped, ep.length, ep.obs["state"].shape, ep.actions.shape)
```

```text
4 0 150 (150, 22) (150, 3)
```

Four episodes were saved and none dropped. The first has 150 steps of 22 observation values and 3 action values.
To write files instead, use `HDF5Writer` (`pip install h5py`) or `LeRobotWriter`.

Now evaluate a policy. `evaluate` runs it for a number of episodes and reports how many ended in success.

```python
result = evaluate(env, policy, episodes=4)
print(result.episodes, result.mean_length)
```

```text
4 150.0
```

Four episodes ran and every one lasted 150 steps. `result.success_rate` and its 95 %
interval are in `result.summary()`; with so few episodes the interval is wide, so use many more for a real number.

## Keep in mind

- `alloy.il` records, replays and evaluates; it trains no policy.
- It works on a `VectorEnv`: the ready-made tasks and a `Task` of your own. The task ports (`alloy.envs.ports`) and
  Gymnasium wrappers around an environment are not supported.
- For a policy without memory, leave the `action[i]` columns out of its input: the state vector ends with the
  previous action, and a single-step behaviour-cloning network that reads it learns to repeat itself.
- For keyboard teleoperation, build the environment with a `max_steps` long enough for one demonstration
  (`max_steps=3000` is 30 s).
- Teleoperation drives the environment's action columns. On a joint-action environment a 7-joint arm is driven
  joint by joint; to drive the hand, build the environment with `alloy.envs.diffik.EEActionSpec` actions (an
  opt-in first version, tested on the CPU).
- Cloth, liquid and other particle objects are not in a recording, so such an episode cannot be replayed; neither
  can a clamped or soft cable on the rod solver, or an environment with an actuator model (`ActuatorSpec`).
- A replay of a scene with per-world random values (`scene.randomize`) is not exact: an episode can run with
  another world's values.
- On a GPU [runs do not repeat bit for bit](../../known_gaps/determinism-gpu-rigid.md) once bodies touch, so a
  replay is not exact.

## Next

- [Record a dataset to HDF5 or LeRobot](../how-to/make-a-dataset.md).
- [Train with RL](rl.md): train a policy in the same environments.
- [Robots](robots.md): load a robot and drive its joints.
- [Keep in mind](../keep-in-mind.md)
