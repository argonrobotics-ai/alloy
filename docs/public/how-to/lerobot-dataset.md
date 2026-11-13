# How do I record a LeRobot dataset from simulation?

With the Alloy physics engine (`pip install alloy-engine`), `alloy.il.LeRobotWriter` saves the episodes of a vector
environment as a LeRobotDataset v3.0 on local disk. The `lerobot` extra needs Python 3.12. This page runs a scripted
gripper in 4 worlds, keeps 4 successful episodes and reads the dataset back with LeRobot's own loader.

```bash
pip install "alloy-engine[envs,lerobot]"
```

## 1. Generate demonstrations into a LeRobot dataset

```python
import alloy.envs
from alloy.il import LeRobotWriter, demonstrator, generate

env = alloy.envs.make("GripperLift-v0", num_envs=4, seed=0)
writers = [LeRobotWriter("demos/lift_lerobot", repo_id="local/lift", robot_type="gantry_gripper")]
result = generate(env, demonstrator(env), writers, episodes=4, task="lift the cube and hold it")
print("saved", result.saved, "of", result.wanted)
```

```text
saved 4 of 4
```

`generate` runs the scripted `demonstrator` that ships for the task in every world, keeps the episodes that succeed
and closes the writers. Nothing is sent to the Hugging Face Hub.

## 2. Read it back

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/lift", root="demos/lift_lerobot")
print(ds.num_episodes, ds.num_frames, ds.fps)
print(ds[0]["observation.state"].shape, ds[0]["action"].shape)
```

```text
4 600 100
torch.Size([22]) torch.Size([3])
```

The dataset holds 4 episodes of 150 frames at 100 frames per second. Each frame has `observation.state`, `action`,
`next.reward`, `next.done`, `next.success` and the task text. A camera adds an `observation.images.<name>` video
(`video=False` stores exact PNG images). To keep an HDF5 file in the robomimic layout too, pass an `HDF5Writer` in the
same list.

One measured number: behaviour cloning from 1,000 scripted demonstrations on the shipped `PickPlace-v0` task (a gantry
gripper, state observations) placed the object in 256 of 256 held-out episodes; alloy-engine 0.1.0rc2, one training
run, one NVIDIA L4, the tester's own trainer (Alloy ships none).

## Keep in mind

- Alloy records, replays and evaluates; it trains no policy, and training a LeRobot policy on the dataset or running
  `lerobot-eval` against an Alloy environment needs glue of your own.
- The dataset is readable only after the writer is closed (`generate` closes it): a recording killed before that
  leaves a folder LeRobot cannot open. For a long run, record to HDF5.
- On Python 3.11 the `lerobot` extra installs nothing: use Python 3.12, or record to HDF5 there.
- The control rate must be a whole number of steps per second (`dt=0.01` is 100 per second).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Environments and the Gymnasium API: [how-to](gymnasium-vector-api.md).
- Imitation data with liquids: [how-to](imitation-data-with-liquids.md).
- Imitation learning in more detail: [Imitation](../guides/imitation.md).
