# How do I record a dataset of robot demonstrations?

Run a scripted demonstrator in the Alloy physics engine (`pip install alloy-engine`, with its `envs` extra) and let
`generate` save each successful episode with `HDF5Writer`. The file follows the robomimic layout, so h5py can open it.
This page records eight demos of a gripper lifting a cube, in four worlds, and reads the file back.

```python
import h5py
import alloy.envs
from alloy.il import HDF5Writer, demonstrator, generate

env = alloy.envs.make("GripperLift-v0", num_envs=4)
writer = HDF5Writer("demos/lift.hdf5", overwrite=True)           # needs: pip install h5py
result = generate(env, demonstrator(env), [writer], episodes=8, seed=0)
print(result.saved, result.attempted, result.invalid)

with h5py.File("demos/lift.hdf5", "r") as f:
    print(len(f["data"]), f["data"].attrs["total"], list(f["mask"]))
    demo = f["data"]["demo_0"]
    print(demo.attrs["num_samples"], demo["actions"].shape, demo["obs/state"].shape, demo["rewards"].shape)
```

```text
8 8 0
8 1200 ['all', 'success']
150 (150, 3) (150, 22) (150,)
```

`generate` ran the shipped script in every world until eight episodes had been saved and closed the writer. All eight
attempts succeeded and none was invalid. The file holds 8 demos and 1,200 steps in all. Each demo has 150 steps of
3 action values and 22 observation values, and the file lists the demos under the filter keys `all` and `success`.

To write a LeRobot dataset instead, install `pip install 'lerobot[dataset]'` (Python 3.12) and pass a `LeRobotWriter`
in the same list; a list may hold both writers:

```py
from alloy.il import LeRobotWriter

writer = LeRobotWriter("demos/lift_lerobot", repo_id="local/lift", robot_type="gantry_gripper")
generate(env, demonstrator(env), [writer], episodes=4, seed=0)
```

## Keep in mind

- `demonstrator` ships for `GripperLift-v0`, `GripperLiftCamera-v0` and `PickPlace-v0` only; for other tasks write a
  `PhaseScript`.
- `keep="success"` is the default: failed and invalid episodes are not saved.
- A LeRobot dataset has no start states, so use the HDF5 file when you want to replay episodes.
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Record and evaluate in more detail: [Imitation learning](../guides/imitation.md).
- Train a policy with RL instead: [Train with RL](../guides/rl.md).
- Check a run before you trust it: [World health](../guides/health.md).
