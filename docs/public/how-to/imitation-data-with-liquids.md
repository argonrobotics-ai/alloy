# How do I generate imitation-learning data for a task with liquids?

With the Alloy physics engine (`pip install alloy-engine`, with the `envs` and `il` extras), wrap a liquid scene
in a `VectorEnv`, step it with your own scripted actions, and a `Recorder` writes each episode to HDF5 with
`HDF5Writer`, in the robomimic layout. This page records six episodes of a pusher in water.

```python
import numpy as np
import h5py
import alloy
from alloy.envs import ActionSpec, Task, VectorEnv, checks as C
from alloy.il import HDF5Writer, Recorder

PUSHER = """<mujoco><worldbody><body name="carriage" pos="0 0 0.035">
  <joint name="x" type="slide" axis="1 0 0" range="-0.1 0.1" damping="20"/>
  <geom type="sphere" size="0.005" mass="0.1" contype="0" conaffinity="0"/>
  <body name="tip"><joint name="y" type="slide" axis="0 1 0" range="-0.1 0.1" damping="20"/>
    <geom type="cylinder" size="0.012 0.02" mass="0.5" friction="0.5"/></body></body></worldbody>
  <actuator><position joint="x" kp="400" kv="40"/><position joint="y" kp="400" kv="40"/></actuator></mujoco>"""

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("cup", inner=(0.1, 0.06, 0.05), wall=0.005, pos=(0, 0, 0))
scene.add_liquid("water", size=(0.04, 0.05, 0.02), pos=(0.02, 0, 0.005), material="water")
scene.add_robot("pusher", mjcf=PUSHER, pos=(-0.035, 0.0, 0.0))
inside = C.fraction_inside("water", (-0.05, -0.03, 0.0), (0.05, 0.03, 0.06))
task = Task(name="PushWater", obs=["pusher", inside],
            actions=ActionSpec(["pusher/x", "pusher/y"], mode="delta", delta=0.004,
                               low=[-0.1, -0.1], high=[0.1, 0.1]),
            reward={"in_cup": (1.0, inside)}, max_steps=20)
env = VectorEnv(scene, task, worlds=2, backend="numpy")
print(env.action_names(), env.sim.routes()["water"]["route"])
```

```text
['pusher/x', 'pusher/y'] liquid_sph
```

The scene is a rigid pusher on two slides beside 38 ml of water in a cup. `fraction_inside` is the task's check on the
water's particles. Now write your own demonstrator: push in for ten steps, pull back for ten, and record.

```python
rec = Recorder(env, [HDF5Writer("push_water.hdf5", overwrite=True)], task="push the water", keep="all", states=False)
obs, info = rec.reset(seed=0)
for step in range(3 * task.max_steps):
    forward = 1.0 if step % task.max_steps < 10 else -1.0           # your own script: push in, then pull back
    obs, reward, terminated, truncated, info = rec.step(np.tile([forward, 0.0], (2, 1)).astype(np.float32))
rec.close()
print(rec.saved, env.sim.check()["ok"])
with h5py.File("push_water.hdf5", "r") as f:
    print(len(f["data"]), f["data"].attrs["total"], list(f["mask"]))
    demo = f["data"]["demo_0"]
    print(demo.attrs["num_samples"], demo["actions"].shape, demo["obs/state"].shape, demo["rewards"].shape)
```

```text
6 True
6 120 ['all', 'success']
20 (20, 2) (20, 5) (20,)
```

Two worlds ran three rounds of 20 steps, so the file holds 6 demos and 120 steps. Each demo has 20 steps of 2 action
values and 5 observation values. `keep="all"` saves every episode, since this script has no success test;
`states=False` skips the simulator state, which cannot hold particles anyway.

One measured number: on the shipped `Pour-v0` task, 118 of 128 held-out episodes (92 %) put at least 75 % of the water
in the bowl, for the checkpoint at 38.5 minutes of training in 512 worlds; alloy-engine 0.1.0rc2, one training run,
one NVIDIA L4.
The checkpoint and its evaluation script are on the
[Hugging Face Hub](https://huggingface.co/argonrobotics-ai/alloy-pour).

## Keep in mind

- `demonstrator` ships for `GripperLift-v0`, `GripperLiftCamera-v0` and `PickPlace-v0` only; `Pour-v0` needs a script
  of your own, as above ([Imitation learning](../guides/imitation.md)).
- A recording with liquid cannot be replayed: `replay` raises `EnvError`, as particles are not in the saved state.
- A container narrower than 24 mm is unreliable ([details](../../known_gaps/liquid-narrow-containers.md)).
- A GPU run does not repeat bit for bit, so a second run can save other episodes
  ([GPU determinism](../../known_gaps/determinism-gpu-rigid.md)).
- Full list: [Keep in mind](../keep-in-mind.md).

## Next

- Write the same data as a LeRobot dataset: [LeRobot dataset](lerobot-dataset.md).
- Mixed scenes with liquid: [Robots, cloth and liquids in one scene](../answers/robots-cloth-liquids-one-scene.md).
- Replay and evaluate recordings: [Imitation learning](../guides/imitation.md).
