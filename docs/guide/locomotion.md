# Legged locomotion: the Go2 velocity environment

`alloy.envs.loco` is one reference environment for legged robots: the Unitree Go2 (a library robot, MuJoCo Menagerie
file, BSD-3-Clause) on flat ground or on rough terrain tiles, tracking velocity commands. It is built on
`alloy.envs.VectorEnv` (see `envs.md`). It is opt-in: `import alloy.envs.loco` loads it, and nothing else in Alloy
changes.

Install: `pip install "alloy-engine[envs]"`. Observations and actions are torch tensors when PyTorch is installed (on
the GPU when there is one) and NumPy arrays when it is not. The snippets below work with both: they read env values
through `env.xp.to_numpy(...)`, which gives a NumPy copy of either (a plain `np.asarray` stops on a GPU tensor with
`TypeError: can't convert cuda:0 device type tensor to numpy`). They were run on the CPU with NumPy and with PyTorch;
the GPU run of the snippets as they read now is not done yet.

## Quick start

```python
import numpy as np
from alloy.envs.loco import go2_velocity

env = go2_velocity(num_envs=4)                  # GPU if there is one; the first build compiles kernels
obs, info = env.reset(seed=0)
print(env.xp.name, env.obs_groups, tuple(obs["policy"].shape))   # numpy without PyTorch, torch with it
zeros = np.zeros((4, 12), np.float32)           # action 0 = hold the stand
obs, reward, terminated, truncated, info = env.step(zeros)
print(tuple(reward.shape), float(reward.mean()))
print(sorted(info))   # ['command', 'invalid', 'success', 'targets', 'terms']
env.close()
```

A warm build takes about 7 s on four worlds on an AMD EPYC 7B13 (two builds in fresh processes: 7.2 s and 7.1 s). The
first build ever compiles kernels and takes minutes. The control period is 0.02 s (50 Hz). Each `env.step` runs 16
physics steps (`env.sim.physics_steps` grows by 16). A build prints some `NewtonWarning` lines about collision
margins for the Go2 file; they are harmless.

## Observations

`obs` is a dict of two groups, `obs["policy"]` and `obs["critic"]`. The policy group is what a robot could
measure. The critic group is the policy's numbers plus privileged ones that only a simulator knows.

| key | size | unit | meaning |
|---|---|---|---|
| `base.lin_vel_b` | 3 | m/s | base velocity in the base frame |
| `base.ang_vel_b` | 3 | rad/s | base angular velocity in the base frame (a gyroscope) |
| `base.gravity_b` | 3 | 1 | unit gravity direction in the base frame; (0, 0, -1) when upright |
| `base.accel_b` | 3 | g | accelerometer, scaled by 1/9.81; (0, 0, 1) at rest |
| `command` | 3 | m/s, m/s, rad/s | forward speed, left speed, yaw rate wanted |
| `joints.pos` | 12 | rad | joint angles; 0 is the stand |
| `joints.vel` | 12 | rad/s | joint speeds |
| `action` | 12 | 1 | the action of the last step |
| `feet.contact` | 4 | 1 | 1 if the foot touches the ground |

That is 55 numbers. The critic group adds 24 (79 in all):

| key | size | unit | meaning |
|---|---|---|---|
| `base.height` | 1 | m | base origin above z = 0 |
| `feet.air_time` | 4 | s | time each foot has been off the ground |
| `feet.height` | 4 | m | sole of each foot above z = 0 |
| `joints.torque` | 12 | N m x 0.05 | servo torque estimate, scaled by 0.05 |
| `push.force` | 3 | N x 0.01 | the push on the base now, scaled by 0.01 |

```python
import numpy as np
from alloy.envs.loco import go2_velocity

env = go2_velocity(num_envs=2)
obs, info = env.reset(seed=0)
print(env.obs_groups)                                 # ('policy', 'critic')
print(len(env.obs_names()), len(env.obs_names("critic")))   # 55 79: a label per column
print(env.obs_names("critic")[55:60])
print({k: tuple(v.shape) for k, v in obs.items()})   # {'policy': (2, 55), 'critic': (2, 79)}
obs, reward, terminated, truncated, info = env.step(np.zeros((2, 12), np.float32))
print({k: tuple(v.shape) for k, v in obs.items()})   # a step returns the same two groups
print(env.xp.to_numpy(env.state["base.gravity_b"]).round(2))   # to_numpy: also from a GPU tensor
env.close()
```

`step` and `reset` return `obs` as a dict with both groups; `env.single_observation_space` is a Dict space of the
same two. Give an actor `obs["policy"]` and a value function `obs["critic"]`. How the groups reach an RL library
(rsl_rl takes both, skrl and Stable-Baselines3 take the policy group) is in `rl.md`.

Choose your own lists. An entry is a state key or a `(key, scale)` pair. The critic group is `policy_obs` followed by
`privileged_obs`:

```python
from alloy.envs.loco import Go2VelocityConfig, go2_velocity

config = Go2VelocityConfig(policy_obs=["base.gravity_b", "command", "joints.pos", "joints.vel", "action"],
                           privileged_obs=[("base.height", 3.0)])
env = go2_velocity(num_envs=2, config=config)
obs, info = env.reset(seed=0)
print(tuple(obs["policy"].shape), len(env.obs_names()), len(env.obs_names("critic")))   # (2, 42) 42 43
env.close()
```

### State keys

`env.state[key]` gives the value now, one row per world (W worlds, J = 12 joints, F = 4 feet). Frames: the base frame
has x forward, y left, z up. The world frame has z up and gravity -9.81 m/s^2.

| key | shape | unit | meaning |
|---|---|---|---|
| `base.lin_vel_b`, `base.ang_vel_b` | (W, 3) | m/s, rad/s | base velocity and angular velocity, base frame |
| `base.lin_vel_w`, `base.quat_xyzw` | (W, 3), (W, 4) | m/s | base velocity, world frame; base orientation |
| `base.gravity_b` | (W, 3) | 1 | gravity direction in the base frame |
| `base.height` | (W,) | m | base origin above z = 0 |
| `base.accel_b` | (W, 3) | m/s^2 | accelerometer: the base's mean acceleration over the last step minus gravity |
| `joints.pos`, `joints.vel`, `joints.acc` | (W, J) | rad, rad/s, rad/s^2 | leg joints; 0 is the stand |
| `joints.torque` | (W, J) | N m | kp (target - q) - kd qd, clipped to the motor limit; an estimate, not a solver read |
| `joints.target` | (W, J) | rad | the servo targets of the last step |
| `feet.pos`, `feet.height` | (W, F, 3), (W, F) | m | foot sphere centres; sole above z = 0 |
| `feet.contact`, `feet.touchdown` | (W, F) | 1 | foot touches; 1 on the step a foot lands |
| `body.contacts` | (W,) | 1 | non-foot collision shapes of the robot with a contact force over 1 N (0 with `contact="height"`) |
| `feet.vel` | (W, F, 3) | m/s | foot centres' mean velocity over the last control step, world frame |
| `feet.air_time` | (W, F) | s | time off the ground; on the landing step, the length of the flight that ended |
| `feet.force` | (W, F) | N | contact force (see below) |
| `command`, `action`, `action.prev`, `push.force` | (W, 3), (W, J), (W, J), (W, 3) | | kept by the environment |

The accelerometer reads (0, 0, +9.81) at rest on the reset pose and in free fall reads 0.
After a second of stand it reads about (0.13, 0, 9.81).

Foot contact comes from the rigid solver's contact forces (`contact="force"`, the default of the Go2 environment).
`feet.force` is the upward (world z) force on each foot sphere in newtons, and a foot touches when it is over 1 N. A
standing Go2's four feet add up to its weight (149.2 N for 15.2 kg; front feet 32.4 N, rear feet 42.2 N each). The
value is the force of the last physics step of the control period (1.25 ms of the 20 ms), not an average over it. It
counts every contact on the foot sphere, and none on the shank. The simulation itself does not change with this mode
(bit for bit on the CPU). It costs time: on one L4 with 1,024 worlds a step loop ran 16 % slower with it on.

`Go2VelocityConfig(contact="height")` is the cheaper choice: a foot touches when its sole is within 5 mm of z = 0.
That works on flat ground only and `feet.force` reads zeros. A foot that slides without lifting stays inside the 5 mm
threshold, so it still reads as touching (a standing Go2's soles read 0.1 to 0.2 mm below z = 0).

```python
import numpy as np
from alloy.envs.loco import Go2VelocityConfig, go2_velocity

env = go2_velocity(num_envs=2, config=Go2VelocityConfig(pushes=None))
env.reset(seed=0)
for _ in range(50):
    env.step(np.zeros((2, 12), np.float32))
force = env.xp.to_numpy(env.state["feet.force"])
print(force.round(1), force.sum(1).round(1))          # per foot, and their sum: the robot's weight, about 149 N
env.close()
```

## Actions

An action has 12 numbers, one per leg joint, in this order (`env.action_names()`): `go2/FL_hip_joint`,
`go2/FL_thigh_joint`, `go2/FL_calf_joint`, then the same three for FR, RL and RR.

The servo target in radians is `0.25 * action`, so an action of zero is the stand. Actions are clipped to +-4 and
then to each servo's control range. The servos are position servos from the robot file (kp 50 N m/rad, kd 0.5 N m
s/rad). Ramp the targets: a large jump makes a stiff servo kick.

A held stand sags: the base is 0.29 m up at reset and about 0.26 m after a second with zero actions,
with all four feet down.

## Commands

A command is (forward m/s, left m/s, yaw rate rad/s) in the robot's heading frame. Each world gets one at reset and
a new one every 10 s. The defaults draw uniformly from forward -1 to 1 m/s, left -0.5 to 0.5 m/s and yaw -1 to 1
rad/s. 10 % of draws are (0, 0, 0). A draw with planar speed under 0.1 m/s becomes 0 in x and y.

```python
import numpy as np
from alloy.envs.loco import Go2VelocityConfig, Pushes, VelocityCommands, go2_velocity

config = Go2VelocityConfig(
    commands=VelocityCommands(lin_x=(0.0, 0.8), lin_y=(0.0, 0.0), yaw=(0.0, 0.0), standing=0.0),
    pushes=Pushes(every_s=1.0, duration_s=0.1, force=(60.0, 60.0, 0.0)))
env = go2_velocity(num_envs=2, config=config)
env.reset(seed=0)
print(env.xp.to_numpy(env.commands.value).round(2))   # drawn at reset: forward speeds in 0 to 0.8
env.commands.set((0.8, 0.0, 0.0))                     # fix one command in every world (or an array (2, 3))
zeros = np.zeros((2, 12), np.float32)
peak = 0.0
for _ in range(60):
    obs, reward, terminated, truncated, info = env.step(zeros)
    peak = max(peak, float(abs(env.xp.to_numpy(env.state["push.force"])).max()))
print(peak, env.xp.to_numpy(info["command"]))         # the largest push force seen, and the fixed command
env.commands.release()                                # random commands again from the next draw
print({k: env.xp.to_numpy(v).round(3).tolist() for k, v in list(info["terms"].items())[:3]})
env.close()
```

`VelocityCommands` has `lin_x`, `lin_y`, `yaw`, `resample_s`, `standing` and `deadband`. `info["command"]` is the
command in each world. `env.commands.set` takes a tuple or a NumPy array of shape (worlds, 3); a GPU tensor is refused
with the same `TypeError`, so pass `env.xp.to_numpy(tensor)`.

## Pushes

Every 8 s the base gets a push of 0.1 s: a random force, up to +-80 N along world x and +-80 N along world y, held
at the base's centre of mass. All worlds are pushed at the same moments, each with its own force. In the snippet
above, `Pushes(every_s=1.0, duration_s=0.1, force=(60.0, 60.0, 0.0))` pushes every second. `pushes=None` in the
config turns pushes off. `env.state["push.force"]` is the force now (zeros between pushes). To push by hand, call
`env.pusher.hold(force)` with an array (worlds, 3) in newtons and `env.pusher.clear()` to end it.

A `processes=` build cannot be given to the environment at all: `LocoEnv(sim, ...)` on
`alloy.build(go2_scene(2), worlds=2, device="cpu", processes=2)` raises `FeatureUnavailableError` ("sim.engine is not
available with processes (the engine objects live in the worker processes)"; fix: build with `processes=None`),
pushes or not.

## Rewards

Each step adds `weight * dt * value` for every term except `fell` and `feet_air_time`, which are not scaled by dt
(`dt` is 0.02 s). The weights are in `DEFAULT_REWARD`:

| term | weight | value |
|---|---|---|
| `track_lin_vel` | 2.0 | exp(-(squared error of planar velocity vs command) / 0.25); 1 when exact |
| `track_yaw_rate` | 1.0 | exp(-(yaw-rate error)^2 / 0.25) |
| `lin_vel_z` | -2.0 | vertical base speed squared |
| `ang_vel_xy` | -0.05 | roll and pitch rates squared, summed |
| `flat_orientation` | -5.0 | horizontal part of `base.gravity_b`, squared; 0 when level |
| `torques` | -2e-4 | sum of squared servo torques |
| `joint_acc` | -2.5e-7 | sum of squared joint accelerations |
| `action_rate` | -0.01 | sum of squared change of the action since the last step |
| `feet_air_time` | 1.0 | on each landing, flight time (at most 0.5 s) minus 0.2 s, summed over feet; 0 when the command is to stand |
| `feet_slide` | -0.25 | squared planar speed of every foot that touches the ground, summed over feet |
| `body_contacts` | -5.0 | how many non-foot collision shapes of the robot (base, hips, thighs, shanks) touch something |
| `pose_deviation` | -0.1 | sum of squared leg joint angles: distance from the stand |
| `hip_deviation` | -1.0 | sum of squared hip (sideways) joint angles |
| `base_height` | -50.0 | squared error of the base height against 0.28 m |
| `feet_too_high` | -10.0 | how far each sole is above 0.10 m, summed over feet |
| `stand_still` | -0.5 | sum of the joint angles' absolute values while the command is to stand |
| `fell` | -2.0 | 1 on the step the robot falls |

```python
import numpy as np
from alloy.envs.loco import DEFAULT_REWARD, Go2VelocityConfig, go2_velocity

reward = dict(DEFAULT_REWARD, torques=0.0, fell=-5.0)      # a weight of 0 drops the term
config = Go2VelocityConfig(reward=reward, episode_s=1.0, pushes=None)
env = go2_velocity(num_envs=2, config=config)
env.reset(seed=0)
to_numpy = env.xp.to_numpy                                  # a NumPy copy of an env value, also from a GPU tensor
zeros = np.zeros((2, 12), np.float32)
for i in range(51):
    obs, rew, terminated, truncated, info = env.step(zeros)
    if "_episode" in info:                                  # these worlds ended and were reset
        print("step", i + 1, "truncated", to_numpy(truncated), "terminated", to_numpy(terminated))
        print("return", to_numpy(info["episode"]["r"]), "length", to_numpy(info["episode"]["l"]))
        break
print("torques" in info["terms"])                           # False: dropped
env.close()
```

A name that is not in `DEFAULT_REWARD` raises `EnvError` and lists the valid names. `info["terms"]` is a dict from
term name to its value in each world (unscaled by weight).

## Episode ends

- Fall: the base is below 0.15 m, or tilted more than 60 degrees from upright. The world reports `terminated`.
- Time: after 20 s (`episode_s`, 1000 steps) the world reports `truncated`.

Finished worlds reset inside `step` as `envs.md` describes (`info["final_obs"]`, `info["episode"]`). At each reset
the robot is back in the stand and the world gets a new command. `Go2VelocityConfig(joint_jitter=0.05)` starts every
leg joint within +-0.05 rad of the stand instead (off by default: a bent leg can start with its foot in the floor).

## Randomizing the robot

`Go2VelocityConfig(scene_randomize=[(name, field, low, high), ...])` passes each entry to
`scene.randomize(name, field, low, high)` before the build, so every world gets its own value. The environment adds
no fields of its own: an entry works when `scene.randomize` accepts it for the robot in your installed version, and
raises that call's error (which names the valid fields) when it does not. Nothing is randomized by default.
Robot fields are described in `randomize_robots.md`:

```python
import numpy as np
from alloy.envs.loco import Go2VelocityConfig, go2_velocity

config = Go2VelocityConfig(pushes=None, scene_randomize=[
    ("go2", "mass_scale", 0.8, 1.2),       # every link's mass, one draw per world
    ("go2", "friction", 0.5, 1.1),         # the robot file's geoms and its floor
    ("go2", "kp_scale", 0.9, 1.1)])        # servo stiffness
env = go2_velocity(num_envs=4, config=config)
env.reset(seed=0)
for _ in range(50):
    env.step(np.zeros((4, 12), np.float32))
print({k: np.asarray(v).round(2).tolist() for k, v in env.sim.randomized().items()})
print(env.xp.to_numpy(env.state["feet.force"]).sum(1).round(1))    # each world's weight: 149.2 N x its mass_scale
env.close()
```

Scenes with randomized values rebuild the GPU's captured step at every reset, so training with them runs
slower; we have not measured how much.

## Training a walk

Needs PyTorch and rsl_rl (`pip install rsl-rl-lib`), and in practice a GPU:

```bash
python -m alloy.envs.loco.train --num-envs 4096 --iterations 400 --out runs/go2
```

It trains PPO through `alloy.envs.wrappers` (`rl.md`): the actor sees the policy group, the critic the critic
group. Every 25 iterations it prints one JSON line (episode return, episode length in steps, the share of episodes
that ended in a fall, each reward term's mean) and appends it to `runs/go2/curve.jsonl`; the checkpoint is
`runs/go2/model_last.pt`. At the end it rolls the policy out under six fixed commands and prints the speed reached.
From Python: `from alloy.envs.loco.train import train, evaluate, ppo_config`.

What we measured with that command on one NVIDIA L4, 4,096 worlds (98,304 samples per iteration), 400 iterations,
two seeds (`--seed 0` and `--seed 1`):

- Speed: 2.3 s per iteration; 400 iterations took 15.3 and 15.2 minutes.
- Learning:

| iteration | episode return (seed 0 / seed 1) | episodes ended by a fall | `track_lin_vel` mean |
|---|---|---|---|
| 25 | -19.7 / -19.3 | 33 % / 30 % | 0.36 / 0.36 |
| 100 | 35.7 / 33.9 | 1.8 % / 3.4 % | 0.88 / 0.86 |
| 200 | 59.9 / 59.6 | 0.2 % / 0.3 % | 0.95 / 0.94 |
| 300 | 63.6 / 63.5 | 0.5 % / 0.6 % | 0.95 / 0.95 |
| 400 | 66.6 / 65.5 | 0.3 % / 0.4 % | 0.95 / 0.95 |

- The end-of-run evaluation (4,096 worlds, 6 s per command, pushes on, mean over the last 4.5 s):

| command | reached, seed 0 | reached, seed 1 | worlds that fell (of 4,096) |
|---|---|---|---|
| forward 0.5 m/s | 0.49 | 0.50 | 0 / 0 |
| forward 1.0 m/s | 0.99 | 0.98 | 0 / 0 |
| left 0.4 m/s | 0.39 | 0.40 | 0 / 18 |
| yaw 1.0 rad/s | 1.02 | 1.01 | 0 / 0 |
| stand | 0.00 | 0.00 | 0 / 0 |
| backward 0.5 m/s | 0.50 | 0.50 | 0 / 0 |

  In every row no part of the robot other than its feet touched the ground (`non_foot_contacts` 0.0).

Read these numbers with their limits:

- The gait: we looked at rendered frames of both policies. The trunk stays level and the legs stay under it while
  it walks forward, sideways and turns. The feet are off the ground 74-80 % of the time while moving, so about
  one foot is down at a time: a quick, light-footed gait, not a tuned trot. Nothing rewards a particular footfall
  pattern.
- Two runs of 400 iterations. We have not measured how often a run fails to learn, nor longer training.
- Earlier reward weights produced policies that tracked the command while sliding without lifting their feet,
  walking on their shanks, standing still, or kicking the hind legs above the trunk. `feet_slide`,
  `body_contacts`, `pose_deviation`, `hip_deviation`, `base_height` and `feet_too_high` are there to stop
  those. If you change the weights, render the result (`camera.md`) and read `feet_off_ground_share` and
  `non_foot_contacts` in the evaluation lines; the return alone does not show such a gait.
- No policy trained here has been run on a real robot.

### Playing a trained policy

Needs PyTorch and rsl_rl, as training does. Build the environment again, load the checkpoint into an rsl_rl runner on
it, take the runner's inference policy and step:

```python
import torch
from rsl_rl.runners import OnPolicyRunner
from alloy.envs.loco import go2_velocity
from alloy.envs.loco.train import ppo_config, train
from alloy.envs.wrappers import wrap

checkpoint = train(num_envs=4, iterations=2, out="runs/demo")["checkpoint"]   # a 2-iteration stand-in that has not
                                                                             # learned: use "runs/go2/model_last.pt"
env = go2_velocity(num_envs=4, backend="torch")            # the GPU if there is one; device="cpu" forces the CPU
venv = wrap(env, "rsl_rl")
cfg = ppo_config(1, 0)
cfg["obs_groups"] = venv.default_obs_groups()
runner = OnPolicyRunner(venv, cfg, log_dir=None, device=str(venv.device))
runner.load(checkpoint, map_location=str(venv.device))     # map_location: onto this machine's device
policy = runner.get_inference_policy(device=str(venv.device))
env.commands.set((0.6, 0.0, 0.0))                          # forward at 0.6 m/s in every world
obs, _ = venv.reset()
for _ in range(250):                                       # 5 s
    with torch.no_grad():
        obs, reward, dones, extras = venv.step(policy(obs))
print(env.xp.to_numpy(env.state["base.lin_vel_b"])[:, 0].round(2))    # forward speed in each world, m/s
env.sim.render(width=640, height=480, path="go2.png")      # a picture of world 0 (its collision shapes)
venv.close()
```

- The `train(...)` line makes a 2-iteration stand-in so that the snippet runs by itself; it has learned nothing and
  prints speeds near 0. For your own run, drop that line and pass `"runs/go2/model_last.pt"` to `runner.load`.
- `map_location` puts the checkpoint on this machine's device. Without it torch stops when a checkpoint trained on a
  GPU is loaded on a machine that has none.
- `runner.load` prints the network's layers. `sim.render` draws world 0's collision shapes; call it in the loop with
  a new `path` for a sequence of frames (`camera.md`).
- Run as written on the CPU only (about 2 min on an AMD EPYC 7B13, most of it the stand-in's training and
  evaluation). Not run by us: this snippet on a GPU, and a GPU checkpoint loaded on a CPU machine.

## Rough terrain

`from alloy.envs.loco import go2_rough` is the same velocity task on a field of terrain tiles. The field is one
heightfield, shared by every world's copy of the scene. It has `levels` x kinds tiles: by default 5 levels along x and
3 kinds along y, each tile 6 m square with 0.1 m cells, so 30 m x 18 m of tiles, centred on the origin, with 4 m of
flat ground around them (`border`). The kinds are `"rough"` (a random height per cell, up to 0.06 m at the last
level), `"waves"` (smooth bumps 0.15 m high and 2 m long) and `"blocks"` (a checkerboard of 0.5 m squares up to
0.08 m high). The sizes grow linearly with the level: level 0 is flat and the last level has the full size. Every
tile has a flat 1 m start platform in its middle and falls back to height 0 at its border, so tiles join without
cliffs.

Every world's robot starts on the platform of one tile: world `w` takes kind `w % 3` and a level from 0 to
`start_levels - 1` (default 2, so levels 0 and 1). When a world's episode ends, its level changes for the next one:

- up one level if the robot ended more than half a tile (3 m) from its start;
- down one level if it covered less than half the distance its command asked for (the command's planar speed times
  the episode time);
- otherwise unchanged; a robot that clears the last level goes to a random level.

`env.levels` is each world's level and `env.state["terrain.level"]` the same as a state key (a float).
`Go2RoughConfig(curriculum=False)` freezes the levels, and `start_levels=` sets how many levels the first episodes
use. An episode also ends, as a termination, when the robot leaves the field (its base comes within 0.3 m of the
edge). `base.height` and `feet.height` are measured above the terrain under each point (bilinear between grid nodes),
not above z = 0. The solver's heightfield surface is made of triangles, so between nodes it differs slightly from that
bilinear value.

The policy walks blind: there is no height scan of the ground ahead in the observations, which are the same 55 and 79
numbers as on flat ground. The reward is the flat task's.

```python
import numpy as np
from alloy.envs.loco import Go2RoughConfig, TerrainSpec, go2_rough

def host(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

config = Go2RoughConfig(pushes=None, start_levels=3, terrain=TerrainSpec(levels=3, tile=4.0))
env = go2_rough(num_envs=9, config=config)       # 9 worlds: 3 levels x 3 kinds, one robot per tile
env.reset(seed=0)
for _ in range(50):                              # 1 s holding the stand
    env.step(env.xp.zeros((9, 12)))
t = env.terrain
print("field", t.heights.shape, "levels", host(env.levels), "kinds", host(env.kinds))  # (201, 201) [0 0 0 1 1 1 2 2 2] [0 1 2 0 1 2 0 1 2]
pos = host(env.state["go2/base.pos"])
centre = np.array([t.tile_center(float(l), float(k)) for l, k in zip(host(env.levels), host(env.kinds))])
print("base minus tile centre (xy), m:", (pos[:, :2] - centre).round(2).tolist())
print("base.height:", host(env.state["base.height"]).round(3))
print("feet force sum, N:", host(env.state["feet.force"]).sum(1).round(1))
health = env.sim.health()
print("DEGRADED worlds:", health["degraded_worlds"], "flag of world 0:", health["worlds"][0]["flags"])
print("check ok:", env.sim.check()["ok"])
env.close()
```

On 9 worlds (3 levels x 3 kinds, tiles of 4 m) on an AMD EPYC 7B13 this prints: a (201, 201) field, every robot 0.03 m
behind its tile centre in x, `base.height` 0.264 m and a feet force sum of 149.2 N (the robot's weight) in all nine,
and `check ok: True`. Eight of the nine worlds list the notice `FAR_FROM_ORIGIN`: `sim.health()` lists it for a robot
more than 4 m from the origin, and these robots stand 4 m or more away. It leaves those worlds OK (it is a notice, not
DEGRADED), so on any field wider than 8 m read the other flags. Run on the CPU with NumPy only;
not run with PyTorch.

A different field:
`Go2RoughConfig(terrain=TerrainSpec(levels=3, kinds=("rough", "waves", "blocks"), roughness=0.04))`.
`levels` and the number of kinds must be odd, so that the middle tile is centred on the origin, where the robot file
stands. `TerrainSpec` also has `tile`, `cell`, `platform`, `wave_height`, `wave_length`, `block_height`, `block_size`
and `seed`. `Terrain(spec)` holds the grid: `.heights` (metres, `heights[j, i]` at x = x0 + i cell, y = y0 + j cell),
`height_at(xp, xy)` (the ground height under points, bilinear, 0 outside the field; `xp` is an environment's `env.xp`)
and `tile_center(level, kind)`. `terrain_mjcf(path, terrain)` writes a copy of another robot file whose floor plane,
a geom named `"floor"`, is replaced by the field.

```python
import numpy as np
from alloy.envs.backend import NumpyBackend
from alloy.envs.loco import Terrain, TerrainSpec

terrain = Terrain(TerrainSpec(levels=3, tile=4.0))
print("heights", terrain.heights.shape, "cell", terrain.cell, "half extents", terrain.half)
for level in range(3):
    for kind, name in enumerate(terrain.spec.kinds):
        n = int(round(terrain.spec.tile / terrain.cell))      # cells per tile edge
        j, i = terrain.pad + kind * n, terrain.pad + level * n     # the tiles start after the border's cells
        tile = terrain.heights[j:j + n + 1, i:i + n + 1]
        print(f"level {level} {name:6s} highest {tile.max():.3f} m  centre {terrain.tile_center(level, kind)}")
xp = NumpyBackend("cpu")                  # height_at takes a backend: an environment's is env.xp
xy = np.array([[4.0, 0.0], [4.5, 0.0], [5.0, 0.0], [5.05, 0.0], [20.0, 0.0]], np.float32)   # across a wave of level 2
print(terrain.height_at(xp, xy).round(4))
```

This prints a (201, 201) grid, 0.000 m for every kind of level 0, and the highest point of level 1 and level 2 as
0.030 / 0.075 / 0.038 m and 0.060 / 0.150 / 0.080 m (rough / waves / blocks). Across the level-2 wave the height is
0 at x = 4.0 and 4.5, 0.15 m at 5.0 and 0.1482 m at 5.05.

Limits of this terrain:

- A "blocks" side is a ramp one cell (0.1 m) wide, not a vertical step. There are no true stairs or vertical edges,
  no gaps and no tiles that are only slopes.
- The robot sees no heights, so it cannot plan a step; it can only learn to cope with what it feels.
- `contact="height"` is for flat ground only (see above); the rough environment uses the default `contact="force"`.
- Random actions can drive a part deep into the field. With random actions instead of a policy (96 worlds, 1,000
  steps, one L4), 6 of 13 runs ended with 1 or 2 worlds INVALID (PEN_FAIL: a part 2 to 18 cm into the ground), most
  of them on level-0 tiles; 9 runs on the flat environment had none. Health flags such a world, its
  `info["invalid"]` is true and the environment resets it, and the observations stayed finite. The cause is not
  found.

Training on it, on a GPU:

```bash
python -m alloy.envs.loco.train --terrain rough --num-envs 4096 --iterations 300 --out runs/go2_rough
```

The printed lines then also carry `terrain_level`, the mean level of the worlds. A hand-written trot on a three-level
field, with the curriculum moving the robots, is `examples/envs/go2_walk/rough_terrain.py`.

What one run reached (seed 0, 4,096 worlds, 300 iterations, 29.5 M samples, 21-71 minutes in two runs on one L4; the
default field of five levels, robots starting on levels 0 and 1):

| iteration | mean terrain level | episodes that ended in a fall | `track_lin_vel` term mean |
|---|---|---|---|
| 25 | 0.20 | 22 % | 0.36 |
| 50 | 0.09 | 11 % | 0.58 |
| 100 | 0.66 | 8 % | 0.89 |
| 200 | 1.31 | 4 % | 0.94 |
| 300 | 1.56 | 5 % | 0.94 |

The level first falls (robots that cannot walk yet are moved down), then rises as the walk appears. At the end, with
each robot on the tile the curriculum left it on and pushes on, the policy reached 0.50 m/s on a 0.5 m/s command, 1.02
on 1.0, 0.38 sideways on 0.4, 0.99 rad/s on 1.0 and 0.48 backward on 0.5; in the worst of the six 6 s rollouts under
1 % of worlds fell (1 and 24 of 4,096 in two runs), and no part other than the feet touched the ground in more than
0.1 % of samples.

Read this with its limits: one run, one seed. The mean level was still rising at 1.56 of a top level of 4, so most
robots were on terrain of a quarter to a half of the full height (level 1: 1.5 cm roughness, 3.75 cm waves, 2 cm
blocks); we have not trained long enough to say whether this reward reaches the top level. We have not looked at
rendered frames of this policy, only at its numbers (feet off the ground 75-81 % of the time while moving, as on flat
ground). A terrain field steps slower than a plane: the slower of the two runs took about 11 s per iteration from
iteration 50 on (47.5 minutes for 250 iterations; the first 50, with the first-run compile, took 24 of its 71
minutes), the faster, on another L4 machine, about 4 s.

## Your own legged robot

`LeggedRobot` names the base body, the leg joints, the feet (`Foot`: body, offset, radius), the servo gains, the
torque limits and the stand height. `LocoConfig` holds the robot, its scene name, the commands, the pushes and the
action scale. `LocoEnv(scene, task, config, worlds=...)` builds the environment. This runs with the Go2 pieces; for
your robot, use your own scene (`Scene(worlds=2, dt=0.02)` and `scene.add_robot(...)`), a `Task` and your names.
`go2_velocity_task()` already carries the two observation groups; build your own task the same way, with
`Task(obs={"policy": [...], "critic": [...]})`.

```python
import numpy as np
from alloy.envs.loco import GO2, Foot, LeggedRobot, LocoConfig, LocoEnv, go2_scene, go2_velocity_task

robot = LeggedRobot(
    base="base",
    joints=GO2.joints,                                   # 12 names, in action order
    feet=tuple(Foot(f"{leg}_calf", (-0.002, 0.0, -0.213), 0.022) for leg in ("FL", "FR", "RL", "RR")),
    kp=50.0, kd=0.5, torque_limit=GO2.torque_limit, stand_height=0.288)
scene = go2_scene(worlds=2)
env = LocoEnv(scene, go2_velocity_task(), LocoConfig(robot=robot, name="go2"), worlds=2)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(np.zeros((2, 12), np.float32))
print({k: tuple(v.shape) for k, v in obs.items()}, env.action_names()[:2])
env.close()
```

## What is and is not covered

Covered:
- One robot, the Go2, on a flat floor at z = 0, or on a heightfield of rough, wavy and block tiles with a difficulty
  curriculum (`go2_rough`, "Rough terrain").
- Velocity commands (forward, left, yaw rate), redrawn on a timer and at reset.
- Pushes on the base, on a timer, the same moments in every world.
- Position-servo actions around the stand.
- The reward terms in `DEFAULT_REWARD` and the two observation groups.
- Foot contact from the solver's contact forces, or from foot height (`contact="height"`).
- Your own robot through `LeggedRobot` and `LocoEnv`.

Not covered:
- Ground other than a flat floor and the heightfield tiles: no vertical steps or stairs, gaps or pure slopes, and no
  height scan of the ground in the observations.
- Curricula other than the terrain level rule (no command or push curricula).
- Other robots ready-made (G1, H1).
- Actuator models: motor torque-speed curves and learned actuators.
- Observation noise, action delay and observation history.
- Per-world push timing.
- Heading commands.
- Foot contact from height (`contact="height"`) on anything but flat ground at z = 0 (the rough environment uses
  force).
- Contact forces averaged over a control period (the value is the last physics step's).
- A policy trained in this environment has not been run on a real robot.
