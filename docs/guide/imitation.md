# Imitation learning: record, export, teleoperate, replay, evaluate

`alloy.il` records demonstrations from a [Gym environment](envs.md) (`alloy.envs.VectorEnv`), writes them as
datasets other tools read (HDF5 in the robomimic layout, LeRobotDataset v3.0), lets a person drive a robot from the
keyboard, replays recordings, and measures a policy's success rate. `import alloy` does not load it: write
`from alloy.il import ...`.

```python
import numpy as np
import alloy.envs
from alloy.il import MemoryWriter, Recorder, evaluate, replay

env = alloy.envs.make("GripperLift-v0", num_envs=4)       # GPU if there is one, else CPU

def policy(obs):                                          # a scripted demonstrator: squeeze, then lift
    act = np.tile(np.array([0.0, 1.0, 1.0], np.float32), (env.num_envs, 1))
    act[env.xp.to_numpy(env.state["steps"]) >= 25, 0] = 1.0
    return env.xp.asarray(act)

writer = MemoryWriter()                                   # keeps episodes in a list; file writers are below
rec = Recorder(env, [writer], task="lift the cube and hold it", keep="success")
obs, info = rec.reset(seed=0)
while rec.saved < 8:                                      # step through the recorder instead of the env
    obs, reward, terminated, truncated, info = rec.step(policy(obs))
rec.close()

ep = writer.episodes[0]                                   # one episode of one world, NumPy arrays, time first
print(ep.length, ep.success, ep.obs["state"].shape, ep.actions.shape)       # 150 True (150, 22) (150, 3)
print(replay(env, writer.episodes).summary())             # restore each start, apply the actions, compare
print(evaluate(env, policy, episodes=16).summary())       # success rate with a 95 % interval
```

Runnable scripts in the repository (they need `pip install h5py`): `examples/il/01_record_demos.py` (scripted demos
into an HDF5 file; `--camera --lerobot` adds images and a LeRobot dataset), `examples/il/02_teleop_keyboard.py` (drive
the gripper yourself; `--script demo` plays a key script without a terminal), `examples/il/03_replay_evaluate.py`
and `examples/il/04_generate_demos.py` (thousands of scripted demos).

## What to install

| For | Install | Without it |
|---|---|---|
| `Recorder`, `MemoryWriter`, `replay`, `evaluate`, keyboard teleop | nothing more than Alloy | |
| `HDF5Writer`, `load_hdf5` | the `il` extra or `pip install h5py` | `FeatureUnavailableError` naming the pip command |
| `LeRobotWriter` | the `lerobot` extra (Python 3.12) or `pip install 'lerobot[dataset]'` (tested: 0.7.0) | `FeatureUnavailableError` naming the pip command |
| `SpaceMouseTeleop` | `pip install hidapi` | `FeatureUnavailableError` naming this command |

Torch is optional, as for the envs: with it observations are torch tensors on the simulation device, and the
recorder copies each step to host NumPy arrays.

## Recording

`Recorder(env, writers, task=None, keep="success", states=True, max_episodes=None, env_kwargs=None,
manual=False)` wraps the env from outside. The env is a `VectorEnv` (one built from a `Task`): a task-port env
from `alloy.envs.make` or a Gymnasium wrapper around a `VectorEnv` raises `EnvError` here and in `evaluate`,
`generate` and `replay`; record the `VectorEnv` itself and apply a wrapper's effect in your policy. Call
`rec.reset(...)` and `rec.step(actions)` where you would call the env's; they return what the
env returns. The env and the simulation are not changed: a run steps bit for bit the same with or without the
recorder (a test holds this on the CPU).

- Each world records one episode at a time. When a world's episode ends (the task ended it, or `max_steps`), it goes
  through `keep` and then to every writer: `keep="success"` keeps episodes whose last step passes the task's success
  check, `"all"` keeps every one, a function `keep(episode) -> bool` decides itself. `rec.saved` and `rec.dropped`
  count them; `max_episodes=n` stops saving after n (`rec.full`).
- `task` is the instruction text stored with the dataset ("lift the cube"); it defaults to the env's name.
- All three autoreset modes of the env work. With `"next_step"` the step after an episode's end (whose action the
  env ignores) is in no episode; with `"disabled"` a world records again after you reset it by mask through
  `rec.reset(options={"reset_mask": m})`.
- `rec.finish(save=True)` ends every world's running episode now, writes it (its last step is marked truncated),
  resets the env and returns the new observation. Teleoperation uses it: the operator decides when a demonstration
  is done, so it writes an episode that has not passed the success check under `keep="success"` too. A `keep`
  function still decides, and an INVALID episode is written only with `keep="all"`.
- `manual=True` (which `teleop_loop` sets while it runs) leaves the saving to `finish`: an episode the env itself
  ends is kept only when it passed the success check, so an attempt that ran into `max_steps` is dropped.
- Cost: in one run on an NVIDIA L4 (GripperLift-v0, 64 worlds, a scripted policy) stepping through the recorder ran
  84 control steps per second against 92 without it.
- Episodes are held in memory until they end: `worlds x steps x observation size`. With camera observations that
  is `worlds x steps x H x W x 3` bytes per camera; lower the world count for long episodes with large images.
  An episode's `obs` and `next_obs` are two views of one `(T + 1, ...)` array per key. A `MemoryWriter` keeps
  every episode it is given, so with it the same amount is held again; the file writers do not keep them.

An `Episode` has `obs[key]` (what the policy saw before each action), `next_obs[key]` (after it; for the last step
the observation the episode ended on, before the reset), `actions`, `rewards`, `terminated`, `truncated`, `dones`,
`success`, `invalid`, `states`, `start`, `world` and `index`. `obs["state"]` is the env's state vector; a task with
cameras adds one `uint8 (T, H, W, 3)` array per camera, under the camera's name. With observation groups
(`Task(obs={"policy": [...], "critic": [...]})`) there is one vector per group under the group's name and no
`"state"`; `spec.state_key` names the main one (`"state"`, or the first group). `spec.obs_key_names[key]` labels the
columns of each vector as the policy gets them, after the task's layers (`ObsHistory(3)` gives three labels per
column, `name@t-2` to `name@t`), and `spec.obs_names` is the main vector's.
`actions` are the policy's actions as the env clips them: in the default `"delta"` action mode -1 to 1 per joint
(a step of the joint's target, not a joint target); in `"absolute"` mode the `ActionSpec`'s `low` to `high`.

`states` (one row per step) and `start` are the simulator state of that world: joint coordinates and velocities
(free rigid bodies are joints there), the control targets and the task's goal points. **Cloth, liquid and other
particle objects are not in them.** `states=False` skips them (a little faster; such a recording cannot be
replayed).

## HDF5 in the robomimic layout

```py
from alloy.il import HDF5Writer, Recorder, load_hdf5      # needs: pip install h5py

rec = Recorder(env, [HDF5Writer("demos/lift.hdf5", overwrite=True)], task="lift the cube and hold it")
obs, info = rec.reset(seed=0)                             # env and policy: the first snippet's
while rec.saved < 8:
    obs, reward, terminated, truncated, info = rec.step(policy(obs))
rec.close()
spec, episodes = load_hdf5("demos/lift.hdf5")             # back as Episode objects, for replay
```

The file is readable after every episode (a run that crashes keeps what it wrote), and
`HDF5Writer(path, resume=True)` adds to a file an earlier run wrote (same observations and actions).
`next_obs_images=False` leaves the camera frames out of `next_obs` (they repeat `obs` one step later), which halves
the image data; keep the default for robomimic algorithms that read `next_obs` images. Each demo also stores an
`invalid` attribute. `copy_hdf5(path, writer)` writes a file's episodes to another writer, one at a time.

| In the file | Holds |
|---|---|
| `data` attributes `total`, `env_args` | sample count; JSON with `env_name`, `type` 2 (robomimic's gym type), `env_kwargs` |
| `data/demo_<i>` attribute `num_samples` | steps T of that episode (also `success`, `world`, `episode_return`) |
| `data/demo_<i>/actions`, `rewards`, `dones` | `(T, A)` float32, `(T,)` float32, `(T,)` with 1 on the last step |
| `data/demo_<i>/obs/<key>`, `next_obs/<key>` | `state`: `(T, D)` float32; a camera: `(T, H, W, 3)` uint8, gzip |
| `data/demo_<i>/states` | `(T, S)` float32 simulator state (when recorded) |
| `mask/all`, `mask/success` | robomimic filter keys: the demo names |
| `data/demo_<i>/terminated`, `truncated`, `start/<key>`; `data` attribute `alloy_spec` | Alloy's additions (robomimic ignores them): what `load_hdf5` and `replay` need |

What is tested: robomimic 0.3.0's own `SequenceDataset`, `get_env_metadata_from_dataset` and
`get_shape_metadata_from_dataset` read such a file, also through the `success` filter key. What is not: a robomimic
training run on it, and robomimic's rollout evaluation, which needs a robomimic env wrapper for Alloy that does not
exist: evaluate with `alloy.il.evaluate`. Observation keys are the env's (`state`, camera names); list them in your
robomimic config's observation spec. `load_hdf5` reads files written by `HDF5Writer`, not other robomimic files.

## LeRobotDataset v3.0

```py
from alloy.il import LeRobotWriter, Recorder              # needs: pip install 'lerobot[dataset]', Python 3.12

writers = [LeRobotWriter("demos/lift_lerobot", repo_id="local/lift", robot_type="gantry_gripper")]
rec = Recorder(env, writers, task="lift the cube and hold it")
obs, info = rec.reset(seed=0)                             # env and policy: the first snippet's
while rec.saved < 4:
    obs, reward, terminated, truncated, info = rec.step(policy(obs))
rec.close()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/lift", root="demos/lift_lerobot")
print(ds.num_episodes, ds.num_frames, ds.fps, ds[0]["observation.state"].shape)
```

The dataset is written through `lerobot`'s own `LeRobotDataset.create / add_frame / save_episode / finalize`, on
local disk only (nothing is sent to the Hub). Per frame: `observation.state`, `action`, one
`observation.images.<camera>` per camera (an MP4 video, lossy; `video=False` stores exact PNG images),
`observation.<key>` for any other float vector, `next.reward`, `next.done`, `next.success` and the task text.

- With observation groups the first group (`policy`) is `observation.state` and each other group is
  `observation.<group>`, with its column labels. A task with cameras and no state columns has no
  `observation.state`. A `/` in a camera or group name is stored as `_` (`gripper/wrist` becomes
  `observation.images.gripper_wrist`); two names that end up equal raise `EnvError`.
- LeRobot's ACT and Diffusion policy configs refuse a dataset that has neither an image nor
  `observation.environment_state`. `LeRobotWriter(..., env_state=["cube"])` moves the state columns whose label
  starts with one of those names into `observation.environment_state` and leaves the rest in `observation.state`
  (a test holds that `ACTConfig` then accepts a state-only dataset; no policy was trained on one).
- **The dataset is readable only after `close()`.** A recording killed before `close()` leaves a directory that
  `LeRobotDataset` cannot open (it then asks the Hub and fails); `overwrite=True` replaces such a directory. For a
  long run, record with `HDF5Writer` and make the LeRobot dataset at the end:
  `copy_hdf5("demos/lift.hdf5", LeRobotWriter("demos/lift_lerobot", repo_id="local/lift"))`.
  `LeRobotWriter(..., resume=True)` appends to a closed dataset with the same features and fps, and raises
  `EnvError` for one that was not closed. A writer closed with no episode removes its empty directory.
- `lerobot` 0.7.0 needs Python 3.12. On Python 3.11 the `lerobot` extra installs nothing and the writer raises
  `FeatureUnavailableError` saying so: record with `HDF5Writer` there.

Tested with lerobot 0.7.0: three episodes read back with the right counts, fps, version string `v3.0`, exact state
and action values, and images within the video codec's error. Not covered: `next_obs`, `states` and `start` are
not in the LeRobot layout (use the HDF5 file for replay; a `Recorder` takes both writers at once); observation
keys that are neither images nor float vectors are skipped with a warning; the control rate must be a whole number
of steps per second (`dt=0.01` is 100); reading a dataset with videos can print "TorchCodec could not load the
FFmpeg libraries ... Falling back to 'pyav'", a `lerobot` / `torchcodec` message: the read still worked in our test
(lerobot 0.7.0); training a LeRobot policy on the dataset and running `lerobot-eval` against
an Alloy env are not tested and need glue of your own.

## Teleoperation

`KeyboardTeleop` reads keys from the terminal, so it works over ssh and needs no display. Each action column gets a
key pair; `teleop_loop` paces a `Recorder` to the env's control rate (real time when the simulation keeps up; a slow
scene runs slower than real time), one world or many (every world gets the same action).

```py
from alloy.il import HDF5Writer, KeyboardTeleop, Recorder, teleop_loop

env = alloy.envs.make("GripperLift-v0", num_envs=1, max_steps=3000)      # 30 s per attempt
device = KeyboardTeleop(env.action_names(), dt=env.sim.dt,
                        link={"gripper/close_left": ["gripper/close_right"]})     # one key pair, both fingers
print(device.help())                                      # w/s: lift up/down, a/d: close/open
rec = Recorder(env, [HDF5Writer("demos/teleop.hdf5")], keep="all")
rec.reset(seed=0)
teleop_loop(rec, device)                                  # Enter: save episode, Backspace: discard, Esc: quit
rec.close()
```

- Default keys, by action column: `w/s`, `a/d`, `q/e`, `r/f`, `t/g`, `y/h`, `u/j`, `i/k`, `o/l` (first key +1,
  second -1); `bindings={"key": (column, value)}` sets your own, and more than 9 columns need it. Space zeroes every
  column.
- A terminal sends no key-up events, so a pressed key drives its column for `hold` seconds (0.15 by default) after
  its last press; holding the key down keeps it driven through the keyboard's auto-repeat.
- What you drive is the env's action columns. On a joint-action env those are joint targets nudged up or down, so
  a 7-joint arm is driven joint by joint, and gantry-style robots such as the GripperLift gripper drive naturally.
  For hand motion, build the env with hand-motion actions (`alloy.envs.diffik.EEActionSpec`, or the ready-made
  `alloy.envs.diffik.panda_reach_ee()`): its columns are `ee_dx`, `ee_dy`, `ee_dz` (and `ee_rx`, `ee_ry`, `ee_rz`
  with `orientation=True`), and the keyboard or a SpaceMouse drives them like any other column. Those actions are
  an opt-in first version, tested on the CPU: one body, no collision avoidance (`docs/guide/rl.md`).
- **There is no live view in `teleop_loop`.** `examples/il/02_teleop_keyboard.py` prints a status line and can write a
  preview image every few steps (`--preview look.png`) for an image viewer that reloads.
- **Only what you approve is saved.** While `teleop_loop` runs, an episode the env ends by itself is kept only
  when it passed the task's success check; an attempt that runs into the task's `max_steps` without Enter is
  dropped. Enter saves the running episode as it is (with `keep="success"` too; a `keep` function still decides),
  Backspace and Esc drop it. The returned `saved` and `discarded` count episodes, one per world, the ones
  running at Esc included. Build the env with a `max_steps` long enough for one demonstration.
- The snippet above needs a real terminal (it raises `EnvError` when the input is a pipe, a notebook or a job
  without one). Without a terminal, play keys from a string:
  `KeyboardTeleop(env.action_names(), dt=env.sim.dt, keys=ScriptKeys.parse("a*30 w*60 enter"))` (`ScriptKeys` is in
  `alloy.il`; tests and unattended runs use it).
- `SpaceMouseTeleop(action_names, axes={"z": ("gripper/lift", 1.0)}, buttons={...})` maps a 3Dconnexion mouse's six
  axes and two buttons to action columns (both buttons together: save). `link=` couples columns as the keyboard's
  does (one button, both fingers). The mouse has no button left for discard or quit: pass `keys=TerminalKeys()` and
  Enter, Backspace, Esc and Space work as with the keyboard; without `keys` the loop ends with Ctrl-C or
  `max_steps` and nothing can be discarded (`device.help()` prints which of the two applies). Its report parser is
  tested on synthetic reports; **the device has not been run on SpaceMouse hardware.** Linux needs read access to the HID
  device.

## Generating demonstrations with a script

`generate(env, policy, writers, episodes)` runs a scripted policy in every world until `episodes` episodes are
saved, then closes the writers. It is the `Recorder` loop above, with the counting done for you.

```python
import alloy.envs
from alloy.il import MemoryWriter, Phase, PhaseScript, demonstrator, generate

env = alloy.envs.make("GripperLift-v0", num_envs=4)
writer = MemoryWriter()                                   # or HDF5Writer(...), LeRobotWriter(...), several at once
result = generate(env, demonstrator(env, noise=0.1), [writer], episodes=8, seed=0)
print(result.saved, result.attempted, result.success_rate, result.interval)     # 8 8 1.0 (0.67..., 1.0)

script = PhaseScript(env, [                               # your own demonstrator: phases, per world
    Phase([0.0, 1.0, 1.0], steps=25, name="squeeze"),     # fixed action columns for 25 steps
    Phase([1.0, 1.0, 1.0], until=lambda s: s["cube.pos"][:, 2] > 0.12, name="lift"),
    Phase([0.0, 1.0, 1.0], name="hold"),                  # the last phase lasts to the episode's end
])
print(generate(env, script, [MemoryWriter()], episodes=4).summary()["success_rate"])
```

- `PhaseScript(env, phases, noise=0.0, seed=0)` is a `policy(obs) -> actions`. Every world walks the phases in order
  from the start of its own episode, so worlds that reset at different times are in different phases. A `Phase` has
  an action (one value per action column, or a function `fn(env.state) -> (num_envs, A)` for closed-loop moves that
  read named state), and ends after `steps` steps or once `until(env.state)` holds for that world; the last phase
  stays. `noise` adds uniform noise in `[-noise, noise]` to every action; the noisy action is what the env runs and
  what is recorded, so a replay runs the recorded actions (exact on the CPU; on a GPU see Replay below).
- `demonstrator(env, noise=0.0, seed=0)` returns the script that ships for the env's task: `GripperLift-v0`,
  `GripperLiftCamera-v0` and `PickPlace-v0`. **No script ships for the other ready-made envs** (`PandaReach-v0`,
  `PegInsert-v0`, `DrawerOpen-v0`, `Pour-v0`, `ClothFold-v0`): `demonstrator` raises `EnvError` for them; write a
  `PhaseScript`. The PickPlace script follows the object and the palm and never checks that the grasp held.
- `generate(..., keep="success", task=None, states=True)` are the `Recorder`'s. `max_attempts` (default 20 per
  wanted episode) stops a script that rarely succeeds: read `result.complete`. The result holds `saved`,
  `attempted` (every episode that ended), `successes`, `success_rate` with its `interval`, `steps`, `wall_s`,
  `episodes_per_s` and `world_steps_per_s`; `summary()` is JSON-ready. `progress=fn` is called with the counts so
  far whenever episodes end.
- `result.invalid` (an int) counts the attempted episodes whose world went INVALID (the engine froze it) at any
  step. Such an episode is never a success, `keep="success"` does not save it, and with `keep="all"` it is saved
  with `Episode.invalid` True (the HDF5 file stores the flag per demo; the LeRobot layout does not, so for it use
  `keep="success"`, or `keep=lambda ep: not ep.invalid`).
- `generate` works with every autoreset mode of the env: with `"disabled"` it resets the worlds whose episode
  ended itself.
- **Training a policy without memory on these demos: leave the `action` columns out of its input.** The
  ready-made envs' state vector ends with the previous action (`spec.obs_names` labels them `action[i]`). A
  single-step behaviour-cloning network that reads them learns to repeat its own last output and stalls: on
  GripperLift-v0 (CPU, 16 and 64 scripted demos, a 2-layer MLP, 8 held-out worlds) it lifted in 0 of 8 worlds with
  the `action` columns and in 8 of 8 without them (one training run each); a 16-step observation history also
  reached 8 of 8.
- The variety in the data is the env's own start randomization (object positions, joint starts, goals), the
  per-world build randomization and `noise`. **There is no trajectory augmentation**: no new demonstrations are
  made from a few human ones (MimicGen-style), no motion planner and no inverse kinematics for the scripts.
- Episodes are held in host memory until they end and are written one at a time; with cameras that is
  `worlds x max_steps` images in memory.

Measured on one NVIDIA L4 (`examples/il/04_generate_demos.py`, warm kernels, HDF5 with gzip, the time includes
writing; one run each; on a GPU a second run of the same command can save other episodes, since contacts there do
not repeat bit for bit):

| Env, worlds | Saved / tried | Script success (95 % interval) | Wall | Episodes/s | File |
|---|---|---|---|---|---|
| GripperLift-v0, 1,024 | 10,000 / 10,240 | 100 % [99.96, 100] | 101 s | 99 | 623 MB |
| GripperLift-v0, 4,096 | 20,000 / 20,480 | 100 % [99.98, 100] | 221 s | 91 | 2.1 GB |
| GripperLiftCamera-v0 (64x64 RGB), 256 | 1,000 / 1,024 | 100 % [99.6, 100] | 76 s | 13 | 312 MB |
| the same, 64, with `--lerobot` (MP4 video) | 256 / 256 | 100 % [98.5, 100] | 189 s | 1.4 | 80 MB (HDF5) |
| PickPlace-v0, 256 | 1,000 / 1,280 | 97.0 % [96.0, 97.8] | 97 s | 10 | 101 MB |
| PickPlace-v0, 256, `--noise 0.1` | 1,000 / 1,280 | 97.8 % [96.9, 98.5] | 106 s | 9.4 | 101 MB |

The writers run on one thread, so 4,096 worlds saved no faster than 1,024; a LeRobot dataset with video is
bound by the video encoder. On the CPU (an AMD EPYC 7B13, one process), the example's default run, 64 GripperLift
episodes in 16 worlds, took 28 s, and 4 PickPlace episodes in 4 worlds 19 s (all 4 succeeded and replayed exactly).
"Tried" counts episodes that ended; a run stops saving at the number asked for, so the last round's extra
episodes are tried but not saved.

`examples/il/04_generate_demos.py --env GripperLift-v0 --episodes 10000 --worlds 1024` is the command-line form.

## Replay

`replay(env, episodes)` puts one episode into each world: it restores the episode's recorded `start`, applies its
actions, and returns a `ReplayResult` with, per episode, `success`, `recorded_success`, `invalid` (the engine froze
the world during the replay), `recorded_invalid`, `matches` (same success and same INVALID, not ended early),
`state_error` (largest difference from the recorded `states`, in the state's own units: m, rad, m/s, rad/s) and
`obs_error`. Build the env like the one that recorded (same scene and task, default autoreset); the world count
may differ when the scene has no per-world randomization.

- Each episode replays in the world that recorded it (`episode.world`), since `Scene.randomize` gives every world
  its own values; in an env with fewer worlds it takes a free world. An episode's `start` keeps what
  `sim.randomized()` held for its world (servo gains, masses and the other robot and gravity draws), and `replay`
  raises `EnvError` when the world it would run in holds other values: build the env with the recording's world
  count and seed. Values drawn at every reset (`when="reset"`) cannot be put back, so such episodes are refused;
  compare those policies with `evaluate`. Object draws that `sim.randomized()` does not list (density, friction)
  are not checked: they repeat only in an env with the same world count and seed.
- A world whose episode is over gets actions that leave it alone (zeros, or in `"absolute"` mode the targets it
  holds), not the episode's last action.

- On the CPU, replaying in the env that recorded reproduces the recording exactly (`state_error == 0`; tests hold
  this for GripperLift and for PandaReach, whose arm starts at random joints, and that a changed action shows up as
  an error). The recorded `start` holds the joint targets and the controls of MuJoCo-native actuators
  (`target.ctrl`).
- **On a GPU a replay is not exact once contact is chaotic**, also on the same GPU with the same world count and seed:
  contacts there do not repeat bit for bit (`docs/known_gaps/determinism-gpu-rigid.md`). Compare with a tolerance
  (`state_error`) and judge by success (`matches`), not by an exact repeat. Two measurements: GripperLift-v0 on an
  NVIDIA L4 (64 worlds, 128 episodes, replayed in the env that recorded): every episode reproduced its recorded
  success, and the largest state difference per episode had a median of 5.5e-5 and a maximum of 0.79. PickPlace-v0 in
  a training audit (64 worlds, 64 demonstrations, same GPU, world count and seed, a new env): 1-2 of 64 flipped from
  success to failure, the state error had a median of 4.6 and a maximum of 30, and two new envs given the same actions
  ended with the object up to 0.20 m apart in 19 of 64 worlds. So on a GPU a dataset cannot be checked, or re-rendered
  with new cameras, by replaying its actions. In an env with another world count or build seed expect drift on the CPU
  too.
- Scenes with cloth, liquid or other particles cannot be replayed (`replay` raises `EnvError`): their particle
  state is not recorded. The same holds for a rod that steps on the rod solver (`sim.routes()` says `rod_vbd`: a
  clamped or soft cable), whose segment poses are not in the joint state, and for an env whose actions go through
  an actuator model (`ActuatorSpec`), whose held targets and delays are not recorded. The `Recorder` warns once
  when it is given such an env: its `states` and `start` lack that part. Episodes from the LeRobot writer or
  recorded with `states=False` have no start state and are refused too.
- `obs_error` covers every vector observation (each group of `Task(obs={...})`), not camera images. Observation
  history and delay layers restart from the restored state, so `ObsHistory` replays with zero error on the CPU;
  layers that draw random numbers (`ObsNoise`, a ranged `ActionDelay`) do not repeat their draws in a replay.
- Replay re-runs the physics from the recorded actions. Setting the recorded state at every step instead (to
  render new camera views of old demonstrations) is not provided.

## Policy evaluation

`evaluate(env, policy, episodes=100, seed=0)` rolls `policy(obs) -> actions` out and returns an `EvalResult`:
`success_rate`, `interval` (95 % Wilson score interval), `mean_return`, `mean_length`, the per-episode arrays and
`summary()`. Every world runs the same number of episodes (`episodes` is rounded up to a multiple of the world
count), so quick failures are not over-counted. Success is the task's success check on an episode's last step.
`result.invalid` (a bool array, one entry per episode) marks the episodes whose world went INVALID (the engine
froze it) at any step, and `summary()` counts them; they stay in the episode count and are never successes, also
when the frozen state passes the check.
The count is read from `info["invalid"]` at every step: the env's reset clears the engine's health record, so
`sim.health()` and `sim.check()` after a run do not show those worlds. DEGRADED worlds are not counted.
Pass a `Recorder` instead of the env to keep the rollouts as a dataset: it then saves the counted episodes only
(the same number from every world, through its `keep`), not the extra ones quick worlds run while the others
finish. The policy is any function of the batched
observation; loading a trained robomimic or LeRobot policy into that function is yours to write.

## Not covered

- Demonstration generation beyond a scripted policy: trajectory augmentation in the style of MimicGen, motion
  planning, subtask annotation, DAgger-style correction. Scripts ship for three of the nine ready-made envs.
- VR, phone, gamepad and leader-arm teleoperation.
- Training code: Alloy writes datasets and evaluates policies; it trains none.
- Running a trained robomimic or LeRobot policy in Alloy, and their own training and evaluation commands on Alloy
  envs: not tested, glue of your own.
- Depth and segmentation images in datasets: the env's cameras give RGB today.
