# Training with rsl_rl, Stable-Baselines3 and skrl

`alloy.envs.wrappers` adapts an Alloy environment ([envs.md](envs.md)) to three RL libraries, and
`python -m alloy.envs.wrappers.train` runs PPO training with one command (what it reaches:
[below](#what-it-reaches)). Everything on this page works from the
installed package; no source checkout is needed.

## Install

```bash
pip install "alloy-engine[envs]"        # Alloy + PyTorch
pip install rsl-rl-lib                  # and / or: stable-baselines3, skrl
```

The libraries are optional and none is installed by Alloy. A wrapper used without its library raises `EnvError` with
the pip command. Tested versions: rsl-rl-lib 5.5.2, stable-baselines3 2.9.2, skrl 2.1.1 (torch 2.15, Python 3.12).
Other versions are not tested: rsl_rl changed its environment interface between major versions, so use 5.x.

## Train with one command

```bash
python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl
python -m alloy.envs.wrappers.train PandaReach-v0 --lib skrl --num-envs 1024 --iterations 150
python -m alloy.envs.wrappers.train GripperLift-v0 --lib sb3 --num-envs 4 --iterations 2 --device cpu   # quick check
```

- `env`: any ready-made environment (`python -c "import alloy.envs; print(sorted(alloy.envs.REGISTRY))"`).
  Your own environment: `module:function`, e.g. `my_package.my_envs:make_env`, a function
  `make_env(num_envs=, device=, seed=)` that returns a `VectorEnv` on the torch backend (the module must be
  importable from where you run the command).
- `--lib rsl_rl | sb3 | skrl` (default `rsl_rl`), `--num-envs` (512), `--iterations` (200), `--seed` (0),
  `--device` (`cuda:0` when there is a GPU, else `cpu`), `--log-dir` (default `runs/<lib>/<env>`),
  `--report-every` (10).
- One iteration is 32 steps of every world, then one PPO update: 200 iterations at 512 worlds are 3.28 M samples.
- It prints one JSON line per 10 iterations and a summary line at the end, and saves the policy in `--log-dir`.
  The lines of the first command above, from one of its runs on one L4 (sharing the GPU with two other runs;
  another run of the same command gives other numbers, see below):

```text
{"lib": "rsl_rl", "env": "GripperLift-v0", "num_envs": 512, "iter": 10, "samples": 163840, "minutes": 0.51, "episodes": 1024, "success_rate": 0.0, "mean_return": 20.79}
{"lib": "rsl_rl", "env": "GripperLift-v0", "num_envs": 512, "iter": 20, "samples": 327680, "minutes": 0.88, "episodes": 1024, "success_rate": 0.0, "mean_return": 28.237}
...
{"lib": "rsl_rl", "env": "GripperLift-v0", "num_envs": 512, "iter": 200, "samples": 3276800, "minutes": 6.42, "episodes": 1024, "success_rate": 0.999, "mean_return": 641.261}
{"summary": {"success_rate": 0.999, "minutes": 6.42, "samples": 3276800, "checkpoint": "runs/rsl_rl/GripperLift-v0/model_200.pt"}}
```

`success_rate` is the share of the episodes that ended in those iterations that ended in success (for
`GripperLift-v0`: the cube up 8 cm and held when the episode ends). The first run of a new environment also compiles
kernels (a few minutes on a GPU, cached afterwards), so `minutes` of a first run is longer than of the next. The
512-world command prints nothing until its first 10 iterations are done (a minute or two, more on a first run). The
4-world quick check takes 35-50 s the first time on a CPU (a 2.8 GHz Xeon cloud VM, GCP n2-standard-16) and about
15 s after that; it prints
`"success_rate": null`, because no episode ends in 2 iterations (64 steps; a GripperLift episode is 150 steps).

The same from Python:

```python
from alloy.envs.wrappers.train import train

summary = train("GripperLift-v0", lib="rsl_rl", num_envs=4, iterations=2, device="cpu")   # a 2-iteration check
print(summary["success_rate"], summary["checkpoint"])
```

### What it reaches

Measured on one NVIDIA L4 with the commands above (seeds 0 to 3), kernels already compiled. A run's result is the
success rate of the episodes that ended in its last 10 iterations; the bar is 90 %. The policy is still sampling its
actions during training, so these are training rates, not an evaluation of the final policy. 200 iterations at 512
worlds are 3.28 M samples; 150 at 1,024 are 4.92 M.

| Library | Environment | Worlds x iterations | Runs | Reached 90 % | Lowest to highest result |
|---|---|---|---|---|---|
| rsl_rl | `GripperLift-v0` | 512 x 200 | 6 | 6 | 91.5 % to 99.9 % |
| rsl_rl | GripperLift, privileged critic | 512 x 200 | 1 | 0 | 74 % (its best window: 99.6 %) |
| rsl_rl | `PandaReach-v0` | 1,024 x 150 | 1 | 1 | 100 % |
| skrl | `GripperLift-v0` | 512 x 200 | 4 | 4 | 97 % to 100 % |
| skrl | GripperLift, privileged critic | 512 x 200 | 1 | 1 | 100 % |
| skrl | `PandaReach-v0` | 1,024 x 150 | 2 | 2 | 100 % |
| Stable-Baselines3 | `GripperLift-v0` | 512 x 200 | 3 | 3 | 92 % to 99 % |
| Stable-Baselines3 | GripperLift, policy group only | 512 x 200 | 1 | 0 | 39 % (its best window: 83 %) |
| Stable-Baselines3 | `PandaReach-v0` | 1,024 x 150 | 2 | 0 | 65 % to 85 % |

The rows above were measured before delta and end-effector targets were kept within a window of the measured joints
([envs.md](envs.md), `ActionSpec`). With the window, one rsl_rl run each (seed 0, one L4): `GripperLift-v0` 96.6 %,
`PandaReach-v0` 100 % (55 % at iteration 30, 100 % from iteration 60), the end-effector reach below 100 % (75 % at
iteration 30, 100 % from iteration 60; 71 % and 100 % in a run without the window).

How to read it:

- **A seed does not fix the outcome on a GPU.** Two runs of the first command with seed 0 on one machine ended at
  99.8 % and 91.5 %; with earlier skrl settings, seed 0 ended at 0 % in one run and 96 % in another. Which part
  does not repeat was not isolated: on a GPU the simulation itself does not repeat bit for bit once the robot
  touches things ([worlds.md](worlds.md), Determinism), GPU arithmetic in the training libraries is a second
  source, and training amplifies small differences. So no row is the result of "its" seed: a row says how many
  runs there were and where they ended, and with one to six runs per row that is a small sample.
- **rsl_rl and skrl reached the bar on both tasks** in every run of the ready-made environments. skrl reached it
  after we changed its settings (the entropy term and action clipping, below): with the earlier ones two of five
  runs never left 0 %. Late in a
  `GripperLift-v0` run the rate still moves (the rsl_rl privileged-critic run was between 71 % and 99.6 % over its
  last 50 iterations and ended on a low window).
- **Stable-Baselines3 reached the bar on `GripperLift-v0`, not on `PandaReach-v0`** (65 % and 85 %), and its
  policy-group run of the example ended at 39 %.
- Minutes: the first command took 3.2 min alone on the L4 and 5 to 7 min when three runs shared it.

These are small tasks of ours, not a standard benchmark. The PPO settings are one fixed set per library (two 128-unit
tanh layers, 4 epochs, 4 mini-batches, learning rate 3e-4, discount 0.99; rsl_rl and skrl adapt the learning rate to
a KL target of 0.01; skrl runs without value clipping, with an entropy term of 0.001 and a policy that leaves
clipping of its actions to the environment; Stable-Baselines3 stops an update at a KL of 0.02 and normalises
observations with `VecNormalize(epsilon=0.01)`); they are a starting point, not tuned per environment.
`PickPlace-v0` is not learned by these settings ([envs.md](envs.md),
Training).

### Arm pick: `PickCube-v0`

A Franka Panda (the library model) picks a 4 cm, 50 g cube off its table and carries it to a point drawn 4 to 30 cm
above the table at every reset. Success: the cube's centre within 2.5 cm of the point. Actions: joint deltas for the
7 arm servos (up to 0.05 rad per 20 ms step, each target kept within 0.05 rad of its measured joint) and a gripper
that opens or closes in one step; 100-step (2 s) episodes. The cube counts as held only between the pads and 1 cm off
the table: a reward that paid fingers closed around a cube resting on the table taught policies to hold it there.

```bash
python -m alloy.envs.wrappers.train PickCube-v0 --lib rsl_rl --num-envs 2048 --iterations 500
```

rsl_rl takes this env's own settings (`train_rsl_rl.RECIPES["PickCube-v0"]`: discount 0.95, three 256-unit layers, 8
epochs and 8 mini-batches, no entropy term).

With these settings training does not yet learn the pick reliably: some runs learn it and others do not. Treat
them as a starting point; tuning continues. Not covered: other cube sizes, masses or friction, other objects, camera
observations, and Stable-Baselines3 and skrl (not run on it).

Speed under contact, two opt-in settings (both off by default): `ALLOY_NOSLIP_SWEEPS=20` caps Alloy's no-slip pass at
20 sweeps per world, and `ALLOY_NOSLIP_STALL=1e-5` stops a world's sweeps once they no longer improve. On
`PickPlace-v0` (2,048 worlds, random actions, one L4) they raised 15.5k env steps/s to 23.6k and 26.3k, with end
states within run-to-run noise. On an arm pressing on its table the end state moved by more than run-to-run noise, so
compare your task's success with and without them before training with them. Not measured on `PickCube-v0`.

`PickPlace-v0` does not learn to place: its "held" term rewards fingers open around the object, and no term rewards
the release into the bin, so a policy trained on it does not place the object. A reworked reward is planned; no date.

## Use a wrapper with the library's own trainer

```python
from alloy.envs import make
from alloy.envs.wrappers import wrap

env = make("GripperLift-v0", num_envs=4, device="cpu", backend="torch")
venv = wrap(env, "rsl_rl")                       # or "sb3", "skrl"
print(type(venv).__name__, venv.num_envs)        # RslRlVecEnv 4
venv.close()
```

| `wrap(env, ...)` | Class | What the library gets |
|---|---|---|
| `"rsl_rl"` | `RslRlVecEnv` | rsl_rl's `VecEnv`: observations as a TensorDict with one key per observation group, `extras["time_outs"]` for time limits, `extras["log"]` with the success rate, return and length of finished episodes and each reward term. Tensors stay on the simulation device. |
| `"sb3"` | `Sb3VecEnv` | Stable-Baselines3's `VecEnv`: NumPy arrays (every step copies observations, rewards and ends from the GPU to the host), `terminal_observation`, `TimeLimit.truncated`, `is_success` and `episode` in each finished world's info. |
| `"skrl"` | `SkrlEnv` | skrl's torch `Wrapper`: actions and observations stay torch tensors on the simulation device (skrl 2.1.1's Gymnasium wrapper converts actions to NumPy). |

A world the engine froze (INVALID) ends its episode as terminated in all three wrappers, so no value is bootstrapped
from it, and its reward (except the task's `invalid_penalty`) and last observation read 0 on that step
([envs.md](envs.md), Ready-made environments).

rsl_rl with your own settings:

```python
from rsl_rl.runners import OnPolicyRunner

from alloy.envs import make
from alloy.envs.wrappers import wrap
from alloy.envs.wrappers.train_rsl_rl import default_config

venv = wrap(make("GripperLift-v0", num_envs=4, device="cpu", backend="torch"), "rsl_rl")
cfg = default_config(iterations=2, seed=0)       # a plain dict: edit what you need
cfg["obs_groups"] = venv.default_obs_groups()    # actor <- "policy", critic <- "critic" when the env has it
cfg["algorithm"]["learning_rate"] = 1e-3
OnPolicyRunner(venv, cfg, log_dir=None, device=str(venv.device)).learn(num_learning_iterations=2)
venv.close()
```

Stable-Baselines3:

```python
from stable_baselines3 import PPO

from alloy.envs import make
from alloy.envs.wrappers import wrap

venv = wrap(make("GripperLift-v0", num_envs=4, device="cpu"), "sb3")
model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=4, device="cpu")
model.learn(total_timesteps=4 * 32 * 2)
venv.close()
```

With many worlds set `n_steps` yourself: Stable-Baselines3's default of 2,048 steps per world is meant for a few
environments (512 worlds x 2,048 steps is a million samples per update).

## Observation groups: a critic that sees more than the policy

By default an environment returns one flat observation. `Task(obs={...})` with named groups returns one array per
group instead, so the critic (or a teacher policy) can read values the policy cannot have on a real robot. A group
named `"policy"` is required.

```python
import dataclasses

from alloy.envs import VectorEnv
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task
from alloy.envs.wrappers.train import train

groups = {
    "policy": ["gripper", "cube.pos", "action"],     # joints (q, qd), cube position, last action
    "critic": ["gripper", "cube", "action"],         # + cube orientation, linear and angular velocity
}
task = dataclasses.replace(gripper_lift_task(), obs=groups)
env = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu", backend="torch")
obs, info = env.reset(seed=0)
print({name: tuple(value.shape) for name, value in obs.items()})   # {'policy': (4, 12), 'critic': (4, 22)}
print(env.obs_groups, env.obs_names("critic")[:3])
summary = train("GripperLift-v0", lib="rsl_rl", env=env, iterations=2)   # trains on this env, then closes it
```

- Group entries are the same as in a flat `obs` list ([envs.md](envs.md), Your own task): names, state keys, checks,
  and `(label, fn)` with `fn(state) -> (worlds, k)` for anything else.
- `reset` and `step` return the dict; `info["final_obs"]` is a dict too; `env.single_observation_space` is a Dict
  space; `env.obs_names(group)` labels a group's columns. Camera images keep their own keys beside the groups.
- A list (`obs=[...]`) keeps the one flat array. Groups change what is returned, not what is simulated: the same
  seed and actions give the same states either way.
- rsl_rl: each group is a TensorDict key; `default_obs_groups()` gives the critic the `"critic"` group. skrl: the
  `"critic"` group is skrl's state, read by the value model only. Stable-Baselines3 has one observation: its policy
  and value networks both read the `"policy"` group, and other groups are not passed on.
- Group names other than `"policy"` and `"critic"` (a `"teacher"` group, say) reach rsl_rl as TensorDict keys for
  your own `obs_groups` mapping; the train command uses only those two.

## Task-space actions: move the hand, not the joints

`alloy.envs.diffik.EEActionSpec` makes the action a motion of one body (a hand): `(dx, dy, dz)`, and with
`orientation=True` three more columns for a small rotation. Each step the environment turns it into joint targets by
damped least squares on the arm's Jacobian (differential inverse kinematics), for every world at once. It is opt-in:
an `ActionSpec` keeps joint-space actions.

```python
import numpy as np

from alloy.envs.diffik import panda_reach_ee

env = panda_reach_ee(num_envs=4, device="cpu", backend="numpy")     # PandaReach with hand actions
obs, info = env.reset(seed=0)
print(env.action_names())                                           # ['ee_dx', 'ee_dy', 'ee_dz']
for _ in range(60):                                                 # a proportional controller in task space
    to_target = env.state["target.pos"] - env.state["panda/hand.pos"]
    obs, reward, terminated, truncated, info = env.step(np.clip(to_target / 0.02, -1.0, 1.0))
print(np.linalg.norm(env.state["target.pos"] - env.state["panda/hand.pos"], axis=1))   # metres, per world
env.close()
```

Your own arm: `Task(actions=EEActionSpec(body="arm/hand", joints=[...arm joints or actuators...],
extra=["arm/gripper"]))`. An action of 1 moves the hand `delta` metres (default 0.02) or turns it `rot_delta`
radians (0.1) in that step; `extra` joints follow the columns after the hand's, in delta mode; `damping` (0.01) is
the damped-least-squares term. Each step starts from the last joint targets pulled back toward the measured joints
until every arm joint is within `window` of its own (the arm's lead shrinks as one vector, so the hand keeps the
direction it was sent; `extra` joints: 5 x their `delta`, each on its own), so a hand pressed on a table or a part
does not run its targets on. That is why it is the default: after a 3 s press on the table, an up command lifted the
Panda's hand 1 cm in 4 steps, in 12-23 with no window (CPU, the Panda only, seeds 0-3). Its cost: the servos set the
hand's top speed, so steer it closed loop, as a policy does; an open-loop command of 10 cm over 5 steps moved the
Panda's hand 4.1 cm (9.7 cm with `window=float("inf")`). The default, 0.03 rad, is set for the shipped Panda's gains;
choose it again for another arm. The window bounds the windup, not the press: a blocked servo still pushes up to its
force limit. With the shipped Panda's firm contacts, a scripted 100-step push of a cube along the table (64 worlds)
leaves no world INVALID with or without the window (CPU, and one L4), so on the shipped arms the window does not
prevent a freeze; with soft contacts (MuJoCo's default 0.02 s on the arm and the table) that push left 34-45 of 64
worlds INVALID with no window and 14-22 with it. Train it like any environment:
`python -m alloy.envs.wrappers.train alloy.envs.diffik:panda_reach_ee --lib rsl_rl --num-envs 1024 --iterations 150`.

Measured (CPU, the shipped Franka Panda, `tests/envs/test_diffik.py`): the Jacobian agrees with MuJoCo's
`mj_jacBody` to 1e-5; with no window, a commanded 10 cm straight move ends within 5 mm along the move and within 8 mm
off its line (the Panda file's servos have no gravity compensation, so the arm's sag changes as it reaches); with
`orientation=True` the hand turned 0.9 degrees over a 10 cm move. One training run of the
command above on one L4 (rsl_rl, 1,024 worlds, 150 iterations, seed 0): 91 % of episodes within 3 cm at iteration
30, 100 % from iteration 60 to the end, 2.7 min; one run, and a GPU run with one seed does not repeat (above).

Not covered: one body per spec (no two-arm or whole-body solve); hinge and slide joints only; rotations in the world
frame; no null-space posture control, no collision avoidance, no operational-space (force) control; joint limits
only clip the targets; near a singularity the hand undershoots (damping) instead of the targets growing; targets
build on the last target, not the measured pose, so a blocked arm keeps winding up until a limit. Tested on the
Panda's `<general>` position servos only; joint position targets of `<position>` servos are not tested. On a GPU
only that training row and the example were run (the tests run on the CPU).

## Noise, delay and history layers

`Task(layers=[...])` puts stages between the simulation and the policy, per world. They change what the policy sees
and when its action arrives; the simulation is not changed. With no layers nothing changes, bit for bit.

```python
import dataclasses

from alloy.envs import VectorEnv
from alloy.envs import realism as R
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

layers = [
    R.ObsNoise({"cube.pos": 0.002, "gripper/lift.q": 0.001}),   # Gaussian, std per column (m, rad); 0.01 = all
    R.ObsDelay(1),                                              # the policy sees the last step's observation
    R.ObsHistory(3),                                            # the last 3 observations, oldest first
    R.ActionDelay((0, 2)),                                      # each world's action arrives 0 to 2 steps late
]
groups = {"policy": ["gripper", "cube.pos", "action"], "critic": ["gripper", "cube", "action"]}
task = dataclasses.replace(gripper_lift_task(), obs=groups, layers=layers)
env = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu", backend="numpy")
obs, info = env.reset(seed=0)
print(obs["policy"].shape, obs["critic"].shape)                 # (4, 36) (4, 22): 3 x 12 for the policy
print(R.obs_names(env, "policy")[:2], R.obs_names(env, "policy")[-1])
env.close()
```

- `ObsNoise(std or {column pattern: std}, uniform=None, groups=("policy",))`, `ObsDelay(steps or (lo, hi))`,
  `ObsHistory(length)`, `ActionDelay(steps or (lo, hi))`. A `(lo, hi)` range is drawn per world at every reset. Layers
  act on the `"policy"` group unless `groups=` names others, so a critic group stays clean. On a task whose `obs` is a
  flat list, every layer acts on that list. The layers run in the order given; noise, then delay, then history is the
  usual order.
- After a reset a delay or history buffer holds that world's first observation in every slot, and a delayed action
  is zero until the first one arrives. `info["final_obs"]` is the processed observation.
- `env.obs_names()` keeps the plain labels; `realism.obs_names(env, group)` gives the labels with history
  (`name@t-2` ... `name@t`).
- Checked in `tests/envs/test_realism.py` (CPU): rewards and states equal a plain environment's bit for bit under
  noise; each world's observation is the clean one from its own k steps ago; the measured noise is within 10 % of
  the set value.

Not covered: noise is white, drawn anew each step (no bias, drift or quantization); delays are whole control steps
(no latency inside a step); camera images pass through untouched (no image noise, delay or history); `ActionDelay`
with absolute-target actions is tested on a stand-in writer only; the layers were tested on the CPU and, on a GPU,
only run by the example (`examples/envs/14_sim2real_layers.py`, 64 worlds: measured noise 0.020 for 0.02 set). No
policy trained with these layers has been run on a real robot.

## Actuator models: a torque law between the policy and the joint

`alloy.envs.actuators.ActuatorSpec` makes the action a joint position target that a model turns into torque:
`kp (target - q) - kd qd`, kept inside an effort limit and, for `model="dc_motor"`, under a torque-speed line that
falls from `saturation_effort` at rest to zero at `velocity_limit`; `delay_steps` delays the target by whole
control steps. The torque is recomputed at every physics step (1.25 ms at `dt=0.01`), not once per control period.

```python
import numpy as np

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs.actuators import ActuatorSpec, motorized

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.5">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""

spec = ActuatorSpec(["arm/shoulder"], model="dc_motor", kp=60.0, kd=2.0, effort_limit=12.0,
                    saturation_effort=14.0, velocity_limit=8.0, delay_steps=(0, 2))
scene = alloy.Scene(worlds=4, dt=0.01)
scene.add_ground()
scene.add_robot("arm", mjcf=motorized(ARM))                     # the file's servo becomes a plain motor
sim = alloy.build(scene, device="cpu", plugins=[spec.plugin()])   # the model steps inside the simulation
env = VectorEnv(sim, Task(obs=["arm"], actions=spec, max_steps=200), backend="numpy")
obs, info = env.reset(seed=0)
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(np.full((4, 1), 0.25, np.float32))
print(env.writer.torque[:, 0])                                  # N m applied to the joint now, per world
env.close()
```

- The robot must be motorized: `motorized(path or MJCF text)` rewrites `<position>`, `<velocity>`, `<general>` and
  similar servo actuators into `<motor>` ones. A joint that still has a file servo is refused with the fix, because
  the servo and the model would both act.
- The model is a plug-in, so build the simulation first (`alloy.build(scene, plugins=[spec.plugin()])`) and pass it
  to `VectorEnv(sim, task)`.
- Checked in `tests/envs/test_actuators.py` (CPU, one hinge): the steady error under gravity is load / kp and the
  applied torque equals the load (ratio 1.0000); a far target gives exactly the effort limit and an acceleration of
  limit / inertia; a DC motor tops out at 9.999 of a 10 rad/s no-load speed; a motorized joint at zero gains moves
  like an unactuated one.
- Stability: the PD needs `kp h^2 + 2 kd h < 4 I` (h the physics step, I the joint's inertia). Past that the world
  goes INVALID and freezes; there is no check at build.

Not covered: learned actuator networks; joint friction, backlash, motor heating, supply voltage; delays shorter than
a control step; tendon and site actuators (not rewritten by `motorized`); one action column per hinge or slide
joint. Not run on a legged robot. On a GPU one check only: the hinge above, 64 worlds, held the same angle and
torque as on the CPU (0.5213 rad, 1.276 N m) after 3 s.

## What is not covered

- Environments: the wrappers take `VectorEnv` environments (`alloy.envs.make` of a ready-made name, or your own
  `VectorEnv(scene, task)`). The task ports ([envs.md](envs.md), Ports) are not `VectorEnv`s and are refused.
- Libraries: rsl_rl, Stable-Baselines3 and skrl only. RL-Games, TorchRL, RLlib and JAX libraries have no wrapper.
- The train command runs PPO only, with fixed settings, and has no play / evaluate / resume command and no video
  recording; load the saved policy with the library's own tools.
- Camera environments: no shipped trainer learns from pixels. `--lib rsl_rl` on a camera environment
  (`GripperLiftCamera-v0`) renders the images every step and `RslRlVecEnv` passes them as extra TensorDict keys, but
  the command's actor and critic are MLPs over the `"policy"` state group only, so the images never reach the policy.
  It trains blind and says nothing: the success it reports comes from the state, which holds the cube's pose (an
  outside audit measured 98.8 % that way, at 3.4x the cost per step of the state environment). `--lib skrl` refuses an
  environment with cameras, with an error. `--lib sb3` gives the images to a CNN (`MultiInputPolicy`), beside the same
  state, and copies every image to the CPU; it is not trained in our tests. Workaround: for a policy that must see,
  write the training loop yourself over the environment's image observations (one key per camera name), and leave the
  object's pose out of the task's `obs`, or the policy can ignore the image.
- Time limits: rsl_rl bootstraps from `time_outs`, Stable-Baselines3 from `terminal_observation`. skrl's own PPO
  bootstraps from the observation after the step, which for a world that just ended is the first observation of its
  next episode. The train command uses a PPO subclass that takes the ended episode's last observation instead
  (`alloy.envs.wrappers.train_skrl.agent_class()`; `SkrlEnv.step` puts it in `info["bootstrap_observations"]`); with
  skrl's stock agents on `SkrlEnv` the value target at a time limit is approximate.
- Stable-Baselines3 runs on host NumPy arrays (a copy per step). Its train command saves the observation
  statistics beside the policy (`vecnormalize.pkl`): load both to use the policy. With `VecNormalize`'s default
  epsilon `GripperLift-v0` stayed at 0 % in our runs.
- Recurrent policies, multi-agent environments, distillation (teacher-student) runs and multi-GPU training are not
  tested.
- Environments that reset all worlds together (`env.together`, [envs.md](envs.md)) work, but every episode then ends
  at the step limit.
- Actions are clipped to the environment's action space ([-1, 1] for most) by the rsl_rl wrapper
  (`clip_actions=False` turns that off); environments whose actions are absolute joint targets are passed
  unclipped.
- Task-space actions, the noise / delay / history layers and the actuator models are each a first version: their
  own "Not covered" paragraphs above say what is missing.

Runnable example, in the repository's `examples` folder (it needs only the installed package):
`examples/envs/11_train_rl.py [lib] [num_envs] [iterations] [device]` trains GripperLift with the privileged critic
above. `examples/envs/13_ee_actions.py [num_envs] [device]` drives the Panda's hand to each world's target
with task-space actions; `examples/envs/14_sim2real_layers.py [num_envs] [device]` runs the noise, delay and
history layers.
