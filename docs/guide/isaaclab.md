# Switch a training task's physics to Alloy with one setting

`alloy.isaaclab` (optional) makes Alloy a physics choice inside Isaac Lab 3.0 (and Isaac Lab `main`, see Verified
tasks), next to the solvers Isaac Lab's Newton backend ships. A task that offers `physics=newton_mjwarp` runs on Alloy
with `physics=alloy`: same task, same robot, same rewards and training script, nothing edited. `import alloy` does not
load it, and Alloy without Isaac Lab behaves exactly as before.

```bash
isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=alloy
```

## Install

1. Install Isaac Lab 3.0 as its documentation says. The kit-less route (no Isaac Sim) is enough:
   `uv pip install isaaclab==3.0.0 --index https://pypi.nvidia.com --torch-backend cu130` (Python 3.12).
2. Install Alloy in the same environment: `uv pip install alloy-engine` (or `pip install alloy-engine`).

Step 2 moves two packages Isaac Lab 3.0.0 installed: MuJoCo 3.12.0 to 3.17.0 and Warp 1.17.0 to 1.18.0, the versions
Alloy is built against (Alloy's own Newton and MuJoCo Warp are bundled under private names, so Isaac Lab's Newton
1.6.1 stays as it is). `uv pip install` does this without a flag; `pip` does it too and then prints that `isaaclab`
and `newton` declared other versions: that message is expected. If you reinstall Isaac Lab later, install Alloy again
afterwards.

Isaac Lab's own `physics=newton_mjwarp` keeps working in that environment. One thing needs a fix for it: MuJoCo 3.17
stores 12 numbers per equality constraint where Isaac Lab's MuJoCo Warp 3.12 reads 11 (the first 11 mean the same;
the 12th is new), so robots with an equality constraint, such as Isaac Lab's Franka with coupled fingers, fail to
load on Isaac Lab's own path. `alloy.isaaclab` passes MuJoCo Warp 3.12 the first 11 numbers. It acts only on that
exact pair of versions and changes nothing else (`alloy/isaaclab/compat.py`). Isaac Lab's `isaaclab` command loads
it by itself. **In a script of your own** that runs Isaac Lab's `newton_mjwarp` with such a robot, put
`import alloy.isaaclab` at the top (before the scene is built); without it, that run fails with
`cannot reshape array of size 12 into shape (1,1,11)` once Alloy is installed in the environment. Isaac Lab's own
test suite needs the same: run it as `python -m pytest -p alloy.isaaclab.register ...`.

## Switch

Command line: Isaac Lab's own `physics=` token, on any `isaaclab train`, `play`, `zero_agent` or `random_agent` run.
Alloy registers itself with Isaac Lab's `isaaclab.tasks` entry point, so nothing needs importing.

```bash
isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD physics=alloy
isaaclab zero_agent --task Isaac-Reach-Franka --num_envs 16 physics=alloy
```

Python (your own script):

```python
import alloy.isaaclab                                     # physics=alloy now resolves in resolve_task_config
from alloy.isaaclab import AlloySolverCfg, alloy_physics
from isaaclab_newton.physics import NewtonCfg

env_cfg.sim.physics = alloy_physics(env_cfg.sim.physics)  # a NewtonCfg with MuJoCo Warp -> the same on Alloy
physics = NewtonCfg(solver_cfg=AlloySolverCfg(njmax=80, nconmax=25, cone="elliptic"))   # or name it directly
```

`physics=alloy` takes the task's `newton_mjwarp` configuration and changes only the solver: every MuJoCo Warp
setting the task chose (constraint and contact capacities, cone, integrator, iterations, whether MuJoCo or Newton
finds the contacts) carries over, and other settings a task keys by physics name (an armature value for
`newton_mjwarp`, say) take their `newton_mjwarp` value. If any MuJoCo Warp configuration in the resolved task is still
Newton's own, the run stops with the config path instead of silently running Newton's solver. A task that offers no
`newton_mjwarp` (PhysX only) gets an error saying so.

A run's log names the physics that stepped it:

```text
[INFO] [alloy] physics: Alloy 0.1.2 rigid route (Newton 1.8.0rc1, MuJoCo Warp 3.17.0) on 4096 worlds; step: the task's own (rigid_step='task'); noslip off; model bridged: ...
```

## What changes with Alloy

- **The solver.** Alloy steps Isaac Lab's model with its bundled MuJoCo Warp 3.17 through its bundled Newton 1.8,
  with the run-time guards an Alloy build has. Isaac Lab keeps everything else: scene building, assets, sensors,
  actuators, resets, CUDA graph capture. The model Isaac Lab built is used as is: Alloy's model shares every array
  with it (`alloy/isaaclab/bridge.py`), so a mass or friction Isaac Lab randomizes is the one Alloy steps.
- **The step.** Alloy takes the task's own physics step, as the task's MuJoCo Warp preset does.
  `rigid_step="alloy"` opts into the step Alloy's own robot scenes use, 1/600 s or finer: a 5 ms physics step then
  runs as 4 steps of 1.25 ms, an 8.3 ms step as 8 of 1.04 ms. That is finer contact at a cost: training wall time
  grows, by less than the step count, because the rest of a training step does not change.
- **Health checks.** Every physics step, every world is checked on the GPU for non-finite state, penetration past
  1 cm, speeds past 20 m/s, more contacts or constraint rows than MuJoCo Warp's buffers hold (contacts then dropped),
  joints pulled apart and more (`docs/guide/health.md` lists them). Isaac Lab resets worlds on its own schedule, so
  Alloy counts how each episode ended before the reset clears it, and prints a line when something was flagged:

  ```text
  [INFO] [alloy] health: 0 of 4096 worlds INVALID now, 3 DEGRADED; episodes ended INVALID 2, DEGRADED 41; worlds per flag: PEN_WARN 44, PEN_FAIL 2; first: ...
  ```

  Worlds are flagged, never frozen or changed. In Python, `alloy.isaaclab.active_solver().health.report()` returns
  the same summary and `.health.invalid_now()` an int32 per world (`wp.to_torch(...)` for a termination term).

## Settings (`AlloySolverCfg`)

Every `MJWarpSolverCfg` field, plus:

| Field | Default | Meaning |
|---|---|---|
| `rigid_step` | `"task"` | `"task"`: the task's own step; `"alloy"`: steps of at most 1/600 s inside each physics step (finer contact, slower); a number: a cap in seconds |
| `health` | `"report"` | `"report"`: check every world every step and print summaries; `"off"`: no checks |
| `health_speed_limit` | `20.0` | m/s; a body faster than this makes its world INVALID (SPEED_FAIL). Raise it for tasks that are fast on purpose (an untrained cartpole at full force passes 20 m/s) |
| `noslip_iterations` | `0` | MuJoCo's noslip post-solve after each step (Alloy's own gripper scenes use 100): a held object creeps out of a firm grip far less (a 0.25 kg cube in a Franka grasp: 65 % less slip); costs solver time (the Franka lift task, 4,096 worlds on an L4: collection at 0.34x the speed without it, training 2.1x as long). Elliptic cone: Alloy's version, scoped to the robot's contacts; pyramidal: MuJoCo Warp's own. Not with `integrator="discrete"` (MuJoCo Warp refuses the pair) |
| `noslip_tolerance` | `1e-6` | its convergence tolerance |
| `health_report_every` | `2000` | physics steps between checks; a line is printed when a world went INVALID since the last one, and a full summary when the run ends; `0`: only at the end |

The task's own `integrator` setting carries over and also takes `"discrete"` on Alloy (MuJoCo 3.17's discrete
integrator): the H1 flat-ground task trains with it to the same reward as with `"implicitfast"`, at 0.93x the
collection throughput.

## Verified tasks

These ran on Isaac Lab 3.0.0, installed kit-less, with Alloy beside it.
Training, on one NVIDIA L4 GPU (driver 580), with this command:

```bash
isaaclab train --rl_library rsl_rl --task <task id> --max_iterations <iterations> --seed 42 physics=alloy
```

Every training run used 4,096 worlds, seed 42 and the health checks; reward is the mean of the last 10 iterations.
At the default step the health checks took 4-7 % of the collection throughput; `health="off"` gives that back.

| Task | Iterations | Mean reward, default step | Mean reward, `rigid_step="alloy"` |
|---|---|---|---|
| Cartpole | 150 | 4.94 | 4.90 |
| AnymalD, flat ground | 300 | 18.49 | 20.07 |
| H1, flat ground | 500 | 32.06 (success 0.99) | 31.27 |
| Franka reach | 1000 | 0.13 (success 1.00) | 0.10 (success 1.00; a run before the health checks existed) |

Task ids, in table order: `Isaac-Cartpole-Direct`, `Isaac-Velocity-Flat-AnymalD`, `Isaac-Velocity-Flat-H1`,
`Isaac-Reach-Franka`.

Stepping and resetting was checked on 18 tasks: 1,024 worlds, 300 random-action steps, the same GPU,
at `rigid_step="alloy"` (five of them, the humanoid and the hand tasks, also at the default step).
Those four, and also `Isaac-Ant`, `Isaac-Humanoid`, `Isaac-Lift-Franka`, `Isaac-Lift-KukaAllegro`,
`Isaac-Open-Drawer-Franka`, `Isaac-Reach-UR10`, `Isaac-Reorient-Cube-Allegro`, `Isaac-Reorient-Cube-Shadow` (MuJoCo
fixed tendons), `Isaac-Reorient-KukaAllegro`, `Isaac-Shadow-Handover`, `Isaac-Velocity-Flat-Cassie`,
`Isaac-Velocity-Flat-G1`, `Isaac-Velocity-Flat-UnitreeGo2` and `Isaac-Velocity-Rough-AnymalD` (heightfield terrain),
with Isaac Lab's own sensors (contact sensors included), actuators and resets. With `physics=alloy` installed in the
same environment, Isaac Lab's own `physics=newton_mjwarp` still runs these tasks.

Isaac Lab `main` (commit 7bca809, Newton 1.9 development build): the four trained tasks step and reset on Alloy in
small CPU runs, contact sensors included; not yet run on a GPU or trained. Isaac Lab changes `main` often, so a later
commit may need a plug-in update.

Tests: `tests/isaaclab` (bridge and settings run without Isaac Lab; the rest skip without it), the CPU and GPU
smoke runs of `examples/isaaclab/01_switch_physics.py`.

## Not supported yet

- Only Newton's MuJoCo Warp solver is replaced. Isaac Lab's XPBD, Featherstone, Kamino, VBD and MPM solvers, and
  PhysX, are left as they are; `alloy_physics` raises for them. Tasks that offer only Isaac Lab's coupled
  MuJoCo Warp + VBD preset (`newton_mjwarp_vbd_proxy`: the cloth, soft-body and cable lift tasks) are refused with a
  message saying so.
- Cloth, soft bodies and liquid in an Isaac Lab scene: not on Alloy yet.
- `noslip_iterations` with `integrator="discrete"`: refused (MuJoCo Warp has no noslip for that integrator).
- Repeatable runs: the task's `deterministic_mode` request is passed to the solver unchanged and is not supported yet.
  With `"run_to_run"`, the flat-ground AnymalD task repeated bit for bit at 64 and 256 worlds but stepped at only 14
  and 39 env steps/s, and at 1,024 worlds the run stopped inside Warp's deterministic mode (an array past 2^31 bytes,
  or out of memory).
- Not run yet: the MuJoCo CPU backend (`use_mujoco_cpu=True`) beyond small CPU runs.
