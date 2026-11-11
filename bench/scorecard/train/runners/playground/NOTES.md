# MuJoCo Playground (row 4, time to train): notes

Framework keys: `playground_mjx` (JAX MJX, `--impl jax`), `playground_warp` (MuJoCo Warp, `--impl warp`).
Pilots: seed 0, one L4 (g2-standard-8), spot.

## Versions (install.sh, venv ~/pg/.venv, python 3.12 via uv)
playground 0.2.0 (PyPI; the classic suite Go1Joystick / PandaPickCube / LeapCubeReorient is here, v1.0.x moved it to a
legacy tree and I did not use it), mujoco + mujoco-mjx 3.17.0 (MuJoCo Warp is vendored in mjx), jax/jaxlib 0.12.0.1
(cuda12), warp-lang 1.19.0, brax 0.14.2, rsl-rl-lib 5.5.0, torch 2.15.0, numpy 2.5.4. Freeze: ~/pg/freeze.txt on the VM.
Env assets: Playground clones mujoco_menagerie on first env load (network needed).

## Gotchas (all handled in the runner, nothing installed is edited)
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` (JAX takes 75 % of the GPU otherwise; torch shares it). Recorded in settings.
- Playground's `wrapper_torch.RSLRLBraxWrapper` and `learning/train_rsl_rl.py` target rsl_rl 3.x (ActorCritic cfg,
  `get_observations()` resets). rsl_rl 5.5.0 has a different VecEnv/config API, so `vec.py` is a 5.x VecEnv; the
  iteration loop in `run.py` is the same four calls as `OnPolicyRunner.learn` (act, process_env_step, compute_returns,
  update) so the metric point is written after the update, and wall_s counts it. Differences from their wrapper: none in
  time_outs (info["truncation"]), action clip +-1, asymmetric obs (actor "state", critic "privileged_state"), post-reset
  obs, episode_length. Their own rsl_rl config differs from the protocol (fixed lr 3e-4, entropy 0.001).
- jax 0.12 removed `jnp.clip(a_max=)` (Leap env) and `jax.device_put_replicated` (brax 0.14.2): shims in compat.py.
- JIT: reset ~22 s + step ~20 s compile (MJX), excluded from wall_s (warmup before `start_training`); wall_total_s has it.
  The step jit must see identical signatures: reset state is `device_put` and weak types stripped, else it recompiles at
  iteration 2 (17 s).
- MuJoCo Warp prints "solver iterations limit reached" on every step (C level): a 23 GB log and a slower run. Pipe through
  grep -v (queue.sh does). The Go1 xml uses 1 solver iteration as shipped.
- Warp contact buffers (naconmax/naccdmax) are shipped totals sized for 8192 (Go1, Leap) / 2048 (Panda) envs; scaled up
  linearly only when num_envs exceeds that.
- Spot VM preemption happened once (VM -1); queue.sh is restartable.
- Auto-reset is the cached first state (`full_reset=False`, what both shipped scripts use): each env keeps its command /
  target / goal-info for life; only data and obs are reset.

## Matched variant (run.py)
rsl_rl 5.5.0 PPO: MLP [512,256,128] ELU actor and critic, actor AND critic obs normalisation on (protocol names the
actor; critic is on too, as the reference rsl_rl configs), Gaussian scalar std init 1.0, 24 steps x 4096 envs, 5 epochs, 4
minibatches, lr 1e-3 adaptive (KL 0.01), lam 0.95, clip 0.2, value coef 1, max grad norm 1.0, gamma/entropy per task
(0.99/0.01, 0.995/0.005, 0.998/0.002). Env as shipped by the rsl_rl route: domain randomization ON where the env has one
(Go1: yes; Panda: check settings.domain_randomization; Leap: yes).
Metrics (env.py MetricsWrapper, computed from state.data before auto-reset, averaged over all envs and steps of the
iteration): Go1 vel_err_xy = |cmd_xy - local_linvel_xy| (norm), vel_err_yaw = |cmd_yaw - gyro_z|, fall_rate = falls /
episodes ended (fall = done that is not a time-out); also cmd_xy and vel_xy means (evidence). Panda: pos_err_end etc. at
the last step of episodes ended in the iteration, box from xpos, lifted = box z > start z + 4 cm. Leap: rot_err = the env's
own cube orientation error against the goal the step was scored with, goals_per_min = share of steps with error < 0.1 rad
(the env's success_threshold) x 60 / ctrl_dt, drop_rate = falls / episodes ended. `reward` = mean episode return of
episodes ended this iteration (carried forward when none ended: NaN before the first end); `reward_step` = mean step reward.

## Native variant (native.py)
Brax PPO via the train_jax_ppo.py recipe with brax_ppo_config(env): Go1 8192 envs, 200 M steps, unroll 20, 32 minibatches x
4 updates, lr 3e-4, gamma 0.97, entropy 0.01, policy [512,256,128] on "state", value on "privileged_state"; no domain
randomization (script default). One Point per Brax eval callback (num_evals raised from the shipped 10 to 20/40 for
finer curves, training unchanged); metrics are from the deterministic eval rollout (128 eval envs, one episode each), so
they are not comparable one-to-one with the matched training-policy averages. The training-step JIT compile falls in the
first interval; it is estimated (interval 1 minus interval 2) and wall_s is shifted by it from point 2 on (compile_est_s
metric; point 1 keeps the compile in wall_s, flagged compile_in_wall=1).
