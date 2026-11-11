# mjlab runner notes (row 4)

- mjlab 1.6.0 (PyPI wheel), mujoco + mujoco-warp 3.11.0 (the wheel pins `~=3.11.0`), warp-lang 1.19.0, torch 2.15,
  rsl-rl-lib overridden to 5.5.2 (the wheel pins 5.4.2; the protocol says 5.5.x; runs unchanged). Python 3.12 in a uv venv.
- Headless: `MUJOCO_GL=disable` (run.py sets it; otherwise `import mujoco` dies on an EGL ctypes error), `libegl1`,
  tensorboard logger (wandb is mjlab's default).
- Run: `python run.py --task go2_flat|franka_lift|cube_reorient --seed N --num-envs 4096 --max-iterations K
  --max-wall-s S --out F.jsonl [--variant native] [--robot go2]`. schema.py is copied next to run.py on the VM.
- go2_flat = `Mjlab-Velocity-Flat-Unitree-Go1` (substitution). `--robot go2` = OUR Go2 port (go2.py, Menagerie
  unitree_go2 at 98a70c5, Go1 env cfg reused, feet-only collision, 10 Hz position actuators); not mjlab's.
- franka_lift = `Mjlab-Lift-Cube-Yam` (Yam arm). cube_reorient: mjlab has no task; run.py exits 3.
- Env facts (Go1 flat): step 20 ms (physics 5 ms x decimation 4), episode 20 s (1000 steps), command resample 3-8 s,
  pushes every 1-3 s, friction/COM/encoder-bias randomization, 12 actions, fell_over (70 deg) terminates; the
  command-velocity curriculum only widens after 5000 iterations. Rewards as shipped.
- Matched vs native: the shipped Go1 agent cfg already equals the protocol settings (MLP 512-256-128 ELU, 24 steps,
  5 epochs, 4 minibatches, lr 1e-3 adaptive KL .01, gamma .99, entropy .01, init std 1). Matched applies them
  explicitly; critic obs normalisation is left as shipped (on). Native differs only in not overriding.
  Lift: shipped gamma .99 / entropy .005; matched uses gamma .995.
- Metrics come from a zero-weight termination-term probe (metrics.py): it sees the last state of ending episodes.
  vel_err_xy / vel_err_yaw: mean over all envs and steps of |command - base velocity| (body frame), per step; mjlab's
  own `error_vel_xy` is normalised per episode and not used. fall_rate = terminated (fell_over) / ended episodes.
  reward = mean step reward (mjlab reward manager, dt-scaled). Checkpoints/ONNX export disabled in the timed loop.
- Throughput L4, 4096 envs: ~90k env-steps/s, 1.1 s per iteration (Go1 flat).
