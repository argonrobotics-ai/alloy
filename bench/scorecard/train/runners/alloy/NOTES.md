# Alloy runner notes (row 4, time to train)

Runner: `run.py` (rsl_rl PPO through `alloy.envs.wrappers.for_rsl_rl.RslRlVecEnv`), matched variant only (Alloy has
no separate native trainer; its `python -m alloy.envs.wrappers.train` uses the matched wrapper too). Matched PPO
settings in `matched.py`; metrics in `metrics.py`; envs in `bench/scorecard/train/alloy_envs/`.
Alloy tree on the VM: `git archive` of origin/main (FEAT-RL and FEAT-LOCO merged) with bench/scorecard/ overlaid
(commit in the status file). main has no root install.sh: `install.sh` here does the install (uv, `.[envs]`, rsl-rl-lib 5.5.2).
VM jobs: `queue_go2.sh` (smoke, matched pilot, native; skips finished jobs; @reboot cron).

## Envs
| | FrankaLift (franka_lift) | LeapReorient (cube_reorient) |
|---|---|---|
| robot | shipped Panda (emb_e2_panda) on its table, props removed | shipped LEAP hand (emb_e9_leap), palm up, its own floor |
| object | 5 cm cube, 0.1 kg, friction 1 (add_box) | 4 cm cube, 50 g (from the robot file) |
| control | dt 0.02 s (50 Hz), 250 steps = 5 s | dt 0.05 s (20 Hz), 200 steps = 10 s |
| actions | 8: delta joint targets (arm 0.05 rad/step, gripper 8 mm/step) | 16: delta joint targets (0.1 rad/step) |
| obs | 42: Panda q/qd (18), cube pose+vel (13), goal pos (3), last action (8) | 78: hand q/qd (32), cube pose+vel (13), goal quat (4), last action (16) |
| randomization | cube x,y +-(0.1, 0.25) m around (0.5, 0), yaw +-180 deg; arm joints +-0.1 rad about home; goal in box x .4-.6, y -.25-.25, z .25-.5 | cube xy +-1 cm, yaw +-180 deg; goal orientation uniform on SO(3), redrawn at reset and each time reached |
| reward | reach + lift + goal + fine goal terms, small action and joint-velocity penalties (the lift reward shape of the common reference tasks; weights in franka_lift.py) | 1/(rot_err+0.1) + 250 reach (tol 0.2 rad) - 50 drop - 1e-4 action |
| ends | time-out; cube below -5 cm | time-out; cube > 10 cm from its start (drop) |

Metrics are read from state before the autoreset (`MeteredEnv.last_metrics`). Episode-end metrics use episodes that
ended in the iteration (carried forward when none ended); rsl_rl runs with `init_at_random_ep_len=True` so ends are
staggered. `goals_per_min` counts goal-reached events per env per simulated minute at tolerance 0.2 rad.

## Change log
- 01:25Z first version of both envs; zero action holds cube, action moves joints, per-world reset ok (tests/bench/test_score_train_envs.py).

## go2_flat (Go2Velocity-v0, FEAT-LOCO; docs/guide/locomotion.md)
Go2 (Menagerie file) on flat ground; 12 position-servo actions (target = 0.25 * action, kp 50, kd 0.5); control 50 Hz
(16 physics steps of 1.25 ms); episode 20 s = 1000 steps; obs 55 (policy) / 79 (critic: + privileged); commands
vx +-1, vy +-0.5, yaw +-1, new every 10 s, 10 % stand. As shipped: pushes ON (every 8 s, 0.1 s, up to +-80 N in x and y),
no domain randomization, no obs noise, contact "force". Falls end the episode (base < 0.15 m or tilt > 60 deg).
Metrics (metrics.py): read from the state the reward is computed on (before the same-step autoreset): vel_err_xy =
norm of (command - base lin vel, xy, body frame), vel_err_yaw = |command - yaw rate|, averaged over envs and the 24
steps; fall_rate = share of ended episodes that were terminations (not time-outs).
Variants: matched (matched.py) and native = `alloy.envs.loco.train.ppo_config` (the documented
`python -m alloy.envs.loco.train --num-envs 4096 --iterations 1000`): the same numbers as matched except the critic
also normalises its input; run through this runner so the record uses the protocol metrics.
