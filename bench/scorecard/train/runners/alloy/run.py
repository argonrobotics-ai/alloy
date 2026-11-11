# SPDX-License-Identifier: Apache-2.0
"""Row 4 (time to train), Alloy runner: rsl_rl PPO through Alloy's wrapper, protocol record out.

  python run.py --task franka_lift --seed 0 --num-envs 4096 --max-iterations 3000 --max-wall-s 4500 --out run.jsonl
Tasks: franka_lift, cube_reorient (bench envs in bench/scorecard/train/alloy_envs); go2_flat (Go2Velocity-v0, FEAT-LOCO).
Variants: matched (protocol PPO) and native (the settings of alloy.envs.loco.train, see NOTES.md).
"""

from __future__ import annotations

import argparse
import sys
import time
from importlib.metadata import version
from pathlib import Path

T_PROCESS = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from schema import RunHeader, RunWriter  # copied next to the runner on the VM
except ImportError:  # in the worktree
    from bench.scorecard.train.schema import RunHeader, RunWriter

import torch  # noqa: E402

from matched import STEPS_PER_ENV, matched_config, native_config  # noqa: E402
from metrics import Meter, hook_go2_metrics  # noqa: E402

TASK_IDS = {"go2_flat": "Go2Velocity-v0", "franka_lift": "FrankaLift", "cube_reorient": "LeapReorient"}


class MaxWallReached(Exception):
    """Raised from the logging hook when the wall-clock budget is spent."""


def make_env(task: str, num_envs: int, seed: int):
    """The torch-backend Alloy env for a protocol task key."""
    if task == "franka_lift":
        from bench.scorecard.train.alloy_envs import make_franka_lift
        return make_franka_lift(num_envs, seed=seed, backend="torch")
    if task == "cube_reorient":
        from bench.scorecard.train.alloy_envs import make_leap_reorient
        return make_leap_reorient(num_envs, seed=seed, backend="torch")
    from alloy.envs.loco import go2_velocity
    env = go2_velocity(num_envs=num_envs, seed=seed, backend="torch")
    hook_go2_metrics(env)
    return env


def metered_vecenv(env, meter: Meter):
    """An rsl_rl VecEnv over ``env`` that feeds ``meter`` after every step."""
    from alloy.envs.wrappers.for_rsl_rl import RslRlVecEnv

    class Metered(RslRlVecEnv):
        def step(self, actions):
            obs, rew, dones, extras = super().step(actions)
            meter.record(self.env.last_metrics, rew, dones, extras.get("time_outs"))
            return obs, rew, dones, extras

    return Metered(env)


def env_facts(task: str, env) -> dict:
    """Env facts for the header: go2_flat pushes / randomization / commands as the env ships them."""
    if task != "go2_flat":
        return {}
    from dataclasses import asdict
    cfg = env.config
    return {"pushes": asdict(cfg.pushes) if cfg.pushes else None, "commands": asdict(cfg.commands),
            "domain_randomization": "none (scene_randomize empty, joint_jitter 0)", "action_scale": cfg.action_scale,
            "contact": str(cfg.contact), "physics_steps_per_control": 16}


def versions() -> dict[str, str]:
    import alloy
    out = {p: version(p) for p in ("warp-lang", "mujoco", "rsl-rl-lib", "torch")}
    out["alloy"] = alloy.__version__
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASK_IDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, required=True)
    ap.add_argument("--max-wall-s", type=float, default=4500.0)
    ap.add_argument("--variant", choices=("matched", "native"), default="matched")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from rsl_rl.runners import OnPolicyRunner

    cfg = (matched_config if args.variant == "matched" else native_config)(args.task, args.seed, args.max_iterations)
    env = make_env(args.task, args.num_envs, args.seed)
    meter = Meter(args.task, args.num_envs, float(env.sim.dt))
    venv = metered_vecenv(env, meter)
    cfg["obs_groups"] = venv.default_obs_groups()
    cfg["logger"] = {"class_name": "alloy.envs.wrappers.train_rsl_rl:QuietWriter"}
    runner = OnPolicyRunner(venv, cfg, log_dir=None, device=str(venv.device))
    header = RunHeader(
        framework="alloy", task=args.task, task_id=TASK_IDS[args.task], seed=args.seed, num_envs=args.num_envs,
        variant=args.variant, gpu=torch.cuda.get_device_name(0), versions=versions(),
        settings={**cfg["algorithm"], "num_steps_per_env": STEPS_PER_ENV, "max_wall_s": args.max_wall_s,
                  "control_dt": float(env.sim.dt), "max_steps": int(env.task.max_steps),
                  **env_facts(args.task, env)})
    writer = RunWriter(args.out, header, process_start=T_PROCESS)
    warm_up(venv)
    return train(runner, meter, writer, args)


def warm_up(venv) -> None:
    """A few steps so kernel compile and graph capture stay out of the timed loop, then back to a fresh start."""
    zeros = torch.zeros(venv.num_envs, venv.num_actions, device=venv.device)
    for _ in range(6):
        venv.step(zeros)
    venv.reset()


def train(runner, meter: Meter, writer: RunWriter, args) -> int:
    steps_per_iter = STEPS_PER_ENV * args.num_envs
    n = {"it": 0}
    orig_log = runner.logger.log

    def log_and_record(*a, **kw):
        torch.cuda.synchronize()
        n["it"] += 1
        m = meter.flush()
        pt = writer.point(n["it"], n["it"] * steps_per_iter, m)
        print(f"it {pt.iteration} wall {pt.wall_s:.0f}s " + " ".join(f"{k}={v:.3g}" for k, v in m.items()), flush=True)
        if pt.wall_s > args.max_wall_s:
            raise MaxWallReached
        return orig_log(*a, **kw)

    runner.logger.log = log_and_record
    runner.save = lambda *a, **k: None
    meter._reset()
    writer.start_training()
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except MaxWallReached:
        print("max wall reached; stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
