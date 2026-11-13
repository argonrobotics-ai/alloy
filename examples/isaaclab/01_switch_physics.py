# SPDX-License-Identifier: Apache-2.0
"""Runs an unmodified Isaac Lab task on Alloy: the same task, one setting changed (``physics=alloy``).

Needs Isaac Lab 3.0 in the same environment (docs/guide/isaaclab.md): install Isaac Lab as its docs say, then
``pip install alloy-engine`` beside it.
Run:  python 01_switch_physics.py [--task Isaac-Cartpole-Direct] [--num_envs 16] [--steps 100] [--device cuda:0]
      python 01_switch_physics.py physics=newton_mjwarp         # the task's own Newton MuJoCo Warp, to compare
(``physics=`` is Isaac Lab's own token; this script adds ``physics=alloy`` when you give none.)
Prints: which physics stepped the task, env steps per second over the timed steps, how many episodes reset, the
robot's mean base height when the task has a "robot", and Alloy's health summary. The first Alloy run compiles
kernels (minutes); later runs start in seconds. The same switch on Isaac Lab's own command:
``isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=alloy``.
"""
from __future__ import annotations

import argparse
import sys
import time

import warp as wp

wp.config.enable_backward = False   # before any kernel module loads, as Isaac Lab's own scripts do

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

import alloy.isaaclab  # noqa: E402  (makes physics=alloy resolve; Isaac Lab's isaaclab command loads it by itself)


def parse() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="Isaac-Cartpole-Direct")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    add_launcher_args(p)
    p.set_defaults(visualizer=[])
    return p.parse_known_args()


def run(env: gym.Env, steps: int, warmup: int = 5) -> tuple[float, int]:
    """Random actions; returns (env steps per second over the timed steps, episodes reset)."""
    space, dev, n = env.action_space, env.unwrapped.device, env.unwrapped.num_envs
    resets, t0 = 0, time.perf_counter()
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
        with torch.inference_mode():
            _, _, term, trunc, _ = env.step(2 * torch.rand(space.shape, device=dev) - 1)
        resets += int((term | trunc).sum())
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return n * steps / (time.perf_counter() - t0), resets


def main() -> None:
    args, hydra_args = parse()
    if not any(a.split("=", 1)[0] in ("physics", "presets") for a in hydra_args):
        hydra_args.append("physics=alloy")
    sys.argv = [sys.argv[0], *hydra_args]
    import isaaclab_tasks  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    if args.device:
        env_cfg.sim.device = args.device
    solver = type(env_cfg.sim.physics.solver_cfg).__name__
    print(f"task {args.task}: {' '.join(hydra_args)} -> {solver}")
    torch.manual_seed(args.seed)
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        env.reset(seed=args.seed)
        rate, resets = run(env, args.steps)
        robot = env.unwrapped.scene.keys() if hasattr(env.unwrapped.scene, "keys") else []
        height = (f", robot base height {float(env.unwrapped.scene['robot'].data.root_pos_w[:, 2].mean()):.3f} m"
                  if "robot" in robot else "")
        print(f"{args.steps} steps x {args.num_envs} envs: {rate:,.0f} env steps/s, {resets} episodes reset{height}")
        alloy_solver = alloy.isaaclab.active_solver()
        if solver == "AlloySolverCfg" and alloy_solver is not None and alloy_solver.health is not None:
            print(alloy_solver.health.report().line())
        env.close()


if __name__ == "__main__":
    main()
