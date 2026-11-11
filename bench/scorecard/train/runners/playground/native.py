# SPDX-License-Identifier: Apache-2.0
"""Row 4 native runner: Brax PPO with the env's shipped config (``learning/train_jax_ppo.py`` logic), same record.

  python native.py --task go2_flat --seed 0 --out f.jsonl [--impl jax|warp] [--max-wall-s 4500] [--num-evals N]
Brax evaluates every ``total_steps / num_evals`` steps instead of every iteration: one Point per eval callback, its
metrics from the deterministic eval rollout (128 eval envs, one episode each, no exploration noise), flagged in
settings. Training itself is exactly the shipped config (env count, unroll, lr, network, no domain randomization,
as the script's default). JIT compile of the training step lands in the first interval; the runner estimates it
(first interval minus second) and shifts ``wall_s`` back by that once known (point 1 keeps the raw value).
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

T_PROCESS = time.perf_counter()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import registry  # noqa: E402
from mujoco_playground.config import locomotion_params, manipulation_params  # noqa: E402

try:
    from schema import RunHeader, RunWriter
except ImportError:
    from bench.scorecard.train.schema import RunHeader, RunWriter

import compat  # noqa: E402
import versions  # noqa: E402
from env import PREFIX, wrap_stack  # noqa: E402
from tasks import TASKS  # noqa: E402


class WallCap(Exception):
    pass


def ppo_params(env_name: str, impl: str):
    if env_name in registry.manipulation._envs:
        return manipulation_params.brax_ppo_config(env_name, impl)
    return locomotion_params.brax_ppo_config(env_name, impl)


def protocol_metrics(spec, m: dict, ctrl_dt: float) -> dict[str, float]:
    """Brax eval metrics -> protocol names. Eval sums per-step metrics over each episode."""
    length = float(m["eval/avg_episode_length"])
    out = {"reward": float(m["eval/episode_reward"]), "eval_episode_length": length}
    for n in spec.step_names:
        total = float(m[f"eval/episode_{PREFIX}{n}"])
        out[n] = total / max(length, 1.0)
    for n in spec.end_names:
        out[n] = float(m[f"eval/episode_{PREFIX}endg_{n}"])
    fell = float(m[f"eval/episode_{PREFIX}fell"])
    if spec.key == "go2_flat":
        out["fall_rate"] = fell
    if spec.key == "cube_reorient":
        out["drop_rate"] = fell
        out["goals_per_min"] = out.pop("goal_hit") * 60.0 / ctrl_dt
    return out


def main() -> int:
    compat.apply()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impl", choices=["jax", "warp"], default="jax")
    ap.add_argument("--max-wall-s", type=float, default=4500.0)
    ap.add_argument("--num-evals", type=int, default=0, help="0 = the shipped value")
    ap.add_argument("--num-timesteps", type=int, default=0, help="0 = the shipped value")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = TASKS[args.task]

    cfg = registry.get_default_config(spec.env_name)
    params = ppo_params(spec.env_name, args.impl)
    if args.num_evals:
        params.num_evals = args.num_evals
    if args.num_timesteps:
        params.num_timesteps = args.num_timesteps
    if args.impl == "warp":
        scale = max(1.0, params.num_envs / spec.ref_envs)
        for k in ("naconmax", "naccdmax"):
            if k in cfg:
                cfg[k] = int(cfg[k] * scale)
    cfg_over = {"impl": args.impl}
    env = registry.load(spec.env_name, config=cfg, config_overrides=cfg_over)
    eval_env = registry.load(spec.env_name, config=registry.get_default_config(spec.env_name), config_overrides=cfg_over)
    train_kwargs = dict(params)
    nf = train_kwargs.pop("network_factory")
    num_eval_envs = train_kwargs.pop("num_eval_envs", 128)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf)

    def wrap_fn(e, episode_length=1000, action_repeat=1, randomization_fn=None):
        return wrap_stack(e, spec, None, episode_length)

    header = RunHeader(
        framework="playground_mjx" if args.impl == "jax" else "playground_warp", task=args.task,
        task_id=spec.env_name, seed=args.seed, num_envs=int(params.num_envs), variant="native",
        gpu=jax.devices()[0].device_kind, versions=versions.collect(),
        settings={"trainer": "brax ppo.train via the train_jax_ppo.py recipe", "impl": args.impl,
                  "ppo_params": params.to_dict(), "env_config": cfg.to_dict(),
                  "metrics_source": "deterministic eval rollouts at Brax eval callbacks (num_eval_envs=%d, one episode each)" % num_eval_envs,
                  "domain_randomization": False, "note": spec.notes, "points": "one per eval callback; iteration = eval index"},
    )
    writer = RunWriter(args.out, header, process_start=T_PROCESS)
    state = {"i": 0, "t": [], "shift": 0.0}
    ctrl_dt = float(env.dt)

    def progress(num_steps, metrics):
        now = time.perf_counter()
        if state["i"] == 0:
            writer.start_training()  # the eval at step 0 happens before the first training epoch
        state["t"].append(now)
        t = state["t"]
        if len(t) == 3 and state["shift"] == 0.0:  # interval 1 holds the training-step compile
            state["shift"] = max(0.0, (t[1] - t[0]) - (t[2] - t[1]))
            writer._t_train += state["shift"]
        m = protocol_metrics(spec, metrics, ctrl_dt)
        m["compile_in_wall"] = 1.0 if len(t) == 2 else 0.0
        m["compile_est_s"] = state["shift"]
        writer.point(state["i"], num_steps, m)
        print(f"eval {state['i']} steps={num_steps} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
        state["i"] += 1
        if now - t[0] - state["shift"] > args.max_wall_s:
            raise WallCap()

    try:
        ppo.train(environment=env, eval_env=eval_env, progress_fn=progress, network_factory=network_factory,
                  seed=args.seed, wrap_env_fn=wrap_fn, num_eval_envs=num_eval_envs, **train_kwargs)
    except WallCap:
        print("wall cap reached", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
