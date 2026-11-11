# SPDX-License-Identifier: Apache-2.0
"""Row 4 matched runner: rsl_rl 5.5 PPO at the protocol settings on a MuJoCo Playground env (JAX MJX or Warp).

  python run.py --task go2_flat --seed 0 --num-envs 4096 --max-iterations 1000 --max-wall-s 4500 --out f.jsonl [--impl warp]
Writes the protocol record (schema.py): one RunHeader, then one Point per iteration.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

T_PROCESS = time.perf_counter()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX takes 75 % of the GPU otherwise; torch shares it

import jax  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from schema import RunHeader, RunWriter  # copied next to this file on the VM
except ImportError:
    from bench.scorecard.train.schema import RunHeader, RunWriter

import compat  # noqa: E402

compat.apply()
from env import build_stack, wrap  # noqa: E402
from tasks import TASKS  # noqa: E402
from vec import PlaygroundVecEnv  # noqa: E402
import versions  # noqa: E402


def ppo_config(spec, hidden=(512, 256, 128)) -> dict:
    """rsl_rl 5.5 OnPolicyRunner-style config at the protocol settings."""
    return {
        "num_steps_per_env": 24,
        "multi_gpu": None,
        "obs_groups": {"actor": ["state"], "critic": ["privileged_state"]},  # fixed up for symmetric envs
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "clip_param": 0.2,
            "gamma": spec.gamma, "lam": 0.95, "value_loss_coef": 1.0, "entropy_coef": spec.entropy,
            "learning_rate": 1e-3, "max_grad_norm": 1.0, "optimizer": "adam", "use_clipped_value_loss": True,
            "schedule": "adaptive", "desired_kl": 0.01,
        },
        "actor": {"class_name": "MLPModel", "hidden_dims": list(hidden), "activation": "elu",
                  "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": list(hidden), "activation": "elu",
                   "obs_normalization": True},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument("--max-wall-s", type=float, default=4500.0)
    ap.add_argument("--impl", choices=["jax", "warp"], default="jax")
    ap.add_argument("--init-std", type=float, default=1.0, help="protocol value 1.0; other values are labelled diagnostics")
    ap.add_argument("--tag", default="", help="free text stored in settings (diagnostic variants)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = TASKS[args.task]

    from rsl_rl.algorithms import PPO

    torch.manual_seed(args.seed)
    raw, cfg, rand = build_stack(spec, args.num_envs, args.impl)
    stack, key_reset = wrap(raw, spec, rand, args.seed, args.num_envs, cfg.episode_length)
    env = PlaygroundVecEnv(stack, key_reset, spec, args.num_envs, cfg.episode_length, raw)
    env.warmup()

    ppo = ppo_config(spec)
    ppo["actor"]["distribution_cfg"]["init_std"] = args.init_std
    if not env.asymmetric:
        ppo["obs_groups"]["critic"] = ["state"]
    device = "cuda:0"
    obs = env.get_observations()
    alg = PPO.construct_algorithm(obs, env, ppo, device)

    header = RunHeader(
        framework="playground_mjx" if args.impl == "jax" else "playground_warp", task=args.task,
        task_id=spec.env_name, seed=args.seed, num_envs=args.num_envs, variant="matched",
        gpu=torch.cuda.get_device_name(0), versions=versions.collect(),
        settings={**{k: v for k, v in ppo.items() if k != "multi_gpu"}, "impl": args.impl,
                  "env_config": cfg.to_dict(), "ctrl_dt": env.ctrl_dt, "episode_length": cfg.episode_length,
                  "domain_randomization": rand is not None, "autoreset": "cached first state (full_reset=False)",
                  "xla_preallocate": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "note": spec.notes, "tag": args.tag,
                  "critic_obs": ppo["obs_groups"]["critic"]},
    )
    writer = RunWriter(args.out, header, process_start=T_PROCESS)
    steps_per_it = ppo["num_steps_per_env"] * args.num_envs
    last: dict[str, float] = {}
    obs = env.get_observations()
    alg.train_mode()
    writer.start_training()
    t0 = time.perf_counter()
    for it in range(1, args.max_iterations + 1):
        with torch.inference_mode():
            for _ in range(ppo["num_steps_per_env"]):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            alg.compute_returns(obs)
        loss = alg.update()
        m = env.pop_metrics(last)  # reads the GPU: the iteration is finished here
        last = m
        m["lr"] = float(alg.learning_rate)
        m["value_loss"] = float(loss.get("value", float("nan"))) if isinstance(loss, dict) else float("nan")
        writer.point(it, it * steps_per_it, m)
        if it % 10 == 0 or it == 1:
            print(f"it {it} t={time.perf_counter() - t0:.0f}s " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
        if time.perf_counter() - t0 > args.max_wall_s:
            print("wall cap reached", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
