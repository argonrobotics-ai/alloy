# SPDX-License-Identifier: Apache-2.0
"""Train the Go2 to walk: PPO (rsl_rl) on the Go2 velocity environment, through ``alloy.envs.wrappers``.

Run (needs PyTorch and ``pip install rsl-rl-lib``; a GPU is strongly advised)::

    python -m alloy.envs.loco.train --num-envs 4096 --iterations 400 --out runs/go2

It prints one JSON line per 25 iterations (episode return and length, the share of episodes that ended in a fall,
each reward term's mean) and appends it to ``<out>/curve.jsonl``; ``<out>/model_last.pt`` is the checkpoint. At
the end it rolls the policy out under fixed commands and prints the speed it reached (``evaluate``). From Python::

    from alloy.envs.loco.train import train
    summary = train(num_envs=4096, iterations=400, out="runs/go2")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from alloy.envs.errors import EnvError
from alloy.envs.loco.go2 import Go2VelocityConfig, go2_velocity

STEPS_PER_ENV = 24
REPORT_EVERY = 25
EVAL_COMMANDS = ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0))


def ppo_config(iterations: int, seed: int) -> dict[str, Any]:
    """The rsl_rl runner settings used for the Go2 walk. Example: ``ppo_config(1000, 0)["algorithm"]["gamma"]``."""
    from alloy.envs.wrappers.train_rsl_rl import default_config  # noqa: PLC0415

    cfg = default_config(iterations, seed)
    net = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True}
    cfg["actor"] = {**net, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0,
                                                "std_type": "scalar"}}
    cfg["critic"] = dict(net)
    cfg["num_steps_per_env"] = STEPS_PER_ENV
    cfg["algorithm"].update(num_learning_epochs=5, learning_rate=1e-3, entropy_coef=0.01)
    return cfg


class _Tally:
    """Means over a reporting window: reward terms per step, and the episodes that ended."""

    def __init__(self) -> None:
        self.terms: dict[str, float] = {}
        self.steps = self.episodes = self.falls = 0
        self.returns = self.lengths = 0.0

    def add(self, terminated: Any, info: dict[str, Any]) -> None:
        self.steps += 1
        for name, value in info["terms"].items():
            self.terms[name] = self.terms.get(name, 0.0) + float(value.float().mean())
        if "_episode" not in info:
            return
        done, episode = info["_episode"], info["episode"]
        self.episodes += int(done.sum())
        self.falls += int((terminated & done).sum())
        self.returns += float(episode["r"][done].sum())
        self.lengths += float(episode["l"][done].float().sum())

    def take(self) -> dict[str, Any]:
        n, steps = max(self.episodes, 1), max(self.steps, 1)
        row = {"episodes": self.episodes, "episode_return": round(self.returns / n, 3),
               "episode_steps": round(self.lengths / n, 1), "fell_share": round(self.falls / n, 3),
               "terms": {k: round(v / steps, 4) for k, v in self.terms.items()}}
        self.__init__()
        return row


def _watch(env: Any, tally: _Tally) -> None:
    """Make ``env.step`` feed ``tally`` (the wrapper keeps only what rsl_rl logs)."""
    inner = env.step

    def step(actions: Any) -> Any:
        out = inner(actions)
        tally.add(out[2], out[4])
        return out

    env.step = step


def train(num_envs: int = 4096, iterations: int = 400, seed: int = 0, out: str = "runs/go2",
          device: str | None = None, config: Go2VelocityConfig | None = None,
          terrain: str = "flat") -> dict[str, Any]:
    """Train PPO on the Go2 velocity environment; returns ``{"checkpoint", "curve", "minutes", "evaluation"}``.
    ``terrain="rough"`` trains on the terrain field with its level curriculum (``alloy.envs.loco.go2_rough``); each
    printed line then also has ``terrain_level``, the mean level of the worlds.

    Example: ``train(num_envs=4096, iterations=400, out="runs/go2")["evaluation"][0]``."""
    from alloy.envs.wrappers import wrap  # noqa: PLC0415
    from alloy.envs.wrappers._common import require  # noqa: PLC0415

    require("rsl_rl")
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    os.makedirs(out, exist_ok=True)
    env = _make_env(terrain, num_envs, device, seed, config)
    tally = _Tally()
    _watch(env, tally)
    venv = wrap(env, "rsl_rl")
    cfg = ppo_config(iterations, seed)
    cfg["obs_groups"] = venv.default_obs_groups()
    runner = OnPolicyRunner(venv, cfg, log_dir=out, device=str(venv.device))
    checkpoint, curve = os.path.join(out, "model_last.pt"), os.path.join(out, "curve.jsonl")
    done, t0 = 0, time.perf_counter()
    while done < iterations:
        chunk = min(REPORT_EVERY, iterations - done)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=done == 0)
        done += chunk
        row = {"iter": done, "minutes": round((time.perf_counter() - t0) / 60, 2),
               "samples": done * STEPS_PER_ENV * num_envs, **tally.take()}
        if hasattr(env, "levels"):
            row["terrain_level"] = round(float(env.levels.float().mean()), 3)
        print(json.dumps(row), flush=True)
        with open(curve, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        runner.save(checkpoint)
    rows = evaluate(venv, runner.get_inference_policy(device=str(venv.device)))
    minutes = round((time.perf_counter() - t0) / 60, 2)
    venv.close()
    return {"checkpoint": checkpoint, "curve": curve, "minutes": minutes, "evaluation": rows}


def _make_env(terrain: str, num_envs: int, device: str | None, seed: int, config: Any) -> Any:
    if terrain == "flat":
        return go2_velocity(num_envs=num_envs, device=device, seed=seed, config=config, backend="torch")
    if terrain != "rough":
        raise EnvError(f"terrain is {terrain!r}", "use 'flat' or 'rough'", field="terrain")
    from alloy.envs.loco.rough import go2_rough  # noqa: PLC0415

    return go2_rough(num_envs=num_envs, device=device, seed=seed, config=config, backend="torch")


def evaluate(venv: Any, policy: Any, seconds: float = 6.0) -> list[dict[str, Any]]:
    """Roll ``policy`` out under each of ``EVAL_COMMANDS`` (pushes stay as the environment has them) and print one
    JSON line per command: the commanded and the reached base velocity (mean over worlds and over the last
    three quarters of the rollout), how many worlds fell, how often feet were off the ground and non-foot contacts.

    Example: ``evaluate(wrap(env, "rsl_rl"), runner.get_inference_policy())``."""
    import torch  # noqa: PLC0415

    env, rows = venv.unwrapped, []
    steps = max(4, int(round(seconds / float(env.sim.dt))))
    for command in EVAL_COMMANDS:
        env.commands.set(command)
        obs, _ = venv.reset()
        vel, swing, other, fell = [], [], [], torch.zeros(env.num_envs, dtype=torch.bool, device=venv.device)
        for k in range(steps):
            with torch.no_grad():
                obs, _reward, _dones, _extras = venv.step(policy(obs))
            s = env.state
            fell |= s["steps"] < k + 1                              # the world's episode restarted: it fell
            if k >= steps // 4:
                vel.append(torch.cat([s["base.lin_vel_b"][:, 0:2], s["base.ang_vel_b"][:, 2:3]], -1))
                swing.append(1.0 - s["feet.contact"].mean())
                other.append(s["body.contacts"].mean())
        reached = torch.stack(vel).mean((0, 1))
        rows.append({"command": list(command), "reached": [round(float(v), 3) for v in reached],
                     "worlds_fell": int(fell.sum()), "worlds": env.num_envs,
                     "feet_off_ground_share": round(float(torch.stack(swing).mean()), 3),
                     "non_foot_contacts": round(float(torch.stack(other).mean()), 4)})
        print(json.dumps(rows[-1]), flush=True)
    env.commands.release()
    return rows


def main(argv: list[str] | None = None) -> int:
    """Command line: ``python -m alloy.envs.loco.train --num-envs 4096 --iterations 400 --out runs/go2``."""
    parser = argparse.ArgumentParser(description="Train the Go2 to walk with rsl_rl PPO.")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/go2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--terrain", default="flat", choices=("flat", "rough"))
    args = parser.parse_args(argv)
    summary = train(args.num_envs, args.iterations, args.seed, args.out, args.device, terrain=args.terrain)
    print(json.dumps({"done": True, "checkpoint": summary["checkpoint"], "minutes": summary["minutes"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
