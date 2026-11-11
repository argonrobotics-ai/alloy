# SPDX-License-Identifier: Apache-2.0
"""Train Stable-Baselines3 PPO on an Alloy env: ``train("GripperLift-v0")``.

One iteration is 32 env steps of every world. PPO uses a 128x128 tanh network, a small initial action spread (log
std -1.0), a constant learning rate of 3e-4 with updates stopped early at a KL of 0.02, and 4 epochs over 4
minibatches. Observations are normalised by Stable-Baselines3's ``VecNormalize`` with ``epsilon=0.01``: with its
default epsilon GripperLift-v0 stayed at 0 % success in our runs (columns that barely vary while the cube rests are
scaled up to noise), without normalisation it learned part-way and fell back, and with 0.1 PandaReach-v0 did not
learn. What these settings reached, per
seed: docs/guide/rl.md. It prints one JSON line per ``report_every`` iterations. Needs PyTorch and
Stable-Baselines3 (``pip install torch stable-baselines3``).

Example::

    from alloy.envs.wrappers.train_sb3 import train
    summary = train("GripperLift-v0", num_envs=64, iterations=100)
    print(summary["success_rate"], summary["checkpoint"])
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from alloy.envs.wrappers._common import require

log = logging.getLogger("alloy.envs")
HORIZON = 32
OBS_EPSILON = 0.01


def ppo_settings(num_envs: int, iterations: int) -> dict[str, Any]:
    """Keyword arguments of the train command's PPO. Example: ``ppo_settings(512, 200)["n_steps"]`` -> 32."""
    return {"n_steps": HORIZON, "batch_size": max(64, num_envs * HORIZON // 4), "n_epochs": 4, "gamma": 0.99,
            "gae_lambda": 0.95, "ent_coef": 0.0, "max_grad_norm": 1.0, "verbose": 0,
            "learning_rate": 3e-4, "target_kl": 0.02,
            "policy_kwargs": {"log_std_init": -1.0, "net_arch": {"pi": [128, 128], "vf": [128, 128]}}}


def normalized(base: Any) -> Any:
    """``base`` with running observation normalisation (``VecNormalize``, ``epsilon=0.01``: a floor of about 0.1 on
    each column's standard deviation); an env with camera images is returned as it is.

    Example: ``venv = normalized(Sb3VecEnv(env))``."""
    if base._images:
        return base
    from stable_baselines3.common.vec_env import VecNormalize  # noqa: PLC0415

    return VecNormalize(base, norm_obs=True, norm_reward=False, clip_obs=10.0, epsilon=OBS_EPSILON)


def _make_callback(venv: Any, meta: dict[str, Any], every: int, total: int, t0: float) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415

    class Report(BaseCallback):
        """Prints a JSON line per ``every`` iterations and remembers the last chunk."""

        def __init__(self) -> None:
            super().__init__()
            self.iter, self.last = 0, {}

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self.iter += 1
            if self.iter % every == 0 or self.iter == total:
                self.last = self._line()
                print(json.dumps(self.last), flush=True)

        def _line(self) -> dict[str, Any]:
            episodes, successes = venv.tally.take()
            buf = self.model.ep_info_buffer
            ret = sum(e["r"] for e in buf) / len(buf) if buf else None
            rate = round(successes / episodes, 4) if episodes else None
            return {"lib": "sb3", **meta, "iter": self.iter, "samples": self.num_timesteps,
                    "minutes": round((time.perf_counter() - t0) / 60, 2), "episodes": episodes,
                    "success_rate": rate, "mean_return": None if ret is None else round(ret, 3),
                    "invalid_episodes": venv.tally.take_invalid()}

    return Report()


def train(env_id: str, *, num_envs: int = 512, iterations: int = 200, seed: int = 0, device: str | None = None,
          log_dir: str | None = None, report_every: int = 10, env_kwargs: dict | None = None,
          env: Any = None) -> dict[str, Any]:
    """Train PPO on ``env_id`` (or on the ready ``VectorEnv`` ``env``) and return a summary dict with
    ``success_rate`` (of the last report chunk), ``minutes``, ``samples`` and ``checkpoint``.

    Example: ``train("GripperLift-v0", num_envs=64, iterations=100, log_dir="runs/sb3/demo")``."""
    require("sb3")
    from stable_baselines3 import PPO  # noqa: PLC0415

    from alloy.envs.tasks import make  # noqa: PLC0415
    from alloy.envs.wrappers.for_sb3 import Sb3VecEnv  # noqa: PLC0415

    t0 = time.perf_counter()
    raw = env if env is not None else make(env_id, num_envs=num_envs, device=device, seed=seed, **(env_kwargs or {}))
    base = Sb3VecEnv(raw)
    venv = normalized(base)
    out = Path(log_dir or f"runs/sb3/{env_id}")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"env": env_id, "num_envs": raw.num_envs}
    try:
        policy = "MultiInputPolicy" if base._images else "MlpPolicy"
        model = PPO(policy, venv, seed=seed, device="cuda" if str(raw.device).startswith("cuda") else "cpu",
                    **ppo_settings(raw.num_envs, iterations))
        callback = _make_callback(base, meta, report_every, iterations, t0)
        model.learn(iterations * HORIZON * raw.num_envs, callback=callback)
        checkpoint = out / "model.zip"
        model.save(checkpoint)
        if venv is not base:
            venv.save(str(out / "vecnormalize.pkl"))      # the observation statistics the policy needs
    finally:
        raw.close()
    log.info("sb3 training of %s finished: %d samples", env_id, model.num_timesteps)
    return {"success_rate": callback.last.get("success_rate"), "minutes": round((time.perf_counter() - t0) / 60, 2),
            "samples": int(model.num_timesteps), "checkpoint": str(checkpoint)}


__all__ = ["ppo_settings", "train"]
