# SPDX-License-Identifier: Apache-2.0
"""Train rsl_rl PPO on an Alloy env in one call; prints one JSON line per chunk of iterations.

Needs ``pip install rsl-rl-lib`` (TensorBoard is optional). Example::

    from alloy.envs.wrappers.train_rsl_rl import train

    summary = train("GripperLift-v0", num_envs=512, iterations=200)
    print(summary["success_rate"], summary["checkpoint"])
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from alloy.envs.wrappers._common import require
from alloy.envs.wrappers.for_rsl_rl import RslRlVecEnv

log = logging.getLogger("alloy.envs")

STEPS_PER_ENV = 32
_BASE_CONFIG: dict[str, Any] = {
    "class_name": "OnPolicyRunner", "num_steps_per_env": STEPS_PER_ENV, "save_interval": 10_000,
    "experiment_name": "alloy", "run_name": "",
    "actor": {"class_name": "MLPModel", "hidden_dims": [128, 128], "activation": "tanh", "obs_normalization": True,
              "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [128, 128], "activation": "tanh", "obs_normalization": True},
    "algorithm": {"class_name": "PPO", "num_learning_epochs": 4, "num_mini_batches": 4, "learning_rate": 3e-4,
                  "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "entropy_coef": 0.001, "desired_kl": 0.01,
                  "max_grad_norm": 1.0, "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2},
}


def __getattr__(name: str) -> Any:
    """``QuietWriter``, built on first use so that importing this module needs no rsl_rl.

    Example: ``cfg["logger"] = {"class_name": "alloy.envs.wrappers.train_rsl_rl:QuietWriter"}``."""
    if name != "QuietWriter":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from rsl_rl.utils.log_writer import LogWriter  # noqa: PLC0415

    class QuietWriter(LogWriter):
        """A log writer that keeps nothing: rsl_rl still tracks episodes and saves checkpoints without TensorBoard."""

        def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
            """Drop the value."""

    QuietWriter.__module__ = __name__
    globals()["QuietWriter"] = QuietWriter
    return QuietWriter


#: Settings that differ from the base for one ready-made env, as "section.key": value (chosen from training runs on
#: one L4)
RECIPES: dict[str, dict[str, Any]] = {
    "PickCube-v0": {"algorithm.gamma": 0.95, "algorithm.num_learning_epochs": 8, "algorithm.num_mini_batches": 8,
                    "algorithm.entropy_coef": 0.003, "actor.hidden_dims": [256, 256, 256],
                    "critic.hidden_dims": [256, 256, 256]},
}


def default_config(iterations: int, seed: int, env_id: str | None = None) -> dict[str, Any]:
    """The PPO settings tuned on GripperLift-v0 (512 worlds, 200 iterations, one L4), with ``RECIPES[env_id]`` on
    top for the envs that have one; success differs from run to run on a GPU.

    ``obs_groups`` is a placeholder: ``train`` fills it from the env (``RslRlVecEnv.default_obs_groups``).
    Example: ``cfg = default_config(200, 0, "PickCube-v0"); cfg["obs_groups"] = venv.default_obs_groups()``."""
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg.update(seed=seed, max_iterations=iterations, obs_groups={"actor": ["policy"], "critic": ["policy"]},
               logger=_logger_config())
    for key, value in RECIPES.get(env_id or "", {}).items():
        section, name = key.split(".")
        cfg[section][name] = copy.deepcopy(value)
    return cfg


def _logger_config() -> Any:
    if importlib.util.find_spec("tensorboard") is not None:
        return "tensorboard"
    return {"class_name": f"{__name__}:QuietWriter"}


def train(env_id: str, *, num_envs: int = 512, iterations: int = 200, seed: int = 0, device: str | None = None,
          log_dir: str | None = None, report_every: int = 10, env_kwargs: dict[str, Any] | None = None,
          env: Any = None) -> dict[str, Any]:
    """Train PPO on ``env_id`` and return ``{success_rate, minutes, samples, checkpoint}``.

    One JSON line per ``report_every`` iterations goes to stdout. The final checkpoint is saved in ``log_dir``
    (default ``runs/rsl_rl/<env_id>``). A ready torch-backend ``VectorEnv`` passed as ``env`` is used instead of
    making one (it is closed at the end). Example: ``train("GripperLift-v0", num_envs=512, iterations=200)``."""
    require("rsl_rl")
    from alloy.envs import make  # noqa: PLC0415

    t0 = time.perf_counter()
    log_dir = log_dir or os.path.join("runs", "rsl_rl", env_id)
    if env is None:
        env = make(env_id, num_envs=num_envs, device=device, seed=seed, backend="torch", **(env_kwargs or {}))
    venv = RslRlVecEnv(env)
    try:
        return _learn(venv, env_id, iterations, seed, log_dir, max(1, report_every), t0)
    finally:
        venv.close()


def _learn(venv: RslRlVecEnv, env_id: str, iterations: int, seed: int, log_dir: str, chunk: int,
           t0: float) -> dict[str, Any]:
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    cfg = default_config(iterations, seed, env_id)
    cfg["obs_groups"] = venv.default_obs_groups()
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(venv, cfg, log_dir=log_dir, device=str(venv.device))
    last: dict[str, Any] = {}

    def report(done: int) -> None:
        last.update(_report(venv, env_id, done, t0))

    learn_reporting(runner, iterations, chunk, report)
    checkpoint = os.path.join(log_dir, f"model_{iterations}.pt")
    runner.save(checkpoint)                      # records iter = iterations (rsl_rl's own final save: the same file)
    log.info("rsl_rl: %s trained %d iterations, checkpoint %s", env_id, iterations, checkpoint)
    return {"success_rate": last.get("success_rate"), "minutes": last.get("minutes"),
            "samples": iterations * STEPS_PER_ENV * venv.num_envs, "checkpoint": checkpoint}


def learn_reporting(runner: Any, iterations: int, every: int, report: Callable[[int], None],
                    random_start: bool = False) -> None:
    """Run ``iterations`` PPO iterations in ONE ``runner.learn`` call, calling ``report(done)`` after every ``every``
    iterations and after the last; afterwards ``runner.current_learning_iteration`` is the count done.

    rsl_rl 5.x's ``learn`` opens a new log writer (a new TensorBoard file) at every call and leaves the iteration
    count one short (the last 0-based index), so learning in chunks lost one iteration per chunk in every checkpoint's
    ``iter`` and split the log. Here the report rides on the runner's per-iteration ``logger.log``, which also sets
    the count to the iterations done, so rsl_rl's own saves record it and a resume starts where training stopped.
    ``random_start`` is rsl_rl's ``init_at_random_ep_len`` (episodes start at random lengths).
    Example: ``learn_reporting(runner, 200, 10, lambda done: print(done))``."""
    start, logger = runner.current_learning_iteration, getattr(runner, "logger", None)
    original = getattr(logger, "log", None)
    if original is None:                         # no per-iteration logger: chunks, the count kept here
        done = 0
        while done < iterations:
            step = min(every, iterations - done)
            runner.learn(num_learning_iterations=step, init_at_random_ep_len=random_start and done == 0)
            done += step
            runner.current_learning_iteration = start + done
            report(done)
        return
    done = [0]

    def log_and_report(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        done[0] += 1
        runner.current_learning_iteration = start + done[0]
        if done[0] % every == 0 or done[0] == iterations:
            report(done[0])
        return out

    logger.log = log_and_report
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=random_start)
    finally:
        del logger.log                           # the class's own method again


def _report(venv: RslRlVecEnv, env_id: str, done: int, t0: float) -> dict[str, Any]:
    episodes, successes, mean_return = venv.take_stats()
    line = {"lib": "rsl_rl", "env": env_id, "num_envs": venv.num_envs, "iter": done,
            "samples": done * STEPS_PER_ENV * venv.num_envs, "minutes": round((time.perf_counter() - t0) / 60, 2),
            "episodes": episodes, "success_rate": round(successes / episodes, 3) if episodes else None,
            "mean_return": None if mean_return is None else round(mean_return, 3),
            "invalid_episodes": venv.tally.take_invalid()}
    print(json.dumps(line), flush=True)
    return line


__all__ = ["RECIPES", "default_config", "learn_reporting", "train"]
