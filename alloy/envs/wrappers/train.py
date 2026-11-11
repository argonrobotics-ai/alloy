# SPDX-License-Identifier: Apache-2.0
"""Train a policy on a ready-made environment with an RL library, in one command.

    python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl
    python -m alloy.envs.wrappers.train PandaReach-v0 --lib skrl --num-envs 1024 --iterations 150
    python -m alloy.envs.wrappers.train GripperLift-v0 --lib sb3 --num-envs 4 --iterations 2 --device cpu   # smoke
    python -m alloy.envs.wrappers.train my_package.my_envs:make_env --lib rsl_rl      # your own environment

The environment is a ready-made name, or ``module:function`` for your own: a function
``make_env(num_envs=, device=, seed=)`` that returns a ``VectorEnv`` (or a subclass) on the torch backend.

``--lib``: rsl_rl (``pip install rsl-rl-lib``), sb3 (``pip install stable-baselines3``) or skrl (``pip install
skrl``); each needs PyTorch. One iteration is 32 steps of every world, then a PPO update. Prints one JSON line per
``--report-every`` iterations (episodes that ended in them and their success rate) and a last line with the summary;
the trained policy is saved under ``--log-dir`` (default ``runs/<lib>/<env>``). From Python::

    from alloy.envs.wrappers.train import train
    summary = train("GripperLift-v0", lib="rsl_rl", num_envs=512, iterations=200)
    print(summary["success_rate"], summary["checkpoint"])

The PPO settings are one fixed set per library, chosen on GripperLift-v0; they are a starting point, not tuned per
environment. For your own settings use the wrapper with the library's own trainer (``alloy.envs.wrappers.wrap``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from alloy.envs.errors import EnvError
from alloy.envs.wrappers._common import LIBRARIES, require

log = logging.getLogger("alloy.envs")
STEPS_PER_ITERATION = 32


def train(env_id: str, *, lib: str = "rsl_rl", **options: Any) -> dict[str, Any]:
    """Train PPO from library ``lib`` on environment ``env_id``; returns the summary dict.

    Options (all optional): ``num_envs`` (512), ``iterations`` (200), ``seed`` (0), ``device`` (GPU if there is
    one), ``log_dir``, ``report_every`` (10), ``env_kwargs`` (passed to ``alloy.envs.make``) or ``env`` (a ready
    ``VectorEnv`` to train on instead of making one).

    Example: ``train("GripperLift-v0", lib="skrl", num_envs=256, iterations=100)["success_rate"]``."""
    require(lib)
    module = importlib.import_module(f"alloy.envs.wrappers.train_{lib}")
    if ":" in env_id and options.get("env") is None:
        options["env"] = env_from_factory(env_id, options.get("num_envs", 512), options.get("device"),
                                          options.get("seed", 0), options.pop("env_kwargs", None))
    log.info("training %s with %s: %s", env_id, lib, {k: v for k, v in options.items() if k != "env"})
    return module.train(env_id, **options)


def env_from_factory(spec: str, num_envs: int, device: str | None, seed: int,
                     env_kwargs: dict[str, Any] | None = None) -> Any:
    """Build an environment from ``"module:function"``: imports the module and calls
    ``function(num_envs=, device=, seed=, **env_kwargs)``.

    Example: ``env_from_factory("my_package.my_envs:make_env", 64, None, 0)``."""
    module_name, _, attr = spec.partition(":")
    fix = "pass a ready-made name (GripperLift-v0) or 'module:function', e.g. my_package.my_envs:make_env"
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError, ValueError) as exc:
        raise EnvError(f"cannot load the environment function {spec!r} ({exc})", fix, field="env") from exc
    env = factory(num_envs=num_envs, device=device, seed=seed, **(env_kwargs or {}))
    if not hasattr(env, "step") or not hasattr(env, "num_envs"):
        raise EnvError(f"{spec!r} returned {type(env).__name__}, not an environment",
                       "return an alloy.envs.VectorEnv (or a subclass) from the function", field="env")
    return env


def parser() -> argparse.ArgumentParser:
    """The command line's argument parser. Example: ``parser().parse_args(["PandaReach-v0"])``."""
    ap = argparse.ArgumentParser(prog="python -m alloy.envs.wrappers.train", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("env", help="a ready-made environment, e.g. GripperLift-v0 (python -c 'import alloy.envs; "
                                "print(sorted(alloy.envs.REGISTRY))' lists them), or module:function for your own")
    ap.add_argument("--lib", default="rsl_rl", choices=sorted(LIBRARIES), help="the RL library (default rsl_rl)")
    ap.add_argument("--num-envs", type=int, default=512, help="parallel worlds (default 512)")
    ap.add_argument("--iterations", type=int, default=200,
                    help=f"PPO iterations of {STEPS_PER_ITERATION} steps per world (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default: the GPU if there is one)")
    ap.add_argument("--log-dir", default=None, help="where the policy is saved (default runs/<lib>/<env>)")
    ap.add_argument("--report-every", type=int, default=10, help="iterations per printed JSON line (default 10)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code. Example: ``main(["GripperLift-v0", "--lib", "sb3"])``."""
    args = parser().parse_args(argv)
    try:
        summary = train(args.env, lib=args.lib, num_envs=args.num_envs, iterations=args.iterations, seed=args.seed,
                        device=args.device, log_dir=args.log_dir, report_every=args.report_every)
    except EnvError as exc:
        print(f"error: {exc.message}\nfix: {exc.fix}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": summary}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
