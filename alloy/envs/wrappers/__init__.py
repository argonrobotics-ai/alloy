# SPDX-License-Identifier: Apache-2.0
"""Adapters from ``alloy.envs.VectorEnv`` to RL libraries: rsl_rl, Stable-Baselines3 and skrl.

Each library is optional and imported only when its wrapper is used (``pip install rsl-rl-lib``,
``stable-baselines3`` or ``skrl``, each with PyTorch). Example::

    from alloy.envs import make
    from alloy.envs.wrappers import wrap

    env = wrap(make("GripperLift-v0", num_envs=512), "rsl_rl")     # or "sb3", "skrl"

Train with one command: ``python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl``.
Observation groups (``Task(obs={"policy": [...], "critic": [...]})``) reach rsl_rl as TensorDict groups and skrl as
its state (critic) input; Stable-Baselines3 has one observation, so it gets the policy group.
"""

from __future__ import annotations

from typing import Any

from alloy.envs.wrappers._common import LIBRARIES, require, require_vector_env

_CLASSES = {"rsl_rl": ("alloy.envs.wrappers.for_rsl_rl", "RslRlVecEnv"),
            "sb3": ("alloy.envs.wrappers.for_sb3", "Sb3VecEnv"),
            "skrl": ("alloy.envs.wrappers.for_skrl", "SkrlEnv")}


def wrap(env: Any, lib: str, **kwargs: Any) -> Any:
    """Wrap a ``VectorEnv`` for an RL library: ``lib`` is "rsl_rl", "sb3" or "skrl".

    Example: ``wrap(make("PandaReach-v0", num_envs=1024), "rsl_rl")``."""
    require(lib)
    require_vector_env(env, lib)
    return _load(*_CLASSES[lib])(env, **kwargs)


def _load(module: str, name: str) -> Any:
    import importlib  # noqa: PLC0415

    return getattr(importlib.import_module(module), name)


def __getattr__(name: str) -> Any:
    for module, cls in _CLASSES.values():
        if cls == name:
            return _load(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LIBRARIES", "RslRlVecEnv", "Sb3VecEnv", "SkrlEnv", "wrap"]
