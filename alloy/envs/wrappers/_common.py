# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the RL-library wrappers: the import guard, observation groups and episode counters."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alloy.envs.errors import EnvError

#: wrapper name -> (module to import, pip package)
LIBRARIES = {"rsl_rl": ("rsl_rl", "rsl-rl-lib"), "sb3": ("stable_baselines3", "stable-baselines3"),
             "skrl": ("skrl", "skrl")}
POLICY = "policy"


def require(lib: str) -> Any:
    """Import the RL library behind wrapper ``lib`` or raise ``EnvError`` with the pip command.

    Example: ``require("rsl_rl")`` -> the ``rsl_rl`` module."""
    if lib not in LIBRARIES:
        raise EnvError(f"no wrapper named {lib!r}", f"use one of {sorted(LIBRARIES)}", field="lib")
    module, package = LIBRARIES[lib]
    try:
        importlib.import_module("torch")
        return importlib.import_module(module)
    except ImportError as exc:
        raise EnvError(f"the {lib} wrapper needs PyTorch and {package}, and one of them is not installed ({exc})",
                       f"pip install torch {package}", field="lib") from exc


def require_vector_env(env: Any, lib: str) -> None:
    """Raise ``EnvError`` unless ``env`` is a ``VectorEnv`` (the task-port envs are not: they have no array
    backend). Example: ``require_vector_env(env, "sb3")``."""
    if getattr(env, "xp", None) is None or not hasattr(env, "task") or not hasattr(env, "stats"):
        raise EnvError(f"the {lib} wrapper takes an alloy.envs.VectorEnv, not {type(env).__name__}",
                       "pass an env from alloy.envs.make(<a ready-made name>) or VectorEnv(scene, task); the task "
                       "ports (alloy.envs.ports) are not supported by the wrappers", field="env")


def require_torch_env(env: Any, lib: str) -> None:
    """Raise ``EnvError`` unless ``env`` returns torch tensors. Example: ``require_torch_env(env, "skrl")``."""
    require_vector_env(env, lib)
    if getattr(env.xp, "name", "") != "torch":
        raise EnvError(f"the {lib} wrapper needs an environment on the torch backend, this one is "
                       f"{getattr(env.xp, 'name', '?')}", "make it with backend='torch' (the default when PyTorch "
                       "is installed)", field="backend")


def group_names(env: Any) -> tuple[str, ...]:
    """The env's state observation groups; a flat env is the one group "policy".

    Example: ``group_names(env)`` -> ("policy", "critic")."""
    return tuple(getattr(env, "obs_groups", ()) or (POLICY,))


def state_groups(env: Any, obs: Any) -> dict[str, Any]:
    """``{group: (W, D) array}`` from an env observation, whichever of its three forms it has: one flat array, a
    camera dict (``"state"`` + images) or a group dict. Images are left out (see ``image_keys``).

    Example: ``state_groups(env, obs)["policy"].shape`` -> (W, D)."""
    if not isinstance(obs, Mapping):
        return {POLICY: obs}
    named = tuple(getattr(env, "obs_groups", ()) or ())
    if named:
        return {g: obs[g] for g in named}
    return {POLICY: obs["state"]}


def image_keys(env: Any) -> list[str]:
    """Names of the env's camera observations. Example: ``image_keys(env)`` -> ["front"]."""
    return list(getattr(env, "cameras", {}) or {})


@dataclass
class EpisodeTally:
    """Counts finished episodes and successes from ``info`` of ``VectorEnv.step``.

    Example: ``tally.add(info); tally.success_rate()``."""

    episodes: int = 0
    successes: int = 0
    invalids: int = 0          # episodes that ended in a frozen (INVALID) world

    def add(self, info: Mapping[str, Any]) -> tuple[int, int]:
        """Add the episodes that ended in this step; returns ``(ended, succeeded)`` of the step."""
        if "_episode" not in info:
            return 0, 0
        done = info["_episode"]
        ended = int(done.sum())
        won = int((info["episode"]["success"] & done).sum())
        frozen = info["episode"].get("invalid")
        self.invalids += 0 if frozen is None else int((frozen & done).sum())
        self.episodes += ended
        self.successes += won
        return ended, won

    def success_rate(self) -> float | None:
        """Successes / episodes so far, None before the first episode ends."""
        return self.successes / self.episodes if self.episodes else None

    def take_invalid(self) -> int:
        """Return the INVALID episodes counted since the last call and start again from zero."""
        out, self.invalids = self.invalids, 0
        return out

    def take(self) -> tuple[int, int]:
        """Return ``(episodes, successes)`` and start again from zero."""
        out = (self.episodes, self.successes)
        self.episodes = self.successes = 0
        return out


__all__ = ["LIBRARIES", "POLICY", "EpisodeTally", "group_names", "image_keys", "require", "require_torch_env",
           "require_vector_env", "state_groups"]
