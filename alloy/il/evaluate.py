# SPDX-License-Identifier: Apache-2.0
"""Policy evaluation: roll a policy out in a ``VectorEnv`` and report its success rate with an interval.

Every world runs the same number of episodes, so short (failed or quick) episodes are not over-counted the way
"the first N episodes to finish" over-counts them. Example::

    import alloy.envs
    from alloy.il import evaluate

    env = alloy.envs.make("GripperLift-v0", num_envs=16)
    result = evaluate(env, policy, episodes=64, seed=1)       # policy(obs) -> (num_envs, A) actions
    print(result.success_rate, result.interval, result.mean_return)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.il.support import require_vector_env


@dataclass
class EvalResult:
    """Per-episode outcomes of an evaluation. Example: ``evaluate(env, policy, episodes=64).summary()``."""

    success: np.ndarray        # (N,) bool: the task's success check on each episode's last step; False if invalid
    returns: np.ndarray        # (N,) float: sum of rewards
    lengths: np.ndarray        # (N,) int: steps
    worlds: np.ndarray         # (N,) int: the world each episode ran in
    seed: int | None = None
    invalid: np.ndarray | None = None      # (N,) bool: the engine marked the world INVALID (froze it) at some step

    def __post_init__(self) -> None:
        if self.invalid is None:
            self.invalid = np.zeros(self.success.shape, bool)
        self.success = np.asarray(self.success, bool) & ~self.invalid        # a frozen world is never a success

    @property
    def episodes(self) -> int:
        """Number of episodes N. Example: ``result.episodes``."""
        return int(self.success.shape[0])

    @property
    def success_rate(self) -> float:
        """Share of successful episodes, 0 to 1. Example: ``result.success_rate``."""
        return float(self.success.mean()) if self.episodes else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        """95 % Wilson score interval of the success rate. Example: ``lo, hi = result.interval``."""
        return wilson_interval(int(self.success.sum()), self.episodes)

    @property
    def mean_return(self) -> float:
        """Mean episode return. Example: ``result.mean_return``."""
        return float(self.returns.mean()) if self.episodes else float("nan")

    @property
    def mean_length(self) -> float:
        """Mean episode length in steps. Example: ``result.mean_length``."""
        return float(self.lengths.mean()) if self.episodes else float("nan")

    def summary(self) -> dict[str, Any]:
        """A JSON-ready summary. Example: ``json.dumps(result.summary())``."""
        lo, hi = self.interval
        return {"episodes": self.episodes, "successes": int(self.success.sum()),
                "invalid": int(self.invalid.sum()), "success_rate": self.success_rate,
                "success_interval_95": [lo, hi], "mean_return": self.mean_return, "mean_length": self.mean_length,
                "seed": self.seed}


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval of a proportion (95 % by default). Example: ``wilson_interval(45, 50)``."""
    if n <= 0:
        return (0.0, 1.0)
    p, z2 = successes / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(env: Any, policy: Callable[[Any], Any], episodes: int = 100, *, seed: int | None = 0,
             max_steps: int | None = None) -> EvalResult:
    """Run ``policy`` for at least ``episodes`` episodes (rounded up to the same number in every world) and return
    their outcomes. ``env`` is a ``VectorEnv`` or a ``Recorder`` around one (the rollouts are then recorded too);
    its autoreset mode must be "same_step" (the default). ``policy(obs)`` gets the env's batched observation and
    returns ``(num_envs, A)`` actions. ``max_steps`` bounds the total control steps (default: enough for every
    episode to reach the task's own ``max_steps``). ``result.invalid`` marks the episodes whose world the engine
    froze (INVALID) at any step; they stay in the count and are never successes. It is read from ``info["invalid"]``
    at every step, because the env's reset clears the health record ``sim.health()`` reads.

    Example: ``evaluate(env, lambda obs: env.action_space.sample(), episodes=32).success_rate``."""
    recorder = env if hasattr(env, "world_quota") else None
    base = recorder.env if recorder is not None else env   # a Recorder keeps its VectorEnv in .env
    require_vector_env(base, "evaluate")
    if base.autoreset != "same_step":
        raise EnvError(f"evaluate needs autoreset='same_step', this env has {base.autoreset!r}",
                       "build the env with the default autoreset", field="autoreset")
    if episodes < 1:
        raise EnvError(f"episodes is {episodes}", "pass a positive number of episodes", field="episodes")
    worlds = int(base.num_envs)
    per_world = math.ceil(episodes / worlds)
    budget = max_steps if max_steps is not None else per_world * (int(base.task.max_steps) + 1)
    if recorder is None:
        return _rollouts(env, base, policy, per_world, budget, seed)
    quota, recorder.world_quota = recorder.world_quota, per_world      # it keeps the counted episodes, no others
    try:
        return _rollouts(env, base, policy, per_world, budget, seed)
    finally:
        recorder.world_quota = quota


def _rollouts(env: Any, base: Any, policy: Callable[[Any], Any], per_world: int, budget: int,
              seed: int | None) -> EvalResult:
    """Step until every world has ended ``per_world`` episodes; later episodes of a world are not counted."""
    worlds, to_np = int(base.num_envs), base.xp.to_numpy
    count = np.zeros(worlds, np.int64)
    bad = np.zeros(worlds, bool)                          # worlds INVALID at some step of their running episode
    rows: list[tuple[np.ndarray, ...]] = []
    obs, _ = env.reset(seed=seed)
    for _ in range(budget):
        obs, _, _, _, info = env.step(policy(obs))
        if "invalid" in info:
            bad |= np.asarray(to_np(info["invalid"]), bool)
        if "_episode" in info:
            ended = np.asarray(to_np(info["_episode"]), bool)
            ids = np.flatnonzero(ended & (count < per_world))
            ep = info["episode"]
            rows.append((np.asarray(to_np(ep["success"]), bool)[ids], np.asarray(to_np(ep["r"]), np.float64)[ids],
                         np.asarray(to_np(ep["l"]), np.int64)[ids], ids, bad[ids]))
            count[ids] += 1
            bad[ended] = False
        if np.all(count >= per_world):
            break
    else:
        raise EnvError(f"{int(count.sum())} of {per_world * worlds} episodes ended within {budget} steps",
                       "raise max_steps, or give the task a max_steps so every episode ends", field="max_steps")
    success, returns, lengths, ids, invalid = (np.concatenate(c) for c in zip(*rows))
    return EvalResult(success=success, returns=returns, lengths=lengths, worlds=ids, seed=seed, invalid=invalid)


__all__ = ["EvalResult", "evaluate", "wilson_interval"]
