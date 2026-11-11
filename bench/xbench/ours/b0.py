# SPDX-License-Identifier: Apache-2.0
"""SB-7 B0 random-legal action traces, numpy only (P3-2: imported by every engine's runner, no Alloy import).

The same arithmetic as ``alloy.bully.stress.random_legal_traces`` + ``alloy.bully.actions.legalize`` (a CPU test,
tests/xbench/test_ours_b0.py, checks the two give bitwise-equal traces), so an engine fed seed ``s`` drives exactly
the actions Alloy's SB-7 campaign drives with seed ``s``:

- Ornstein-Uhlenbeck per actuator in units of the half-range: ``x += -theta x dt + sigma sqrt(2 theta dt) N(0, 1)``
  (stationary std ``sigma``), one ``rng.standard_normal((n, A))`` draw per tick from ``np.random.default_rng(seed)``;
- mapped to SI around the box centre, clipped to [low, high], then rate-limited by ``max_delta`` per tick starting
  from ``initial`` (the recorded first command); float64 throughout, float32 at the end.

A trace is ``[n, T, A]``: world ``k`` of a batch run with seed ``s`` is episode ``(s, k)``. Engines play
``applied_traces`` (legalised twice, as Alloy's ``EpisodeRunner`` does) or ``batch_traces``.
"""

from __future__ import annotations

import numpy as np

HORIZON_S = 2.0
THETA = 2.0
SIGMA = 0.5


def tick_count(horizon_s: float, rate_hz: float) -> int:
    """Action ticks in ``horizon_s`` at ``rate_hz`` (at least 1)."""
    if horizon_s <= 0.0 or rate_hz <= 0.0:
        raise ValueError(f"horizon_s and rate_hz must be positive, got {horizon_s}, {rate_hz}")
    return max(1, int(round(horizon_s * rate_hz)))


def legalize(trace: np.ndarray, low: np.ndarray, high: np.ndarray, max_delta: np.ndarray,
             initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Clip ``trace`` [W, T, A] to the box and rate-limit it from ``initial``; returns (float32 trace, clips [W])."""
    raw = np.asarray(trace, dtype=np.float64)
    if raw.ndim == 2:
        raw = raw[None]
    if np.any(np.asarray(max_delta) <= 0.0):
        raise ValueError("max_delta_per_tick must be positive")
    boxed = np.clip(raw, low, high)
    tol = 1e-6 * np.maximum(np.abs(np.asarray(high) - np.asarray(low)), 1.0)
    changed = np.abs(boxed - raw) > tol
    out = np.empty_like(boxed)
    prev = boxed[:, 0] if initial is None else np.broadcast_to(np.clip(initial, low, high), boxed[:, 0].shape)
    for t in range(boxed.shape[1]):
        out[:, t] = prev + np.clip(boxed[:, t] - prev, -max_delta, max_delta)
        prev = out[:, t]
    changed |= np.abs(out - boxed) > tol
    clips = changed.reshape(changed.shape[0], -1).sum(axis=1).astype(np.int64)
    return out.astype(np.float32), clips


def random_legal_traces(space: dict, n: int, seed: int, horizon_s: float = HORIZON_S, theta: float = THETA,
                        sigma: float = SIGMA) -> np.ndarray:
    """``n`` legal OU traces [n, T, A] (float32, SI) for ``space`` = the pack's ``action_space`` block
    (``low``, ``high``, ``max_delta_per_tick``, ``initial``, ``rate_hz``)."""
    low, high = np.asarray(space["low"], np.float64), np.asarray(space["high"], np.float64)
    max_delta, initial = np.asarray(space["max_delta_per_tick"], np.float64), np.asarray(space["initial"], np.float64)
    rate = float(space["rate_hz"])
    ticks = tick_count(horizon_s, rate)
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate
    x = np.zeros((n, low.size))
    unit = np.empty((n, ticks, low.size))
    for t in range(ticks):
        x += -theta * x * dt + sigma * np.sqrt(2.0 * theta * dt) * rng.standard_normal(x.shape)
        unit[:, t] = x
    mid, half = 0.5 * (low + high), 0.5 * (high - low)
    return legalize(mid + unit * half, low, high, max_delta, initial)[0]


def applied_traces(space: dict, n: int, seed: int, horizon_s: float = HORIZON_S) -> np.ndarray:
    """What Alloy's SB-7 actually plays: ``EpisodeRunner.run`` legalises the float32 traces once more
    (alloy/bully/replay.py ``_legal``), which is not bitwise idempotent; engines play this."""
    low, high = np.asarray(space["low"], np.float64), np.asarray(space["high"], np.float64)
    md, init = np.asarray(space["max_delta_per_tick"], np.float64), np.asarray(space["initial"], np.float64)
    return legalize(random_legal_traces(space, n, seed, horizon_s), low, high, md, init)[0]


SEED_BATCH = 2048        # traces per seed: world k's trace depends on n, so every engine draws n = 2,048 per seed
FIRST_SEED = 100_000     # WP7's 303,104-episode T12 campaign: seeds 100000 .. 100147 (37 pieces x 4 seeds)


def campaign_seeds(episodes: int, first: int = FIRST_SEED) -> list[int]:
    """Seeds covering ``episodes`` episodes at ``SEED_BATCH`` per seed (WP7's ``sb7 --seed 100000+4i`` pieces of
    8,192 at 2,048 worlds use seeds ``first + b``); an engine batch of B worlds takes B / 2,048 consecutive seeds."""
    return [first + b for b in range(-(-episodes // SEED_BATCH))]


def batch_traces(space: dict, seeds: list[int], horizon_s: float = HORIZON_S) -> np.ndarray:
    """Traces for consecutive seeds, ``SEED_BATCH`` each, concatenated: row ``i`` is episode
    ``(seeds[i // 2048], i % 2048)``."""
    return np.concatenate([applied_traces(space, SEED_BATCH, s, horizon_s) for s in seeds], axis=0)


__all__ = ["FIRST_SEED", "HORIZON_S", "SEED_BATCH", "applied_traces", "batch_traces", "campaign_seeds", "legalize",
           "random_legal_traces", "tick_count"]
