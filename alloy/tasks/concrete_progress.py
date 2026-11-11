# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything concrete-finishing progress-v1 score, ported (host-side numpy).

Source (read-only): gym-anything-robotics/benchmarks/roboworld/environments/concrete_finishing/
task_set_progress.py (cited as TSP:line) and task.py (cited as T:line).

A task measures `Quantity` terms and multiplicative `gates`; this module turns an end and a start measurement
into the progress value (TSP:251-281).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FAIL_TERM = 0.9        # a failing quantity earns at most this share of its raw fraction (T:71, TSP:263-265)
FAIL_FACTOR = 0.95     # score of an unsuccessful run = 0.95 x progress (TSP:18, 275)


def clip(value: float) -> float:
    """Clip to [0, 1] (TSP:20-22)."""
    return float(np.clip(value, 0.0, 1.0))


def falloff(value: float, threshold: float, zero: float) -> float:
    """1 at the threshold, linearly 0 at `zero` (TSP:25)."""
    return clip((zero - value) / (zero - threshold))


def band(values: np.ndarray, lo: float, hi: float, scale: float) -> np.ndarray:
    """Per-cell credit: 1 inside [lo, hi], linear to 0 over `scale` outside it (TSP:29)."""
    return np.clip(1.0 - np.maximum(lo - values, values - hi) / scale, 0.0, 1.0)


@dataclass(frozen=True)
class Quantity:
    """One graded quantity (TSP:35-42): `value` and `target` on a common axis (larger is better)."""

    value: float
    target: float
    passed: bool
    keep: float = 1.0       # gate value when it fails after passing at the start
    group: str = ""
    weight: float = 1.0


@dataclass
class Progress:
    value: float
    work: float
    terms: dict[str, float] = field(default_factory=dict)        # per group
    quantities: dict[str, float] = field(default_factory=dict)   # per working quantity
    gates: dict[str, float] = field(default_factory=dict)


def progress_v1(end: dict[str, Quantity], gates: dict[str, float], start: dict[str, Quantity]) -> Progress:
    """TSP:251-281: quantities passing at the start become gates; the others earn 1 when passing else 0.9 x the
    fraction of the way from the start value to the target; groups are weight-averaged, `work` is their mean and
    progress = clip(work x product of gates)."""
    gates = dict(gates)
    groups: dict[str, list[tuple[float, float]]] = {}
    details: dict[str, float] = {}
    for name, q in end.items():
        s = start[name]
        if not all(math.isfinite(x) for x in (q.value, q.target, q.keep, s.value)):
            raise ValueError(f"Nonfinite task quantity: {name}")
        if s.passed:
            gates[name] = 1.0 if q.passed else q.keep
            continue
        span = q.target - s.value
        raw = clip((q.value - s.value) / span) if span > 1e-12 else 0.0
        term = 1.0 if q.passed else FAIL_TERM * raw
        details[name] = term
        groups.setdefault(q.group, []).append((term, q.weight))
    terms = {g: sum(v * w for v, w in items) / sum(w for _, w in items) for g, items in groups.items()}
    work = float(np.mean(list(terms.values()))) if terms else 0.0
    if not all(math.isfinite(v) for v in gates.values()):
        raise ValueError("Nonfinite task defect factor")
    return Progress(clip(work * math.prod(gates.values())), work, terms, details, gates)
