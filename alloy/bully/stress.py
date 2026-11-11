# SPDX-License-Identifier: Apache-2.0
"""B0 fixed stress phases (SPEC §3.11, bench §4.7, playbook VAL-115).

VAL-115's four legal phases, written in action space so they apply to any target: (1) every actuator switching
between 80 % of its range at 2 Hz, (2) actuators driven toward opposite extremes and swapped (the "arms into each
other" phase for paired actuators), (3) a 5 Hz squeeze between 50 % and 100 % of the range (gripper on a rim at its
force limit), (4) a drive to each limit at the maximum rate, held (through an object at motor limits). Then the
bench §4.7 extras, in action space: hold high then release to low (drop from the highest reachable pose) and one
max-rate sweep per actuator with the rest centred (push into a wall). Drops onto specific supports and pushes into
named thin walls need scene kinematics; a target supplies them as extra traces (``stress_extra``). Every phase is
``horizon_s`` long (default 2 s, the B1 horizon) and legal.
"""

from __future__ import annotations

import logging

import numpy as np

from alloy.bully.actions import legalize, tick_count
from alloy.bully.types import ActionSpace, BullyTarget
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")


def _square(t: np.ndarray, hz: float) -> np.ndarray:
    """+1 / -1 square wave of frequency ``hz`` starting at +1."""
    return np.where(np.floor(2.0 * hz * t) % 2 == 0, 1.0, -1.0)


def _phases_unit(ticks: int, rate_hz: float, dim: int) -> list[tuple[str, np.ndarray]]:
    """Phases in units of the half-range around the centre: -1 = low, +1 = high; each [T, A]."""
    t = np.arange(ticks) / rate_hz
    ones = np.ones((ticks, dim))
    sign = np.where(np.arange(dim) % 2 == 0, 1.0, -1.0)
    phases = [
        ("val115_1_joints_80pct_2hz", 0.8 * _square(t, 2.0)[:, None] * ones),
        ("val115_2_opposed_extremes", _square(t, 0.5)[:, None] * sign[None, :]),
        ("val115_3_squeeze_5hz", 0.75 + 0.25 * _square(t, 5.0)[:, None] * ones),
        ("val115_4_drive_high_hold", ones.copy()),
        ("val115_4_drive_low_hold", -ones),
        ("drop_hold_high_release", np.where(t <= 0.5 * t[-1], 1.0, -1.0)[:, None] * ones),
    ]
    for a in range(dim):
        push = np.zeros((ticks, dim))
        push[:, a] = np.where(t < 0.25 * t[-1], -1.0, 1.0)  # start low, then sweep high at the max rate
        phases.append((f"push_actuator_{a}", push))
    return phases


def named_stress_phases(target: BullyTarget, horizon_s: float = 2.0) -> list[tuple[str, np.ndarray]]:
    """(name, legal SI trace [T, A]) for every B0 phase of ``target``, including the target's own extras."""
    sp: ActionSpace = target.actions
    ticks = tick_count(horizon_s, sp.rate_hz)
    mid, half = 0.5 * (sp.low + sp.high), 0.5 * (sp.high - sp.low)
    out = []
    for name, unit in _phases_unit(ticks, sp.rate_hz, sp.dim):
        raw = mid + unit * half
        legal = legalize(raw[None], sp.low, sp.high, sp.max_delta_per_tick, sp.initial)
        out.append((name, legal.trace[0]))
    extra = target.stress_extra
    for k, trace in enumerate(extra(target) if extra is not None else ()):
        legal = legalize(np.asarray(trace)[None], sp.low, sp.high, sp.max_delta_per_tick, sp.initial)
        out.append((f"extra_{k}", _fit_ticks(legal.trace[0], ticks)))
    log.info("B0 %s: %d phases of %d ticks", target.name, len(out), ticks)
    return out


def stress_phases(target: BullyTarget) -> list[np.ndarray]:
    """B0 traces [T, A] (SI, legal) for ``target`` (SPEC §3.11 signature)."""
    return [trace for _, trace in named_stress_phases(target)]


def tile_phases(phases: list[np.ndarray], worlds: int) -> np.ndarray:
    """[W, T, A]: world ``w`` plays phase ``w % len(phases)`` (B0 in one batched episode)."""
    if not phases:
        raise BullyError("no stress phases")
    stack = np.stack(phases, axis=0)
    return stack[np.arange(worlds) % stack.shape[0]]


def _fit_ticks(trace: np.ndarray, ticks: int) -> np.ndarray:
    """Truncate, or hold the last command, so a target's extra trace has the common length."""
    if trace.shape[0] >= ticks:
        return trace[:ticks]
    pad = np.repeat(trace[-1:], ticks - trace.shape[0], axis=0)
    return np.concatenate([trace, pad], axis=0)


def random_legal_traces(target: BullyTarget, n: int, horizon_s: float = 2.0, seed: int = 0,
                        theta: float = 2.0, sigma: float = 0.5) -> np.ndarray:
    """SB-7 rollouts: ``n`` legal Ornstein-Uhlenbeck traces [n, T, A] around the box centre (bench §5 metric 6).

    In units of the half-range: ``x += -theta·x·dt + sigma·sqrt(2·theta·dt)·N(0, 1)`` (stationary std ``sigma``),
    then mapped to SI, clipped and rate-limited. ``seed`` makes the batch reproducible.
    """
    sp = target.actions
    ticks = tick_count(horizon_s, sp.rate_hz)
    rng = np.random.default_rng(seed)
    dt = 1.0 / sp.rate_hz
    x = np.zeros((n, sp.dim))
    unit = np.empty((n, ticks, sp.dim))
    for t in range(ticks):
        x += -theta * x * dt + sigma * np.sqrt(2.0 * theta * dt) * rng.standard_normal(x.shape)
        unit[:, t] = x
    mid, half = 0.5 * (sp.low + sp.high), 0.5 * (sp.high - sp.low)
    return legalize(mid + unit * half, sp.low, sp.high, sp.max_delta_per_tick, sp.initial).trace
