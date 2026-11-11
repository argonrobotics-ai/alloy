# SPDX-License-Identifier: Apache-2.0
"""E08 checker on synthetic rollouts from a numpy sprung door: right physics passes, wrong ones fail."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import spring_door as sd

N, DT = 16, 0.01
SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "newenv" / "scripts" / "spring_door.py"
spec = importlib.util.spec_from_file_location("spring_door_script", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _phase(name: str, steps: int, law, start, k=2.0, damp=1.0, inertia=2.1333, limited=True, gain=3.0):
    theta, rate = (np.array(v, float) for v in start)
    th, probes = [], []
    for i in range(steps + 1):
        th.append(theta.copy())
        probes.append(rate.copy())
        u = np.clip(law(i * DT, {"door_angle": theta, "door_rate": rate}), -1, 1).reshape(-1) * gain
        for _ in range(20):
            h = DT / 20
            rate = rate + (u - k * theta - damp * rate) / inertia * h
            theta = theta + rate * h
            if limited:
                hit = (theta < 0) | (theta > 1.5)
                theta, rate = np.clip(theta, 0, 1.5), np.where(hit, 0.0, rate)
    return c.make_phase(name, np.arange(steps + 1) * DT, {"door_angle": np.array(th), "door_rate": np.array(probes)}), \
        (theta, rate)


def _rollout(**kw) -> c.Rollout:
    start = (np.full(N, 0.002), np.zeros(N))
    act = lambda ph, t, p: script.action(ph, t, p, 1)
    rest, _ = _phase("rest", 200, lambda t, p: np.zeros(N), start, **kw)
    half, _ = _phase("act_half", 400, lambda t, p: act("act_half", t, p), start, **kw)
    full, _ = _phase("act_full", 400, lambda t, p: act("act_full", t, p), start, **kw)
    op, end = _phase("act_open", 250, lambda t, p: act("act_open", t, p), start, **kw)
    rel, _ = _phase("release", 200, lambda t, p: np.zeros(N), end, **kw)
    return c.synthetic([rest, half, full, op, rel], dt=DT, episode_steps=500)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in sd.check(r, {}).checks if not x.ok and x.level == 2]


def test_correct_physics_passes() -> None:
    assert _failed(_rollout()) == []


def test_stiff_spring_fails_step_response() -> None:
    failed = _failed(_rollout(k=4.0))
    assert "act_half_follows_model" in failed


def test_no_limit_fails_limit_and_overshoot() -> None:
    failed = _failed(_rollout(limited=False))
    assert "act_full_reaches_limit" in failed and "release_no_overshoot" in failed


def test_heavier_door_fails_release_time() -> None:
    assert "release_half_time_ratio" in _failed(_rollout(inertia=4.5))
