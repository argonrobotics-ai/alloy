# SPDX-License-Identifier: Apache-2.0
"""E06 checker on synthetic rollouts from a numpy cart-pole: the right physics passes; wrong ones fail."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bench.newenv.checkers import cartpole
from bench.newenv.checkers import common as c

N, DT = 16, 0.01
SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "newenv" / "scripts" / "cartpole.py"
spec = importlib.util.spec_from_file_location("cartpole_script", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _accel(s: np.ndarray, force: np.ndarray, g: float) -> np.ndarray:
    _, th, _, w = s.T
    m, big_m, l = 0.1, 1.0, 0.25
    inertia = m * 0.5 ** 2 / 3
    a11, a12 = big_m + m, m * l * np.cos(th)
    a21, a22 = m * l * np.cos(th), inertia
    r1, r2 = force + m * l * w ** 2 * np.sin(th), m * g * l * np.sin(th)
    det = a11 * a22 - a12 * a21
    return np.stack([(r1 * a22 - a12 * r2) / det, (a11 * r2 - a21 * r1) / det], axis=1)


def _simulate(steps: int, policy, g: float = 9.81, scale: float = 10.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    state = np.zeros((N, 4))
    state[:, 1] = rng.uniform(-0.05, 0.05, N)
    probes, dones = [], []
    fresh = np.zeros(N, bool)
    for k in range(steps + 1):
        probes.append(state.copy())
        dones.append(fresh)
        if k == steps:
            break
        probe = {"cart_x": state[:, 0], "pole_angle": state[:, 1], "cart_v": state[:, 2], "pole_rate": state[:, 3]}
        f = np.clip(policy(k * DT, probe), -1, 1)[:, 0] * scale
        for _ in range(4):
            acc = _accel(state, f, g)
            state[:, 2] += acc[:, 0] * DT / 4
            state[:, 3] += acc[:, 1] * DT / 4
            state[:, 0] += state[:, 2] * DT / 4
            state[:, 1] += state[:, 3] * DT / 4
        fresh = (np.abs(state[:, 1]) > 0.4) | (np.abs(state[:, 0]) > 2.0)
        state[fresh] = np.stack([np.zeros(fresh.sum()), rng.uniform(-0.05, 0.05, fresh.sum()),
                                 np.zeros(fresh.sum()), np.zeros(fresh.sum())], axis=1)
    return np.array(probes), np.array(dones)


def _phase(name: str, steps: int, policy, **kw) -> c.Phase:
    p, d = _simulate(steps, policy, **kw)
    probe = {"cart_x": p[:, :, 0], "pole_angle": p[:, :, 1], "cart_v": p[:, :, 2], "pole_rate": p[:, :, 3]}
    return c.make_phase(name, np.arange(steps + 1) * DT, probe, done=d)


def _rollout(g: float = 9.81, scale: float = 10.0, hold_policy=None) -> c.Rollout:
    zero = lambda t, p: np.zeros((N, 1))
    law = hold_policy or (lambda t, p: script.action("act_hold", t, p, 1))
    phases = [_phase("rest", 200, zero, g=g), _phase("act_hold", 500, law, g=g, scale=scale)]
    return c.synthetic(phases, dt=DT, episode_steps=500)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in cartpole.check(r, {}).checks if not x.ok and x.level == 2]


def test_correct_physics_passes() -> None:
    assert _failed(_rollout()) == []


def test_gravity_off_fails_rest() -> None:
    failed = _failed(_rollout(g=0.0))
    assert "rest_worlds_fell" in failed or "rest_fall_time" in failed


def test_weak_force_fails_hold() -> None:
    assert "hold_worlds" in _failed(_rollout(scale=1.0))


def test_frozen_pole_fails_growth() -> None:
    zero = lambda t, p: np.zeros((N, 1))
    r = _rollout(g=0.0, hold_policy=zero)
    assert "growth_rate" in _failed(r) or "rest_worlds_fell" in _failed(r)
