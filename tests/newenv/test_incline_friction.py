# SPDX-License-Identifier: Apache-2.0
"""E04 checker on synthetic rollouts: good, frozen puck, no gravity on the slope, friction ignored, narrow friction draw."""
from __future__ import annotations

import math

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import incline_friction as inc

N, DT, G, TH = 16, 0.02, 9.81, math.radians(15.0)
UP = np.array([math.cos(TH), 0.0, math.sin(TH)])


def _pos(a: np.ndarray, steps: int) -> np.ndarray:
    t = (np.arange(steps) * DT)[:, None]
    d = 0.5 * a[None, :] * t ** 2
    return np.array([0.0, 0.0, 0.5]) + d[:, :, None] * UP


def _accel_rest(mu: np.ndarray, gravity: float = G, friction_on: bool = True) -> np.ndarray:
    a = gravity * (math.sin(TH) - (mu * math.cos(TH) if friction_on else 0.0))
    return -np.maximum(np.broadcast_to(a, mu.shape), 0.0)


def _rollout(mu: np.ndarray | None = None, gravity: float = G, friction_on: bool = True, frozen: bool = False) -> c.Rollout:
    mu = np.linspace(0.1, 0.6, N) if mu is None else mu
    probe_mu = lambda steps: np.broadcast_to(mu, (steps, N)).copy()
    a_rest = np.zeros(N) if frozen else _accel_rest(mu, gravity, friction_on)
    rest = c.make_phase("rest", np.arange(101) * DT, {"puck_pos": _pos(a_rest, 101), "friction": probe_mu(101)})
    a_up = np.zeros(N) if frozen else np.maximum(1.0 / 0.2 - G * (math.sin(TH) + mu * math.cos(TH)), 0.0)
    act = c.make_phase("act_up", np.arange(26) * DT, {"puck_pos": _pos(a_up, 26), "friction": probe_mu(26)})
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * DT, {"puck_pos": _pos(np.zeros(N), 3), "friction": probe_mu(3)}, done=done)
    return c.synthetic([rest, act, rand], dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in inc.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_rollout_checks_out() -> None:
    assert _failed(_rollout()) == []


def test_frozen_puck_fails() -> None:
    assert "slide_accel_ratio" in _failed(_rollout(frozen=True))


def test_weak_gravity_fails() -> None:
    assert "slide_accel_ratio" in _failed(_rollout(gravity=5.0))


def test_friction_ignored_fails() -> None:
    assert "slide_accel_ratio" in _failed(_rollout(friction_on=False))


def test_narrow_friction_fails_spread() -> None:
    assert "friction_spread" in _failed(_rollout(mu=np.linspace(0.3, 0.4, N)))
