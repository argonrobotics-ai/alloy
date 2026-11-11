# SPDX-License-Identifier: Apache-2.0
"""E15 checker on synthetic water: a cup that pours into the bowl passes; leaks, a miss and a wrong rate fail."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import cup_pour

N, P, DT = 4, 400, 0.02
PV = 200e-6 / P
BOWL = np.array([0.17, 0.0, 0.0])


def _water(rng: np.random.Generator, centre: np.ndarray, radius: float, z0: float, dz: float) -> np.ndarray:
    ang, rr = rng.uniform(0, 2 * np.pi, P), radius * np.sqrt(rng.uniform(0, 1, P))
    pts = np.stack([centre[0] + rr * np.cos(ang), centre[1] + rr * np.sin(ang), z0 + dz * rng.uniform(0, 1, P)], axis=1)
    return np.broadcast_to(pts, (N, P, 3)).copy()


def _probe(points: np.ndarray, tilt: np.ndarray, pv: float = PV) -> dict[str, np.ndarray]:
    steps = len(points)
    return {"liquid_points": points, "particle_volume": np.full((steps, N), pv), "cup_tilt": tilt,
            "bowl_pos": np.broadcast_to(BOWL, (steps, N, 3)), "bowl_radius": np.full((steps, N), 0.1)}


def _rollout(volume_scale: float = 1.0, leak: bool = False, miss: bool = False, rate: float = 0.5) -> c.Rollout:
    rng = np.random.default_rng(0)
    cup = _water(rng, np.zeros(3), 0.035, 0.23, 0.04)
    rest_pts = np.broadcast_to(cup, (101, N, P, 3)).copy()
    if leak:
        rest_pts[50:, :, : P // 4, 0] += 0.3
    rest = c.make_phase("rest", np.arange(101) * DT, _probe(rest_pts, np.zeros((101, N)), PV * volume_scale))
    steps = 401
    tilt = np.minimum(np.arange(steps) * DT * rate, 2.2 / 1.0)[:, None] * np.ones((1, N))
    bowl_pts = _water(rng, BOWL if not miss else BOWL + 0.5, 0.08, 0.01, 0.03)
    frames = np.where((np.arange(steps) < steps - 1)[:, None, None, None], cup[None], bowl_pts[None])
    act = np.where(tilt[..., None] < 2.2, 0.5, 0.0)
    act[0] = 0.0
    pour = c.make_phase("act_pour", np.arange(steps) * DT, _probe(frames, tilt), action=act)
    return c.synthetic([rest, pour], dt=DT, episode_steps=500)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in cup_pour.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_pour_passes() -> None:
    assert _failed(_rollout()) == []


def test_wrong_volume_fails() -> None:
    assert "volume_ml" in _failed(_rollout(volume_scale=2.0))


def test_leaking_cup_fails_rest() -> None:
    assert "rest_in_cup" in _failed(_rollout(leak=True))


def test_missing_the_bowl_fails_pour() -> None:
    assert "pour_in_bowl" in _failed(_rollout(miss=True))


def test_wrong_tilt_rate_fails() -> None:
    assert "tilt_rate" in _failed(_rollout(rate=1.0))
