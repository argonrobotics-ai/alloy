# SPDX-License-Identifier: Apache-2.0
"""E17 checker on synthetic water: a tap that fills 150 ml and stops passes; a leaking tap, a tap that never opens,
a tap that fills at rest and one that keeps adding water after closing each fail at their check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import fill_level

N, DT, P = 3, 0.05, 4000
PV = 1e-7                                            # 4000 points = 400 ml of supply
SPACING = PV ** (1 / 3)
GLASS = np.array([0.0, 0.0, 0.0])


def _column(n: int, rng: np.random.Generator) -> np.ndarray:
    """n points filling the glass from its floor up: n x PV of water over the glass's area."""
    h = n * PV / fill_level.AREA
    r = fill_level.RADIUS * 0.98 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(a), r * np.sin(a), SPACING / 2 + (h - SPACING) * rng.uniform(0, 1, n)], axis=1)


def _frames(fill_ml: np.ndarray, stray: int = 0) -> np.ndarray:
    """(T, N, P, 3): per row, ``fill_ml`` (T,) millilitres in the glass; the rest of the supply in a drain pile far away."""
    rng = np.random.default_rng(2)
    out = np.zeros((len(fill_ml), N, P, 3))
    drain = np.stack([0.3 + 0.01 * rng.uniform(size=P), np.zeros(P), 0.05 + 0.0 * rng.uniform(size=P)], axis=1)
    for k, ml in enumerate(fill_ml):
        n = int(round(ml * 1e-6 / PV))
        out[k] = drain
        out[k, :, :n] = _column(n, rng)[None]
    return out


def _probe(points: np.ndarray) -> dict[str, np.ndarray]:
    steps = len(points)
    return {"liquid_points": points.astype(np.float32), "particle_volume": np.full((steps, N), PV),
            "glass_pos": np.broadcast_to(GLASS, (steps, N, 3)).copy()}


def _rollout(rate: float = 50.0, rest_fill: float = 0.0, late: float = 0.0, open_s: float = 3.0) -> c.Rollout:
    rest = c.make_phase("rest", np.arange(41) * DT, _probe(_frames(np.full(41, rest_fill))))
    t = np.arange(121) * DT
    fill = np.minimum(t, open_s) * rate + np.maximum(t - open_s - 1.0, 0.0) * late
    act = np.where(t[:, None, None] < open_s, 1.0, 0.0) * np.ones((121, N, 1))
    act[0] = 0.0
    phase = c.make_phase("act_fill", t, _probe(_frames(fill)), action=act)
    return c.synthetic([rest, phase], dt=DT, episode_steps=160)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in fill_level.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_tap_passes() -> None:
    assert _failed(_rollout()) == []


def test_tap_that_never_opens_fails() -> None:
    assert "glass_ml" in _failed(_rollout(rate=0.0))


def test_tap_that_leaks_half_fails_share() -> None:
    assert "glass_share_of_commanded" in _failed(_rollout(rate=25.0))


def test_water_at_rest_fails() -> None:
    assert "rest_adds_no_water_ml" in _failed(_rollout(rest_fill=10.0))


def test_water_after_close_fails_drift() -> None:
    assert "amount_drift_after_close" in _failed(_rollout(late=5.0))
