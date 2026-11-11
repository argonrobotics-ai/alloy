# SPDX-License-Identifier: Apache-2.0
"""E16 checker on synthetic water: a tilting, accelerating tank passes; no tilt, a wrong volume, a tank that
teleports and water that does not ride along each fail at their check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import tank_slosh

N, P, DT = 3, 3000, 0.02
PV = 1.5e-3 / P


def _slab() -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.stack([rng.uniform(-0.098, 0.098, P), rng.uniform(-0.048, 0.048, P), rng.uniform(0.004, 0.0706, P)], axis=1)


def _tank_x(t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return scale * np.where(t < 0.5, t ** 2, 0.25 + 1.0 * (t - 0.5))


def _probe(points: np.ndarray, x: np.ndarray, pv: float) -> dict[str, np.ndarray]:
    steps = len(x)
    pos = np.zeros((steps, N, 3))
    pos[..., 0] = x[:, None]
    return {"liquid_points": np.broadcast_to(points, (steps, N, P, 3)).copy().astype(np.float32),
            "particle_volume": np.full((steps, N), pv), "tank_pos": pos}


def _rollout(pv: float = PV, tilt: float = 0.2, tank_scale: float = 1.0, leave: bool = False,
             ride: bool = True) -> c.Rollout:
    base = _slab()
    rest = c.make_phase("rest", np.arange(101) * DT, _probe(base, np.zeros(101), pv))
    t = np.arange(76) * DT
    x = _tank_x(t, tank_scale)
    frac = (base[:, 2] - 0.004) / 0.0666
    tilted = base.copy()
    tilted[:, 2] -= tilt * base[:, 0] * frac
    probe = _probe(tilted, x, pv)
    if ride:
        probe["liquid_points"][..., 0] += x[:, None, None].astype(np.float32)
    if leave:
        probe["liquid_points"][40:, :, : P // 10, 0] += 0.4
    act = np.where(t[:, None, None] < 0.5, 2.0 / 3.0, 0.0) * np.ones((76, N, 1))
    return c.synthetic([rest, c.make_phase("act_slosh", t, probe, action=act)], dt=DT, episode_steps=200)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in tank_slosh.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_tank_passes() -> None:
    assert _failed(_rollout()) == []


def test_no_tilt_fails_slope() -> None:
    assert "surface_slope" in _failed(_rollout(tilt=0.0))


def test_wrong_volume_fails() -> None:
    assert "volume_l" in _failed(_rollout(pv=2 * PV))


def test_tank_that_jumps_fails_follow() -> None:
    assert "tank_displacement_m" in _failed(_rollout(tank_scale=2.0))


def test_water_left_behind_fails_contained() -> None:
    assert "contained_share_outside" in _failed(_rollout(ride=False))


def test_leak_fails_contained() -> None:
    assert "contained_share_outside" in _failed(_rollout(leave=True))
