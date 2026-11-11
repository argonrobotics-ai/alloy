# SPDX-License-Identifier: Apache-2.0
"""E18 checker on a synthetic bed of points: a furrowed, heaped bed passes; a levelled bed (water), too much mass, a
scoop at the wrong speed and a heap still sliding at rest each fail at their check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import sand_scoop

N, P, DT = 2, 12000, 0.1
PM = 7.0 / P
FLOOR, TOP = 0.0122, 0.0597
X_START, X_END, Y0 = -0.10, 0.05, 0.0


def _bed(furrow: bool, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-0.147, 0.147, P), rng.uniform(-0.147, 0.147, P)
    h = np.full(P, TOP)
    if furrow:
        dy = np.abs(y - Y0)
        drop = np.where(dy <= 0.04, 0.02, np.clip(0.02 - 0.5 * (dy - 0.04), 0.0, None))
        h = h - np.where((x >= X_START) & (x <= X_END), drop, 0.0)
        h = h + np.where((x > X_END) & (x <= X_END + 0.05) & (dy <= 0.04), 0.02, 0.0)
    return np.stack([x, y, FLOOR + (h - FLOOR) * rng.uniform(0, 1, P)], axis=1)


def _scoop(speed: float, steps_down: int, steps_push: int, steps_hold: int) -> tuple[np.ndarray, np.ndarray]:
    act = np.zeros((1 + steps_down + steps_push + steps_hold, 4))
    act[1:1 + steps_down, 2] = -1.0
    act[1 + steps_down:1 + steps_down + steps_push, 0] = 1.0
    vel = act[:, :3] * speed
    pos = np.array([X_START, Y0, 0.14]) + np.cumsum(vel * DT, axis=0)
    return pos, act


def _rollout(furrow: bool = True, pm_scale: float = 1.0, speed: float = 0.1, creep: float = 0.0) -> c.Rollout:
    start = _bed(False)
    rest_pts = np.broadcast_to(start, (21, N, P, 3)).copy()
    rest_pts[10:, :, :, 0] += creep * np.arange(11)[:, None, None] * DT
    pm = np.full((21, N), PM * pm_scale)
    rest = c.make_phase("rest", np.arange(21) * 0.1, {"sand_points": rest_pts, "particle_mass": pm,
                                                      "scoop_pos": np.zeros((21, N, 3)), "scoop_pitch": np.zeros((21, N))})
    pos, act = _scoop(speed, 11, 15, 20)
    steps = len(pos)
    pts = np.broadcast_to(start, (steps, N, P, 3)).copy()
    pts[-1] = _bed(furrow)[None]
    probe = {"sand_points": pts, "particle_mass": np.full((steps, N), PM * pm_scale),
             "scoop_pos": np.broadcast_to(pos[:, None, :], (steps, N, 3)).copy(), "scoop_pitch": np.zeros((steps, N))}
    act_phase = c.make_phase("act_dig", np.arange(steps) * DT, probe, action=np.broadcast_to(act[:, None, :], (steps, N, 4)).copy())
    return c.synthetic([rest, act_phase], dt=DT, episode_steps=50)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in sand_scoop.check(r, {}).checks if not x.ok and x.level == 2]


def test_furrowed_bed_passes() -> None:
    assert _failed(_rollout()) == []


def test_levelled_bed_fails_furrow() -> None:
    failed = _failed(_rollout(furrow=False))
    assert "furrow_depth_m" in failed and "wall_slope" in failed and "heap_ahead_m" in failed


def test_too_heavy_fails_mass() -> None:
    assert "total_mass_kg" in _failed(_rollout(pm_scale=3.0))


def test_fast_scoop_fails_speed() -> None:
    assert "scoop_push_speed" in _failed(_rollout(speed=0.2))


def test_sliding_heap_fails_rest_speed() -> None:
    assert "rest_speed_mps" in _failed(_rollout(creep=0.05))
