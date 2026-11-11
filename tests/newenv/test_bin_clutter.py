# SPDX-License-Identifier: Apache-2.0
"""E03 checker on synthetic rollouts: good, a frozen bin, objects overlapping, objects outside the bin, a fast bin."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import bin_clutter
from bench.newenv.checkers import common as c

N, K, DT = 16, 10, 0.02


def _objects(spread: float = 0.08, overlap: bool = False, outside: bool = False) -> np.ndarray:
    grid = np.array([[(i % 4 - 1.5) * spread, (i // 4 - 1.0) * spread] for i in range(K)])
    if overlap:
        grid[1] = grid[0] + 0.01
    if outside:
        grid[2] = (0.4, 0.0)
    z = np.full((K, 1), 0.025)
    return np.broadcast_to(np.concatenate([grid, z], axis=1), (N, K, 3)).copy()


def _phase(name: str, steps: int, objs: np.ndarray, vel: tuple[float, float], bin0: np.ndarray) -> c.Phase:
    t = np.arange(steps) * DT
    b = bin0[None, :, :] + t[:, None, None] * np.array([vel[0], vel[1], 0.0])
    o = objs[None] + (b - bin0[None])[:, :, None, :]
    return c.make_phase(name, t, {"object_pos": o, "bin_pos": b})


def _rollout(speed: float = 0.2, **kw) -> c.Rollout:
    bin0 = np.zeros((N, 3))
    objs = _objects(**kw)
    rest = _phase("rest", 101, objs, (0, 0), bin0)
    more = _phase("rest_more", 51, objs, (0, 0), bin0)
    ax = _phase("act_x", 36, objs, (speed, 0), bin0)
    ay = _phase("act_y", 36, objs, (0, -speed), bin0)
    rng = np.random.default_rng(0)
    start = rng.uniform(-0.1, 0.1, (3, N, K, 3))
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * DT, {"object_pos": start, "bin_pos": np.zeros((3, N, 3))}, done=done)
    return c.synthetic([rest, more, ax, ay, rand], dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in bin_clutter.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_rollout_checks_out() -> None:
    assert _failed(_rollout()) == []


def test_frozen_bin_fails_speed() -> None:
    assert "bin_speed_x" in _failed(_rollout(speed=0.0))


def test_fast_bin_fails_speed() -> None:
    assert "bin_speed_x" in _failed(_rollout(speed=0.4))


def test_overlap_fails_pair_distance() -> None:
    assert "pair_distance" in _failed(_rollout(overlap=True))


def test_object_outside_fails_inside() -> None:
    assert "rest_all_inside" in _failed(_rollout(outside=True))
