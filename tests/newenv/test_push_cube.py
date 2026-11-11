# SPDX-License-Identifier: Apache-2.0
"""E01 checker on synthetic rollouts: a good one, a frozen cube, a teleported cube, a floating cube."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import push_cube

N = 16


def _pos(z: float, x0: np.ndarray, steps: int, vx: float = 0.0, dt: float = 0.02) -> np.ndarray:
    t = np.arange(steps)[:, None] * dt
    x = x0[None, :] + 0.5 * vx * t * t
    return np.stack([x, np.broadcast_to(np.linspace(-0.3, 0.3, N), x.shape), np.full(x.shape, z)], axis=-1)


def _rollout(accel: float = 10.0, rest_z: float = 0.525, jump: bool = False) -> c.Rollout:
    x0 = np.linspace(-0.3, 0.3, N)
    rest_t = np.arange(101) * 0.02
    rest = c.make_phase("rest", rest_t, {"cube_pos": _pos(rest_z, x0, 101), "target_pos": _pos(0.5, x0[::-1], 101)})
    push_pos = _pos(0.525, x0, 11, accel)
    if jump:
        push_pos[1:, :, 0] += 0.5
    push = c.make_phase("act_push", np.arange(11) * 0.02, {"cube_pos": push_pos, "target_pos": _pos(0.5, x0, 11)})
    rng = np.random.default_rng(0)
    cube = np.concatenate([rng.uniform(-0.4, 0.4, (3, N, 2)), np.full((3, N, 1), 0.525)], axis=-1)
    target = np.concatenate([rng.uniform(-0.4, 0.4, (3, N, 2)), np.full((3, N, 1), 0.5)], axis=-1)
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * 0.02, {"cube_pos": cube, "target_pos": target}, done=done)
    return c.synthetic([rest, push, rand], dt=0.02, episode_steps=250)


def _levels(r: c.Rollout) -> tuple[int, list[str]]:
    res = push_cube.check(r, {})
    return res.level, [x.name for x in res.checks if not x.ok and x.level == 2]


def test_good_rollout_checks_out() -> None:
    _, failed = _levels(_rollout(accel=7.0))
    assert failed == []


def test_frozen_cube_fails_push() -> None:
    _, failed = _levels(_rollout(accel=0.0))
    assert "push_dx" in failed


def test_teleported_cube_fails_push() -> None:
    _, failed = _levels(_rollout(accel=7.0, jump=True))
    assert "push_dx" in failed


def test_floating_cube_fails_rest() -> None:
    _, failed = _levels(_rollout(accel=7.0, rest_z=0.60))
    assert "rest_height" in failed
