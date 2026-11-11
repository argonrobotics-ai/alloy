# SPDX-License-Identifier: Apache-2.0
"""E05 checker on synthetic rollouts: good, nothing falls, only the first falls, all fall at once (no chain), bad success."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import dominoes

N, K, DT = 16, 10, 0.02
STEPS = 226


def _quat(angle: np.ndarray) -> np.ndarray:
    """Rotation about y by angle (T, N, K) as wxyz."""
    q = np.zeros(angle.shape + (4,))
    q[..., 0], q[..., 2] = np.cos(angle / 2), np.sin(angle / 2)
    return q


def _angles(delays: np.ndarray, count: int) -> np.ndarray:
    t = (np.arange(STEPS) * DT)[:, None, None]
    ang = np.clip((t - delays[None, None, :]) * 4.0, 0.0, np.pi / 2)
    ang[:, :, count:] = 0.0
    return np.broadcast_to(ang, (STEPS, N, K)).copy()


def _pos() -> np.ndarray:
    row = np.stack([np.arange(K) * 0.033, np.zeros(K), np.full(K, 0.024)], axis=1)
    return np.broadcast_to(row, (STEPS, N, K, 3)).copy()


def _rollout(delays: np.ndarray | None = None, count: int = K, success_late: bool = True, lie: bool = False) -> c.Rollout:
    delays = np.arange(K) * 0.2 + 0.3 if delays is None else delays
    ang = _angles(delays, count)
    rest = c.make_phase("rest", np.arange(51) * DT, {"domino_pos": _pos()[:51], "domino_quat": _quat(ang * 0)[:51]})
    done_all = (ang[-1] > np.radians(60)).all(axis=-1)
    success = np.zeros((STEPS, N), bool)
    success[-1] = done_all if not lie else ~done_all
    push = c.make_phase("act_push", np.arange(STEPS) * DT,
                        {"domino_pos": _pos(), "domino_quat": _quat(ang)}, success=success)
    rng = np.random.default_rng(0)
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * DT, {"domino_pos": _pos()[:3] + rng.normal(0, 1e-4, (3, N, K, 3)),
                                                      "domino_quat": _quat(ang * 0)[:3]}, done=done)
    return c.synthetic([rest, push, rand], dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in dominoes.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_rollout_checks_out() -> None:
    assert _failed(_rollout()) == []


def test_nothing_falls_fails() -> None:
    assert "all_toppled_in_time" in _failed(_rollout(count=0))


def test_only_first_falls_fails() -> None:
    assert "all_toppled_in_time" in _failed(_rollout(count=1))


def test_scrambled_order_fails() -> None:
    delays = np.array([0.3, 1.5, 0.5, 1.9, 0.7, 1.1, 0.9, 2.1, 1.3, 1.7])
    assert "topple_in_order" in _failed(_rollout(delays=delays))


def test_wrong_success_fails() -> None:
    assert "success_agrees" in _failed(_rollout(lie=True))
