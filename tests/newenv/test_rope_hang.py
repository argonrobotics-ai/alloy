# SPDX-License-Identifier: Apache-2.0
"""E14 checker on synthetic rollouts: a good rope, a stretched rope, a rope with gravity off, a slipping anchor."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c, rope_hang

N, K, DT = 3, 21, 0.05
A = np.array([0.0, 0.0, 1.5])


def _chain(end: np.ndarray, sag: float, length: float = 1.0) -> np.ndarray:
    """(K, 3) points from the anchor to ``end``: a parabola of the given sag, scaled to the given rope length."""
    s = np.linspace(0, 1, K)[:, None]
    line = A + s * (end - A)
    pts = line - np.array([0, 0, 1.0]) * (4 * sag * s * (1 - s))
    if np.allclose(end[:2], A[:2]):
        pts = A + s * (end - A)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return A + (pts - A) * (length / seg)


def _phase(name: str, steps: int, e0: np.ndarray, e1: np.ndarray, sag: float, length: float = 1.0,
           slip: float = 0.0, done: np.ndarray | None = None) -> c.Phase:
    k = np.linspace(0, 1, steps + 1)
    ends = e0 + k[:, None] * (e1 - e0)
    pts = np.stack([_chain(e, sag * kk, length) for e, kk in zip(ends, k)])
    pts[:, 0, 2] -= slip * k
    pts = np.broadcast_to(pts[:, None], (steps + 1, N, K, 3)).copy()
    end = np.broadcast_to(ends[:, None], (steps + 1, N, 3)).copy()
    target = np.broadcast_to((A + [0.2, 0.1, -0.3])[None, None] + np.linspace(0, 0.1, N)[None, :, None] * [1, 1, 1],
                             (steps + 1, N, 3)).copy()
    return c.make_phase(name, np.arange(steps + 1) * DT, {"rope_points": pts, "end_pos": end, "target_pos": target},
                        done=done)


def _rollout(sag: float = 0.3625, length: float = 1.0, lift: float = 0.3, slip: float = 0.0) -> c.Rollout:
    hang = np.array([0.0, 0.0, 0.5])
    rest = _phase("rest", 40, hang, hang, 0.0, length, slip)
    speed = _phase("act_speed", 20, hang, hang + [0, 0, lift], 0.0, length)
    goal = A + [0.6, 0.0, 0.0]
    done = np.zeros((101, N), bool)
    raise_ = _phase("act_raise", 100, hang, goal, sag, length, done=done)
    rand = _phase("random", 4, hang, hang, 0.0, length, done=np.array([[0] * N, [1] * N, [0] * N, [0] * N, [0] * N], bool))
    return c.synthetic([rest, speed, raise_, rand], dt=DT, episode_steps=200)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in rope_hang.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_rope_passes() -> None:
    assert _failed(_rollout()) == []


def test_stretched_rope_fails() -> None:
    assert "rope_length" in _failed(_rollout(length=1.3))


def test_taut_rope_gravity_off_fails() -> None:
    assert "catenary_sag" in _failed(_rollout(sag=0.02))


def test_slipping_anchor_fails() -> None:
    assert "anchor_holds" in _failed(_rollout(slip=0.05))


def test_frozen_gripper_fails() -> None:
    assert "speed_travel" in _failed(_rollout(lift=0.0))
