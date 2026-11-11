# SPDX-License-Identifier: Apache-2.0
"""E11 checker on synthetic rollouts: a good hang, a frozen scene, a 2x stretched cloth, gravity off, a slipping corner."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import cloth_hang, common as c

N, G, DT = 4, 21, 0.05
SIDE = 0.4


def _sheet(depth: float, stretch: float = 1.0) -> np.ndarray:
    """(V, 3) cloth hanging in the x = 0 plane: top edge along y at z = 1, extending ``depth`` down."""
    y, z = np.meshgrid(np.linspace(-SIDE / 2, SIDE / 2, G), np.linspace(0, -depth, G))
    return np.stack([np.zeros(y.size), y.ravel() * stretch, 1.0 + z.ravel()], axis=-1)


def _phase(name: str, steps: int, dx: float = 0.0, dz: float = 0.0, depth: float = 0.39, stretch: float = 1.0,
           slip: float = 0.0, follow: float = 1.0) -> c.Phase:
    k = np.linspace(0, 1, steps + 1)[:, None, None]
    corners = np.array([[0.0, -SIDE / 2 * stretch, 1.0], [0.0, SIDE / 2 * stretch, 1.0]])
    cor = corners[None] + k * np.array([dx, 0.0, dz]) * np.ones((1, 2, 1))
    cor = np.broadcast_to(cor[:, None], (steps + 1, N, 2, 3)).copy()
    cor[..., 2] -= slip * k[:, 0, 0][:, None, None]
    cloth = _sheet(depth, stretch)[None, None] + follow * k[..., None] * np.array([dx, 0.0, dz])
    cloth = np.broadcast_to(cloth, (steps + 1, N, G * G, 3)).copy()
    return c.make_phase(name, np.arange(steps + 1) * DT, {"cloth_points": cloth, "corner_pos": cor})


def _rollout(**kw) -> c.Rollout:
    mv = kw.pop("move", 0.2)
    rest = _phase("rest", 40, **kw)
    act_move = _phase("act_move", 20, dx=mv, **kw)
    act_lift = _phase("act_lift", 20, dz=mv, **kw)
    cloth = np.broadcast_to(_sheet(0.39), (3, N, G * G, 3)).copy()
    cor = np.broadcast_to(np.array([[0, -0.2, 1.0], [0, 0.2, 1.0]]), (3, N, 2, 3)).copy()
    cor[..., 0] += np.linspace(0, 0.05, N)[None, :, None]
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * DT, {"cloth_points": cloth, "corner_pos": cor}, done=done)
    return c.synthetic([rest, act_move, act_lift, rand], dt=DT, episode_steps=100)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in cloth_hang.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_hang_passes() -> None:
    assert _failed(_rollout()) == []


def test_frozen_corners_fail_speed() -> None:
    assert "act_move_corner_travel" in _failed(_rollout(move=0.0))


def test_stretched_cloth_fails() -> None:
    failed = _failed(_rollout(stretch=2.0, depth=0.8))
    assert "cloth_stretch" in failed


def test_gravity_off_fails_hang_depth() -> None:
    assert "rest_hang_depth" in _failed(_rollout(depth=0.05))


def test_slipping_corner_fails_hold() -> None:
    assert "rest_corner_hold" in _failed(_rollout(slip=0.02))


def test_cloth_left_behind_fails_follow() -> None:
    assert "act_move_cloth_follows" in _failed(_rollout(follow=0.0))
