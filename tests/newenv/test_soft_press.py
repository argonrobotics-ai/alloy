# SPDX-License-Identifier: Apache-2.0
"""E13 checker on synthetic rollouts: a good press, a rigid cube, a frozen plate, a cube that does not recover."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c, soft_press

N, DT, M = 3, 0.02, 5
H0, W0 = 0.0995, 0.10


def _cube(h: float, w: float) -> np.ndarray:
    g = np.linspace(-0.5, 0.5, M)
    x, y, z = np.meshgrid(g * w, g * w, np.linspace(0, h, M))
    pts = np.stack([x.ravel(), y.ravel(), z.ravel() + 0.0025 * 0], axis=-1)
    return np.broadcast_to(pts, (N, *pts.shape))


def _phase(name: str, steps: int, h0: float, h1: float, plate: tuple[float, float], f0: float, f1: float,
           bulge: float = 1.118, rise: bool = False) -> c.Phase:
    k = np.linspace(0, 1, steps + 1)
    hs = h0 + k * (h1 - h0)
    ws = W0 * (1 + (bulge - 1) * (H0 - hs) / 0.02) if bulge != 1 else np.full_like(hs, W0)
    pts = np.stack([_cube(h, w) for h, w in zip(hs, ws)])
    pz = plate[0] + k * (plate[1] - plate[0])
    force = np.broadcast_to((f0 + k * (f1 - f0))[:, None], (steps + 1, N))
    return c.make_phase(name, np.arange(steps + 1) * DT, {"soft_points": pts, "plate_z": np.broadcast_to(pz[:, None], (steps + 1, N)),
                                                           "plate_force": force})


def _rollout(squeeze: float = 0.02, force: float = 210.0, plate_travel: float = 0.02, recover: float = 0.0,
             bulge: float = 1.118) -> c.Rollout:
    h_end = H0 - squeeze
    rest = _phase("rest", 20, H0, H0, (0.1, 0.1), 0.5, 0.5, bulge=1.0)
    down = _phase("act_down", 50, H0, h_end, (0.1, 0.1 - plate_travel), 0.5, force, bulge=bulge)
    up = _phase("act_up", 50, h_end, H0 - recover, (0.1 - plate_travel, 0.1 - plate_travel + 0.02), force, 0.5, bulge=bulge)
    settle = _phase("act_settle", 25, H0 - recover, H0 - recover, (0.1, 0.1), 0.5, 0.5, bulge=1.0)
    rand = c.make_phase("random", np.arange(3) * DT, {"soft_points": np.stack([_cube(H0, W0)] * 3),
                                                      "plate_z": np.full((3, N), 0.1), "plate_force": np.zeros((3, N))},
                        done=np.array([[0] * N, [1] * N, [0] * N], bool))
    return c.synthetic([rest, down, up, settle, rand], dt=DT, episode_steps=100)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in soft_press.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_press_passes() -> None:
    assert _failed(_rollout()) == []


def test_rigid_cube_fails_follow() -> None:
    assert "press_cube_follows" in _failed(_rollout(squeeze=0.0, force=900.0, bulge=1.0))


def test_wrong_force_fails() -> None:
    assert "press_force_end" in _failed(_rollout(force=3.0))


def test_frozen_plate_fails() -> None:
    assert "press_plate_travel" in _failed(_rollout(plate_travel=0.0))


def test_no_recovery_fails() -> None:
    assert "release_height" in _failed(_rollout(recover=0.015))


def test_no_bulge_fails() -> None:
    assert "press_bulge" in _failed(_rollout(bulge=1.0))
