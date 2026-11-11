# SPDX-License-Identifier: Apache-2.0
"""E12 checker on synthetic rollouts: a good cover, a cloth through the cube, a floating cloth, a cube that does not slide."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import cloth_cover, common as c

N, G, DT = 4, 26, 0.05
TABLE = 0.5


def _cloth(cube: np.ndarray, top_z: float, side_z: float = 0.0015, flat: float = 0.5) -> np.ndarray:
    """(N, V, 3) a draped 50 cm cloth: on the cube's top face inside 5 cm, on the table beyond 13 cm, a ramp between."""
    g = np.linspace(-flat / 2, flat / 2, G)
    xx, yy = np.meshgrid(g, g)
    cheb = np.maximum(np.abs(xx), np.abs(yy)).ravel()
    z = np.where(cheb < 0.06, top_z, np.where(cheb > 0.13, TABLE + side_z, top_z - (cheb - 0.06) / 0.07 * (top_z - TABLE)))
    pts = np.stack([xx.ravel(), yy.ravel(), z], axis=-1)
    return pts[None] + cube[:, None, :] * np.array([1, 1, 0])


def _phase(name: str, steps: int, top_z: float, dx: float = 0.0, cube_dz: float = 0.0, flat: float = 0.5,
           side_z: float = 0.0015) -> c.Phase:
    k = np.linspace(0, 1, steps + 1)[:, None, None]
    start = np.stack([np.linspace(-0.1, 0.1, N), np.linspace(0.1, -0.1, N), np.full(N, TABLE + 0.05)], axis=-1)
    cube = start[None] + k * np.array([dx, 0.0, cube_dz])
    cloth = np.stack([_cloth(cube[i], top_z, side_z, flat) for i in range(steps + 1)])
    table = np.full((steps + 1, N), TABLE)
    return c.make_phase(name, np.arange(steps + 1) * DT, {"cloth_points": cloth, "cube_pos": cube, "table_z": table})


def _rollout(top_z: float = 0.6015, dx: float = 0.1, **kw) -> c.Rollout:
    rest = _phase("rest", 40, top_z, **kw)
    settle = _phase("settle", 20, top_z, **kw)
    slide = _phase("act_slide", 20, top_z, dx=dx, **kw)
    rng = np.random.default_rng(0)
    cube = np.concatenate([rng.uniform(-0.1, 0.1, (3, N, 2)), np.full((3, N, 1), TABLE + 0.05)], axis=-1)
    cloth = np.stack([_cloth(cube[i], top_z) for i in range(3)])
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * DT, {"cloth_points": cloth, "cube_pos": cube,
                                                      "table_z": np.full((3, N), TABLE)}, done=done)
    return c.synthetic([rest, settle, slide, rand], dt=DT, episode_steps=100)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in cloth_cover.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_cover_passes() -> None:
    assert _failed(_rollout()) == []


def test_cloth_through_cube_fails() -> None:
    assert "rest_inside_cube" in _failed(_rollout(top_z=0.57))


def test_cloth_floating_above_fails() -> None:
    assert "cloth_on_top_face" in _failed(_rollout(top_z=0.66))


def test_frozen_cube_fails_slide() -> None:
    assert "slide_travel" in _failed(_rollout(dx=0.0))


def test_stretched_cloth_fails() -> None:
    assert "cloth_stretch" in _failed(_rollout(flat=1.0))


def test_cloth_hovering_off_table_fails() -> None:
    assert "far_points_on_table" in _failed(_rollout(side_z=0.05))
