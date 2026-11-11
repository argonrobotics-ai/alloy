# SPDX-License-Identifier: Apache-2.0
"""Dry-whisk task (alloy/tasks/soil_whisk.py): geometry, spoonfuls and the Lacey mixing index (no simulation)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from alloy.tasks import soil_whisk as W


def _corners(size, pos, yaw):
    lx, ly = 0.5 * size[0], 0.5 * size[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(pos[0] + c * x - s * y, pos[1] + s * x + c * y) for x in (-lx, lx) for y in (-ly, ly)]


VOXEL = 0.0045                       # the flour's MPM voxel in the default scene (router: min(d_feature / 4, 5 mm))


def test_the_star_keeps_two_voxels_from_the_wall_at_every_angle_and_position() -> None:
    s = W.WhiskSpec()
    far = 0.0
    for phi in np.linspace(0, 2 * math.pi, 17):
        centre = (s.orbit_radius * math.cos(phi), s.orbit_radius * math.sin(phi))
        for theta in np.linspace(0, 0.5 * math.pi, 9):
            for size, pos, rpy in W.star_arms(s, theta, centre).values():
                far = max(far, *(math.hypot(x, y) for x, y in _corners(size, pos, rpy[2])))
    assert far <= W.reach(s) + 1e-9 and s.bowl_radius - far >= 2 * VOXEL


def test_arms_do_not_overlap_and_stand_off_the_floor() -> None:
    """Overlapping kinematic parts read as penetrations from step 0 (OSS-3's fixture note)."""
    s = W.WhiskSpec()
    arms = list(W.star_arms(s, 0.3, (0.0, 0.0)).values())
    assert len(arms) == 4
    for size, pos, _ in arms:
        assert math.hypot(pos[0], pos[1]) - 0.5 * size[0] == pytest.approx(s.arm[3])     # inner end at r0
        assert pos[2] - 0.5 * size[2] == pytest.approx(s.bowl_wall + s.floor_gap)
        assert size[1] >= VOXEL                                                         # at least a voxel thick
    assert s.arm[3] > 0.5 * s.arm[1]          # an arm along x starts past the half-thickness of the arm along y


def test_a_whisk_reaching_the_wall_is_refused() -> None:
    with pytest.raises(ValueError, match="two voxels"):
        W.build(replace(W.WhiskSpec(), orbit_radius=0.035))


def test_spoonfuls_lie_on_the_flour_top_inside_the_flour() -> None:
    s = W.WhiskSpec()
    top = s.bowl_wall + s.flour_height
    regions = W.spoon_regions(s)
    assert len(regions) == len(W.SPECIES)
    for k, r in enumerate(regions):
        x0, y0, z0, x1, y1, z1 = r[:6]
        assert z0 < top < z1
        assert max(math.hypot(x, y) for x in (x0, x1) for y in (y0, y1)) < s.flour_radius
        assert list(r[6:]) == [1.0 if j == k else 0.0 for j in range(len(W.SPECIES))]


def _grid(n: int, spacing: float) -> np.ndarray:
    g = (np.arange(n) + 0.5) * spacing
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)


def test_lacey_is_zero_for_whole_cells_of_one_species_and_one_for_a_random_mix() -> None:
    x = _grid(24, 0.0025)                                    # 60 mm cube, 13,824 particles, 64 per 10 mm cell
    lab = np.zeros((len(x), len(W.SPECIES)))
    lab[x[:, 0] < 0.02, 0] = 1.0                             # sugar: two whole layers of cells
    lab[(x[:, 0] >= 0.02) & (x[:, 0] < 0.03), 1] = 1.0
    lab[(x[:, 0] >= 0.03) & (x[:, 0] < 0.04), 2] = 1.0
    seg = W.lacey(x, lab, 0.01)
    assert seg["cells"] == 216 and all(abs(seg[k]) < 1e-9 for k in W.SPECIES)
    rng = np.random.default_rng(0)
    mixed = W.lacey(x, lab[rng.permutation(len(x))], 0.01)
    assert all(abs(mixed[k] - 1.0) < 0.02 for k in W.SPECIES)


def test_occupied_volume_counts_voxels_with_particles() -> None:
    x = _grid(24, 0.0025)                                    # 60 mm cube
    assert W.occupied_volume(x, 0.01) == pytest.approx(216 * 1e-6)
    assert W.occupied_volume(np.concatenate([x, [[0.5, 0.5, 0.5]]]), 0.01) == pytest.approx(217 * 1e-6)
    assert W.occupied_volume(np.concatenate([x, [[0.5, 0.5, 0.5]]]), 0.01, min_count=4) == pytest.approx(216 * 1e-6)


def test_lacey_skips_sparse_cells() -> None:
    x = np.concatenate([_grid(8, 0.0025), [[0.5, 0.5, 0.5]]])               # one stray particle in its own cell
    lab = np.zeros((len(x), len(W.SPECIES)))
    lab[: len(x) // 2, :] = 1.0
    assert W.lacey(x, lab, 0.01)["cells"] == 8
