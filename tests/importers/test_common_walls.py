# SPDX-License-Identifier: Apache-2.0
"""common.walls: min_wall re-measured where Alloy's probe reads a remesh defect; plausible readings kept."""

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from alloy.cards.geometry import MeshData  # noqa: E402
from alloy.importers.common import cards as cw  # noqa: E402
from alloy.importers.common import walls  # noqa: E402
from alloy.importers.common.checks import THIN  # noqa: E402


def _box(size, centre, max_edge):
    m = trimesh.creation.box(size)
    m.apply_translation(centre)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge)
    return np.asarray(v, float), np.asarray(f, int)


def test_a_sliver_defect_is_ignored_and_the_real_wall_is_read():
    plate = _box((0.02, 0.02, 0.002), (0.0, 0.0, 0.0), 0.0015)               # a 2 mm plate
    pillow = _box((0.0006, 0.0006, 4e-6), (0.0, 0.0, 0.001 - 5e-6), 1.0)    # a 0.7 mm^2 shell 3 um under its top
    v = np.vstack([plate[0], pillow[0]])
    mesh = MeshData(v, np.vstack([plate[1], pillow[1] + len(plate[0])]))
    d, area = walls.ray_walls(mesh, within=0.005)
    assert d.min() < 1e-5                                                   # the defect answers some rays
    assert walls.wall(mesh, within=0.005) == pytest.approx(0.002, rel=1e-6)  # 10 mm^2 of faces agree on 2 mm
    assert walls.min_wall(mesh, 1.5e-3, cap=2e-3) == 1.5e-3                  # a plausible probe reading is kept
    assert walls.min_wall(mesh, 1.2e-3, cap=2e-3) == 1.2e-3
    thin = MeshData(*_box((0.02, 0.02, 0.0002), (0.0, 0.0, 0.0), 0.0015))    # a real 0.2 mm sheet
    assert walls.min_wall(thin, 1.5e-4, cap=2e-3) == 1.5e-4                  # re-measure agrees: the probe's value stays
    assert walls.min_wall(thin, 1e-6, cap=2e-3) == pytest.approx(2e-4, rel=1e-6)   # a defect reading is replaced
    solid = MeshData(*_box((0.02, 0.02, 0.02), (0.0, 0.0, 0.0), 0.004))
    assert walls.min_wall(solid, 1e-6, cap=2e-3) == math.nextafter(2e-3, 0.0)   # no thin wall: just under the cap


def test_the_note_says_how_min_wall_was_measured():
    assert cw.min_wall_note(1e-3, 1e-3) == cw.MIN_WALL_NOTE
    assert "plain probe read 2.3e-05 m on remesh defects" in cw.min_wall_note(1.17e-3, 2.3e-5)
    assert "read 0.00027 m from its 1,024 sampled faces" in cw.min_wall_note(1.14e-4, 2.7e-4)   # found a thinner wall
    capped = cw.min_wall_note(math.nextafter(THIN, 0.0), 1e-6)
    assert capped.startswith("no wall of the physics mesh is thinner than") and "1e-06 m" in capped
