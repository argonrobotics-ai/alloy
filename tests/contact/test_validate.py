# SPDX-License-Identifier: Apache-2.0
"""validate_pair: every §3.5 rule has one failing case (WP3 acceptance), plus mesh measurements."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.contact.types import PrecisionPairSpec
from alloy.contact.validate import is_watertight, mesh_info, sdf_bytes_estimate, validate_pair
from alloy.core.errors import PrecisionSpecError
from alloy.testing.contact import PegHoleSpec, blind_hole_tris, box_tris, chord_error, cylinder_tris


class _Mesh:
    """Duck-typed newton.Mesh (vertices, indices) so the rules run without building Warp meshes."""

    def __init__(self, verts: np.ndarray, tris: np.ndarray) -> None:
        self.vertices, self.indices = verts, tris.reshape(-1)


GOOD = PrecisionPairSpec(0, 1, mode="hydroelastic", clearance=2e-4, voxel_size=5e-5)
PEG = _Mesh(*cylinder_tris(6e-3, 6e-3, 128))          # 12 mm long so no crop is needed at 5e-5
HOLE = None                                           # primitives / analytic partner skip mesh rules


def _expect(rule: str, spec: PrecisionPairSpec, mesh_a=PEG, mesh_b=HOLE, **kw) -> None:
    args = dict(entry_a="mj", entry_b="mj", interface_pos_tol=0.0) | kw
    with pytest.raises(PrecisionSpecError, match=rule):
        validate_pair(spec, mesh_a, mesh_b, **args)


def test_good_pair_passes():
    validate_pair(GOOD, PEG, HOLE, entry_a="mj", entry_b="mj", interface_pos_tol=0.0)


def test_voxel_vs_clearance():
    _expect("voxel_vs_clearance", dataclasses.replace(GOOD, clearance=5e-5))      # radial 2.5e-5 < voxel 5e-5


def test_voxel_vs_wall():
    _expect("voxel_vs_wall", dataclasses.replace(GOOD, wall_thickness=1.5e-4))   # wall/4 = 3.75e-5


def test_voxel_vs_feature_radius():
    small = _Mesh(*cylinder_tris(4e-5, 1e-3, 64))                               # 40 um radius pin
    _expect("voxel_vs_feature", GOOD, mesh_a=small)


def test_voxel_range():
    _expect("voxel_range", dataclasses.replace(GOOD, voxel_size=2e-4))
    _expect("voxel_range", dataclasses.replace(GOOD, voxel_size=1e-4))           # 12 mm part < 40 mm


def test_band_covers_gap():
    _expect("band_covers_gap", dataclasses.replace(GOOD, narrow_band=(-5e-4, 4e-4)))


def test_gap_never_builder_default():
    _expect("gap", dataclasses.replace(GOOD, gap=0.1, narrow_band=(-0.2, 0.2)))


def test_chord_rule():
    coarse = _Mesh(*cylinder_tris(6e-3, 6e-3, 24))      # chord 6e-3*(1-cos(pi/24)) = 5.1e-5 > 5e-6
    _expect("chord", GOOD, mesh_a=coarse)


def test_watertight_for_hydroelastic():
    v, t = cylinder_tris(6e-3, 6e-3, 128)
    open_tube = _Mesh(v, t[: 2 * 128])                  # walls only
    _expect("watertight", GOOD, mesh_a=open_tube)
    validate_pair(dataclasses.replace(GOOD, mode="sdf"), open_tube, None, entry_a="mj", entry_b="mj",
                  interface_pos_tol=0.0)


def test_crop_required_for_long_parts():
    rod = _Mesh(*cylinder_tris(6e-3, 0.06, 128))        # 120 mm / 5e-5 = 2400 voxels > 2048
    _expect("crop", GOOD, mesh_a=rod)
    validate_pair(dataclasses.replace(GOOD, crop_a=((-7e-3,) * 3, (7e-3,) * 3)), rod, None, entry_a="mj",
                  entry_b="mj", interface_pos_tol=0.0)


def test_sdf_budget():
    hole = _Mesh(*blind_hole_tris(PegHoleSpec(segments=128)))   # ~4.6e-3 m^2 at 5e-5 -> ~140 MB > 64 MB
    _expect("sdf_budget", dataclasses.replace(GOOD, mode="sdf"), mesh_a=hole)


def test_cross_entry_tolerance():
    # max_pen 5e-5 -> 0.25*max_pen = 1.25e-5 m
    _expect("cross_entry", GOOD, entry_a="mj", entry_b="vbd", interface_pos_tol=1e-4)
    validate_pair(GOOD, PEG, HOLE, entry_a="mj", entry_b="vbd", interface_pos_tol=1e-5)


def test_mesh_info_values():
    v, t = cylinder_tris(6e-3, 6e-3, 96)
    info = mesh_info(v, t)
    assert info.watertight
    assert info.max_chord_error == pytest.approx(chord_error(6e-3, 96), rel=1e-3)
    assert info.min_feature_radius == pytest.approx(6e-3, rel=1e-3)
    assert info.extent[2] == pytest.approx(12e-3)
    bv, bt = box_tris(0.01, 0.02, 0.03)
    binfo = mesh_info(bv, bt)
    assert binfo.watertight and math.isinf(binfo.min_feature_radius) and binfo.max_chord_error == 0.0
    assert binfo.area == pytest.approx(8 * (0.01 * 0.02 + 0.02 * 0.03 + 0.01 * 0.03))


def test_is_watertight_detects_flipped_triangle():
    v, t = box_tris(1.0, 1.0, 1.0)
    assert is_watertight(t)
    t2 = t.copy()
    t2[0] = t2[0][::-1]
    assert not is_watertight(t2)


def test_sdf_bytes_estimate_matches_research_example():
    # contact §4.7: 12 mm x 30 mm peg, A ~ 1.4e-3 m^2, b = 1 mm -> ~44 MB at 5e-5 and ~7 MB at 1e-4
    assert sdf_bytes_estimate(1.4e-3, 1e-3, 5e-5) / 2**20 == pytest.approx(44, rel=0.05)
    assert sdf_bytes_estimate(1.4e-3, 1e-3, 1e-4) / 2**20 == pytest.approx(7, rel=0.12)
