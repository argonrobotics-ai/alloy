# SPDX-License-Identifier: Apache-2.0
"""alloy.testing.contact: closed, outward meshes with the analytic polygon volumes; chord rule."""

from __future__ import annotations

import math

import pytest

from alloy.contact.validate import is_watertight
from alloy.testing import contact as tc


def _ngon_area(r: float, n: int) -> float:
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n)


def test_cylinder_volume_and_closed():
    v, t = tc.cylinder_tris(6e-3, 15e-3, 96)
    assert tc.signed_volume(v, t) == pytest.approx(_ngon_area(6e-3, 96) * 30e-3, rel=1e-6)
    assert is_watertight(t)


def test_blind_hole_volume_and_closed():
    s = tc.PegHoleSpec()
    v, t = tc.blind_hole_tris(s)
    expected = _ngon_area(0.5 * s.outer_d, s.segments) * s.height - _ngon_area(0.5 * s.bore_d, s.segments) * s.depth
    assert tc.signed_volume(v, t) == pytest.approx(expected, rel=1e-6)
    assert is_watertight(t)
    assert s.radial_clearance == pytest.approx(1e-4)
    assert s.bottom_z == pytest.approx(5e-3)


def test_ring_and_box():
    v, t = tc.ring_tris(6.1e-3, 15e-3, 0.01, 64)
    assert tc.signed_volume(v, t) == pytest.approx((_ngon_area(15e-3, 64) - _ngon_area(6.1e-3, 64)) * 0.01, rel=1e-6)
    assert is_watertight(t)
    bv, bt = tc.box_tris(0.01, 0.02, 1.25e-4)
    assert tc.signed_volume(bv, bt) == pytest.approx(0.02 * 0.04 * 2.5e-4, rel=1e-6)
    assert is_watertight(bt)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        tc.blind_hole_tris(tc.PegHoleSpec(depth=30e-3))
    with pytest.raises(ValueError):
        tc.ring_tris(2e-3, 1e-3, 0.01, 32)


def test_chord_rule():
    # 12.2 mm bore at 5 um needs N >= 78 (contact §4.4); CON-46's 96 sides give 3.3 um
    assert tc.segments_for_chord(6.1e-3, 5e-6) == 78
    assert tc.chord_error(6.1e-3, 78) <= 5e-6 < tc.chord_error(6.1e-3, 77)
    assert tc.chord_error(6.1e-3, 96) == pytest.approx(3.27e-6, rel=1e-2)


def test_peg_hole_mesh_builds(device):
    peg, hole, spec = tc.peg_hole_mesh()
    assert peg.vertices.shape[1] == 3 and hole.indices.size % 3 == 0
    assert spec.segments == 96
