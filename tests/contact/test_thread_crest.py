# SPDX-License-Identifier: Apache-2.0
"""Crest-flat rows on the analytic thread pair (SCR-SC3-2, C5): an off-centre nut's crest meets the bolt's root."""

from __future__ import annotations

import numpy as np

from alloy.contact.analytic_samples import CREST_BAND, ThreadFeature, thread_samples
from alloy.contact.thread_sdf import nut_countersink, parse_thread_id, thread_sdf_np, thread_sdf_params

BOLT, NUT = "iso_thread:M10x1.5:6g", "iso_thread:M10x1.5:6H"
Z0, Z1, OUTER, SPACING = -4e-3, 4e-3, 8.5e-3, 5e-5


def _nut(crest: bool, chamfered: bool = False) -> ThreadFeature:
    cs = nut_countersink(parse_thread_id(NUT)) if chamfered else None
    return ThreadFeature(NUT, Z0, Z1, OUTER, flank_band=0.2, chamfer0=cs, chamfer1=cs, crest=crest)


def _bin_sets(sb) -> list[set]:
    return [set(map(tuple, sb.points[a:b].round(9))) for a, b in zip(sb.start[:-1], sb.start[1:])]


def test_crest_off_by_default_and_flanks_unchanged() -> None:
    assert ThreadFeature(NUT, Z0, Z1, OUTER).crest is False
    for chamfered in (False, True):
        a, b = thread_samples(_nut(False, chamfered), SPACING), thread_samples(_nut(True, chamfered), SPACING)
        assert b.n_bins == a.n_bins // 2 * 3
        flank_a, sets_b = _bin_sets(a), _bin_sets(b)
        flank_b = [s for i, s in enumerate(sets_b) if i % 3 != 2]
        assert flank_a == flank_b                                         # same flank rows, same bins and order


def test_crest_samples_lie_on_the_crest_flat() -> None:
    spec = parse_thread_id(NUT)
    sb = thread_samples(_nut(True), SPACING)
    crest = np.concatenate([sb.points[sb.start[i]:sb.start[i + 1]] for i in range(2, sb.n_bins, 3)])
    assert len(crest) > 0
    assert np.allclose(np.hypot(crest[:, 0], crest[:, 1]), spec.r_minor, atol=1e-8)
    prm = thread_sdf_params(spec, z0=Z0, z1=Z1, outer_radius=OUTER)
    assert np.abs(thread_sdf_np(prm, crest.astype(np.float64))).max() < 1e-7      # on the nut's own surface
    u = crest[:, 2] - spec.lead_2pi * np.arctan2(crest[:, 1], crest[:, 0])
    du = (u - 0.5 * spec.pitch + 0.5 * spec.pitch) % spec.pitch - 0.5 * spec.pitch
    assert np.abs(du).max() <= CREST_BAND * spec.pitch / 8.0 + 1e-6               # inside the flat, corners left out


def _deepest(feat: ThreadFeature, shift: float) -> float:
    """Deepest sample of the nut, moved radially by ``shift`` in +x, in the bolt's SDF (negative: inside)."""
    bolt = thread_sdf_params(parse_thread_id(BOLT), z0=-6e-3, z1=6e-3)
    pts = thread_samples(feat, SPACING).points.astype(np.float64) + np.array([shift, 0.0, 0.0])
    return float(thread_sdf_np(bolt, pts).min())


def test_crest_rows_see_an_off_centre_nut() -> None:
    """Shifted radially past the root gap (|deviation| / 2), the nut's crest goes into the bolt's root flat at the full
    shift, its flanks at half of it (60° profile): only crest rows report the crest overlap. On a lead-in turn the end
    cones trim the flanks and the crest is all that is left (C5's first turn)."""
    bolt, nut = parse_thread_id(BOLT), parse_thread_id(NUT)
    root_gap = nut.r_minor - bolt.r_minor
    assert abs(root_gap - 0.5 * abs(bolt.deviation)) < 1e-9
    over = 24e-6
    flank, crest = _deepest(_nut(False), root_gap + over), _deepest(_nut(True), root_gap + over)
    assert abs(crest + over) < 1e-6 and flank > crest + 0.4 * over


def test_lead_in_cone_faces_sampled_to_the_crest_edge() -> None:
    """crest=True at a countersunk end: samples on the countersink face, its edge with the crest flat included, all on
    the nut's chamfered surface; a flat-cut end gets none."""
    import math
    spec, cs = parse_thread_id(NUT), nut_countersink(parse_thread_id(NUT))
    a, b = thread_samples(_nut(True, True), SPACING).points, thread_samples(_nut(False, True), SPACING).points
    prm = thread_sdf_params(spec, z0=Z0, z1=Z1, outer_radius=OUTER, chamfer0=cs, chamfer1=cs)
    assert np.abs(thread_sdf_np(prm, a.astype(np.float64))).max() < 2e-7
    z_edge = Z0 + (cs.r_face - spec.r_minor) / math.tan(math.radians(cs.angle_deg))
    low = a[a[:, 2] < Z0 + 0.6 * spec.pitch]
    assert low[:, 2].min() <= z_edge + 1e-7                           # down to the countersink/crest edge
    assert low[:, 2].min() < b[b[:, 2] < 0.0][:, 2].min() - 1e-5      # below what the flank grid reaches
    flat = thread_samples(_nut(True, False), SPACING).points
    assert flat[:, 2].min() >= Z0 + 0.5 * spec.pitch - 1e-9           # flat-cut ends: END_TRIM as before


def test_crest_corners_sampled_exactly_on_the_lead_in() -> None:
    """crest=True at a countersunk end: samples on both crest/flank corner helices within END_BAND pitches of it, on
    the nut's surface; none mid-slab beyond the grid's own."""
    from alloy.contact.analytic_samples import END_BAND
    spec, cs = parse_thread_id(NUT), nut_countersink(parse_thread_id(NUT))
    pts = thread_samples(_nut(True, True), SPACING).points.astype(np.float64)
    u = pts[:, 2] - spec.lead_2pi * np.arctan2(pts[:, 1], pts[:, 0])
    du = (u - 0.5 * spec.pitch + 0.5 * spec.pitch) % spec.pitch - 0.5 * spec.pitch
    on_crest = np.abs(np.hypot(pts[:, 0], pts[:, 1]) - spec.r_minor) < 1e-9
    corner = on_crest & (np.abs(np.abs(du) - spec.pitch / 8.0) < 1e-8)
    assert corner.sum() > 1000
    end = END_BAND * spec.pitch + 1e-9
    assert ((pts[corner, 2] <= Z0 + end) | (pts[corner, 2] >= Z1 - end)).all()
    prm = thread_sdf_params(spec, z0=Z0, z1=Z1, outer_radius=OUTER, chamfer0=cs, chamfer1=cs)
    assert np.abs(thread_sdf_np(prm, pts[corner])).max() < 2e-7
