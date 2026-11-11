# SPDX-License-Identifier: Apache-2.0
"""hullfit: one hull for a near-convex mesh, thin CoACD slivers merged or thickened, real thin walls kept."""

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("scipy")

import trimesh  # noqa: E402

from alloy.importers.common import hullfit as hf  # noqa: E402
from alloy.importers.common import hulls as hl  # noqa: E402

NO_SINGLE = hl.HullParams(solid_min=2.0)          # never one hull: exercise thicken on the given hulls


def _box(lo, hi):
    m = trimesh.creation.box(np.subtract(hi, lo))
    m.apply_translation(0.5 * (np.asarray(lo, float) + np.asarray(hi, float)))
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def test_near_convex_meshes_skip_coacd_and_long_ones_are_cut_along_their_length(monkeypatch):
    monkeypatch.setattr(hl, "_run_coacd", lambda *a: pytest.fail("CoACD ran on a convex mesh"))
    v, f = _box((-0.069, -0.0645, 0.0), (0.069, 0.0645, 0.039))           # the YCB pudding box: compact, one hull
    hulls, stats = hl.decompose(v, f, hl.hull_volume(v, f))
    assert stats.single and stats.n_hulls == 1
    v, f = _box((-0.0105, -0.0605, 0.0), (0.0105, 0.0605, 0.019))         # the YCB large marker's bounds
    hulls, stats = hl.decompose(v, f, hl.hull_volume(v, f))
    assert stats.single and stats.n_hulls == 3 and stats.thin == 0          # 121 mm / (2 x 21 mm): 3 pieces
    assert stats.volume_ratio == pytest.approx(1.0, abs=1e-6)               # they tile the bar: no overlap, no gap
    assert sorted(np.ptp(h[0][:, 1]) for h in hulls) == pytest.approx([0.121 / 3] * 3, rel=1e-6)
    torus = trimesh.creation.torus(0.05, 0.01)
    tv, tf = np.asarray(torus.vertices), np.asarray(torus.faces)
    assert hf.single_hull(tv, float(torus.volume), hf.Surface(tv, tf), hl.HullParams()) is None   # a ring: no


def test_slices_of_a_solid_bar_merge_to_thick_hulls():
    bar = _box((0.0, 0.0, 0.0), (0.02, 0.12, 0.02))
    slices = [_box((x, 0.0, 0.0), (x + 0.004, 0.12, 0.02)) for x in np.arange(0.0, 0.02 - 1e-9, 0.004)]
    hulls, fit = hf.thicken(slices, hf.Surface(*bar), NO_SINGLE)
    assert fit.merged >= 1 and fit.thin == 0
    assert min(hl.hull_thickness(*h) for h in hulls) >= hl.THIN_FLOOR
    assert sum(hl.hull_volumes(hulls)) == pytest.approx(hl.hull_volume(*bar), rel=1e-6)   # no overfill


def test_a_thin_wall_stays_thin_and_a_covered_sliver_is_dropped():
    plate = _box((0.0, 0.0, 0.0), (0.1, 0.1, 0.002))                      # a real 2 mm wall: both sides air
    halves = [_box((0.0, 0.0, 0.0), (0.05, 0.1, 0.002)), _box((0.05, 0.0, 0.0), (0.1, 0.1, 0.002))]
    hulls, fit = hf.thicken(halves, hf.Surface(*plate), NO_SINGLE)
    assert fit.thickened == 0 and fit.thin == len(hulls) >= 1
    assert max(hl.hull_thickness(*h) for h in hulls) == pytest.approx(0.002, rel=1e-6)
    block = _box((0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    inner = _box((0.01, 0.01, 0.02), (0.04, 0.04, 0.022))                 # a 2 mm sliver inside the block
    hulls, fit = hf.thicken([block, inner], hf.Surface(*block), NO_SINGLE)
    assert fit.covered == 1 and len(hulls) == 1


def test_thickening_only_goes_into_the_object():
    bar = _box((0.0, 0.0, 0.0), (0.02, 0.12, 0.02))
    skin = _box((0.0, 0.0, 0.018), (0.02, 0.12, 0.02))                    # the top 2 mm of the bar
    surf = hf.Surface(*bar)
    down, up = np.array([0.0, 0.0, -0.003]), np.array([0.0, 0.0, 0.003])
    assert surf.swept_out(skin, down) < hf.INWARD_TOL < surf.swept_out(skin, up)
    assert surf.swept_out(skin, up) == pytest.approx(0.0015, rel=0.2)    # half the step, on average
