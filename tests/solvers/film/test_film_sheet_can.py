# SPDX-License-Identifier: Apache-2.0
"""Law S3 of J2 metal sheets (SCR-WP7-15 P3, sheet_can.py): a tinplate can wall dented by a finger, a measured row.
The J2 hinges on a closed, curved sheet (rest curvature in the authored rest angles) keep part of a deep press and
none of a shallow one; the run stays finite. The dent depth is not validated (sheet_can.py says why). CPU,
mass-scaled quasi-static."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.solvers.film import sheet_can as C

pytestmark = pytest.mark.compile_heavy


def test_the_can_mesh_is_a_closed_ring_with_outward_triangles():
    v, tris = C.can_mesh(0.033, 0.06, 16, 4)
    assert v.shape == (16 * 5, 3) and tris.shape == (2 * 16 * 4, 3)
    assert np.allclose(np.hypot(v[:, 0], v[:, 1]), 0.033)
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    n = np.cross(b - a, c - a)
    centre = (a + b + c) / 3.0
    assert np.all(np.sum(n[:, :2] * centre[:, :2], axis=1) > 0.0)                # every normal points out
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts.tolist()) == {1, 2} and (counts == 1).sum() == 2 * 16      # open only at the two rims


@pytest.mark.law
def test_s3_a_finger_dents_a_coarse_can_and_part_of_the_dent_stays():
    """32 x 8 wall, 40 mm tall, an 8 mm press: measured 2026-11-08 kept 1.07 mm of a 7.57 mm peak, ε̄p 0.72 %, the
    same at half the finger speed (1.07 mm)."""
    r = C.s3_can_dent(n_theta=32, n_z=8, height=0.04, depth=8.0e-3, speed=2.0e-2, hold=0.05, settle=0.3)
    assert r.finite and r.max_strain > 1.0e-3, r
    assert 0.9 * 8.0e-3 < r.peak_dent < 8.0e-3, r
    assert 0.5e-3 < r.kept_dent < 2.0e-3, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_s3_can_dent_rows():
    """The S3 rows, 48 x 14 wall, 60 mm tall (r 33 mm, 0.2 mm tinplate, 8 mm fingertip). Measured 2026-11-08: a 3 mm
    press stays elastic (kept 0.001 mm, 64 x 20), an 8 mm press keeps 0.96 mm of 7.52 mm (ε̄p 0.80 %)."""
    shallow = C.s3_can_dent(n_theta=48, n_z=14, depth=3.0e-3, settle=1.5)
    assert shallow.finite and shallow.max_strain == 0.0 and abs(shallow.kept_dent) < 0.05e-3, shallow
    deep = C.s3_can_dent(n_theta=48, n_z=14, depth=8.0e-3, settle=1.5)
    assert deep.finite and 0.8e-3 < deep.kept_dent < 1.1e-3 and 0.006 < deep.max_strain < 0.010, deep
