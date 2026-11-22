# SPDX-License-Identifier: Apache-2.0
"""Law: the sports closest-point kernels give unit normals at the centre, and tactile deposition stays finite.

``box_closest`` / ``cylinder_closest`` at a point exactly on the axis used to return a normal of length 2; tactile
``_deposit`` used to divide 0 by 0 when the spread was tiny next to the distance.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.plugins.sports import kernels as sports_k
from alloy.plugins.tactile import kernels as tactile_k


@wp.kernel
def _probe_sports(out_box: wp.array[wp.vec4], out_cyl: wp.array[wp.vec4]):
    out_box[0] = sports_k.box_closest(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.1, 0.2, 0.3))
    out_cyl[0] = sports_k.cylinder_closest(wp.vec3(0.0, 0.0, 0.0), 0.1, 0.05)


@wp.kernel
def _probe_tactile(taxel_xy: wp.array[wp.vec2], acc: wp.array[wp.int64]):
    tactile_k._deposit(0, wp.vec3(10.0, 0.0, 0.0), wp.vec2(0.05, 0.05), taxel_xy, 2, 1, 1.0e-6, acc)


def test_sports_box_and_cylinder_closest_unit_normal_at_zero():
    """box_closest and cylinder_closest must emit unit normals at cl == 0 (not length 2)."""
    ob = wp.zeros(1, dtype=wp.vec4, device="cpu")
    oc = wp.zeros(1, dtype=wp.vec4, device="cpu")
    wp.launch(_probe_sports, dim=1, inputs=[ob, oc], device="cpu")
    b, c = ob.numpy()[0], oc.numpy()[0]
    assert np.linalg.norm(b[:3]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(c[:3]) == pytest.approx(1.0, abs=1e-6)


def test_tactile_deposit_stays_finite_for_tiny_spread():
    """_deposit must shift Gaussian exponents by min_d2 so tiny spread never divides 0/0."""
    xy = wp.array([[0.0, 0.0], [0.01, 0.0]], dtype=wp.vec2, device="cpu")
    acc = wp.zeros(6, dtype=wp.int64, device="cpu")
    wp.launch(_probe_tactile, dim=1, inputs=[xy, acc], device="cpu")
    got = acc.numpy()
    assert got[3] > 0 and got[0] == 0
