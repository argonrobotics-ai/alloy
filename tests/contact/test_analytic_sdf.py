# SPDX-License-Identifier: Apache-2.0
"""Analytic SDFs: Warp kernel equals the numpy twin; values against hand-computed distances; GeoType codes pinned."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact import analytic_sdf as asdf
from alloy.core.errors import PrecisionSpecError


def test_geotype_codes_match_newton():
    assert asdf.GEO_PLANE == int(newton.GeoType.PLANE)
    assert asdf.GEO_SPHERE == int(newton.GeoType.SPHERE)
    assert asdf.GEO_CAPSULE == int(newton.GeoType.CAPSULE)
    assert asdf.GEO_CYLINDER == int(newton.GeoType.CYLINDER)
    assert asdf.GEO_BOX == int(newton.GeoType.BOX)


@pytest.mark.parametrize(
    ("geo", "scale", "point", "expected"),
    [
        (asdf.GEO_SPHERE, (0.01, 0, 0), (0.0, 0.0, 0.02), 0.01),
        (asdf.GEO_SPHERE, (0.01, 0, 0), (0.0, 0.003, 0.0), -0.007),
        (asdf.GEO_BOX, (0.01, 0.02, 0.03), (0.015, 0.0, 0.0), 0.005),
        (asdf.GEO_BOX, (0.01, 0.02, 0.03), (0.0, 0.0, 0.0), -0.01),
        (asdf.GEO_BOX, (0.01, 0.02, 0.03), (0.013, 0.024, 0.0), 0.005),       # edge: hypot(3, 4) mm
        (asdf.GEO_CAPSULE, (0.005, 0.02, 0), (0.0, 0.0, 0.03), 0.005),        # cap: 30 - 20 - 5 mm
        (asdf.GEO_CAPSULE, (0.005, 0.02, 0), (0.008, 0.0, 0.01), 0.003),      # side
        (asdf.GEO_CYLINDER, (0.006, 0.015, 0), (0.0061, 0.0, 0.0), 1e-4),     # 0.1 mm radial gap
        (asdf.GEO_CYLINDER, (0.006, 0.015, 0), (0.0, 0.0, 0.0151), 1e-4),     # end face
        (asdf.GEO_CYLINDER, (0.006, 0.015, 0), (0.009, 0.0, 0.019), 0.005),   # rim: hypot(3, 4) mm
        (asdf.GEO_CYLINDER, (0.006, 0.015, 0), (0.0, 0.0, 0.0), -0.006),
        (asdf.GEO_PLANE, (0, 0, 0), (1.0, 2.0, -0.002), -0.002),
    ],
)
def test_numpy_values(geo, scale, point, expected):
    assert float(asdf.sd_primitive_np(geo, scale, np.array(point))) == pytest.approx(expected, abs=1e-12)


@wp.kernel
def _eval(geo: wp.array[int], scale: wp.array[wp.vec3], pts: wp.array[wp.vec3], out: wp.array[float]):
    i = wp.tid()
    out[i] = asdf.sd_primitive(geo[i], scale[i], pts[i])


def test_warp_matches_numpy(device, rng):
    geos = [asdf.GEO_SPHERE, asdf.GEO_BOX, asdf.GEO_CAPSULE, asdf.GEO_CYLINDER, asdf.GEO_PLANE]
    scales = {asdf.GEO_SPHERE: (0.01, 0, 0), asdf.GEO_BOX: (0.01, 0.02, 0.005), asdf.GEO_CAPSULE: (0.004, 0.01, 0),
              asdf.GEO_CYLINDER: (0.006, 0.015, 0), asdf.GEO_PLANE: (0, 0, 0)}
    n = 400
    g = np.array([geos[i % len(geos)] for i in range(n)], np.int32)
    s = np.array([scales[x] for x in g], np.float32)
    p = rng.uniform(-0.03, 0.03, size=(n, 3)).astype(np.float32)
    out = wp.zeros(n, dtype=float, device=device)
    wp.launch(_eval, dim=n, inputs=[wp.array(g, dtype=int, device=device), wp.array(s, dtype=wp.vec3, device=device),
                                    wp.array(p, dtype=wp.vec3, device=device), out], device=device)
    ref = np.array([asdf.sd_primitive_np(int(g[i]), s[i].astype(np.float64), p[i].astype(np.float64))
                    for i in range(n)])
    np.testing.assert_allclose(out.numpy(), ref, atol=2e-8)


def test_unsupported_types():
    with pytest.raises(PrecisionSpecError):
        asdf.sd_primitive_np(int(newton.GeoType.MESH), (1, 1, 1), np.zeros(3))
    assert not asdf.has_exact_sdf(int(newton.GeoType.CYLINDER), (0.006, 0.015, 0.01))   # crowned
    assert asdf.has_exact_sdf(int(newton.GeoType.CYLINDER), (0.006, 0.015, 0.0))
    assert not asdf.has_exact_sdf(int(newton.GeoType.ELLIPSOID), (1, 2, 3))
