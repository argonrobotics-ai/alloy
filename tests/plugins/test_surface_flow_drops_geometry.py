# SPDX-License-Identifier: Apache-2.0
"""``geometry.point_cell`` (surface flow phase 2, design §4.7): the cell under a point and its normal distance, the
inverse of the cell centres (CPU)."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.context import PluginContext
from alloy.plugins.harness.scene import TestScene, body_xform, build_model
from alloy.plugins.surface_flow.charts import Chart, PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.geometry import point_cell
from alloy.plugins.surface_flow.tables import build_tables

R, H = 0.04, 0.1
GLASS = ((0.012, 0.005), (0.035, 0.005), (0.035, 0.1), (0.04, 0.1), (0.04, 0.0), (0.012, 0.0))


def _scene(b: newton.ModelBuilder) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.01), hx=0.2, hy=0.1, hz=0.01, label="plate")
    cup = b.add_body(xform=body_xform(0.5, 0.0, 0.2), label="cup")
    b.add_shape_sphere(cup, radius=0.01, label="cup_hull")


@wp.kernel
def _near(charts: wp.array[Chart], prof: wp.array[wp.vec3], cchart: wp.array[wp.int32], p: wp.array[wp.vec3],
          cell: wp.array[wp.int32], dist: wp.array[wp.float32], normal: wp.array[wp.vec3]):
    c = wp.tid()
    n = point_cell(charts[cchart[c]], prof, p[c])
    cell[c] = n.cell
    dist[c] = n.dist
    normal[c] = n.normal


def _tables():
    model = build_model(TestScene(build=_scene, ground=False, gravity=(0.0, 0.0, 0.0)), "cpu", 1)
    ctx = PluginContext(model, 0.02)
    return build_tables(ctx, [PlaneChart("plate", cells=(20, 10)),
                              RevolutionChart(profile=((R, H), (R, 0.0)), n_theta=12, ds=0.005, body="cup"),
                              RevolutionChart(profile=GLASS, n_theta=24, ds=0.0025)])


def _query(t, points: np.ndarray, charts: np.ndarray):
    n = len(points)
    out = (wp.zeros(n, dtype=wp.int32), wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.vec3))
    wp.launch(_near, dim=n, inputs=[t.charts, t.prof, wp.array(charts.astype(np.int32), dtype=wp.int32),
                                    wp.array(points.astype(np.float32), dtype=wp.vec3)], outputs=list(out),
              device="cpu")
    return tuple(a.numpy() for a in out)


def _chord_dist(t, points: np.ndarray) -> np.ndarray:
    """Host reference: each point's signed distance from its cell's face chord (revolution charts; the plane's z).
    At a profile corner the chord between two face nodes cuts the corner, and the cell centre (on the profile) sits
    off it: up to 0.24 mm on GLASS. point_cell, like ray_hit, measures from the chords."""
    charts, prof, cchart, starts = t.charts.numpy(), t.prof.numpy(), t.cchart.numpy(), t.starts
    out = points[:, 2].astype(np.float64).copy()
    for c, p in enumerate(points):
        ch = charts[cchart[c]]
        if int(ch["kind"]) == 0:
            continue
        k = (c - int(ch["cell_start"])) // int(ch["ny"])
        a, b = prof[int(ch["prof_start"]) + k], prof[int(ch["prof_start"]) + k + 1]
        seg = np.array([b[0] - a[0], b[1] - a[1]])
        n = np.array([-seg[1], seg[0]]) / np.linalg.norm(seg)
        out[c] = np.dot(np.array([np.hypot(p[0], p[1]) - a[0], p[2] - a[1]]), n)
    return out


def test_point_over_each_cell_centre_finds_that_cell():
    t = _tables()
    cpos, cnrm, cchart = t.cpos.numpy(), t.cnrm.numpy(), t.cchart.numpy()
    for off in (5.0e-4, -2.0e-4):                         # above the film side, and just inside the material
        pts = cpos + off * cnrm
        cell, dist, normal = _query(t, pts, cchart)
        assert np.array_equal(cell, np.arange(len(cpos)))
        assert np.allclose(dist, _chord_dist(t, pts), atol=2.0e-6)
        straight = np.abs(_chord_dist(t, cpos)) < 1.0e-7       # cells whose centre is on its chord: exactly `off`
        assert straight.mean() > 0.95 and np.allclose(dist[straight], off, atol=2.0e-6)
        assert np.allclose(normal, cnrm, atol=1.0e-5)


def test_point_off_a_plane_chart_is_outside():
    t = _tables()
    cell, dist, _ = _query(t, np.array([[0.3, 0.0, 0.001], [0.0, 0.0, 0.004]]), np.array([0, 0]))
    assert cell[0] == -1 and cell[1] >= 0 and np.isclose(dist[1], 0.004)


def test_point_past_a_revolution_profile_end_is_outside():
    """WP15 review: a point above a wall's open top end (or below its bottom end) is off the chart, not "2 mm from the
    film" (before, the foot clamped to the end and only the normal part counted); beside the wall it is on it."""
    t = _tables()
    pts = np.array([[R + 0.002, 0.0, H + 0.2], [R + 0.002, 0.0, H + 0.001], [R + 0.002, 0.0, -0.05],
                    [R + 0.002, 0.0, 0.5 * H], [0.0375, 0.0, 0.15]])           # the wall's wetted side is +r
    cell, dist, _ = _query(t, pts, np.array([1, 1, 1, 1, 2]))
    assert cell[0] == -1 and cell[1] == -1 and cell[2] == -1
    assert cell[3] >= 0 and np.isclose(dist[3], 0.002, atol=1e-6)
    assert cell[4] >= 0 and np.isclose(dist[4], 0.05, atol=1e-6)      # over the glass's rim top: a real distance
