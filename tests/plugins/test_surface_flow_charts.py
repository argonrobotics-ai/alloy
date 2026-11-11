# SPDX-License-Identifier: Apache-2.0
"""Surface-flow charts: grid topology, geometry (areas, normals, curvature), device tables and ray hits (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.context import PluginContext
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_model
from alloy.plugins.surface_flow.charts import (Chart, PlaneChart, RevolutionChart, plane_geometry, revolution_geometry,
                                               topology)
from alloy.plugins.surface_flow.geometry import cell_window, chart_xform, ray_hit
from alloy.plugins.surface_flow.tables import build_tables

R, H = 0.04, 0.1                                         # cylinder radius and height [m]
GLASS = ((0.012, 0.005), (0.035, 0.005), (0.035, 0.1), (0.04, 0.1), (0.04, 0.0), (0.012, 0.0))


def _scene(b: newton.ModelBuilder) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.01), hx=0.2, hy=0.1, hz=0.01, label="plate")
    cup = b.add_body(xform=body_xform(0.5, 0.0, 0.2), label="cup")
    b.add_shape_sphere(cup, radius=0.01, label="cup_hull")


def _ctx(worlds: int = 2) -> PluginContext:
    model = build_model(TestScene(build=_scene, ground=False, gravity=(0.0, 0.0, 0.0)), "cpu", worlds)
    return PluginContext(model, 0.02)


def test_topology_faces_and_cells_agree():
    for wrap in (False, True):
        nx, ny = 5, 4
        t = topology(nx, ny, wrap, ("wall", "open", "wall", "open"))
        assert len(t.fa) == (nx + 1) * ny + (nx * ny if wrap else nx * (ny + 1))
        for c in range(nx * ny):
            for k, f in enumerate(t.cface[c]):
                side = t.fb[f] if k % 2 == 0 else t.fa[f]    # a cell is b of its minus faces, a of its plus faces
                assert side == c
        interior = (t.fa >= 0) & (t.fb >= 0)
        assert (t.ftype[interior] == 0).all() and (t.ftype[~interior] > 0).all()
        assert set(t.ftype[:ny][t.fa[:ny] < 0]) == {1} and set(t.ftype[nx * ny:(nx + 1) * ny]) == {2}


def test_plane_and_revolution_areas_normals_curvature():
    g = plane_geometry(10, 5, (0.4, 0.2))
    assert g.area.sum() == pytest.approx(0.08)
    disc = revolution_geometry(RevolutionChart(profile=((0.02, 0.0), (0.1, 0.0)), n_theta=16, ds=0.002))
    assert disc.area.sum() == pytest.approx(math.pi * (0.1**2 - 0.02**2), rel=1e-9)
    assert np.allclose(disc.cnrm, [0.0, 0.0, 1.0]) and np.allclose(disc.curv, 0.0)
    out = revolution_geometry(RevolutionChart(profile=((R, H), (R, 0.0)), n_theta=32, ds=0.002))   # down: outside
    assert out.area.sum() == pytest.approx(2 * math.pi * R * H, rel=1e-9)
    radial = out.cpos[:, :2] / np.linalg.norm(out.cpos[:, :2], axis=1, keepdims=True)
    assert np.allclose(out.cnrm[:, :2], radial) and np.allclose(out.curv, 1.0 / R)   # convex
    inner = revolution_geometry(RevolutionChart(profile=((R, 0.0), (R, H)), n_theta=32, ds=0.002))  # up: inside
    assert np.allclose(inner.cnrm[:, :2], -radial) and np.allclose(inner.curv, -1.0 / R)
    glass = revolution_geometry(RevolutionChart(profile=GLASS, n_theta=24, ds=0.0025))
    length = sum(math.dist(a, b) for a, b in zip(GLASS[:-1], GLASS[1:]))
    assert len(glass.prof) - 1 == round(length / 0.0025)
    assert glass.curv.max() > 0.0 and glass.curv.min() < 0.0            # rim convex, inner corner concave


def test_profile_near_axis_is_refused():
    with pytest.raises(PluginError, match="axis"):
        revolution_geometry(RevolutionChart(profile=((0.001, 0.0), (0.05, 0.0)), n_theta=64, ds=0.002))


def test_tables_concatenate_charts_per_world():
    ctx = _ctx(worlds=2)
    t = build_tables(ctx, [PlaneChart("plate", cells=(20, 10)),
                           RevolutionChart(profile=((R, H), (R, 0.0)), n_theta=12, ds=0.005, body="cup"),
                           RevolutionChart(profile=((R, H), (R, 0.0)), n_theta=12, ds=0.005, body="cup",
                                           shape="cup_hull")])
    assert t.starts == [0, 200, 440] and t.shapes == [(20, 10), (20, 12), (20, 12)]
    assert t.cell_count == 200 + 240 + 240 and t.area_np[:200].sum() == pytest.approx(0.08, rel=1e-6)
    body, shape = t.cbody.numpy(), t.cshape.numpy()
    assert (body[:, 0] == -1).all() and len(set(body[:, 1])) == 2           # each world's own cup
    assert (shape[:, 0] >= 0).all() and len(set(shape[:, 0])) == 2 and (shape[:, 1] == -1).all()
    assert len(set(shape[:, 2])) == 2 and (shape[:, 2] >= 0).all()                # the named hull, per world
    local = t.clocal.numpy()
    assert local[0, 0, 2] == pytest.approx(0.02)                             # the box top face
    fa, fb, cface = t.fa.numpy(), t.fb.numpy(), t.cface.numpy()
    assert ((fa < 0) | (fa >= 200) | (fb < 200)).all()                       # charts do not share faces
    first, count = 21 * 10 + 20 * 11, 21 * 12 + 20 * 12                    # the cup chart's faces
    assert (cface[200:440] >= first).all() and (cface[200:440] < first + count).all()


@wp.kernel
def _cast(charts: wp.array[Chart], prof: wp.array[wp.vec3], body_q: wp.array[wp.transform],
          cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], k: int, o: wp.array[wp.vec3],
          d: wp.array[wp.vec3], cell: wp.array[wp.int32], t: wp.array[wp.float32], n: wp.array[wp.vec3]):
    r = wp.tid()
    h = ray_hit(charts[k], prof, chart_xform(body_q, cbody[0, k], clocal[0, k]), o[r], wp.normalize(d[r]))
    cell[r] = h.cell
    t[r] = h.t
    n[r] = h.normal


@wp.kernel
def _window(charts: wp.array[Chart], prof: wp.array[wp.vec3], k: int, i: int, j: int, radius: float,
            out: wp.array[wp.vec4i]):
    out[0] = cell_window(charts[k], prof, i, j, radius)


def _rays(ctx, t, k, rays):
    o = wp.array([r[0] for r in rays], dtype=wp.vec3, device="cpu")
    d = wp.array([r[1] for r in rays], dtype=wp.vec3, device="cpu")
    n = len(rays)
    cell, dist, normal = (wp.zeros(n, dtype=wp.int32), wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.vec3))
    q = ctx.model.body_q
    wp.launch(_cast, dim=n, inputs=[t.charts, t.prof, q, t.cbody, t.clocal, k, o, d], outputs=[cell, dist, normal],
              device="cpu")
    return cell.numpy(), dist.numpy(), normal.numpy()


def test_ray_hits_plane_and_revolution_from_the_wetted_side():
    ctx = _ctx(worlds=1)
    t = build_tables(ctx, [PlaneChart("plate", cells=(20, 10)),
                           RevolutionChart(profile=((R, H), (R, 0.0)), n_theta=12, ds=0.005, body="cup"),
                           RevolutionChart(profile=GLASS, n_theta=24, ds=0.0025)])
    cell, dist, normal = _rays(ctx, t, 0, [((0.05, 0.03, 0.5), (0.0, 0.0, -1.0)),       # from above: cell (12, 6)
                                           ((0.0, 0.0, -0.5), (0.0, 0.0, 1.0)),        # from below: the dry side
                                           ((0.3, 0.0, 0.5), (0.0, 0.0, -1.0))])       # off the face
    assert cell.tolist() == [12 * 10 + 6, -1, -1]
    assert dist[0] == pytest.approx(0.48, abs=1e-6) and np.allclose(normal[0], [0.0, 0.0, 1.0])
    cell, dist, normal = _rays(ctx, t, 1, [((0.7, 0.0, 0.2523), (-1.0, 0.0, 0.0)),      # outside wall, from outside
                                           ((0.5, 0.0, 0.25), (1.0, 0.0, 0.0))])       # from inside: dry side
    assert cell.tolist() == [200 + 9 * 12 + 0, -1]
    assert dist[0] == pytest.approx(0.2 - R, abs=1e-5) and np.allclose(normal[0], [1.0, 0.0, 0.0], atol=1e-5)
    g0 = 440
    cell, dist, normal = _rays(ctx, t, 2, [((0.02, 0.0, 0.5), (0.0, 0.0, -1.0)),        # into the glass: its floor
                                           ((0.02, 0.0, -0.5), (0.0, 0.0, 1.0))])      # from below: its base
    assert dist[0] == pytest.approx(0.495, abs=1e-5) and np.allclose(normal[0], [0.0, 0.0, 1.0], atol=1e-5)
    assert dist[1] == pytest.approx(0.5, abs=1e-5) and np.allclose(normal[1], [0.0, 0.0, -1.0], atol=1e-5)
    ds = sum(math.dist(a, b) for a, b in zip(GLASS[:-1], GLASS[1:])) / 100           # 100 cells along the profile
    base = 0.023 + 0.095 + 0.005 + 0.1 + (0.04 - 0.02)                                 # s of r = 0.02 on the base
    assert cell.tolist() == [g0 + int(0.008 / ds) * 24, g0 + int(base / ds) * 24]
    cell, dist, normal = _rays(ctx, t, 2, [((0.004, 0.0, 0.5), (0.0, 0.0, -1.0)),       # the floor's uncharted centre
                                           ((0.004, 0.0, -0.5), (0.0, 0.0, 1.0))])     # and the base's
    assert cell.tolist() == [g0, g0 + 99 * 24] and np.allclose(dist, [0.495, 0.5], atol=1e-5)


def test_cell_window():
    ctx = _ctx(worlds=1)
    t = build_tables(ctx, [PlaneChart("plate", cells=(20, 10)),
                           RevolutionChart(profile=((R, H), (R, 0.0)), n_theta=12, ds=0.005, body="cup")])
    out = wp.zeros(1, dtype=wp.vec4i, device="cpu")
    wp.launch(_window, dim=1, inputs=[t.charts, t.prof, 0, 12, 6, 0.03], outputs=[out], device="cpu")
    assert out.numpy()[0].tolist() == [10, 14, 4, 8]
    wp.launch(_window, dim=1, inputs=[t.charts, t.prof, 1, 0, 0, 0.03], outputs=[out], device="cpu")
    i0, i1, j0, j1 = out.numpy()[0].tolist()
    assert (i0, i1) == (0, 6) and j0 == -2 and j1 == 2              # r·dθ = 21 mm: 2 cells either way, unwrapped


def _cup_corner(split: bool):
    """A cup's floor (r 20-50 mm) and wall (to 30 mm up) in 2 mm cells: one profile with a 90° corner, or two charts
    linked there."""
    if not split:
        return (RevolutionChart(((0.02, 0.0), (0.05, 0.0), (0.05, 0.03)), n_theta=8, ds=0.002),), ()
    return (RevolutionChart(((0.02, 0.0), (0.05, 0.0)), n_theta=8, ds=0.002),
            RevolutionChart(((0.05, 0.0), (0.05, 0.03)), n_theta=8, ds=0.002)), ((0, "end", 1, "start"),)


def test_a_profile_corner_takes_gravity_along_its_chord_as_a_link_does():
    """At a profile's corner the face's gravity is the drop between the cell centres over their distance (g/2 at
    90°; the unit mean tangent gave 1/cos(45°) of it), each side keeps its own tangent, and film draining down the
    wall round the corner onto the floor matches the same cup as two linked charts (WP9's links: chord/dist), to
    1e-6 of the depth."""
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415

    t = build_tables(_ctx(), _cup_corner(False)[0])
    corner = 15 * 8                                                    # i-face 15 (the corner node), column 0
    assert t.fdir.numpy()[corner][2] == pytest.approx(0.5, rel=1e-6, abs=0.0)
    assert np.allclose(t.fdira.numpy()[corner], [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0], rtol=0.0,
                       atol=1e-6)
    assert np.allclose(t.fdirb.numpy()[corner], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-6)
    out = []
    for split in (False, True):
        charts, links = _cup_corner(split)
        flow = SurfaceFlow(charts=charts, links=links, material="water", max_depth=3.0e-3)
        h0 = np.zeros((30, 8))
        h0[18:24] = 1.0e-3                                             # on the wall, 4-14 mm up

        def init(sim, flow=flow, h0=h0) -> None:
            flow.set_depth(h0.ravel())
        run = run_scene(TestScene(build=_scene, init=init, ground=False, gravity=(0.0, 0.0, -9.81), periods=10,
                                  worlds=1), [flow], "cpu")
        assert run.error is None, run.error
        out.append(flow.read()["h"][0])
    one, two = out
    assert one.reshape(30, 8)[:15].sum() > 0.0                         # round the corner onto the floor
    assert np.abs(one - two).max() <= 1.0e-6 * h0.max(), np.abs(one - two).max()
