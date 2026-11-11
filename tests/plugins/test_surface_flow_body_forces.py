# SPDX-License-Identifier: Apache-2.0
"""Film forces on bodies (design §4.5, §4.7; WP9): the film's weight on a carried plate (D-7) and the Couette drag of
a film on a body sliding over it (D-8). CPU, the hooks called by hand."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.host import PluginHost
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow

G, DT, TOP = 9.81, 1.0e-3, 0.01


def _model(debris: bool, on_body: bool):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    b.begin_world()
    if on_body:
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), label="carrier")
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.01, label="plate")
    else:
        b.add_shape_box(-1, hx=0.05, hy=0.05, hz=0.01, label="plate")
    if debris:
        grit = b.add_body(xform=wp.transform(wp.vec3(0.004, 0.0, TOP + 1.0e-3 + 0.005), wp.quat_identity()),
                          label="grit", mass=10.0)
        b.add_shape_box(grit, hx=0.01, hy=0.01, hz=0.005)
    b.end_world()
    return b.finalize(device="cpu")


def _setup(depth, debris=False, on_body=True):
    m = _model(debris, on_body)
    drops = {"debris": ({"body": "grit", "radius": 0.01},) if debris else ()}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20)),), material="water", max_depth=3.0e-3,
                       drops=drops)
    PluginHost(m, [flow], control_dt=0.01)
    flow.check_dt(DT)
    flow.set_depth(depth, chart=0)
    s = m.state()
    flow.update(s, s, DT)                     # one film step: the active tiles
    return m, flow, s


def _apply(m, flow, s):
    f = SimpleNamespace(body_f=wp.zeros(m.body_count, dtype=wp.spatial_vector, device="cpu"),
                        particle_f=wp.zeros(max(m.particle_count, 1), dtype=wp.vec3, device="cpu"))
    flow.drops.apply(s, f, DT)
    return f.body_f.numpy()


def test_the_film_weight_acts_at_its_centroid():
    """D-7: the carried plate takes ρ g V down, with the torque of the film's centroid about its COM."""
    x = (np.arange(20) - 9.5) * 0.005
    depth = (1.0e-3 + 1.0e-2 * x)[:, None] * np.ones((1, 20))          # thicker towards x+
    m, flow, s = _setup(depth)
    h = flow.depth(0)[0]
    rho, area = flow.mat.density, 0.005 * 0.005
    mass = rho * (h * area).sum()
    f = _apply(m, flow, s)[0]
    assert np.allclose(f[:3], [0.0, 0.0, -G * mass], rtol=1e-6, atol=0.0)
    cx = (rho * h * area * x[:, None]).sum() / mass                       # the centroid's x (the COM is at 0)
    assert np.allclose(f[3:], [0.0, G * mass * cx, 0.0], rtol=1e-5, atol=1e-12)


def test_an_accelerating_carrier_feels_the_films_inertia():
    """WP15 review: the film pushes its carrier with M·(g - a). A plate accelerating up at 2 m/s² carries M·(g + 2)
    down. The impulse store is plug-in state (reset and restore carry it)."""
    a = 2.0
    m, flow, s = _setup(1.0e-3)
    up = m.state()
    qd = up.body_qd.numpy()
    qd[0] = [0.0, 0.0, a * DT, 0.0, 0.0, 0.0]
    up.body_qd.assign(qd)
    flow.update(s, up, DT)                                                # this exchange: the frame accelerates up
    mass = flow.mat.density * (flow.depth(0)[0] * 0.005 * 0.005).sum()
    f = _apply(m, flow, up)[0]
    assert f[2] == pytest.approx(-(G + a) * mass, rel=1e-5, abs=0.0)
    assert {"drops_impulse", "drops_imp_moment"} <= set(flow.state_arrays())


def test_a_static_chart_puts_no_weight_on_any_body():
    m, flow, s = _setup(1.0e-3, on_body=False)
    assert np.allclose(_apply(m, flow, s), 0.0, atol=0.0)


def test_couette_drag_on_a_body_sliding_over_the_film():
    """D-8: a body sliding at U over a film of depth h at rest feels μ U A / h (the tools' film_shear), over the cells
    of its footprint."""
    h0, U = 1.0e-3, 0.2
    m, flow, s = _setup(h0, debris=True, on_body=False)
    qd = s.body_qd.numpy()
    qd[0] = [U, 0.0, 0.0, 0.0, 0.0, 0.0]
    s.body_qd.assign(qd)
    f = _apply(m, flow, s)[0]
    c = flow.cell_centres(0)                                              # [nx, ny, 3] chart frame
    foot = np.hypot(c[..., 0] - 0.004, c[..., 1]) <= 0.01
    expect = -flow.mat.viscosity * U * foot.sum() * 0.005 * 0.005 / h0
    assert foot.sum() > 5 and f[0] == pytest.approx(expect, rel=1e-5, abs=0.0)
    assert abs(f[1]) < 1e-9 and abs(f[2]) < 1e-9
