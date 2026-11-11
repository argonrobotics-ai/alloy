# SPDX-License-Identifier: Apache-2.0
"""Bodies in pools (supply_float.py: WP15's ask for WP8's pan, can and tomatoes) on the stub film, CPU: a puck and a
tomato float at their Archimedes drafts, and a tipped glass carries its water's weight at the water's centroid."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Pool
from alloy.plugins.surface_flow.supply_float import Float
from alloy.plugins.surface_flow.supply_stub import StubFilm

G, RHO = 9.81, 998.0
FAR = PlaneChart("far", cells=(2, 2))
LEVEL = 0.1                                             # the sink's surface: 0.36 m² at 0.1 m


def _far(b) -> None:
    b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.01), hx=0.05, hy=0.05, hz=0.01, label="far")


def _sink(*floats: Float) -> dict:
    return {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.36 * LEVEL, "area": 0.36}], "edges": [],
            "pools": [Pool("sink", floor=(0.0, 0.0, 0.0), half=(0.3, 0.3))], "floats": list(floats)}


def _float_z(shape: str, z0: float, periods: int) -> tuple[np.ndarray, float]:
    """A puck (cylinder R 50 mm, H 40 mm, 500 kg/m³) or a tomato (sphere 14 mm, 980 kg/m³) in the sink: its height
    every period, and its mass."""
    def build(b) -> None:
        _far(b)
        body = b.add_body(xform=body_xform(0.0, 0.0, z0), label="thing")
        if shape == "puck":
            b.add_shape_cylinder(body, radius=0.05, half_height=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
        else:
            b.add_shape_sphere(body, radius=0.014, cfg=newton.ModelBuilder.ShapeConfig(density=980.0))
    fl = Float("thing", hull=((0.05, 0.0), (0.05, 0.04)), origin=(0.0, 0.0, -0.02)) if shape == "puck" else \
        Float("thing", sphere=0.014)
    p = StubFilm(charts=(FAR,), network=_sink(fl))
    sim = build_simulation(TestScene(build=build, ground=False, worlds=1), [p], "cpu")
    b = int(p.supply.fbody.numpy()[0, 0])
    zs = []
    for _ in range(periods):
        sim.step_period()
        zs.append(float(sim.state.body_q.numpy()[b][2]))
    return np.asarray(zs), float(sim.model.body_mass.numpy()[b])


def test_a_puck_floats_at_its_archimedes_draft():
    """Released at its draft m/(ρ·π·R²) (20 mm: half its height), it stays there; released 5 mm high, it bobs about it
    and drag settles it there within 1 mm in 3 s."""
    draft = 0.02
    zs, m = _float_z("puck", LEVEL + 0.02 - draft, 25)
    assert m / (RHO * math.pi * 0.05 ** 2) == pytest.approx(draft, rel=0.01)
    assert np.abs(zs - (LEVEL + 0.02 - m / (RHO * math.pi * 0.05 ** 2))).max() < 2e-4
    zs, m = _float_z("puck", LEVEL + 0.005, 150)
    eq = LEVEL + 0.02 - m / (RHO * math.pi * 0.05 ** 2)
    assert zs.min() < eq < zs.max() and abs(zs[-25:].mean() - eq) < 1e-3, (zs[-25:].mean(), eq)


def test_a_tomato_floats_98_percent_under():
    """A 980 kg/m³ sphere released with its top at the surface rises to 98 % under: the cap below the surface holds
    0.98 of it (its centre 0.83 R... above where it started), within 0.3 mm after 3 s."""
    r = 0.014
    zs, m = _float_z("tomato", LEVEL - r, 150)
    v_sub = m / RHO
    hc = np.linspace(0.0, 2.0 * r, 200001)
    cap = math.pi * hc ** 2 * (3.0 * r - hc) / 3.0
    h = hc[np.searchsorted(cap, v_sub)]
    eq = LEVEL - h + r                                                        # the centre r above the cap's bottom
    assert abs(zs[-25:].mean() - eq) < 3e-4, (zs[-25:].mean(), eq)


def test_a_tipped_glass_carries_its_waters_weight_at_its_centroid():
    """A glass (R 30 mm, 100 mm deep) holding 60 mm of water, tipped 30° on a free body: the supply's wrench on it is
    the water's weight ρ·V·g at the water's centroid (a 4000-slice quadrature of the tilted cylinder below its
    surface), to 1e-6 in force and 2e-3 in torque."""
    R, HT, th = 0.03, 0.1, math.radians(30.0)
    A, v = math.pi * R * R, math.pi * R * R * 0.06
    q = (math.sin(th / 2.0), 0.0, 0.0, math.cos(th / 2.0))

    def build(b) -> None:
        _far(b)
        glass = b.add_body(xform=body_xform(0.5, 0.0, 0.3, q), label="glass")
        b.add_shape_sphere(glass, xform=body_xform(0.0, 0.0, 0.05), radius=0.005)
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": v, "area": A}], "edges": [],
           "pools": [Pool("glass", body="glass", cavity=((R, 0.0), (R, HT)))]}
    p = StubFilm(charts=(FAR,), network=net)
    sim = build_simulation(TestScene(build=build, ground=False, worlds=1, gravity=(0.0, 0.0, 0.0)), [p], "cpu")
    st = sim.model.state()
    p.supply.gravity = np.array([0.0, 0.0, -G])                              # the supply's weight only
    forces = SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))
    p.supply.apply(st, forces, 0.002)
    b = int(p.supply.pbody.numpy()[0, 0])
    f = forces.body_f.numpy()[b]
    assert np.allclose(f[:3], [0.0, 0.0, -RHO * v * G], rtol=1e-6, atol=1e-9)
    X = st.body_q.numpy()[b]                                                 # (p, q xyzw)
    rot = wp.quat(*X[3:7])
    up = np.array(wp.quat_rotate_inv(rot, wp.vec3(0.0, 0.0, 1.0)))         # world up in the glass's frame
    s, uz = math.hypot(up[0], up[1]), up[2]
    sv = p.supply.cav_surf.numpy()[0, 0]
    h = float(sv[3])
    zg = (np.arange(4000) + 0.5) * HT / 4000
    t = np.clip((h - uz * zg) / s, -R, R)
    seg = R * R * np.arccos(t / R) - t * np.sqrt(R * R - t * t)
    area = math.pi * R * R - seg                                             # the disc on the side x·e <= t
    mx = -2.0 / 3.0 * (R * R - t * t) ** 1.5                                 # ∫ x dA along e
    dz = HT / 4000
    vol = area.sum() * dz
    c_local = np.array([mx.sum() * dz * up[0] / s, mx.sum() * dz * up[1] / s, (area * zg).sum() * dz]) / vol
    assert vol == pytest.approx(v, rel=1e-3)
    com = np.array(wp.transform_point(wp.transform(wp.vec3(*X[:3]), rot), wp.vec3(*sim.model.body_com.numpy()[b])))
    cw = np.array(wp.transform_point(wp.transform(wp.vec3(*X[:3]), rot), wp.vec3(*c_local)))
    tau = np.cross(cw - com, np.array([0.0, 0.0, -RHO * v * G]))
    assert np.allclose(f[3:], tau, rtol=2e-3, atol=2e-3 * np.abs(tau).max()), (f[3:], tau)


@pytest.mark.parametrize(("slices", "rel"), [(48, 1e-2), (192, 1e-3)])
def test_a_tipped_puck_rights_itself_by_its_metacentric_moment(slices, rel):
    """The puck (R 50 mm, H 40 mm) at its 20 mm draft, tipped 5° about x about the point where its axis meets the
    water (its centre of mass): it displaces πR²·d still, and the buoyancy's moment about its centre of mass is the
    wall-sided GZ = sin θ·(GM + ½·BM·tan²θ), with BM = R²/4d and GM = d/2 + BM - H/2, righting it. The slices'
    mid-chord centroids converge as 1/n² (48 slices: 0.6 % of this small moment, 192: 3e-4)."""
    R, H, d, th = 0.05, 0.04, 0.02, math.radians(5.0)
    q = (math.sin(th / 2.0), 0.0, 0.0, math.cos(th / 2.0))

    def build(b) -> None:
        _far(b)
        puck = b.add_body(xform=body_xform(0.0, 0.0, LEVEL, q), label="thing")
        b.add_shape_cylinder(puck, radius=R, half_height=0.5 * H)
    p = StubFilm(charts=(FAR,), network=_sink(Float("thing", hull=((R, 0.0), (R, H)), origin=(0.0, 0.0, -0.5 * H),
                                                         slices=slices)))
    sim = build_simulation(TestScene(build=build, ground=False, worlds=1, gravity=(0.0, 0.0, 0.0)), [p], "cpu")
    p.supply.gravity = np.array([0.0, 0.0, -G])                              # the supply's buoyancy only
    forces = SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))
    p.supply.apply(sim.model.state(), forces, 0.002)
    f = forces.body_f.numpy()[int(p.supply.fbody.numpy()[0, 0])]
    v = math.pi * R * R * d
    bm = R * R / (4.0 * d)
    gz = math.sin(th) * (d / 2.0 + bm - H / 2.0 + 0.5 * bm * math.tan(th) ** 2)
    assert np.allclose(f[:3], [0.0, 0.0, RHO * v * G], rtol=1e-6, atol=1e-6 * RHO * v * G)
    assert f[3] == pytest.approx(-RHO * v * G * gz, rel=rel)                  # back toward upright
    assert abs(f[4]) < 1e-6 and abs(f[5]) < 1e-6


def _wrench(build, fl: Float, vel: tuple[float, ...] | None = None) -> tuple[np.ndarray, StubFilm]:
    """The supply's wrench on the float's body in the sink at the build pose (gravity only in the supply), with the
    body's velocity ``vel`` (linear, angular) if given."""
    p = StubFilm(charts=(FAR,), network=_sink(fl))
    sim = build_simulation(TestScene(build=build, ground=False, worlds=1, gravity=(0.0, 0.0, 0.0)), [p], "cpu")
    p.supply.gravity = np.array([0.0, 0.0, -G])
    st = sim.model.state()
    b = int(p.supply.fbody.numpy()[0, 0])
    if vel is not None:
        qd = st.body_qd.numpy()
        qd[b] = vel
        st.body_qd.assign(qd)
    forces = SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))
    p.supply.apply(st, forces, 0.002)
    return forces.body_f.numpy()[b].astype(np.float64), p


def test_a_submerged_sphere_feels_buoyancy_and_its_drag():
    """A 14 mm sphere 50 mm under, moving at 0.2 m/s along x: ρ·g·V up, and the drag -(c_lin + ½·ρ·cd·a_ref·|v|)·v
    with the defaults (Stokes c_lin = 6π·ρ·ν·r, a_ref = π·r², cd 1), at its centre (no torque)."""
    r, u = 0.014, 0.2

    def build(b) -> None:
        _far(b)
        t = b.add_body(xform=body_xform(0.0, 0.0, LEVEL - 0.05), label="thing")
        b.add_shape_sphere(t, radius=r)
    f, p = _wrench(build, Float("thing", sphere=r), (u, 0.0, 0.0, 0.0, 0.0, 0.0))
    nu = p.supply.pools[0].viscosity
    drag = (6.0 * math.pi * RHO * nu * r + 0.5 * RHO * 1.0 * math.pi * r * r * u) * u
    assert f[0] == pytest.approx(-drag, rel=1e-5) and f[2] == pytest.approx(RHO * G * 4.0 / 3.0 * math.pi * r ** 3,
                                                                             rel=1e-6)
    assert abs(f[1]) < 1e-9 and np.abs(f[3:]).max() < 1e-9


def test_a_flared_hull_displaces_its_frustum():
    """A pan-like hull flaring from 30 mm at its base to 60 mm at 50 mm, upright with 20 mm under: ρ·g times the
    frustum below the waterline, π·d/3·(r0² + r0·r_d + r_d²): each slice is a cylinder at its mid radius, the one
    the waterline cuts too (second order in the slice: 1.4e-4 here)."""
    r0, r1, H, d = 0.03, 0.06, 0.05, 0.02

    def build(b) -> None:
        _far(b)
        pan = b.add_body(xform=body_xform(0.0, 0.0, LEVEL - d), label="thing")
        b.add_shape_sphere(pan, xform=body_xform(0.0, 0.0, 0.01), radius=0.005)
    f, _ = _wrench(build, Float("thing", hull=((r0, 0.0), (r1, H))))
    rd = r0 + (r1 - r0) * d / H
    v = math.pi * d / 3.0 * (r0 * r0 + r0 * rd + rd * rd)
    assert f[2] == pytest.approx(RHO * G * v, rel=5e-4) and np.abs(f[:2]).max() < 1e-9
