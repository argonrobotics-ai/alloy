# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped bodies' drips (lumped_drip.py; WP15's second review of the lumped films): exact lowest points,
resting bodies in puddles, carried species, and drops that fit above the board. CPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_drip import LumpDrip, build_drip, lowest
from alloy.plugins.surface_flow.lumped_specs import Lumped

R, DT, G, RD = 0.014, 2.0e-3, (0.0, 0.0, -9.81), 1.5e-3
DROP = 4.0 / 3.0 * math.pi * RD ** 3


@wp.kernel
def _lowest(ld: LumpDrip, body_q: wp.array[wp.transform], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = lowest(ld, body_q, 0, i, wp.vec3(0.0, 0.0, -1.0))


def _quat(axis, ang):
    return wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), ang)


def _ring(r, z, n=4000):
    th = np.linspace(0.0, 2.0 * math.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z)], 1)


def _support_cases(b):
    """WP15's probe q1: (label, body, {shape: shape-frame surface samples})."""
    S = np.random.default_rng(0).normal(size=(40000, 3))
    S /= np.linalg.norm(S, axis=1)[:, None]
    qb, q1 = _quat((0.3, -1.0, 0.5), 1.1), _quat((1.0, 1.0, 0.0), 0.7)
    corners = lambda h: np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])  # noqa: E731
    out = []

    def body(x, q=qb):
        return b.add_body(xform=wp.transform(wp.vec3(x, 0.0, 1.0), q))
    k = body(0.0)
    s = b.add_shape_box(k, xform=wp.transform(wp.vec3(0.05, -0.02, 0.01), q1), hx=0.03, hy=0.02, hz=0.01)
    out.append(("box", k, {s: corners((0.03, 0.02, 0.01))}))
    k = body(1.0)
    s = b.add_shape_capsule(k, xform=wp.transform(wp.vec3(0.0, 0.04, 0.0), q1), radius=0.01, half_height=0.05)
    out.append(("capsule", k, {s: np.concatenate([0.01 * S + [0, 0, 0.05], 0.01 * S - [0, 0, 0.05]])}))
    k = body(2.0)
    s = b.add_shape_cylinder(k, xform=wp.transform(wp.vec3(0.02, 0.0, 0.03), q1), radius=0.02, half_height=0.04)
    out.append(("cylinder", k, {s: np.concatenate([_ring(0.02, -0.04), _ring(0.02, 0.04)])}))
    k = body(3.0)
    v = np.concatenate([corners((0.5, 0.25, 0.125)), [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]]]).astype(np.float32)
    f = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1,
                  7, 3], np.int32)
    sc = (0.1, 0.2, 0.3)
    s = b.add_shape_mesh(k, xform=wp.transform(wp.vec3(0.01, 0.02, -0.03), q1), mesh=newton.Mesh(v, f), scale=sc)
    out.append(("mesh", k, {s: v.astype(np.float64) * sc}))
    k = body(4.0)
    s1 = b.add_shape_sphere(k, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), radius=0.02)
    s2 = b.add_shape_box(k, xform=wp.transform(wp.vec3(0.06, 0.0, -0.02), q1), hx=0.01, hy=0.01, hz=0.03)
    out.append(("sphere + box", k, {s1: 0.02 * S, s2: corners((0.01, 0.01, 0.03))}))
    k = body(5.0)
    s = b.add_shape_ellipsoid(k, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q1), rx=0.05, ry=0.03, rz=0.015)
    out.append(("ellipsoid", k, {s: S * [0.05, 0.03, 0.015]}))
    k = body(6.0, wp.quat_identity())
    s = b.add_shape_cone(k, radius=0.03, half_height=0.01)
    out.append(("cone", k, {s: np.concatenate([_ring(0.03, -0.01), [[0.0, 0.0, 0.01]]])}))
    k = body(7.0, _quat((1.0, 0.0, 0.0), math.pi / 2.0))
    s = b.add_shape_cylinder(k, radius=0.02, half_height=0.04, barrel_radius=0.1)
    zz = np.linspace(-0.04, 0.04, 401)
    rr = 0.02 + np.sqrt(0.1 ** 2 - zz ** 2) - math.sqrt(0.1 ** 2 - 0.04 ** 2)
    out.append(("barrel", k, {s: np.concatenate([_ring(r, z, 400) for r, z in zip(rr, zz)])}))
    return out


def _compose(a, c):
    X = wp.transform_multiply(wp.transform(wp.vec3(*a[:3]), wp.quat(*a[3:])),
                              wp.transform(wp.vec3(*c[:3]), wp.quat(*c[3:])))
    return np.array([*wp.transform_get_translation(X), *wp.transform_get_rotation(X)])


def _world(xf, p):
    t, q = np.asarray(xf[:3], np.float64), xf[3:]
    Rm = np.array(wp.quat_to_matrix(wp.quat(*q)), np.float64).reshape(3, 3)
    return p @ Rm.T + t


def test_the_lowest_point_is_exact_for_each_shape_kind():
    """WP15's probe q1 as a law: a body's lowest point (lumped_drip.lowest) over rotated boxes, capsules, cylinders,
    a scaled mesh with a local transform, a two-shape body, ellipsoids, a cone and a barrel cylinder on its side
    matches the lowest of dense surface samples (or the mesh's vertices) to 0.1 mm, never above them (it fails for the
    ellipsoid, cone and barrel with a bounding sphere, and for the capsule with its centre)."""
    b = newton.ModelBuilder()
    cases = _support_cases(b)
    model = b.finalize(device="cpu")
    ctx = SimpleNamespace(model=model, array=lambda v, dtype: wp.array(np.asarray(v), dtype=dtype, device="cpu"))
    B = len(cases)
    z = lambda *s: wp.zeros(s, dtype=wp.float64, device="cpu")  # noqa: E731
    ld = build_drip(ctx, np.array([[c[1] for c in cases]], np.int32), [[list(c[2]) for c in cases]],
                    SimpleNamespace(drip=z(1, B), drip_sp=z(1, B, 1), sp_wet=z(1, B, 1), sp_drop=z(1, B, 1),
                                    tank=z(1, B), sp_tank=z(1, B, 1)), 0)
    state = model.state()
    out = wp.zeros(B, dtype=wp.vec3, device="cpu")
    wp.launch(_lowest, dim=B, inputs=[ld, state.body_q, out], device="cpu")
    bq, sxf = state.body_q.numpy(), model.shape_transform.numpy()
    for i, (label, k, shapes) in enumerate(cases):
        ref = min(_world(_compose(bq[k], sxf[s]), pts)[:, 2].min() for s, pts in shapes.items())
        got = float(out.numpy()[i][2])
        assert ref - 1.0e-4 <= got <= ref + 1.0e-6, (label, got, ref)


def _fixed(b, label: str, pos: tuple) -> int:
    xf = body_xform(*pos)
    body = b.add_link(xform=xf, label=label)
    b.add_articulation([b.add_joint_fixed(-1, body, parent_xform=xf)])
    return body


def _slab(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.2, hy=0.2, hz=0.01, label="slab")


def _dummy(b) -> None:
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(d, radius=0.01)


def _resting(b) -> None:
    """A tomato 0.5 mm over the slab in a 2.5 mm puddle (x < 0) and an ellipsoid 0.5 mm over the dry slab (x > 0)."""
    _slab(b)
    b.add_shape_sphere(_fixed(b, "tomato", (-0.1, 0.0, R + 5.0e-4)), radius=R)
    b.add_shape_ellipsoid(_fixed(b, "egg", (0.1, 0.0, 0.015 + 5.0e-4)), rx=0.05, ry=0.03, rz=0.015)
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_body_resting_in_a_puddle_or_on_a_dry_board_wets_it_with_its_soap():
    """WP15's must-fix 1 (probes q2, q4): a tomato resting in a puddle deeper than a drop (2.5 mm, over its 0.5 mm
    gap) and an ellipsoid resting on a dry board give their shed water to the cell under them, measured from the
    board, never as drops under it: none floored, no drops; the cell takes what it holds under max_depth and the rest
    waits. The soap goes with the water: the film's supplied soap = the bodies' soap given to the board (1e-12), and
    shed = wet + waiting (water and soap, 1e-12). An ellipsoid 10 mL over its hold fills its cell to max_depth and the
    rest waits (WP15: a sponge's whole excess in one cell was refused and dropped through the board)."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(20, 20)),), material="water", max_depth=5.0e-3,
                       species=("soap",), drops={"drop_radius": RD}, lumped=(Lumped("tomato", area=0.01),
                                                                             Lumped("egg", area=0.01)))
    q = 2.0e-8

    def init(sim) -> None:
        h = np.zeros((20, 20))
        h[:10] = 2.5e-3
        flow.set_depth(h.ravel())
        inp = flow.lumped.inputs
        inp.vol.fill_(q)
        sp = inp.sp.numpy().copy()
        sp[0, :, 0] = 4.0 * q
        inp.sp.assign(sp)
        flow.lumped.set_water(np.array([[0.0, 5.0e-7 + 1.0e-5]]))      # the egg: 10 mL over its hold
    sim = build_simulation(TestScene(build=_resting, init=init, ground=False, gravity=G, worlds=1, dt=DT,
                                     control_dt=DT), [flow], "cpu")
    for _ in range(2):                                                    # shed in the 1st, given in the 2nd
        sim.step_period()
    d = flow.drops.read()
    full = flow.h.numpy()[0] + flow.src_acc.numpy()[0]
    assert d["lump_wet_volume"][0] > 0.0 and d["lump_pending_volume"][0] > 0.5e-5      # most of the 10 mL waits
    assert full.max() <= 5.0e-3 * (1.0 + 1e-9)
    for _ in range(98):
        sim.step_period()
    r, d, f = flow.lumped.read(), flow.drops.read(), flow.read()
    shed, wet_sp = r["lump_shed"][0], r["lump_species_wet"][0, :, 0]
    assert (shed > 0.0).all() and d["floored_volume"][0] == 0.0 and d["lump_drops_volume"][0] == 0.0
    assert shed.sum() == pytest.approx(d["lump_wet_volume"][0] + d["lump_pending_volume"][0], rel=1e-12, abs=0.0)
    assert (wet_sp > 0.0).all()
    assert r["lump_species_shed"][0, :, 0].sum() == pytest.approx(
        wet_sp.sum() + r["lump_species_drip_pending"][0, :, 0].sum(), rel=1e-12, abs=0.0)
    assert f["species_supplied"][0].sum(axis=0)[0] + float((flow.sp.src.numpy()[0, :, 0]
                                                           * flow.tables.area_np).sum()) == pytest.approx(
        wet_sp.sum(), rel=1e-12, abs=0.0)


def _high(b) -> None:
    _slab(b)
    b.add_shape_sphere(_fixed(b, "tomato", (-0.1, 0.0, R + 0.02)), radius=R)
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_burst_of_drips_leaves_only_the_drops_that_fit_above_the_board():
    """WP15's must-fix 1 (probe q6): a tomato 20 mm over the slab sheds 30 drops' worth at once; only the drops that
    stay a radius above the film start each exchange (6: r + 2r·q down to 3.5 mm over it), the rest wait, and all
    30 land, none floored (23 of 30 started under the slab in one column before)."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(20, 20)),), material="water", max_depth=5.0e-3,
                       drops={"drop_radius": RD}, lumped=(Lumped("tomato", area=0.01),))
    cap = 5.0e-5 * 0.01
    sim = build_simulation(TestScene(build=_high, init=lambda sim: flow.lumped.set_water(cap + 30.2 * DROP),
                                     ground=False, gravity=G, worlds=1, dt=DT, control_dt=DT), [flow], "cpu")
    sim.step_period()
    sim.step_period()
    alive = flow.drops.pool.alive.numpy()[0] == 1
    z = flow.drops.pool.pos.numpy()[0][alive][:, 2]
    assert 0 < len(z) <= 12 and z.min() > RD
    for _ in range(80):
        sim.step_period()
    d = flow.drops.read()
    assert d["lump_drops_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0)
    assert d["landed_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0) and d["floored_volume"][0] == 0.0
