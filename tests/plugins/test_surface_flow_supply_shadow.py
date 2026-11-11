# SPDX-License-Identifier: Apache-2.0
"""Occlusion per footprint cell for sprays (supply_shadow.py, SOFT-3's ask A) on the stub film, CPU: a stencil sheet
over half a slab keeps a cone spray's overlapping footprints off the cells it hides, and the hidden share is booked
as caught by the stencil."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Nozzle
from alloy.plugins.surface_flow.supply_kernels import CHART as HIT_CHART
from alloy.plugins.surface_flow.supply_stub import StubFilm, faucet_network, ledger_sides, run_stub, slab_scene


def _stencil(b) -> None:
    slab_scene()(b)                                                      # slab top at z = 0.02
    b.add_shape_box(-1, xform=body_xform(0.1, 0.0, 0.0205), hx=0.1, hy=0.2, hz=0.0002, label="stencil")


def test_a_stencil_keeps_overlapping_footprints_off_the_cells_it_hides():
    """A 30° cone 0.28 m over a slab of 4 mm cells, a 0.4 mm stencil sheet 0.3 mm above its half x > 0: no paint on any
    cell under the stencil (a footprint spans 1.5 ray spacings, 24 mm, so without occlusion it would), paint up to its
    edge on the open half, and every ray's hidden share is caught by the stencil; the cross ledger closes."""
    spray = Nozzle("can", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="cone", rays=64,
                   angle=math.radians(30.0))
    p = run_stub(StubFilm(charts=(PlaneChart("slab", cells=(100, 100)),), nozzles=(spray,), network=faucet_network()),
                 periods=2, build=_stencil)
    r = p.supply.read()
    h = p.h.numpy()
    x = p.tables.cpos.numpy()[:, 0]
    assert not h[:, x > 0.0].any()                                         # under the stencil: dry
    assert (h[:, (x > -0.004) & (x < 0.0)] > 0.0).any()                   # the open side, right up to the edge
    lid = p.supply.occluders.index("stencil")
    assert (r["caught_by"][:, lid] > 0.0).all()
    assert np.allclose(r["caught_by"].sum(axis=1), r["caught"], rtol=1e-12, atol=0.0)
    lhs, rhs, spout, sent = ledger_sides(p)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0) and np.allclose(spout, sent, rtol=1e-12, atol=0.0)


CONE = Nozzle("can", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="cone", rays=64,
              angle=math.radians(30.0))


def _paint(build, chart_cells: int = 100, nozzle: Nozzle = CONE, ground: bool = True) -> StubFilm:
    scene = TestScene(build=build, ground=ground, worlds=2, periods=2)
    p = StubFilm(charts=(PlaneChart("slab", cells=(chart_cells, chart_cells)),), nozzles=(nozzle,),
                 network=faucet_network())
    run = run_scene(scene, [p], "cpu")
    assert run.error is None, run.error
    return run.sim.plugins[p.label]


def test_a_spray_without_occluders_closes_its_ledger():
    """WP15's probe: a 16-ray cone over a 40 x 40 chart in a scene with no colliding shape and no ground (no
    occluder, so no shadow pass): the gather reads the shade of every cell it paints (it was sized for one) and the
    nozzle's ledger closes."""
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

    def bare(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.01), hx=0.2, hy=0.2, hz=0.01, label="slab", cfg=cfg)
        dummy = b.add_body(xform=body_xform(-2.0, 0.0, 0.5), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01, cfg=cfg)
    cone = Nozzle("can", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="cone", rays=16,
                  angle=math.radians(30.0))
    p = _paint(bare, 40, cone, ground=False)
    assert not p.supply.shadows and (p.supply.occ.numpy() < 0).all()
    lhs, rhs, spout, sent = ledger_sides(p)
    assert (p.supplied.numpy().sum(axis=1) > 0.0).all()
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0) and np.allclose(spout, sent, rtol=1e-12, atol=0.0)


def test_the_open_side_is_painted_as_without_the_stencil():
    """WP15's ramp: a ray whose centre the stencil hides still lands the share of its footprint its nozzle sees, so
    every cell on the open half gets what it gets with no stencil at all (it was 0.51 at the edge, 1.0 only 22 mm
    out), and nothing lands under the stencil."""
    with_ = _paint(_stencil)
    without = _paint(slab_scene())
    x = with_.tables.cpos.numpy()[:, 0]
    h1, h0 = with_.h.numpy(), without.h.numpy()
    assert (h0[:, (x > -0.004) & (x < 0.0)] > 0.0).any(axis=1).all()               # the edge column is painted
    assert np.allclose(h1[:, x < 0.0], h0[:, x < 0.0], rtol=1e-12, atol=0.0)
    assert not h1[:, x > 0.0].any()


def test_stacked_stencils_the_nearest_catches_and_carries_the_soap():
    """Two stencils over the same half, the lower first in the occluder table: the one nearer the nozzle catches
    every hidden share (the lower, behind it, nothing), and on WP15's film a soapy spray's soap goes with its water:
    caught is C·caught and the film holds C times what landed."""
    def stacked(b) -> None:
        _stencil(b)
        b.add_shape_box(-1, xform=body_xform(0.1, 0.0, 0.06), hx=0.1, hy=0.2, hz=0.0002, label="upper")
    p = _paint(stacked)
    r = p.supply.read()
    lower, upper = p.supply.occluders.index("stencil"), p.supply.occluders.index("upper")
    assert lower < upper and (r["caught"] > 0.0).all()
    assert np.allclose(r["caught_by"][:, upper], r["caught"], rtol=1e-12, atol=0.0)
    assert (r["caught_by"][:, lower] == 0.0).all()
    c = 5.0
    spray = Nozzle("can", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=2e-5, pattern="cone", rays=64,
                   angle=math.radians(30.0), species={"soap": c})
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50)),), material="water", species=("soap",),
                       nozzles=(spray,), network=faucet_network(), max_depth=1.0e-2)

    def plate(b) -> None:
        sl.plate_scene(0.2, 0.2)(b)
        b.add_shape_box(-1, xform=body_xform(0.05, 0.0, 0.0006), hx=0.05, hy=0.1, hz=0.0002, label="stencil")
    sim = build_simulation(TestScene(build=plate, ground=False, worlds=2), [flow], "cpu")
    for _ in range(4):
        sim.step_period()
    r = flow.read()
    landed = r["supply_sent"] - r["supply_caught"] - r["supply_overspray"] - r["supply_mist"] - r["supply_lost"]
    assert (r["supply_caught"] > 0.0).all() and (landed > 0.0).all()
    assert np.allclose(r["supply_species_caught"][:, 0], c * r["supply_caught"], rtol=1e-12, atol=0.0)
    assert np.allclose(r["supply_species_sent"][:, 0], c * landed, rtol=1e-9, atol=0.0)
    assert np.allclose(r["species_mass"][:, 0], r["supply_species_sent"][:, 0], rtol=1e-9, atol=0.0)


def test_shapes_off_the_sprays_path_cost_nothing():
    """WP15's cost: occluders whose bounds miss the box of the nozzle's exit and the chart are culled (the ground
    under it, eight boxes off to the side), as is the chart's own slab: the paint is bit for bit what it is without
    the extra boxes, and only the stencil is kept."""
    def cluttered(b) -> None:
        _stencil(b)
        for i in range(8):
            b.add_shape_box(-1, xform=body_xform(1.0 + 0.2 * i, 0.5, 0.1), hx=0.05, hy=0.05, hz=0.05,
                            label=f"clutter{i}")
    p, q = _paint(cluttered), _paint(_stencil)
    assert np.array_equal(p.h.numpy(), q.h.numpy())
    keep = p.supply.keep.numpy()[:, 0]                                       # [W, slots] for the one chart
    names = p.supply.occluders
    kept = {names[i] for i in np.flatnonzero(keep[0][:len(names)])}
    assert kept == {"stencil"}, kept


def test_shade_bits_number_the_sprays_only():
    """Jets take no shade bit; spray nozzles are numbered in order, the first 32 only."""
    from types import SimpleNamespace  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply_shadow import shade_bits  # noqa: PLC0415
    nozzles = [SimpleNamespace(pattern="jet")] * 3 + [SimpleNamespace(pattern="cone")] * 33
    bits = shade_bits(nozzles)
    assert bits[:3].tolist() == [-1] * 3 and bits[3:35].tolist() == list(range(32)) and bits[35] == -1


def _scrap(b) -> None:
    """The slab on a fixed link (its chart has a body) and a free 30 x 30 x 12 mm scrap lying on it."""
    link = b.add_link(xform=body_xform(0.0, 0.0, 0.01), label="slab")
    b.add_shape_box(link, hx=0.2, hy=0.2, hz=0.01, label="slab")
    b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(0.0, 0.0, 0.01))])
    scrap = b.add_body(xform=body_xform(0.05, 0.0, 0.026), label="scrap")
    b.add_shape_box(scrap, hx=0.015, hy=0.015, hz=0.006, label="scrap")


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``v`` rotated by the quaternion ``q`` (x, y, z, w)."""
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _xf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Transform a·b (x, y, z, qx, qy, qz, qw)."""
    (x1, y1, z1, w1), (x2, y2, z2, w2) = a[3:], b[3:]
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    return np.concatenate([a[:3] + _rot(a[3:], b[:3]), q])


def test_a_hidden_share_pushes_its_occluder_and_the_momentum_is_kept():
    """SCR-SC9-2: a 6° cone leaning at a scrap on the slab (SC9's flush-floor hose). The share of each ray the scrap
    hides is caught by it, and its momentum ρ·q_shadow·w goes onto the scrap at the point it hides it (it stayed on
    the slab's body: the scrap never moved), pushing it away from the hose; slab + scrap take what the rays carry,
    force and moment (WP15's review: the moment catches a push at the chart's hit point)."""
    hose = Nozzle("hose", edge="spout", pos=(-0.25, 0.0, 0.25), axis=(0.3, 0.0, -0.23), exit_area=1e-5,
                  pattern="cone", rays=9, angle=math.radians(6.0))
    scene = TestScene(build=_scrap, ground=False, worlds=2, periods=2)
    p = StubFilm(charts=(PlaneChart("slab", cells=(50, 50)),), nozzles=(hose,), network=faucet_network())
    run = run_scene(scene, [p], "cpu")
    assert run.error is None, run.error
    p, sim = run.sim.plugins[p.label], run.sim
    st = sim.model.state()
    forces = SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))
    p.supply.apply(st, forces, 0.02)
    s, hw = p.supply.spray.numpy()[0], p.supply.hit["w"].numpy()[0].astype(np.float64)
    assert (p.supply.hit["kind"].numpy()[0] == HIT_CHART).all()                  # every ray passes to the slab
    rho = float(s["rho"][0])
    shadow = rho * (s["q_shadow"][:, None] * hw).sum(axis=0)                    # what the hidden shares carry
    carried = rho * ((s["q"] + s["q_shadow"])[:, None] * hw).sum(axis=0)        # what the rays carry
    f = forces.body_f.numpy().astype(np.float64)
    scrap, slab = (int(p.ctx.bodies(n).numpy()[0]) for n in ("scrap", "slab"))
    assert s["q_shadow"].sum() > 0.1 * (s["q"] + s["q_shadow"]).sum() and shadow[0] > 0.0   # pushed away from the hose
    assert np.allclose(f[scrap, :3], shadow, rtol=1e-5, atol=0.0)
    assert np.allclose(f[scrap, :3] + f[slab, :3], carried, rtol=1e-5, atol=0.0)
    bp = s["bp"][s["q_shadow"] > 0.0].astype(np.float64)       # within the scrap: where the hidden cells' rays meet it
    lo, hi = np.array([0.035, -0.015, 0.020]) - 1e-5, np.array([0.065, 0.015, 0.032]) + 1e-5
    assert ((bp >= lo) & (bp <= hi)).all(), bp
    assert (p.supply.read()["caught_by"][:, p.supply.occluders.index("scrap")] > 0.0).all()
    bq, com_l = st.body_q.numpy().astype(np.float64), sim.model.body_com.numpy().astype(np.float64)
    com = {b: bq[b, :3] + _rot(bq[b, 3:], com_l[b]) for b in (scrap, slab)}
    push = rho * s["q_shadow"][:, None] * hw
    assert np.allclose(f[scrap, 3:], np.cross(s["bp"] - com[scrap], push).sum(axis=0), rtol=1e-5, atol=1e-12)
    X = _xf(bq[slab], p.supply.charts.clocal.numpy()[0, 0].astype(np.float64))
    pw = np.array([X[:3] + _rot(X[3:], hp) for hp in p.supply.hit["p"].numpy()[0].astype(np.float64)])
    carried_m = (np.cross(pw, rho * s["q"][:, None] * hw) + np.cross(s["bp"], push)).sum(axis=0)
    moment = sum(f[b, 3:] + np.cross(com[b], f[b, :3]) for b in (scrap, slab))
    assert np.allclose(moment, carried_m, rtol=1e-5, atol=1e-9)
