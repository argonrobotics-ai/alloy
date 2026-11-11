# SPDX-License-Identifier: Apache-2.0
"""Surface-flow chart links (design note §2.1; WP9): film crosses between charts joined at an edge. CPU, harness."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow.film import FilmMaterial
from alloy.plugins.surface_flow.kernels_structs import Faces
from alloy.plugins.surface_flow.links import cell_dir

G = 9.81
TILT = math.radians(5.0)
DOWNHILL = (G * math.sin(TILT), 0.0, -G * math.cos(TILT))          # a 5° slope falling along +x
TURN_Y = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))     # +90° about y: local x -> -z, z -> +x
FLIP_Y = (0.0, 1.0, 0.0, 0.0)                                         # 180° about y: local x -> -x, z -> -z


def _dummy(b) -> None:
    body = b.add_body(xform=body_xform(0.0, 0.0, 1.0), label="dummy")      # free; nothing touches it
    b.add_shape_sphere(body, radius=0.01)


@wp.kernel
def _sides(faces: Faces, link: wp.array[wp.int32], out: wp.array2d[wp.vec3]):
    """``cell_dir`` of each link face as its a cell and its b cell read it (their cell sums: velocity, shear)."""
    i = wp.tid()
    f = link[i]
    out[i, 0] = cell_dir(faces, f, faces.a[f])
    out[i, 1] = cell_dir(faces, f, faces.b[f])


def _run(flow: SurfaceFlow, build, init, periods: int, gravity=(0.0, 0.0, -G)):
    scene = TestScene(build=build, init=init, ground=False, gravity=gravity, periods=periods, worlds=1)
    run = run_scene(scene, [flow], "cpu")
    return run


def _read(flow: SurfaceFlow, build, init, periods: int, gravity=(0.0, 0.0, -G)) -> dict:
    run = _run(flow, build, init, periods, gravity)
    assert run.error is None, run.error
    return flow.read()


def _boxes(split: bool):
    """A 0.4 x 0.04 m slab (top at z = 0), whole or as two 0.2 m boxes meeting at x = 0."""
    def build(b) -> None:
        if split:
            for name, x in (("left", -0.1), ("right", 0.1)):
                b.add_shape_box(-1, xform=body_xform(x, 0.0, -0.01), hx=0.1, hy=0.02, hz=0.01, label=name)
        else:
            b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.2, hy=0.02, hz=0.01, label="slab")
        _dummy(b)
    return build


def _slab_flow(split: bool) -> SurfaceFlow:
    common = {"material": "water", "max_depth": 4.0e-3, "species": ("soil",)}
    if not split:
        return SurfaceFlow(charts=(PlaneChart("slab", cells=(40, 4)),), **common)
    return SurfaceFlow(charts=(PlaneChart("left", cells=(20, 4)), PlaneChart("right", cells=(20, 4))),
                       links=(("left", "i+", "right", "i-"),), **common)


@pytest.mark.compile_heavy
def test_two_linked_boxes_match_one_long_box():
    """A dam break down a 5° slope, its ridge sloped along the seam (so the transverse stencil counts), runs across a
    link between two boxes as on one box twice as long (the cells are numbered alike): depth and species bit for bit
    (the link face's geometry is the long box's interior face). Volume and species ledgers close to float64
    rounding."""
    h0 = np.zeros((40, 4))
    h0[:12] = 2.0e-3 * (1.0 + 0.5 * np.arange(4) / 3.0)                # x < -0.28 m
    out = {}
    for split in (False, True):
        flow = _slab_flow(split)

        def init(sim, flow=flow):
            flow.set_depth(h0.ravel())
            flow.set_species(3.0 * h0.ravel(), "soil")
        out[split] = (_read(flow, _boxes(split), init, 25, DOWNHILL), flow)
    (one, f1), (two, f2) = out[False], out[True]
    assert two["h"][0][80:].sum() > 0.05 * h0.sum()                       # the film crossed the seam
    assert np.array_equal(two["h"][0], one["h"][0]) and np.array_equal(two["species"][0], one["species"][0])
    vol0 = (h0.ravel() * f2.tables.area_np).sum()
    assert two["ledger"][0] == pytest.approx(vol0, rel=1e-12, abs=0.0)
    assert two["species_ledger"][0][0] == pytest.approx(3.0 * vol0, rel=1e-12, abs=0.0)
    assert two["spilled_volume"][0] == 0.0


@pytest.mark.compile_heavy
def test_a_flat_film_across_a_link_stays_at_rest():
    """A level 1 mm film over both linked boxes: no face moves and the depth stays exactly 1 mm."""
    flow = _slab_flow(True)
    r = _read(flow, _boxes(True), lambda sim: flow.set_depth(1.0e-3), 10)
    assert np.abs(r["u"]).max() == 0.0 and np.all(r["h"] == 1.0e-3)


def _glass(split: bool, top: float):
    """A static wall (r 30 mm at the floor, ``top`` at 60 mm: a glass or a cone; chart frame = world) in 20 rows: one
    band, or two bands linked at z = 30 mm."""
    mid, ds = 0.5 * (0.03 + top), math.hypot(top - 0.03, 0.06) / 20.0
    if not split:
        return (RevolutionChart(((0.03, 0.0), (top, 0.06)), n_theta=24, ds=ds),), ()
    lower = RevolutionChart(((0.03, 0.0), (mid, 0.03)), n_theta=24, ds=ds)
    upper = RevolutionChart(((mid, 0.03), (top, 0.06)), n_theta=24, ds=ds)
    return (lower, upper), ((0, "end", 1, "start"),)


@pytest.mark.compile_heavy
@pytest.mark.parametrize("top", [0.03, 0.05], ids=["glass", "cone"])
def test_two_linked_wall_bands_match_one_wall(top):
    """A ring of film draining down the inside of a glass wall (or a cone's) from its upper band through a link into
    its lower band (revolution charts: θ wraps, start/end edges) matches one band, symmetric in θ; volume exact. A
    glass bit for bit; a cone's straight join (its two tangents equal but for rounding) adds no fold, so its curvature
    is equal and its film within 1e-15 m (WP15: arccos turned the rounding into 1.5e-8 rad of fold: 3.8e-6 /m, 3e-10 m)."""
    out = {}
    for split in (False, True):
        charts, links = _glass(split, top)
        flow = SurfaceFlow(charts=charts, links=links, material="water", max_depth=3.0e-3)
        h0 = np.zeros((20, 24))
        h0[14:18] = 1.5e-3                                                # z 42-54 mm

        def init(sim, flow=flow, h0=h0):
            flow.set_depth(h0.ravel())
        out[split] = (_read(flow, lambda b: _dummy(b), init, 15), flow, h0)
    (one, f1, h0), (two, f2, _) = out[False], out[True]
    h = two["h"][0].reshape(20, 24)
    assert h[:10].sum() > 0.05 * h0.sum()                                 # into the lower band
    assert np.array_equal(f2.tables.curv.numpy(), f1.tables.curv.numpy())
    if top == 0.03:
        assert np.array_equal(two["h"][0], one["h"][0])
    else:
        assert np.abs(two["h"][0] - one["h"][0]).max() <= 1.0e-15
    assert np.array_equal(h, np.repeat(h[:, :1], 24, axis=1))
    assert two["ledger"][0] == pytest.approx((h0.ravel() * f2.tables.area_np).sum(), rel=1e-12, abs=0.0)


def _block(b) -> None:
    """A 0.2 m block (top at z = 0) and its +x side face as a second shape turned +90° about y."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.05), hx=0.1, hy=0.02, hz=0.05, label="top")
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.05, TURN_Y), hx=0.05, hy=0.02, hz=0.1, label="side")
    _dummy(b)


@pytest.mark.compile_heavy
def test_film_runs_over_a_block_edge_down_its_side():
    """A 90° fold between charts in turned frames (WP15's review): the link face's direction is the chord between
    the cell centres over the arc (5 mm across, 5 mm down over 10 mm), so gravity pulls the true drop g·Δz/dist; each
    side reads the face along its own edge tangent ((1, 0, 0) in each frame); the fold's turning adds (π/4)/dx of
    convex curvature to both seam cells; the side's old edge faces are zero-length walls no cell names. Film running
    off the sloped top flows down the side; volume exact."""
    flow = SurfaceFlow(charts=(PlaneChart("top", cells=(20, 4)), PlaneChart("side", cells=(10, 4))),
                       links=(("top", "i+", "side", "i-"),), material="water", max_depth=4.0e-3)
    h0 = np.zeros(120)
    h0[60:80] = 1.5e-3                                                    # the top's last 5 rows
    r = _read(flow, _block, lambda sim: flow.set_depth(h0), 15, DOWNHILL)
    t = flow.tables
    link = np.arange(20 * 4, 21 * 4)                                      # the top's i+ faces
    d, dist = t.fdir.numpy()[link], t.fdist.numpy()[link]
    assert np.allclose(d, [0.5, 0.0, -0.5], rtol=0.0, atol=1e-6) and np.allclose(dist, 0.01, rtol=1e-6, atol=0.0)
    assert np.allclose(-G * d[:, 2], G * 0.005 / dist, rtol=1e-5, atol=0.0)   # g·Δz/dist (float32 cells: 2e-6)
    assert np.allclose(t.fdira.numpy()[link], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-6)
    assert np.allclose(t.fdirb.numpy()[link], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-6)
    sides = wp.zeros((4, 2), dtype=wp.vec3, device="cpu")
    wp.launch(_sides, dim=4, inputs=[flow.faces, wp.array(link.astype(np.int32), dtype=wp.int32, device="cpu")],
              outputs=[sides], device="cpu")
    assert np.allclose(sides.numpy(), [1.0, 0.0, 0.0], rtol=0.0, atol=1e-6)    # each cell reads its own tangent
    seam = np.r_[76:80, 80:84]                                            # the top's last row, the side's first
    assert np.allclose(t.curv.numpy()[seam], 0.25 * math.pi / 0.01, rtol=1e-6, atol=0.0)
    assert np.array_equal(t.fb.numpy()[link], 80 + np.arange(4)) and (t.ftype.numpy()[link] == 0).all()
    old = t.charts.numpy()[1]["iface_start"] + np.arange(4)               # the side's i- faces
    assert (t.ftype.numpy()[old] == 1).all() and (t.flen.numpy()[old] == 0.0).all()
    assert not np.isin(old, t.cface.numpy()).any()
    side = r["h"][0][80:].reshape(10, 4)
    assert side[5:].sum() > 0.05 * h0.sum()                               # down the side's lower half
    assert r["ledger"][0] == pytest.approx((h0 * t.area_np).sum(), rel=1e-12, abs=0.0)


def _plate(b) -> None:
    """A 2 mm plate (top at z = 0) and its underside: the same box turned 180° about y."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.001), hx=0.1, hy=0.02, hz=0.001, label="top")
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.001, FLIP_Y), hx=0.1, hy=0.02, hz=0.001, label="under")
    _dummy(b)


def _lip_flow(**kw) -> SurfaceFlow:
    return SurfaceFlow(charts=(PlaneChart("top", cells=(20, 4), edges=("wall", "open", "wall", "wall")),
                               PlaneChart("under", cells=(20, 4), edges=("open", "wall", "wall", "wall"))),
                       links=(("top", "i+", "under", "i-"),), material="water", max_depth=4.0e-3, **kw)


@pytest.mark.compile_heavy
def test_film_runs_over_a_plate_lip_onto_its_underside_and_the_linked_edge_sheds_nothing():
    """A 180° fold (the plate's lip, 2 mm thick): the arc is 10 mm + the 2 mm lip and the chord drops 2 mm over it;
    the fold adds (π/2)/dx of convex curvature on both sides. Film on the sloped top's open x+ edge, which is linked,
    goes onto the underside and nothing is shed (no open face is left); volume exact."""
    flow = _lip_flow()
    h0 = np.zeros(160)
    h0[60:80] = 1.5e-3
    r = _read(flow, _plate, lambda sim: flow.set_depth(h0), 15, DOWNHILL)
    t = flow.tables
    link = np.arange(20 * 4, 21 * 4)
    assert np.allclose(t.fdist.numpy()[link], 0.012, rtol=1e-6, atol=0.0)
    assert np.allclose(t.fdir.numpy()[link], [0.0, 0.0, -0.002 / 0.012], rtol=0.0, atol=1e-6)
    assert np.allclose(t.curv.numpy()[np.r_[76:80, 80:84]], 0.5 * math.pi / 0.01, rtol=1e-6, atol=0.0)
    assert r["h"][0][80:].sum() > 0.0 and r["spilled_volume"][0] == 0.0
    assert r["ledger"][0] == pytest.approx((h0 * t.area_np).sum(), rel=1e-12, abs=0.0)


@pytest.mark.compile_heavy
def test_a_push_carries_film_over_a_lip():
    """No gravity: pending momentum on the top's seam row only (src_mom, 5 Pa·dt along its +x each 20 ms exchange: a
    5 Pa traction) pushes film over the lip onto the underside, since each side reads the push along its own tangent
    (WP15's discriminating case: along the lip's chord, (0, 0, -1), the push drops out): 10.0 % goes under in 3
    periods, against 0.79 % along the chord and 0.37 % with no push."""
    flow = _lip_flow(max_stride=1)
    h0 = np.zeros(160)
    h0[64:80] = 1.0e-3
    sim = build_simulation(TestScene(build=_plate, init=lambda sim: flow.set_depth(h0), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1, dt=0.02), [flow], "cpu")
    for _ in range(3):
        mom = flow.src_mom.numpy()
        mom[0, 76:80, 0] += 5.0 * 0.02                                    # one exchange per period
        flow.src_mom.assign(mom)
        sim.step_period()
    r = flow.read()
    assert r["h"][0][80:].sum() > 0.05 * h0.sum()
    assert r["ledger"][0] == pytest.approx((h0 * flow.tables.area_np).sum(), rel=1e-12, abs=0.0)


def _hand_tool(flow: SurfaceFlow, cells: np.ndarray, vel: float, ceil: float) -> None:
    """A tool field set by hand (no applicator stage rewrites it): cover 1 over ``cells``, its underside ``ceil``
    above them, sliding at ``vel`` along each cell's own +x (around a lip: away from the top on both sides)."""
    C = flow.tables.area_np.shape[0]
    cover, top = np.zeros((1, C), np.float32), np.full((1, C), 1.0e3, np.float32)
    v = np.zeros((1, C, 3), np.float32)
    cover[0, cells], top[0, cells], v[0, cells, 0] = 1.0, ceil, vel
    tf = flow.tf
    tf.on = 1
    tf.cover, tf.ceil = wp.array(cover, dtype=wp.float32, device="cpu"), wp.array(top, dtype=wp.float32, device="cpu")
    tf.vel = wp.array(v, dtype=wp.vec3, device="cpu")
    tf.push, tf.press = wp.zeros((1, C), dtype=wp.vec2, device="cpu"), wp.zeros((1, C), dtype=wp.float32, device="cpu")


@pytest.mark.compile_heavy
def test_a_tool_sliding_round_a_lip_drags_film_under():
    """No gravity: a tool over the lip's two seam rows (hand-set: its underside 0.3 mm over them, in a 1 mm film,
    sliding at 1 m/s away from the top on both sides) drags film under, since each side's sliding is read along its own
    tangent (along the lip's chord it drops out): 16.9 % goes under in 3 periods, against 0.30 % along the chord."""
    flow = _lip_flow()
    h0 = np.zeros(160)
    h0[64:80] = 1.0e-3

    def init(sim) -> None:
        flow.set_depth(h0)
        _hand_tool(flow, np.r_[76:80, 80:84], 1.0, 0.3e-3)
    r = _read(flow, _plate, init, 3, (0.0, 0.0, 0.0))
    assert r["h"][0][80:].sum() > 0.05 * h0.sum()
    assert r["ledger"][0] == pytest.approx((h0 * flow.tables.area_np).sum(), rel=1e-12, abs=0.0)


TURN_BACK = (0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))   # -90° about y: local x -> +z, z -> -x


def _corner(b) -> None:
    """A 0.2 m floor (top at z = 0) and a wall rising from its +x edge, its chart face toward -x (an inside corner)."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.05), hx=0.1, hy=0.02, hz=0.05, label="floor")
    b.add_shape_box(-1, xform=body_xform(0.11, 0.0, 0.05, TURN_BACK), hx=0.05, hy=0.02, hz=0.01, label="wall")
    _dummy(b)


@pytest.mark.compile_heavy
def test_an_inside_corner_is_concave_and_its_meniscus_fills():
    """A concave 90° fold (WP15's review: the sign was untested): the chord (5 mm across, 5 mm up) rises off the
    floor's wetted side, so both seam cells get -(π/4)/dx; with no gravity a uniform 1 mm film flows into the corner
    (lower pressure there), so the seam rows gain; volume exact."""
    flow = SurfaceFlow(charts=(PlaneChart("floor", cells=(20, 4), edges=("wall", "open", "wall", "wall")),
                               PlaneChart("wall", cells=(10, 4), edges=("open", "wall", "wall", "wall"))),
                       links=(("floor", "i+", "wall", "i-"),), material="water", max_depth=4.0e-3)
    h0 = np.full(120, 1.0e-3)
    r = _read(flow, _corner, lambda sim: flow.set_depth(h0), 5, (0.0, 0.0, 0.0))
    t = flow.tables
    link = np.arange(20 * 4, 21 * 4)
    assert np.allclose(t.fdir.numpy()[link], [0.5, 0.0, 0.5], rtol=0.0, atol=1e-6)
    seam = np.r_[76:80, 80:84]
    assert np.allclose(t.curv.numpy()[seam], -0.25 * math.pi / 0.01, rtol=1e-6, atol=0.0)
    assert (r["h"][0][seam] > 1.0e-3).all()
    assert r["ledger"][0] == pytest.approx((h0 * t.area_np).sum(), rel=1e-12, abs=0.0)


def _three(b) -> None:
    for name, x, hx in (("a", -0.05, 0.05), ("m", 0.01, 0.01), ("b", 0.07, 0.05)):
        b.add_shape_box(-1, xform=body_xform(x, 0.0, -0.01), hx=hx, hy=0.02, hz=0.01, label=name)
    b.add_shape_box(-1, xform=body_xform(0.25, 0.0, 0.0), hx=0.01, hy=0.01, hz=0.01, label="cup")
    _dummy(b)


BAD = {
    "two + edges": (("top", "i+", "under", "i+"), "joins two + edges"),
    "apart": (("top", "j-", "under", "j+"), "do not meet"),
    "counts": (("top", "i+", "under", "j-"), "cells"),
    "twice": ((("top", "i+", "under", "i-"), ("top", "i+", "under", "j-")), "already linked"),
    "wetting": (("top", "i+", "under", "i-"), "wet differently"),
    "narrow": ((("a", "i+", "m", "i-"), ("m", "i+", "b", "i-")), "3 cells"),
    "ring": (("top", "i+", 2, "start"), "ring"),
}


def _bad_charts(case: str) -> tuple:
    if case == "narrow":
        return PlaneChart("a", cells=(10, 4)), PlaneChart("m", cells=(2, 4)), PlaneChart("b", cells=(10, 4))
    top = PlaneChart("top", cells=(20, 4), contact_angle=30.0 if case == "wetting" else None)
    ring = (RevolutionChart(((0.05, 0.0), (0.05, 0.02)), n_theta=4, ds=0.01),) if case == "ring" else ()
    return (top, PlaneChart("under", cells=(20, 4)), *ring)


@pytest.mark.parametrize("case", sorted(BAD))
def test_bad_links_are_refused(case):
    links, words = BAD[case]
    links = links if isinstance(links[0], tuple) else (links,)
    flow = SurfaceFlow(charts=_bad_charts(case), links=links, material="water")
    run = _run(flow, _three if case == "narrow" else _plate, None, 1)
    assert run.error is not None and words in str(run.error), run.error


def test_a_link_with_a_wrong_key_is_a_plugin_error():
    from alloy.plugins.errors import PluginError  # noqa: PLC0415
    with pytest.raises(PluginError, match="a link takes"):
        SurfaceFlow(charts=(PlaneChart("top"),), links=({"a": "top", "edge": "i+", "b": "under", "edge_b": "i-"},))


def test_a_link_reads_the_effective_contact_angle():
    """WP15's review: a chart without its own angle wets at the material's, so it links to a chart that gives the same
    angle (the raw settings, None and 20°, differ); a different angle is refused."""
    mat = FilmMaterial("water_20", contact_angle=20.0)
    for angle, error in ((20.0, None), (30.0, "wet differently")):
        charts = (PlaneChart("top", cells=(20, 4)), PlaneChart("under", cells=(20, 4), contact_angle=angle))
        flow = SurfaceFlow(charts=charts, links=(("top", "i+", "under", "i-"),), material=mat)
        run = _run(flow, _plate, None, 1)
        assert (run.error is None) if error is None else (error in str(run.error)), run.error
