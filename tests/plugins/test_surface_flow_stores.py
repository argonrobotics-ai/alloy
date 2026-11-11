# SPDX-License-Identifier: Apache-2.0
"""Surface flow, the tools' stores (phase 2): the nip split, no drip at retention, tray pick-up; every volume exact
(CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.errors import PluginError

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import ROLLER_BINS, Store, Tray

VISCOUS = FilmMaterial("viscous_100Pas", density=1000.0, viscosity=100.0, surface_tension=0.0)
R, HW, U = 0.02, 0.03, 0.05                     # roller radius, half width [m]; rolling speed [m/s]
LENGTH, WIDTH, NX = 0.3, 0.05, 120


def roller_scene(axis_z: float, x0: float):
    """A static plate (top face at z = 0, the chart) and a roller on a carriage: the carriage slides along x, the
    roller turns about y. Its collision shape (a small sphere at the axis) never touches the plate."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.5 * LENGTH, hy=0.5 * WIDTH, hz=0.01,
                        label="plate")
        xf = body_xform(x0, 0.0, axis_z)
        cart = b.add_link(xform=xf, label="carriage")
        b.add_shape_sphere(cart, radius=0.005, cfg=newton.ModelBuilder.ShapeConfig(density=1.0e5))
        cover = b.add_link(xform=xf, label="cover")
        b.add_shape_sphere(cover, radius=0.005, cfg=newton.ModelBuilder.ShapeConfig(density=1.0e5))
        slide = b.add_joint_prismatic(-1, cart, parent_xform=xf, axis=newton.Axis.X)
        turn = b.add_joint_revolute(cart, cover, axis=newton.Axis.Y)
        b.add_articulation([slide, turn])
    return build


def roll(store: Store, depth, axis_z: float, periods: int, x0: float = -0.1, blade_gap: float | None = None,
         release: float = 0.0, flow_kw: dict | None = None, force_plough: bool = False, speed: float = U,
         base=None, material: FilmMaterial = VISCOUS, gravity=(0.0, 0.0, 0.0), tool_kw: dict | None = None,
         setup=None):
    """``depth``: a depth [m], or a function of the flow giving the start depths ``[1, C]``. ``blade_gap``: a flat
    blade on the carriage instead of the roller, its underside that far over the plate (no turning). ``release``:
    the tool_release share held from the start; ``flow_kw``: more SurfaceFlow keywords; ``base``: a function of the
    flow giving the base ``[1, C]``; ``tool_kw``: more Applicator keywords; ``setup(flow)``: runs at init."""
    if blade_gap is None:
        tool = Applicator(body="cover", shape="roller", size=(R, HW), axis=(0.0, 1.0, 0.0), stiffness=1.0e3,
                          store=store)
    else:
        tool = Applicator(body="carriage", shape="rect", size=(0.005, HW), center=(0.0, 0.0, blade_gap - axis_z),
                          stiffness=1.0e3, store=store, **(tool_kw or {}))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=material, inertia=False,
                       max_depth=1.0e-3, tools=(tool,), **(flow_kw or {}))

    def init(sim) -> None:
        if base is not None:
            flow.set_input("base", base(flow).astype(np.float32))
        flow.set_depth(depth(flow) if callable(depth) else depth)
        flow.tool_release.fill_(release)
        flow.ploughs = flow.ploughs or force_plough
        if setup is not None:
            setup(flow)
        qd = sim.state.joint_qd.numpy().reshape(-1, 2)
        qd[:] = (speed, speed / R if blade_gap is None else 0.0)  # rolling without slip
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=roller_scene(axis_z, x0), init=init, ground=False, gravity=gravity,
                                     worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(periods):
        sim.step_period()
    return flow, sim


def rolled_over(flow, x0: float, sim) -> np.ndarray:
    """The plate's depth on the cells the roller's centre line crossed (well clear of its start and end)."""
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    return flow.depth(0)[0][(x > x0 + 0.005) & (x < x1 - 0.005)]


def rounding_depths(flow) -> np.ndarray:
    """Depths in [0.2, 0.5] mm for which emptying the cell, h + (-(h·a))/a, rounds below 0 (about 1 % of them)."""
    rng = np.random.default_rng(0)
    area = flow.tables.area_np
    h = np.empty(area.size)
    for c, a in enumerate(area):
        v = rng.uniform(2.0e-4, 5.0e-4, 4096)
        v = v[v + (-(v * a)) / a < 0.0]
        assert v.size, a
        h[c] = v[0]
    return h.reshape(1, -1)


def test_a_store_that_takes_the_whole_film_leaves_exactly_none():
    """split = 1 empties each cell the nip crosses. SOFT-2's trowel left one at -5e-20 m (float64 rounding of
    h - (h·a)/a), the film's limiter divided by 0 and the run went NaN. Over depths that all round below 0, the
    rolled cells read exactly 0, every depth stays finite and >= 0, and the books close."""
    flow, sim = roll(Store(capacity=1.0e-5, split=1.0), rounding_depths, R + 1.0e-4, periods=40)
    h = rolled_over(flow, -0.1, sim)
    assert h.size >= 40 and (h == 0.0).all(), h[h != 0.0]
    every = flow.h.numpy()
    assert np.isfinite(every).all() and (every >= 0.0).all(), every.min()
    r = flow.read()
    start = float((rounding_depths(flow)[0] * flow.tables.area_np).sum())
    assert abs(r["ledger"][0] - start) < 1e-12 * start
    assert float(r["clipped_volume"][0]) == 0.0                           # the nip left exact zeros: nothing to clip
    assert r["tool_load"][0].sum() == pytest.approx(r["to_tools_volume"][0], rel=1e-12, abs=0.0)


@pytest.mark.parametrize("express, lays", [(1.0, True), (0.4, False)])
def test_a_loaded_roller_reaches_only_its_expressed_share(express, lays):
    """The nap reaches e·l below the cover, not its whole free load l (SOFT-3: a disc tip with e = 0.034 painted cells
    0.8 mm above the paper). Held 0.7·l over a dry plate, a cover expressing all its load lays film; one expressing 0.4
    of it does not touch, and lays exactly nothing."""
    cap, bins = 6.0e-6, 32
    nap = 0.75 * (cap / bins) / ((2.0 * math.pi * R / bins) * 2.0 * HW)
    store = Store(capacity=cap, retained=0.25, held=0.75, split=0.5, bins=bins, initial=1.0, drip=0.0,
                  express=express)
    flow, sim = roll(store, 0.0, R + 0.7 * nap, periods=40)
    laid = -float(flow.read()["to_tools_volume"][0])
    if lays:
        assert laid > 0.0
    else:
        assert laid == 0.0 and (flow.h.numpy() == 0.0).all(), laid


def wavy(flow) -> np.ndarray:
    """Start depths 0.15-0.45 mm, a cosine along x with a 10-cell period (a crest where the blade starts)."""
    h = 3.0e-4 + 1.5e-4 * np.cos(2.0 * np.pi * np.arange(NX) / 10.0)
    return np.repeat(h, 4).reshape(1, -1)


def test_a_scraping_blade_cuts_the_crests_and_fills_the_lows():
    """Store(scrape=True), GA's carrying knife (coupling.py:530-596; SOFT-2's and WP6's margin trowels): a loaded
    blade 0.3 mm over a 0.15-0.45 mm film cuts split of what stands above its underside into its carry, and while it
    touches it fills what lies below its bearing band up to its lowest level, from the carry. Crests and troughs end
    at g, to the pose's float32 (a few nm). The books close."""
    g, split = 3.0e-4, 1.0
    store = Store(capacity=1.0e-5, retained=0.0, held=1.0, split=split, pickup=0.0, drip=0.0, initial=1.0,
                  scrape=True)
    flow, sim = roll(store, wavy, 0.05, periods=40, blade_gap=g)
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    h0 = wavy(flow)[0].reshape(NX, 4)[(x > -0.1 + 0.005) & (x < x1 - 0.005)]
    want = np.where(h0 > g, h0 - split * np.maximum(h0 - g, 0.0), g)
    got = rolled_over(flow, -0.1, sim)
    assert got.shape[0] >= 8 and (h0 > g).sum() >= 12 and (h0 < g).sum() >= 12        # columns; cells above, below
    # cut and filled every exchange to the underside, whose height jitters by a few float32 ulps at 5 cm (~1e-8 m)
    assert np.allclose(got, want, rtol=0.0, atol=2.0e-8), np.abs(got - want).max()
    r = flow.read()
    area = flow.tables.area_np
    start = float((wavy(flow)[0] * area).sum())
    assert abs(r["ledger"][0] - start) < 1e-12 * start
    assert abs(float(r["store_ledger"][0]) - 1.0e-5) <= 1e-12 * 1.0e-5
    net = float(((wavy(flow)[0] - flow.h.numpy()[0]) * area).sum())          # cut - filled, into the carry
    assert float(r["to_tools_volume"][0]) == pytest.approx(net, rel=1e-12, abs=0.0)


def test_a_scraping_blade_cuts_its_whole_footprint_and_refuses_a_split():
    """GA's plough cuts every cell under the sole each step (coupling.py:521-525), not only where a centre line
    crossed (SOFT-2: a strip beside a joint never reached the T-junction corners). A blade held still 0.3 mm over
    0.4 mm of film cuts every fully covered cell to g. A scraping store with split < 1 is refused."""
    g = 3.0e-4
    store = Store(capacity=1.0e-5, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, scrape=True)
    flow, sim = roll(store, 4.0e-4, 0.05, periods=5, blade_gap=g, speed=0.0)
    h = flow.h.numpy()[0]
    off = np.repeat(np.abs(flow.cell_centres(0)[:, 0, 0] + 0.1), 4)      # from the blade's centre along x
    half, dx = 0.005, LENGTH / NX
    full, clear = off < half - 0.5 * dx, off > half + 0.5 * dx           # fully under it; clear of it
    assert full.sum() >= 8
    assert np.allclose(h[full], g, rtol=0.0, atol=2.0e-8), h[full]
    assert (h[clear] == 4.0e-4).all()
    with pytest.raises(PluginError, match="split must be 1"):
        Store(capacity=1.0e-5, split=0.5, scrape=True)


def test_a_blade_on_bare_faces_fills_a_groove_only_to_their_level():
    """GA's knife bearing on the rigid base fills only up to the base's level (coupling.py:454-458, 578-581). A loaded
    blade 0.03 mm over dry faces (within GA's 0.05 mm on_base), cutting a row of 0.3 mm tags beside a 2 mm groove,
    fills the groove to the faces' level, leaves the faces bare, and cuts the tags to its underside. Higher than
    that it no longer bears on them, and as in GA only ``paste_mask`` keeps its fill off them."""
    g = 3.0e-5
    rows = np.arange(NX * 4) % 4                                 # across the plate: face, tag, groove, face

    def base(flow) -> np.ndarray:
        return np.where(rows == 2, -2.0e-3, 0.0).reshape(1, -1)

    def depth(flow) -> np.ndarray:
        return np.where(rows == 1, 3.0e-4, 0.0).reshape(1, -1)
    store = Store(capacity=1.0e-5, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, initial=1.0,
                  scrape=True)
    flow, sim = roll(store, depth, 0.05, periods=40, blade_gap=g, base=base)
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    done = np.repeat((x > -0.1 + 0.005) & (x < x1 - 0.005), 4)
    h = flow.h.numpy()[0]
    assert done.sum() >= 32
    assert (h[done & ((rows == 0) | (rows == 3))] == 0.0).all()            # bare faces stay bare
    assert np.allclose(h[done & (rows == 2)], 2.0e-3, rtol=1e-6, atol=0.0)  # the groove to the faces' level
    assert np.allclose(h[done & (rows == 1)], g, rtol=0.0, atol=5.0e-9)     # tags cut to the underside
    r = flow.read()
    start = float((depth(flow)[0] * flow.tables.area_np).sum())
    assert abs(r["ledger"][0] - start) < 1e-12 * start


def test_a_short_carry_fills_the_lows_in_proportion():
    """GA's fill lays min(1, carry/want) of every bearing cell's room (coupling.py:587-596). An empty blade held 0.3 mm
    over a step, 0.4 mm of film then 0.2 mm, can lay no more than it cut: nothing on the low side ends above g, and
    the volume it lays is the volume it took."""
    g = 3.0e-4

    def step(flow) -> np.ndarray:
        return np.repeat(np.where(np.arange(NX) < 30, 4.0e-4, 2.0e-4), 4).reshape(1, -1)
    store = Store(capacity=1.0e-5, retained=0.0, held=1.0, split=1.0, pickup=0.0, drip=0.0, scrape=True)
    flow, sim = roll(store, step, 0.05, periods=40, blade_gap=g)
    h = flow.h.numpy()[0]
    r = flow.read()
    area = flow.tables.area_np
    cut = float(((step(flow)[0] - h).clip(0.0) * area).sum())
    laid = float(((h - step(flow)[0]).clip(0.0) * area).sum())
    assert cut > 0.0 and laid > 0.0
    assert (h[step(flow)[0] < g] <= g + 5.0e-9).all()                    # no low cell filled past the blade
    assert float(r["tool_load"][0].sum()) == pytest.approx(cut - laid, rel=1e-9, abs=0.0)
    assert float(r["tool_load"][0].sum()) >= 0.0


@pytest.mark.parametrize("supply", [False, True])
def test_a_store_lets_go_of_its_free_load_into_the_drip_channel(supply):
    """tool_release (WP6's trowel over DC5's bucket, WP11's brush): each exchange a store lets go of that share of its
    free load (above retained), and still drips above held. With a supply, both leave through DC5's stage-drips channel
    at the lowest bin, with the load's species and that point's velocity (a rolling cover's bottom is at rest).
    Without one they are only booked. Every volume is exact."""
    from alloy.plugins.surface_flow.supply import Drain  # noqa: PLC0415

    cap, n, axis = 6.0e-6, 3, R + 0.05
    store = Store(capacity=cap, retained=0.25, held=0.75, split=0.5, initial=1.0, drip=5.0, species={"soap": 4.0})
    kw: dict = {"species": ("soap",)}
    if supply:
        kw.update(drains=(Drain("grate", chart=0, center=(0.12, 0.0, 0.0), radius=0.01, to="sink"),),
                  network={"nodes": [{"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.01}], "edges": []})
    flow, sim = roll(store, 0.0, axis, periods=n, release=0.01, flow_kw=kw)
    steps = int(flow.tick.numpy()[0])                          # exchanges (one per physics step)
    b, h = cap / ROLLER_BINS, flow._exchange_dt                 # each bin, replayed: release, then the drip
    load, gone, drip = b, 0.0, 0.0
    for _ in range(steps):
        g = 0.01 * max(load - 0.25 * b, 0.0)
        load -= g
        d = max(load - 0.75 * b, 0.0) * (1.0 - math.exp(-5.0 * h))
        load, gone, drip = load - d, gone + g, drip + d
    r = flow.read()
    released, dripped = float(r["tool_released"][0].sum()), float(r["tool_dripped"][0].sum())
    assert steps > n and gone > 0.0 and drip > 0.0
    assert released == pytest.approx(ROLLER_BINS * gone, rel=1e-12, abs=0.0)
    assert dripped == pytest.approx(ROLLER_BINS * drip, rel=1e-12, abs=0.0)
    assert abs(float(r["store_ledger"][0]) - cap) <= 1e-12 * cap
    if not supply:
        assert not hasattr(flow, "src_drip_v")
        return
    staged = r.get("supply_staged")                             # DC5's column once the supply casts the channel
    sent = float(staged[0]) if staged is not None else float(flow.src_drip_v.numpy().sum())
    assert sent == pytest.approx(released + dripped, rel=1e-12, abs=0.0)
    if staged is None:
        assert float(flow.src_drip_sp.numpy().sum()) == pytest.approx(4.0 * sent, rel=1e-12, abs=0.0)
    p, u = flow.src_drip_p.numpy()[0, 0], flow.src_drip_u.numpy()[0, 0]
    assert abs(p[2] - (axis - R)) <= R * (1.0 - math.cos(math.pi / ROLLER_BINS)) + 1e-6, p
    assert np.linalg.norm(u) <= 0.1 * U, u                      # rolling without slip: the bottom is at rest


def test_store_tools_skip_the_plough_bit_for_bit():
    """Only ceiling tools plough, so with store tools alone the substep's plough and roll passes are no-ops and are
    skipped (DC6's sash brush spent 45 % of its CPU there). Forcing them back on changes nothing, bit for bit."""
    store = Store(capacity=6.0e-6, retained=0.25, held=0.75, split=0.5, initial=1.0, drip=0.0)
    runs = [roll(store, 1.0e-4, R + 5.0e-5, periods=20, force_plough=f) for f in (False, True)]
    assert not runs[0][0].ploughs and runs[1][0].ploughs
    assert np.array_equal(runs[0][0].h.numpy(), runs[1][0].h.numpy())
    assert np.array_equal(runs[0][0].tool_load.numpy(), runs[1][0].tool_load.numpy())


def test_an_empty_roller_takes_its_split_of_a_wet_film():
    """The nip split: rolling an empty cover over a film h0 leaves (1 - split)·h0 and takes the rest, exactly."""
    h0, cap = 1.0e-4, 1.0e-5
    flow, sim = roll(Store(capacity=cap, split=0.5), h0, R + 0.5 * h0, periods=40)          # 0.8 s: 4 cm
    h = rolled_over(flow, -0.1, sim)
    assert h.size >= 40
    assert np.allclose(h, 0.5 * h0, rtol=0.02, atol=0.0), h
    r = flow.read()
    start = h0 * flow.tables.area_np.sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start                    # film + what the tools took
    assert abs(r["store_ledger"][0]) < 1e-12 * start                      # the cover started empty
    assert r["tool_load"][0].sum() == pytest.approx(r["to_tools_volume"][0], rel=1e-12, abs=0.0)   # taken from the film


@pytest.mark.parametrize("express", [1.0, 0.4])
def test_a_loaded_roller_lays_its_split_of_its_free_load(express):
    """On a dry plate a loaded cover lays (1 - split)·e·l, l its free load (above retention) over a bin's nap area and
    e the share its nap expresses (SOFT-3: GA expresses about twice the nap's strain, not the whole load)."""
    cap, bins = 6.0e-6, 32
    store = Store(capacity=cap, retained=0.25, held=0.75, split=0.5, bins=bins, initial=1.0,
                  drip=0.0, express=express)                                 # the nip alone: no drip
    flow, sim = roll(store, 0.0, R, periods=40)
    nap = 0.75 * (cap / bins) / ((2.0 * math.pi * R / bins) * 2.0 * HW)
    h = rolled_over(flow, -0.1, sim)
    assert np.allclose(h, 0.5 * express * nap, rtol=0.02, atol=0.0), (h, 0.5 * express * nap)
    r = flow.read()
    assert abs(r["store_ledger"][0] - cap) < 1e-12 * cap                  # loads + laid = the start load
    laid = -r["to_tools_volume"][0]
    assert laid > 0.0 and r["film_volume"][0] == pytest.approx(laid, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("initial", [0.75, 1.0])
def test_a_lifted_store_holds_its_held_share_and_drips_the_rest(initial):
    """No drip at retention (a load-roller's law): lifted, the cover keeps held·capacity and drips only above it,
    at its drip rate."""
    cap, drip = 6.0e-6, 1.0
    flow, sim = roll(Store(capacity=cap, held=0.75, drip=drip, initial=initial), 0.0, R + 0.05, periods=50)  # 1 s
    r = flow.read()
    expect = max(initial - 0.75, 0.0) * cap * (1.0 - math.exp(-drip * 1.0))
    assert r["tool_dripped"][0].sum() == pytest.approx(expect, rel=1e-3, abs=1e-18)
    assert abs(r["store_ledger"][0] - initial * cap) < 1e-12 * cap


def test_a_pad_dipped_in_a_tray_fills_toward_capacity_from_it():
    cap, pickup, vol = 2.0e-6, 2.0, 2.0e-4
    pad = Applicator(body="cover", shape="disc", size=(0.02, 0.02), center=(0.0, 0.0, -R), stiffness=1.0e3,
                     store=Store(capacity=cap, pickup=pickup))
    tray = Tray(center=(-0.1, 0.0, -0.05), half=(0.05, 0.05), volume=vol)  # level: -0.05 + 0.02 m, over the pad
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, inertia=False,
                       max_depth=1.0e-3, tools=(pad,), trays=(tray,))
    sim = build_simulation(TestScene(build=roller_scene(-0.05 + R + 0.01, -0.1), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(25):                                                     # 0.5 s
        sim.step_period()
    r = flow.read()
    load = r["tool_load"][0].sum()
    assert load == pytest.approx(cap * (1.0 - math.exp(-pickup * 0.5)), rel=2e-3, abs=0.0)
    assert r["tray_volume"][0, 0] == pytest.approx(vol - load, rel=1e-12, abs=0.0)
    assert abs(r["tray_ledger"][0] - vol) < 1e-12 * vol


def test_a_roller_faster_than_a_bin_per_exchange_is_flagged():
    """WP2 review 4b: a roller reads its contact bin at its current pose, so rolling past more than one bin's arc per
    exchange skips bins. The check names such rollers (32 bins at R 20 mm: an arc of 3.9 mm)."""
    from alloy.plugins.surface_flow.stores import check_bins  # noqa: PLC0415

    roller = Applicator(body="cover", shape="roller", size=(R, HW), axis=(0.0, 1.0, 0.0), store=Store(capacity=1e-6))
    pad = Applicator(body="cover", shape="disc", size=(0.02, 0.02), store=Store(capacity=1e-6))
    assert check_bins("film", (roller, pad), 0.5, 2.0e-3) == []                    # 1 mm per exchange
    assert check_bins("film", (roller, pad), 0.5, 1.0e-2) == ["tool 0"]            # 5 mm: bins skip
