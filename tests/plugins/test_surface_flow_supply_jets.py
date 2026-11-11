# SPDX-License-Identifier: Apache-2.0
"""Surface-flow supply side (DC5): jet geometry (drowning, folds, cups, oblique and rising jets), the aerosol can,
pool species, drain openings, the tangential share of a jet's push, and the joint case on WP15's real film (CPU).

The stub film and the scenes are in ``alloy.plugins.surface_flow.supply_stub``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform
from alloy.plugins.surface_flow.charts import PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.supply import Drain, Nozzle
from alloy.plugins.surface_flow.supply_stub import (CHART, FAUCET, RHO, StubFilm, faucet_network, ledger_sides,
                                                    run_stub, slab_scene)


def _run(plugin: StubFilm, periods: int = 20, build=None, worlds: int = 2) -> StubFilm:
    return run_stub(plugin, periods, build, worlds)


def _ledger_closes(p: StubFilm) -> None:
    lhs, rhs, spout, sent = ledger_sides(p)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-18), (lhs, rhs)
    assert np.allclose(spout, sent, rtol=1e-12, atol=0.0), (spout, sent)


def _h(p: StubFilm, w: int = 0) -> np.ndarray:
    return p.h.numpy()[w].reshape(40, 40)


def test_a_deep_pool_drowns_the_jump_inward():
    """With 20 mm of water standing on the slab, the jump stands at the conjugate depth: r = (Q/(π·a·h))²/g ≈ 20 mm,
    not R_J ≈ 99 mm."""
    p = StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network())
    h0 = 0.02

    def init(sim):
        plug = sim.plugins[p.label]
        plug.h.assign(np.full(plug.h.shape, h0))
    run = run_scene(TestScene(build=slab_scene(), ground=True, worlds=2, periods=3, init=init), [p], "cpu")
    assert run.error is None, run.error
    p = run.sim.plugins[p.label]
    q = float(-p.supply.read()["discharge"][0, 1])
    w = math.sqrt((q / 1e-4) ** 2 + 2 * 9.81 * 0.28)
    a = math.sqrt(q / (math.pi * w))
    want = (q / (math.pi * a * h0)) ** 2 / 9.81
    gain = np.maximum(_h(p) - h0, 0.0)
    xs = (np.arange(40) + 0.5) * 0.01 - 0.2
    rr = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    r_mean = float((rr * gain).sum() / gain.sum())
    assert abs(r_mean - want) < 0.01, (r_mean, want)
    _ledger_closes(p)


CUP = ((0.006, 0.0), (0.035, 0.0), (0.035, 0.14), (0.038, 0.14), (0.038, 0.0))     # floor, wall, rim, outside


def test_a_jet_into_a_cup_stays_inside_it():
    """A revolution chart (a tall cup, inside and outside on one profile): distances run along the surface, so the
    ring lands inside and nothing reaches the outside wall, 3 mm away through the material."""
    cup = RevolutionChart(profile=CUP, n_theta=32, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    jet = Nozzle("faucet", edge="spout", pos=(0.02, 0.0, 0.3), exit_area=1e-4)
    p = _run(StubFilm(charts=(CHART, cup), nozzles=(jet,), network=faucet_network()),
             build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)), periods=5)
    r = p.supply.read()
    assert np.all(r["lost"] == 0) and np.all(r["sent"] > 0)
    cells = p.tables.cells_of(1)
    h = p.h.numpy()[0, cells].reshape(p.tables.shapes[1])
    prof_s = np.cumsum([0.0] + [math.dist(a, b) for a, b in zip(CUP[:-1], CUP[1:])])
    rows_s = (np.arange(h.shape[0]) + 0.5) * (prof_s[-1] / h.shape[0])
    inside = rows_s < prof_s[2]
    assert h[inside].sum() > 0 and h[~inside].sum() == 0, (h[inside].sum(), h[~inside].sum())
    _ledger_closes(p)


def test_an_aerosol_can_sprays_onto_the_chart():
    """A spray nozzle on a can's expel edge: the can's liquid (its ledger) is what the spray sent, plus in flight."""
    can = {"nodes": [{"name": "can", "volume": 1e-4, "pressure": 4.0e5}],
           "edges": [{"name": "valve", "kind": "expel", "a": "can", "cd_area": 8.2e-8, "density": 1000.0,
                      "dp_lam": 1.0, "capacity": 3e-4}]}
    spray = Nozzle("can", edge="valve", pos=(0.0, 0.0, 0.25), exit_area=8.2e-8, pattern="spray",
                   angle=math.radians(30), rays=32, efficiency=0.7, density=1000.0)
    p = _run(StubFilm(charts=(CHART,), nozzles=(spray,), network=can), periods=10)
    r = p.supply.read()
    assert np.all(r["sent"] > 0) and np.all(r["lost"] == 0)
    assert np.allclose(r["mist"], 0.3 * r["sent"], rtol=1e-6, atol=0.0)                      # 70 % transfer efficiency
    assert np.all(r["overspray"] == 0.0)
    supplied = p.supplied.numpy().sum(axis=1)
    landed = supplied + r["to_tank"] + r["lost"] + r["overspray"] + r["mist"] + r["caught"]
    assert np.allclose(r["sent"] + r["swept"], landed, rtol=1e-12, atol=0.0)
    in_flight = -r["discharge"][:, 0] * p.dt
    assert np.allclose(-r["edge_ledger"][:, 0], r["sent"] + in_flight, rtol=1e-12, atol=0.0)


def test_a_spray_into_a_cup_wraps_and_books_every_drop():
    """A wide spray beside a solid jet into a coarse cup (12 columns: footprints span most of a turn, so the
    parallel gather's box wraps to the whole turn): every cell's share comes from the ray's own window, and the
    ledger closes."""
    cup = RevolutionChart(profile=CUP, n_theta=12, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    spray = Nozzle("can", edge="spout", pos=(0.01, 0.0, 0.3), exit_area=4e-6, pattern="spray",
                   angle=math.radians(60), rays=64)
    jet = Nozzle("faucet", edge="spout", pos=(0.02, 0.0, 0.3), exit_area=1e-4)
    with pytest.raises(PluginError, match="share edge"):
        _run(StubFilm(charts=(CHART, cup), nozzles=(spray, jet), network=faucet_network()), periods=1)
    net = faucet_network()
    net["edges"].append({**net["edges"][1], "name": "can_valve", "cd_area": 4e-6})
    spray = Nozzle("can", edge="can_valve", pos=(0.01, 0.0, 0.3), exit_area=4e-6, pattern="spray",
                   angle=math.radians(60), rays=64)
    p = _run(StubFilm(charts=(CHART, cup), nozzles=(spray, jet), network=net),
             build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)), periods=5)
    r = p.supply.read()
    assert np.all(r["sent"] > 0) and np.all(r["swept"] > 0) and np.all(r["caught"] > 0)        # the ground catches misses
    assert p.supplied.numpy()[:, p.tables.cells_of(1)].sum() > 0
    lhs, rhs, _, _ = ledger_sides(p)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-18), (lhs, rhs)
    out = -r["edge_ledger"][:, 1:3].sum(axis=1)                                   # both nozzle edges
    assert np.allclose(out, r["sent"] - r["discharge"][:, 1:3].sum(axis=1) * p.dt, rtol=1e-12, atol=0.0)


def test_an_oblique_jet_sends_its_sheet_forward():
    """A jet 45° to the slab: the ring's share ahead of the hit follows the inviscid sheet split
    sin³β/(1 - cos β·cos φ)² (Hasson & Peck 1964) integrated over the forward half, to 5 %."""
    hose = Nozzle("hose", edge="spout", pos=(-0.15, 0.0, 0.17), axis=(1.0, 0.0, -1.0), exit_area=1e-5)
    p = _run(StubFilm(charts=(CHART,), nozzles=(hose,), network=faucet_network()), periods=3)
    hp = p.supply.hit["p"].numpy()[0, 0]
    wv = p.supply.hit["w"].numpy()[0, 0].astype(np.float64)
    cos_b = math.hypot(wv[0], wv[1]) / np.linalg.norm(wv)
    sin_b = math.sqrt(1.0 - cos_b ** 2)
    phi = np.linspace(-math.pi / 2, math.pi / 2, 4001)
    f = sin_b ** 3 / (1.0 - cos_b * np.cos(phi)) ** 2
    want = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(phi))) / (2 * math.pi)
    h = _h(p)
    xs = (np.arange(40) + 0.5) * 0.01 - 0.2
    ahead = xs[:, None] - hp[0] > 0
    got = float(h[np.broadcast_to(ahead, h.shape)].sum() / h.sum())
    assert abs(got - want) < 0.05, (got, want, math.degrees(math.acos(cos_b)))
    assert got > 0.75
    _ledger_closes(p)


SHORT_CUP = ((0.006, 0.0), (0.035, 0.0), (0.035, 0.04), (0.038, 0.04), (0.038, 0.0))   # 40 mm tall (WP15's probe)


def _cup_rows(p: StubFilm, profile) -> tuple[np.ndarray, np.ndarray]:
    """Volume per profile row of chart 1, and each row's arc length."""
    cells = p.tables.cells_of(1)
    nx, ny = p.tables.shapes[1]
    vol = (p.h.numpy()[0, cells] * p.tables.area_np[cells]).reshape(nx, ny).sum(axis=1)
    prof_s = np.cumsum([0.0] + [math.dist(a, b) for a, b in zip(profile[:-1], profile[1:])])
    return vol, (np.arange(nx) + 0.5) * (prof_s[-1] / nx)


def test_the_jump_stops_at_a_fold():
    """WP15's review M1: an off-centre faucet into a 40 mm cup. R_J (99 mm) would wrap over the floor-wall corner and
    the rim; the region stops at the fold, so the water enters at the wall's foot, on the floor."""
    cup = RevolutionChart(profile=SHORT_CUP, n_theta=32, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    jet = Nozzle("faucet", edge="spout", pos=(0.02, 0.0, 0.3), exit_area=1e-4)
    p = _run(StubFilm(charts=(CHART, cup), nozzles=(jet,), network=faucet_network()),
             build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)), periods=5)
    vol, rows_s = _cup_rows(p, SHORT_CUP)
    floor = rows_s < 0.029 + 2e-3                                   # the floor and the corner row (the wall's foot)
    assert vol[~floor].sum() == 0.0 and vol[floor].sum() > 0, (vol[floor].sum(), vol[~floor].sum())
    assert rows_s[np.argmax(vol)] > 0.025                                            # at the wall's foot
    _ledger_closes(p)


def test_a_cup_with_water_in_it_drowns_the_jump():
    """WP15's review M2: a tall cup with 20 mm on its floor. The jump stands where the floor's water reaches the
    sheet's conjugate depth (~21 mm from the hit), on the floor: nothing goes up the dry wall."""
    cup = RevolutionChart(profile=CUP, n_theta=32, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    jet = Nozzle("faucet", edge="spout", pos=(0.02, 0.0, 0.3), exit_area=1e-4)
    p = StubFilm(charts=(CHART, cup), nozzles=(jet,), network=faucet_network())

    def init(sim):
        plug = sim.plugins[p.label]
        cells = plug.tables.cells_of(1)
        nx, ny = plug.tables.shapes[1]
        h = plug.h.numpy()
        rows = np.repeat((np.arange(nx) + 0.5) * 2e-3 < 0.029, ny)
        h[:, np.arange(cells.start, cells.stop)[rows]] = 0.02
        plug.h.assign(h)
    run = run_scene(TestScene(build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)), ground=True, worlds=1, periods=3,
                              init=init), [p], "cpu")
    assert run.error is None, run.error
    p = run.sim.plugins[p.label]
    vol, rows_s = _cup_rows(p, CUP)
    assert vol[rows_s >= 0.029 + 2e-3].sum() == 0.0                                 # nothing up the dry wall
    _ledger_closes(p)


def test_a_rising_jet_hits_with_its_rising_speed():
    """WP15's review M4: a jet aimed upward at a plate's underside is still rising there; the impact velocity is the
    rising branch's v0 + g·t, not the falling one."""
    def build(b):
        slab_scene(slab_x=1.0, crate=(-1.0, 0.0))(b)
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5), hx=0.1, hy=0.1, hz=0.005, label="ceiling")
    up = Nozzle("up", edge="spout", pos=(0.0, 0.0, 0.3), axis=(0.0, 0.0, 1.0), exit_area=2e-5)
    p = _run(StubFilm(charts=(CHART,), nozzles=(up,), network=faucet_network()), build=build, periods=2)
    q = float(-p.supply.read()["discharge"][0, 1])
    v0 = q / 2e-5
    w_up = math.sqrt(v0 * v0 - 2 * 9.81 * (0.495 - 0.3))
    assert p.supply.hit["kind"].numpy()[0, 0] == 2                                   # the ceiling (a shape)
    assert p.supply.hit["w"].numpy()[0, 0][2] == pytest.approx(w_up, rel=2e-3)


def test_post_jump_flow_is_subcritical():
    """WP15's review m1: the ring's speed is at most Froude 0.3 of the ring depth (dry slab: the depth floor)."""
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network()), periods=2)
    from alloy.plugins.surface_flow import supply_deposit as D
    q = float(-p.supply.read()["discharge"][0, 1])
    r_j = 0.289 * q ** 0.75 * (RHO / (1e-6 * 0.0728)) ** 0.25
    h_fr = (q / (2 * math.pi * r_j * 0.3 * math.sqrt(9.81))) ** (2 / 3)
    u = q / (2 * math.pi * r_j * h_fr)
    assert u / math.sqrt(9.81 * h_fr) == pytest.approx(0.3, rel=1e-6)
    assert D.N_BANDS == 48
    # the ring's pending momentum, applied with its volume at the last step: ρ·dv·u_post (dv the volume it entered)
    mom = np.linalg.norm(p.last_mom.numpy()[0], axis=1)
    dv = p.last_acc.numpy()[0]
    ring = mom > 0
    assert ring.any() and np.all(dv[ring] > 0)
    assert np.allclose(mom[ring] / (RHO * dv[ring]), u, rtol=1e-3, atol=0.0)     # the last exchange's q (float32)


def test_a_jet_into_a_pool_carries_its_species():
    """WP15's review m3: soapy line water landing in the sink's pool brings its soap with it."""
    net = faucet_network(0.0, pool=True)
    net["nodes"][0]["species"] = (10.0,)
    net["edges"][0]["species"] = (10.0,)                                     # the mains carry 10 kg/m³
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=net), build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)))
    f = p.supply.fluid
    c = f.concentration()[:, 1, 0]
    assert np.allclose(c, 10.0, rtol=1e-9, atol=0.0)
    assert np.allclose(f.species_in_total.numpy()[:, 1, 0], 10.0 * p.supply.read()["to_tank"], rtol=1e-9, atol=0.0)


def test_drain_openings_are_clamped_and_a_part_open_free_drain_takes_its_share():
    from alloy.plugins.surface_flow import supply_kernels as K

    drain = Drain("grate", chart=0, center=(0.0, 0.0, 0.0), radius=0.15, to="sink")
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), drains=(drain,), network=faucet_network()), periods=2)
    s = p.supply
    h = np.full(p.h.shape, 1e-3)
    p.h.assign(h)
    for value, want in ((-1.0, 0.0), (0.5, 0.5 * 1e-3 / p.dt), (2.0, 1e30)):
        s.dopen.assign(np.full(s.dopen.shape, value, np.float32))
        wp.launch(K.drain_capacity, dim=2, inputs=[wp.float64(p.dt), s.dstart, s.dcell, s.dkind, s.dcd, s.dopen, p.h,
                                                   p.tables.area, wp.float64(9.81)], outputs=[p.drain_cap],
                  device="cpu")
        cap = p.drain_cap.numpy()[0][p.drain_of.numpy() == 0]
        assert np.allclose(cap, want, rtol=1e-12, atol=0.0), (value, cap[:3])


def test_joint_faucet_film_drain_tank_waste_books_every_drop():
    """The phase-1 joint case on WP15's real film: mains -> valve -> faucet -> the slab's film -> a drain disc -> the
    sink tank -> its waste orifice. The film's ledger closes (with the sweep), the drains equal the tank's inflow, and
    every m³ the mains gave is in the line, the film, the tank or out of the waste, to 1e-8."""
    from alloy.plugins.surface_flow.plugin import SurfaceFlow

    net = faucet_network()
    net["edges"].append({"name": "waste", "kind": "orifice", "a": "sink", "reservoir": P_ATM, "cd_area": 2e-5,
                         "dp_lam": 1.0})
    chart = PlaneChart("slab", cells=(40, 40), edges=("open", "open", "open", "open"))
    drain = Drain("grate", chart=0, center=(0.12, 0.0, 0.0), radius=0.03, to="sink")
    flow = SurfaceFlow(charts=(chart,), material="water", nozzles=(FAUCET,), drains=(drain,), network=net)
    run = run_scene(TestScene(build=slab_scene(), ground=True, worlds=2, periods=25), [flow], "cpu")
    assert run.error is None, run.error
    p = run.sim.plugins[flow.label]
    r, s = p.read(), p.supply.read()
    assert np.all(s["drained"] > 0) and np.all(s["lost"] == 0)
    assert np.allclose(r["ledger"], 0.0, atol=1e-15)
    assert np.allclose(s["inflow_total"][:, 1], r["drained_total_volume"], rtol=1e-12, atol=0.0)
    in_flight = -s["discharge"][:, 1] * 1.25e-3
    held = (s["amount"][:, 0] - 1e-4) + r["film_volume"] + r["spilled_volume"] + r["evaporated_volume"] + \
        s["amount"][:, 1] - s["edge_ledger"][:, 2] + in_flight
    assert np.allclose(held, s["edge_ledger"][:, 0], rtol=1e-8, atol=0.0)
    assert np.abs(r["u"]).max() < 0.5                                              # under u_max, far under u_cap


@pytest.mark.parametrize("half,on", [(0.2, 1.0), (0.03, 0.0)])
def test_a_jets_tangential_push_follows_its_sheet(half, on):
    """WP15's follow-up (1): a 45° jet on a plate body. When the jump stands on the plate, the sheet's friction hands
    the body all of ρ·q·w; on a plate smaller than R_J the sheet runs off with its tangential momentum, and the body
    keeps only the normal part ρ·q·(w·n)·n."""
    def build(b):
        plate = b.add_body(xform=body_xform(0.0, 0.0, 0.01), label="plate")
        b.add_shape_box(plate, hx=half, hy=half, hz=0.01, label="plate_box")
    hose = Nozzle("hose", edge="spout", pos=(-0.15, 0.0, 0.17), axis=(1.0, 0.0, -1.0), exit_area=1e-5)
    p = StubFilm(charts=(PlaneChart("plate_box", cells=(int(round(2 * half / 0.01)),) * 2),), nozzles=(hose,),
                 network=faucet_network())
    run = run_scene(TestScene(build=build, ground=True, worlds=2, periods=3), [p], "cpu")
    assert run.error is None, run.error
    p = run.sim.plugins[p.label]
    assert p.supply.hit["on"].numpy()[0, 0] == on
    q = float(-p.supply.read()["discharge"][0, 1])
    w = p.supply.hit["w"].numpy()[0, 0].astype(np.float64)
    f = run.probe(run.sim.state.body_q.numpy())
    body = int(p.ctx.bodies("plate").numpy()[0])
    want = RHO * q * np.array([on * w[0], on * w[1], w[2]])
    assert np.allclose(f[body, :3], want, rtol=2e-3, atol=1e-3 * abs(want[2])), (f[body, :3], want)   # resting tilt


@pytest.mark.compile_heavy
def test_surface_flow_with_a_supply_passes_the_harness_hooks_on_cpu():
    """SurfaceFlow with a faucet, a drain and a draining tank: contract, crash and the hook checks (determinism,
    identical and isolated worlds, graph) with the supply in. The film's own laws run without a supply (WP15)."""
    import alloy.plugins as ap
    from alloy.plugins.surface_flow.plugin import SurfaceFlow

    net = faucet_network()
    net["edges"].append({"name": "waste", "kind": "orifice", "a": "sink", "reservoir": P_ATM, "cd_area": 2e-5,
                         "dp_lam": 1.0})
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50), edges=("open",) * 4),), material="water",
                       nozzles=(Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=1e-4),),
                       drains=(Drain("grate", chart=0, center=(0.07, 0.0, 0.0), radius=0.02, to="sink"),),
                       network=net)
    report = ap.check(flow, device="cpu", quick=True, laws=False)
    assert report.passed, report.summary()


@pytest.mark.parametrize("cells", [40, 160])
def test_the_impact_pressure_spreads_over_the_jets_footprint(cells):
    """WP9's ask: the faucet's impact pressure is ½·ρ·w² over its footprint π·a² (a from continuity at the impact
    speed, about 3.6 mm): spread evenly over the cells within a of the hit on 2.5 mm cells, on the hit cell alone on
    1 cm cells (a footprint smaller than a cell); its integral is the same, ½·ρ·w·q, either way. WP15's
    ``impact_share``: Σ share·A is the footprint, and impact_p / share is ½·ρ·w² on every cell it hits."""
    chart = PlaneChart("slab", cells=(cells, cells), edges=("wall", "open", "wall", "wall"))
    p = _run(StubFilm(charts=(chart,), nozzles=(FAUCET,), network=faucet_network()), periods=3)
    s = p.supply.jet.numpy()[0]
    q, w, a = float(s["q"]), float(s["speed"]), float(s["a"])
    imp, share = p.impact_p.numpy()[0], p.impact_share.numpy()[0]
    area = p.tables.area_np
    assert float((imp * area).sum()) == pytest.approx(0.5 * RHO * w * q, rel=1e-5)
    on = imp > 0.0
    assert ((share > 0.0) == on).all() and float((share * area).sum()) == pytest.approx(math.pi * a * a, rel=1e-5)
    assert np.allclose(imp[on] / share[on], 0.5 * RHO * w * w, rtol=1e-5, atol=0.0)
    d = np.linalg.norm(p.tables.cpos.numpy() - np.asarray(s["p"]), axis=1)
    if cells == 40:
        assert on.sum() == 1 and d[on][0] < 0.0071                        # the hit cell (1 cm cells)
    else:
        assert float(s["r_fp"]) == pytest.approx(a, rel=1e-4)              # straight down: the footprint is π·a²
        assert on.sum() >= 4 and (d[on] <= a * (1 + 1e-5)).all() and on[d < 0.999 * a].all()
        assert np.allclose(imp[on], imp[on][0], rtol=1e-5, atol=0.0)
        assert imp[on][0] == pytest.approx(0.5 * RHO * w * w * math.pi * a * a / area[on].sum(), rel=1e-5)


def _two_lines() -> dict:
    """Two independent faucets (mains -> valve -> line -> spout each), so one nozzle's flow does not move the other's."""
    net = faucet_network()
    for k in ("1", "2"):
        net["nodes"].append({"name": "line" + k, "medium": "liquid", "volume": 1e-4})
        net["edges"].append({**net["edges"][0], "name": "valve" + k, "a": "line" + k})
        net["edges"].append({**net["edges"][1], "name": "spout" + k, "a": "line" + k})
    return net


def test_a_spray_and_a_jet_on_one_cell_add_their_impact_fields():
    """WP15's sf9 review: a spray's impact share is its droplet-struck fraction 2·rate/w_n, so impact_p/share is the
    droplets' ½·ρ·w_n² (between its rays' least and largest, a weighted mean) as for a jet, and a soil law
    k_i·max(impact_p/σ - share, 0) does not lose a jet's erosion to a spray on its cell: on every cell the two
    together give the sum of each alone (independent lines)."""
    jet = Nozzle("faucet", edge="spout1", pos=(0.0, 0.0, 0.3), exit_area=1e-4)
    spray = Nozzle("spray", edge="spout2", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="spray",
                   angle=math.radians(20.0), rays=16)

    def fields(*nozzles):
        p = _run(StubFilm(charts=(CHART,), nozzles=nozzles, network=_two_lines()), periods=2)
        return p, p.impact_p.numpy(), p.impact_share.numpy()
    pj, ip_j, sh_j = fields(jet)
    ps, ip_s, sh_s = fields(spray)
    _, ip, sh = fields(jet, spray)
    on = sh_s > 0.0
    assert on.sum(axis=1).min() > 4 and ((sh_j > 0.0) & on).any()                   # they share the hit cell
    wz = np.abs(ps.supply.hit["w"].numpy()[:, :, 2])[ps.supply.hit["kind"].numpy() == 1]
    ratio = ip_s[on] / sh_s[on]
    assert ratio.min() >= 0.5 * RHO * wz.min() ** 2 * (1 - 1e-5) and ratio.max() <= 0.5 * RHO * wz.max() ** 2 * (1 + 1e-5)
    assert np.allclose(ip, ip_j + ip_s, rtol=1e-6, atol=0.0) and np.allclose(sh, sh_j + sh_s, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("rays", [16, 32])
def test_a_fan_lays_an_even_line_whatever_its_ray_count(rays):
    """WP15's sf9 review: a fan's footprints span 2 ray spacings along its line (1.5 still striped it by 10 %), so a 60°
    fan 0.28 m above 2 mm cells lays a line whose across-summed deposit, over the inner 60 % of its length and
    divided by its smooth (quartic) trend, ripples within 4 % (2 % measured) at 16 rays as at 32."""
    noz = Nozzle("f", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="fan", angle=math.radians(60.0),
                 rays=rays, fan=(1.0, 0.0, 0.0))
    p = _run(StubFilm(charts=(PlaneChart("slab", cells=(200, 200)),), nozzles=(noz,), network=faucet_network()),
             periods=1, worlds=1)
    h = p.h.numpy()[0].reshape(200, 200)
    xs = p.tables.cpos.numpy()[:, 0].reshape(200, 200)
    line, x = (h.sum(axis=1), xs[:, 0]) if np.ptp(xs[:, 0]) > 0.0 else (h.sum(axis=0), xs[0])
    core = np.abs(x) < 0.6 * 0.28 * math.tan(math.radians(30.0))
    ripple = line[core] / np.polyval(np.polyfit(x[core], line[core], 4), x[core])
    assert ripple.max() / ripple.min() < 1.04


@pytest.mark.parametrize("rays", [64, 256])
def test_a_cone_sprays_an_even_coat_whatever_its_ray_count(rays):
    """WP15 (SC9's clear coat): ray footprints that only tile the cone striped a coat by 29 % at 64 rays. Each
    footprint now spans 1.5 ray spacings (its flow normalised over it), so a 30° cone 0.28 m above 2 mm cells lays a
    coat even to 6 % over the inner 60 % of its radius at 64 rays as at 256 (it was 62 % at both)."""
    noz = Nozzle("s", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="cone", angle=math.radians(30.0),
                 rays=rays)
    p = _run(StubFilm(charts=(PlaneChart("slab", cells=(200, 200)),), nozzles=(noz,), network=faucet_network()),
             periods=1, worlds=1)
    h = p.h.numpy()[0]
    core = np.linalg.norm(p.tables.cpos.numpy()[:, :2], axis=1) < 0.6 * 0.28 * math.tan(math.radians(15.0))
    assert h[core].min() > 0.0 and h[core].std() / h[core].mean() < 0.06
