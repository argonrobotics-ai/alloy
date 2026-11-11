# SPDX-License-Identifier: Apache-2.0
"""Surface-flow supply side (DC5) on a stub film (``alloy.plugins.surface_flow.supply_stub``: the interface arrays,
no flow): nozzles, pools, drains and the cross ledger (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Drain, Nozzle
from alloy.plugins.surface_flow.supply_stub import (CHART, FAUCET, RHO, StubFilm, faucet_network, ledger_sides,
                                                    run_stub, slab_scene)


# a tank's volume is κ·(P - P_ref) with P absolute (~1e5 Pa): a head of a few Pa keeps ~1e-10 of its rounding
TANK_RTOL = 1e-8
LEVEL_RTOL = 1.5e-7       # tank_level is float32: floor + V/A rounded twice (1.5 ulp; one ulp is 9.9e-8 at 0.075 m)


def _run(plugin: StubFilm, periods: int = 20, build=None, worlds: int = 2) -> StubFilm:
    return run_stub(plugin, periods, build, worlds)


def _ledger_closes(p: StubFilm) -> None:
    lhs, rhs, spout, sent = ledger_sides(p)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=1e-18), (lhs, rhs)
    assert np.allclose(spout, sent, rtol=1e-12, atol=0.0), (spout, sent)


def test_jet_fills_a_chart_on_its_jump_ring_and_the_ledger_closes():
    p = StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network())
    p = _run(p)
    r = p.supply.read()
    assert np.all(r["sent"] > 0) and np.all(r["lost"] == 0)
    h = p.h.numpy()[0].reshape(40, 40)
    q = -r["discharge"][0, 1]
    assert 5e-5 < q < 2e-4, q
    # the water sits on a ring around the hit (the stub film does not flow), none at the centre
    xs = (np.arange(40) + 0.5) * 0.01 - 0.2
    rr = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    wet = h > 1e-3 * h.max()
    assert h[rr < 0.02].sum() == 0.0
    r_mean = float((rr * h).sum() / h.sum())
    r_j = 0.289 * q ** 0.75 * (RHO / (1e-6 * 0.0728)) ** 0.25
    assert abs(r_mean - r_j) < 0.012, (r_mean, r_j)
    assert wet.sum() > 20
    _ledger_closes(p)


def _h(p: StubFilm, w: int = 0) -> np.ndarray:
    return p.h.numpy()[w].reshape(40, 40)


def test_a_jet_into_a_pool_fills_its_tank():
    """The faucet over the sink's free surface (the slab is off to the side): every drop lands in the tank."""
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network(0.004, pool=True)),
             build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)))
    r = p.supply.read()
    assert np.all(r["to_tank"] > 0) and not p.h.numpy().any()
    assert np.allclose(r["amount"][:, 1], 0.004 + r["to_tank"], rtol=1e-10, atol=0.0)  # the tank's inflow
    assert np.allclose(r["level"][:, 0], 0.05 + r["amount"][:, 1] / 0.16, rtol=LEVEL_RTOL, atol=0.0)
    _ledger_closes(p)


def test_a_pools_level_is_set_at_the_build():
    """WP9's ask: ``tank_level`` holds a box pool's surface from the build on (a lumped stage reads it in the first
    exchange, before the supply's first ``after_film``)."""
    p = StubFilm(charts=(CHART,), network=faucet_network(0.008, pool=True))
    build_simulation(TestScene(build=slab_scene(), ground=True, worlds=2), [p], "cpu")
    assert np.allclose(p.supply.level.numpy()[:, 0], 0.05 + 0.008 / 0.16, rtol=LEVEL_RTOL, atol=0.0)


def test_an_occluder_takes_the_jet_and_its_momentum():
    """A lid over the slab catches the faucet: booked as caught, all of it on the lid's occluder slot (SOFT-3's ask
    B: supply_caught_by [W, occluders], slots named by supply.occluders), none lost, the slab dry."""
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network()), build=slab_scene(True), periods=5)
    r = p.supply.read()
    lid = p.supply.occluders.index("lid")
    assert np.all(r["caught"] > 0) and np.all(r["sent"] == r["caught"]) and np.all(r["lost"] == 0)
    assert np.array_equal(r["caught_by"][:, lid], r["caught"]) and r["caught_by"].sum(axis=1).tolist() == \
        r["caught"].tolist() and not p.h.numpy().any()
    _ledger_closes(p)


def test_the_struck_body_feels_rho_q_w():
    """The crate under the faucet (no chart below it): force ρ·Q·w at the hit, the impact velocity from the fall."""
    p = StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network())
    scene = TestScene(build=slab_scene(slab_x=1.0, crate=(0.01, 0.0)), ground=True, worlds=2, periods=5)
    run = run_scene(scene, [p], "cpu")
    assert run.error is None, run.error
    p = run.sim.plugins[p.label]
    q = float(-p.supply.read()["discharge"][0, 1])
    v0 = q / FAUCET.exit_area
    w = math.sqrt(v0 * v0 + 2 * 9.81 * (0.3 - 0.04))
    f = run.probe(run.sim.state.body_q.numpy())
    crate = int(p.ctx.bodies("crate").numpy()[0])
    assert f[crate, 2] == pytest.approx(-RHO * q * w, rel=2e-3)
    assert f[crate, 4] == pytest.approx(-0.01 * RHO * q * w, rel=0.02)            # off the COM by 10 mm in x


def test_a_drain_moves_the_film_into_its_tank_and_a_stopper_holds_it():
    drain = Drain("grate", chart=0, center=(0.0, 0.0, 0.0), radius=0.15, to="sink")
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), drains=(drain,), network=faucet_network()))
    r = p.supply.read()
    assert np.all(r["drained"] > 0)
    assert np.allclose(r["inflow_total"][:, 1], r["drained"], rtol=1e-12, atol=0.0)
    assert np.allclose(r["amount"][:, 1], r["drained"], rtol=TANK_RTOL, atol=0.0)
    inside = np.hypot(*np.meshgrid((np.arange(40) + 0.5) * 0.01 - 0.2, (np.arange(40) + 0.5) * 0.01 - 0.2,
                                   indexing="ij")) <= 0.15
    assert not _h(p)[inside].any()                                                # a free drain takes everything
    _ledger_closes(p)
    stopped = StubFilm(charts=(CHART,), nozzles=(FAUCET,), drains=(drain,), network=faucet_network())
    scene = TestScene(build=slab_scene(), ground=True, worlds=2, periods=10,
                      init=lambda sim: sim.plugins[stopped.label].set_input("drain_open", 0.0))
    run = run_scene(scene, [stopped], "cpu")
    assert run.error is None, run.error
    r = run.sim.plugins[stopped.label].supply.read()
    assert np.all(r["drained"] == 0) and np.all(r["amount"][:, 1] == 0)


def test_an_orifice_drain_takes_cd_a_sqrt_2gh():
    from alloy.plugins.surface_flow import supply_kernels as K

    drain = Drain("hole", chart=0, center=(0.0, 0.0, 0.0), radius=0.15, to="sink", cd_area=1e-5)
    p = _run(StubFilm(charts=(CHART,), nozzles=(FAUCET,), drains=(drain,), network=faucet_network()), periods=10)
    s = p.supply
    wp.launch(K.drain_capacity, dim=2, inputs=[wp.float64(p.dt), s.dstart, s.dcell, s.dkind, s.dcd, s.dopen, p.h,
                                               p.tables.area, wp.float64(9.81)], outputs=[p.drain_cap], device="cpu")
    cap = p.drain_cap.numpy()[0]
    cells = p.drain_of.numpy() == 0
    area = p.tables.area_np[cells]
    hbar = float((p.h.numpy()[0][cells] * area).sum() / area.sum())
    assert hbar > 0
    cd_area = float(np.float32(1e-5))                                             # the drain's Cd·A is float32
    assert np.allclose(cap[cells], cd_area * math.sqrt(2 * 9.81 * hbar) / area.sum(), rtol=1e-12, atol=0.0)
    assert np.all(cap[~cells] == 0)
    assert np.all(p.supply.read()["drained"] > 0)


def test_a_fan_lays_a_line_as_wide_as_its_angle():
    fan = Nozzle("fan", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=1e-5, pattern="fan", angle=math.radians(40),
                 rays=16, fan=(1.0, 0.0, 0.0))
    p = _run(StubFilm(charts=(CHART,), nozzles=(fan,), network=faucet_network()), periods=10)
    h = _h(p)
    xs = (np.arange(40) + 0.5) * 0.01 - 0.2
    wet_x = xs[h.sum(axis=1) > 0]
    wet_y = xs[h.sum(axis=0) > 0]
    half = 0.28 * math.tan(math.radians(20))                                       # nearly straight at 10 m/s
    assert abs(wet_x.max() - half) < 0.02 and abs(wet_x.min() + half) < 0.02, (wet_x.min(), wet_x.max(), half)
    assert wet_y.max() - wet_y.min() < 0.05
    _ledger_closes(p)


def test_a_spray_misses_count_as_overspray_and_its_inefficiency_as_mist():
    spray = Nozzle("can", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=1e-5, pattern="spray",
                   angle=math.radians(170), rays=64, efficiency=0.6)
    p = _run(StubFilm(charts=(CHART,), nozzles=(spray,), network=faucet_network()), periods=5)
    r = p.supply.read()
    assert np.all(r["overspray"] > 0.0) and np.all(r["lost"] == 0)                  # rays past the slab
    mist = 1.0 - float(np.float32(0.6))                                           # 60 % transfer efficiency (float32)
    assert np.allclose(r["mist"], mist * r["sent"], rtol=1e-12, atol=0.0)
    _ledger_closes(p)


def test_a_spray_footprint_follows_its_gaussian():
    """An aerosol 0.28 m above the slab (60° cone, σ = 15°): the share deposited within angle α of the axis matches
    ∫ exp(-θ²/2σ²)·sin θ dθ over the cap, to 5 % of the total."""
    angle, n = math.radians(60), 256
    spray = Nozzle("can", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=4e-6, pattern="spray", angle=angle, rays=n)
    p = _run(StubFilm(charts=(PlaneChart("slab", cells=(80, 80)),), nozzles=(spray,), network=faucet_network()),
             periods=4)
    h = p.h.numpy()[0].reshape(80, 80)
    xs = (np.arange(80) + 0.5) * 0.005 - 0.2
    theta = np.arctan(np.hypot(*np.meshgrid(xs, xs, indexing="ij")) / 0.28)
    sigma = 0.25 * angle
    grid = np.linspace(0.0, 0.5 * angle, 2001)
    dens = np.exp(-0.5 * (grid / sigma) ** 2) * np.sin(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    for a in np.radians([7.5, 15.0, 22.5]):
        got = h[theta <= a].sum() / h.sum()
        want = np.interp(a, grid, cum) / cum[-1]
        assert abs(got - want) < 0.05, (np.degrees(a), got, want)
    _ledger_closes(p)
