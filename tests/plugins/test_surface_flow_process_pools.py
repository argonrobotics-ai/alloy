# SPDX-License-Identifier: Apache-2.0
"""Surface-flow soil under a pool's water (WP15's SCR for JOB3's rinse-garbage-can and WP8's pan; WP9 owns the law,
DC5 writes the submerged jets' shear). CPU, harness: a cell under a flooding pool's water has given its film to the
pool (h = 0) and no jet sheet reaches it, yet it is wet, so its soil soaks and the submerged jets' wall shear
(``pool_jet_tau``) erodes it; the removed soil joins the pool's tank, not the film."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.process import Soil
from alloy.plugins.surface_flow.supply import Pool

DT, N, NX, NY = 0.01, 50, 8, 8
SOIL = Soil(initial=0.2, sigma_dry=1000.0, sigma_wet=150.0, t_soak=0.2, k_e=1.0e-4, k_i=0.0, k_front=0.0,
            species="soil")
TAU = 400.0                                               # Pa of submerged-jet shear on the first two rows of cells


def _plate(b) -> None:
    b.add_shape_box(-1, xform=body_xform(1.0, 0.0, -0.005), hx=0.05, hy=0.05, hz=0.005, label="dry")   # off the pool
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.005), hx=0.05, hy=0.05, hz=0.005, label="plate")   # top z = 0
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(d, radius=0.01)


def _expected() -> tuple[float, float]:
    """The soil law's recurrence over N film steps from dry soil: the loss at each step's starting strength, then
    the exact soak; (hyd, removed under TAU)."""
    soak, hy, removed = math.exp(-DT / SOIL.t_soak), 0.0, 0.0
    for _ in range(N):
        sig = SOIL.sigma_dry * (1.0 - hy) * (1.0 - hy) + SOIL.sigma_wet
        removed += SOIL.k_e * max(TAU - sig, 0.0) * DT
        hy = 1.0 - (1.0 - hy) * soak
    return hy, removed


@pytest.mark.compile_heavy
def test_soil_under_a_pools_water_soaks_and_the_submerged_jets_erode_it_into_the_tank():
    """Two soiled plates, the second (chart 1) 50 mm under a flooding sink's water (DC5's flood=True: every cell
    flooded, its film the pool's), the first off the pool. Every flooded cell soaks by the exact law,
    1 - exp(-t/t_soak), though none holds film; the dry plate does not soak (fails when flooded cells are not wet, or
    the gate does not scan each chart's own pool box: WP15's two-chart law). Under the 400 Pa a submerged jet leaves
    on the first two rows, the soil goes by k_e·(τ - σ_c)+ at each step's starting strength (1e-12), booked as shear;
    elsewhere none goes. The removed soil joins the sink's tank (soil_to_pools = the tank's soil = removed, 1e-12;
    WP15's must-fix: it lay on the floor as dry residue before), none is left on the film; soil + removed = the start
    load (1e-14)."""
    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.15 * 0.04, "area": 0.04, "species": (0.0,)}],
           "edges": [], "species": ("soil",),
           "pools": [Pool("sink", floor=(0.0, 0.0, -0.1), half=(0.2, 0.2), flood=True)]}
    flow = SurfaceFlow(charts=(PlaneChart("dry", cells=(NX, NY)), PlaneChart("plate", cells=(NX, NY))),
                       material="water", max_stride=1, network=net, species=("soil",), process={"soil": SOIL})
    dry, wet = slice(0, NX * NY), slice(NX * NY, 2 * NX * NY)          # the charts' cells, in chart order

    def init(sim) -> None:
        tau = np.zeros(flow.pool_jet_tau.shape, np.float32)
        tau[:, wet.start:wet.start + 2 * NY] = TAU            # chart 1's rows i = 0, 1 (cells i·NY + j)
        flow.pool_jet_tau.assign(tau)
    sim = build_simulation(TestScene(build=_plate, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1,
                                     dt=DT, control_dt=DT), [flow], "cpu")
    for _ in range(N):
        sim.step_period()
    fl = flow.flooded.numpy()[0]
    assert flow.stride == 1 and (fl[wet] == 0).all() and (fl[dry] == -1).all() and flow.h.numpy().max() == 0.0
    hy, removed = _expected()
    d = flow.process.read()
    np.testing.assert_allclose(d["soil_hyd"][0, wet], hy, rtol=1e-12, atol=0.0)
    assert (d["soil_hyd"][0, dry] == 0.0).all() and (d["soil_removed"][0, dry] == 0.0).all()
    np.testing.assert_allclose(d["soil_removed"][0, wet][:2 * NY], removed, rtol=1e-12, atol=0.0)
    assert removed > 0.0 and (d["soil_removed"][0, wet][2 * NY:] == 0.0).all()
    np.testing.assert_allclose(d["soil_removed_by"][0, :, 1], d["soil_removed"][0], rtol=0.0, atol=0.0)
    assert d["soil_mass"][0] + d["soil_removed_mass"][0] == pytest.approx(d["soil_start_mass"][0], rel=1e-14, abs=0.0)
    gone = d["soil_removed_mass"][0]
    assert d["soil_to_pools_mass"][0] == pytest.approx(gone, rel=1e-12, abs=0.0)
    assert flow.supply.fluid.species.numpy()[0, 0, 0] == pytest.approx(gone, rel=1e-12, abs=0.0)
    assert flow.sp.mass.numpy()[0, :, 0].max() == 0.0 and flow.sp.src.numpy()[0, :, 0].max() == 0.0


@pytest.mark.compile_heavy
def test_a_pad_scrubbing_soil_under_a_pools_water_wipes_at_the_wet_floor():
    """WP8's pan (NaN since ddfba5810): a guided pad 1 mm into the soil of a plate flooded by a sink's water slides at
    5 cm/s. The cells hold no film (h = 0, it is the pool's) and the pad's gap is <= 0, so the viscous wipe's
    μ|v|/max(gap, h) divided by zero. The gap now floors at h_wet, as the film's wall shear does: per exchange each cell
    under the pad loses (k_abr·cover·press·|v|·σ_ref + k_wipe·cover·μ|v|/max(gap, h, h_wet))/σ_c·Δt at its starting
    strength (float32 tool fields, 1e-6), booked as the tool's; every flooded cell soaks; all finite; ledger exact."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.surface_flow.host_api import H_WET  # noqa: PLC0415
    from tests.plugins.test_surface_flow_tools import blade, scene  # noqa: PLC0415
    soil = Soil(t_soak=0.2, k_wipe=1.0)
    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.15 * 0.04, "area": 0.04}], "edges": [],
           "pools": [Pool("sink", floor=(0.0, 0.0, -0.1), half=(0.2, 0.2), flood=True)]}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material="water", max_depth=2.0e-3,
                       max_stride=1, network=net, tools=(blade(stiffness=1.0e6),), process={"soil": soil})

    def init(sim) -> None:
        sim.state.joint_qd.assign(np.full(1, 0.05))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    dt = 2.0e-3
    sim = build_simulation(TestScene(build=scene(-1.0e-3, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                     init=init, ground=False, gravity=(0.0, 0.0, 0.0), worlds=1, dt=dt,
                                     control_dt=dt), [flow], "cpu")
    sim.step_period()                                                    # wakes the tiles under the pad
    mu = float(flow.mat.viscosity)
    for _ in range(20):
        d = flow.process.read()
        before, hy = d["soil_removed_by"][0][:, 3].copy(), d["soil_hyd"][0].copy()
        sim.step_period()
        tf = flow.tf
        cov, press = tf.cover.numpy()[0].astype(np.float64), tf.press.numpy()[0].astype(np.float64)
        v = np.linalg.norm(tf.vel.numpy()[0].astype(np.float64), axis=1)
        gap = np.maximum(np.maximum(tf.ceil.numpy()[0].astype(np.float64), flow.h.numpy()[0]), H_WET)
        sig = soil.sigma_dry * (1.0 - hy) ** 2 + soil.sigma_wet
        expect = (soil.k_abr * cov * press * v * soil.sigma_ref + soil.k_wipe * cov * mu * v / gap) / sig * dt
        assert (expect > 0.0).sum() >= 8 and tf.ceil.numpy()[0][expect > 0.0].max() <= 0.0
        np.testing.assert_allclose(flow.process.read()["soil_removed_by"][0][:, 3] - before, expect, rtol=1e-6,
                                   atol=0.0)
    d = flow.process.read()
    assert (flow.flooded.numpy()[0] == 0).all() and flow.h.numpy().max() == 0.0
    assert all(np.isfinite(d[k]).all() for k in ("soil", "soil_hyd", "soil_removed", "soil_removed_by"))
    assert d["soil_hyd"][0].min() > 0.0
    assert d["soil_mass"][0] + d["soil_removed_mass"][0] == pytest.approx(d["soil_start_mass"][0], rel=1e-14, abs=0.0)


ADS = {"species": "soap", "K": 20.0, "k_d": 2.0, "capacity": 3.0e-7}     # Langmuir: m³/kg, 1/s, kg/m² (k_d fast)
V_TANK = 0.15 * 0.04                                                      # m³: the sink's water


def _soapy_sink(flood_level: float = 0.15, **kw) -> SurfaceFlow:
    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": flood_level * 0.04, "area": 0.04, "species": (0.0,)}],
           "edges": [], "species": ("soap",),
           "pools": [Pool("sink", floor=(0.0, 0.0, -0.1), half=(0.2, 0.2), flood=True)]}
    return SurfaceFlow(charts=(PlaneChart("dry", cells=(NX, NY)), PlaneChart("plate", cells=(NX, NY))),
                       material="water", max_stride=1, network=net, species=("soap",), adsorption=ADS, **kw)


def _langmuir(g0: float, m0: float, area: float, n: int) -> tuple[float, float]:
    """The flooded exchange's recurrence over n exchanges for a plate of ``area`` loaded to g0 in a tank holding m0 kg
    of soap: C read at each exchange's start, the exact step toward Langmuir's rest at it; (Γ_n, tank kg)."""
    g, m = g0, m0
    for _ in range(n):
        kc = ADS["K"] * m / V_TANK
        rest = ADS["capacity"] * kc / (1.0 + kc)
        g1 = rest + (g - rest) * math.exp(-ADS["k_d"] * (1.0 + kc) * DT)
        m += (g - g1) * area
        g = g1
    return g, m


@pytest.mark.compile_heavy
def test_a_wall_under_a_pools_water_sheds_its_soap_into_the_tank():
    """SCR-WP8-10 (WP8's pan: dunks in clean rinse water never shed the walls' soap). Two worlds, a plate under a
    flooding sink's water loaded to 2.5e-7 kg/m² (world 0) and 1.5e-7 (world 1) of soap, the sink holding none (world
    0) and 5e-4 kg/m³ (world 1: Langmuir's rest 0.9e-7 < the load), a second plate off the pool loaded alike. Each
    flooded cell's load follows the exact exchange toward Langmuir's rest at the tank's concentration, read once per
    exchange, and the tank gains what the walls give (1e-12); with K·C << 1 that is Γ0·exp(-k_d·t) (world 0); the
    dry plate keeps its load (WP15's film exchange needs film); adsorbed_to_pools = what left the walls; the film's
    species ledger, which counts the load, closes; load + tank conserved (1e-12)."""
    flow = _soapy_sink()
    g0, c0 = np.array([2.5e-7, 1.5e-7]), np.array([0.0, 5.0e-4])
    dry, wet = slice(0, NX * NY), slice(NX * NY, 2 * NX * NY)

    def init(sim) -> None:
        flow.set_adsorbed(np.broadcast_to(g0[:, None, None], (2, NX, NY)), "soap", chart=0)
        flow.set_adsorbed(np.broadcast_to(g0[:, None, None], (2, NX, NY)), "soap", chart=1)
        sp = flow.supply.fluid.species.numpy()
        sp[:, 0, 0] = c0 * V_TANK
        flow.supply.fluid.species.assign(sp)
    sim = build_simulation(TestScene(build=_plate, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=2,
                                     dt=DT, control_dt=DT), [flow], "cpu")
    start = flow.read()["species_ledger"].copy()
    for _ in range(N):
        sim.step_period()
    r = flow.read()
    A = flow.tables.area_np
    load, tank = r["adsorbed"][..., 0], flow.supply.fluid.species.numpy()[:, 0, 0]
    assert (flow.flooded.numpy()[:, wet] == 0).all() and (flow.flooded.numpy()[:, dry] == -1).all()
    for w in range(2):
        g, m = _langmuir(g0[w], c0[w] * V_TANK, A[wet].sum(), N)
        np.testing.assert_allclose(load[w, wet], g, rtol=1e-12, atol=0.0)
        assert (load[w, dry] == g0[w]).all()
        assert tank[w] == pytest.approx(m, rel=1e-12, abs=0.0)
        assert r["adsorbed_to_pools"][w, 0] == pytest.approx((g0[w] - g) * A[wet].sum(), rel=1e-12, abs=0.0)
        assert (load[w, wet] * A[wet]).sum() + tank[w] == pytest.approx(
            g0[w] * A[wet].sum() + c0[w] * V_TANK, rel=1e-12, abs=0.0)
    assert load[0, wet].max() == pytest.approx(g0[0] * math.exp(-ADS["k_d"] * N * DT), rel=2e-3, abs=0.0)  # K·C<<1
    np.testing.assert_allclose(r["species_ledger"], start, rtol=1e-12, atol=0.0)


@pytest.mark.compile_heavy
def test_a_wall_above_the_pool_exchanges_with_its_film_bit_for_bit():
    """WP15's pin: with the sink's water below the plates (a 1 mm deep pool, nothing flooded), a film of 0.5 mm with
    soap on a loaded plate exchanges exactly as with no pool at all (WP15's film branch): loads, film and species
    bit for bit."""
    def run(flow: SurfaceFlow) -> dict:
        def init(sim) -> None:
            flow.set_depth(5.0e-4, chart=1)
            flow.set_species(5.0e-4 * 2.0e-3, "soap", chart=1)
            flow.set_adsorbed(1.0e-7, "soap", chart=1)
        sim = build_simulation(TestScene(build=_plate, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1,
                                         dt=DT, control_dt=DT), [flow], "cpu")
        for _ in range(10):
            sim.step_period()
        return {k: v.copy() for k, v in (("load", flow.adsorption.load.numpy()), ("h", flow.h.numpy()),
                                         ("mass", flow.sp.mass.numpy()))}
    pooled = _soapy_sink(flood_level=1.0e-3)
    alone = SurfaceFlow(charts=(PlaneChart("dry", cells=(NX, NY)), PlaneChart("plate", cells=(NX, NY))),
                        material="water", max_stride=1, species=("soap",), adsorption=ADS)
    a, b = run(pooled), run(alone)
    assert pooled.pool_adsorption is not None and (pooled.flooded.numpy() == -1).all()
    assert a["load"][0, NX * NY:].min() != 1.0e-7                       # the film branch did exchange
    for k in a:
        assert np.array_equal(a[k], b[k]), k


@pytest.mark.compile_heavy
def test_a_wall_left_by_its_pool_goes_back_to_its_film_and_the_books_close():
    """WP15's ask: a horizontal plate (top z = 0.1, 40° contact angle) loaded to 2.5e-7 kg/m² of soap sits under a
    free basin's pool (surface 0.15) that falls past it at 0.10 s. Under the water its load sheds into the tank;
    once out it keeps a puddle (DC5's static puddle) and WP15's film exchange takes over. Every step: film + load +
    tank soap = the start, and the film's species ledger (with the load and adsorbed_to_pools) stays at its start
    (1e-12)."""
    level0 = 0.15
    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.25 * 0.16, "area": 0.16, "species": (0.0,)}],
           "edges": [], "species": ("soap",),
           "pools": [Pool("sink", floor=(0.0, 0.0, -0.3), half=(0.2, 0.2), body="basin", flood=True)]}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, NY), contact_angle=40.0),), material="water",
                       max_stride=1, network=net, species=("soap",), adsorption=ADS)

    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.095), hx=0.05, hy=0.05, hz=0.005, label="plate")
        basin = b.add_body(xform=body_xform(0.0, 0.0, level0 + 0.05), label="basin")
        b.add_shape_sphere(basin, radius=0.01)

    def total() -> np.ndarray:
        r = flow.read()
        return r["species_mass"][:, 0] + r["adsorbed_mass"][:, 0] + flow.supply.fluid.species.numpy()[:, 0, 0]
    sim = build_simulation(TestScene(build=build, init=lambda sim: flow.set_adsorbed(2.5e-7, "soap"), ground=False,
                                     gravity=(0.0, 0.0, -9.81), worlds=1, dt=DT, control_dt=DT), [flow], "cpu")
    t0, led0 = total(), flow.read()["species_ledger"].copy()
    phases = {"under": None, "out": None}
    for _ in range(20):
        sim.step_period()
        np.testing.assert_allclose(total(), t0, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(flow.read()["species_ledger"], led0, rtol=1e-12, atol=0.0)
        fl = flow.flooded.numpy()[0]
        if (fl == 0).all():
            phases["under"] = flow.adsorption.load.numpy()[0, :, 0].copy()
        elif (fl == -1).all() and phases["out"] is None:
            phases["out"] = flow.adsorption.load.numpy()[0, :, 0].copy()
    end = flow.adsorption.load.numpy()[0, :, 0]
    assert phases["under"] is not None and phases["out"] is not None
    assert phases["under"].max() < 2.5e-7                               # it shed under the water
    assert flow.h.numpy().min() > 1.0e-4                                # out, it keeps its puddle
    assert np.abs(end - phases["out"]).max() > 0.0                      # and the film exchange runs on it
    assert flow.read()["adsorbed_to_pools"][0, 0] > 0.0


@pytest.mark.compile_heavy
def test_a_soapy_pool_loads_a_clean_wall_to_langmuirs_rest():
    """SCR-WP8-10 part 2 (DC5's species take): a clean plate under a flooding sink at 2e-3 kg/m³ (world 0) and 1e-2
    (world 1) of soap takes it up by the exact exchange recurrence, the tank losing what the wall takes through
    DC5's pool_take_sp (1e-12, the tank's C falling as it gives), toward Langmuir's rest
    Γ_max·K·C/(1 + K·C) at the tank's last C (the gap under exp(-k_d·t)); load + tank and the ledger
    conserved (1e-12 of the soap)."""
    flow = _soapy_sink()
    c0 = np.array([2.0e-3, 1.0e-2])
    wet = slice(NX * NY, 2 * NX * NY)

    def init(sim) -> None:
        sp = flow.supply.fluid.species.numpy()
        sp[:, 0, 0] = c0 * V_TANK
        flow.supply.fluid.species.assign(sp)
    sim = build_simulation(TestScene(build=_plate, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=2,
                                     dt=DT, control_dt=DT), [flow], "cpu")
    start = flow.read()["species_ledger"].copy()
    for _ in range(N):
        sim.step_period()
    r = flow.read()
    A = flow.tables.area_np
    load, tank = r["adsorbed"][..., 0], flow.supply.fluid.species.numpy()[:, 0, 0]
    for w in range(2):
        g, m = _langmuir(0.0, c0[w] * V_TANK, A[wet].sum(), N)
        assert g > 0.0
        np.testing.assert_allclose(load[w, wet], g, rtol=1e-12, atol=0.0)
        assert tank[w] == pytest.approx(m, rel=1e-12, abs=0.0)
        assert r["adsorbed_to_pools"][w, 0] == pytest.approx(-g * A[wet].sum(), rel=1e-12, abs=0.0)
        kc = ADS["K"] * m / V_TANK
        rest = ADS["capacity"] * kc / (1.0 + kc)
        assert abs(g - rest) / rest < math.exp(-ADS["k_d"] * N * DT) * 1.01      # the gap falls at k_d(1 + K·C)
    assert (np.abs(r["species_ledger"][:, 0] - start[:, 0]) <= 1e-12 * c0 * V_TANK).all()    # it starts at 0 kg


@pytest.mark.compile_heavy
def test_walls_take_no_more_soap_than_their_pool_holds():
    """WP15's clamp: walls of a large capacity under a pool holding little soap (Γ_max 1e-3 kg/m², K 1e4 m³/kg, k_d
    100/s: one exchange would take 15x the tank's 6e-7 kg) take exactly what the tank holds, scaled alike over the
    cells: the tank ends at 0 (never below), the load is the tank's soap spread over the plate (1e-12), and DC5
    refuses at most the float64 rounding of the cells' sum (1e-12 of the tank's soap; 1.4e-15 measured)."""
    ads = {"species": "soap", "K": 1.0e4, "k_d": 100.0, "capacity": 1.0e-3}
    flow2 = _soapy_sink()
    flow2 = SurfaceFlow(charts=flow2.config.charts, material="water", max_stride=1, network=flow2.config.network,
                        species=("soap",), adsorption=ads)
    m0 = 1.0e-4 * V_TANK
    wet = slice(NX * NY, 2 * NX * NY)

    def init(sim) -> None:
        sp = flow2.supply.fluid.species.numpy()
        sp[:, 0, 0] = m0
        flow2.supply.fluid.species.assign(sp)
    sim = build_simulation(TestScene(build=_plate, init=init, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1,
                                     dt=DT, control_dt=DT), [flow2], "cpu")
    sim.step_period()
    r = flow2.read()
    A = flow2.tables.area_np
    tank = flow2.supply.fluid.species.numpy()[0, 0, 0]
    assert 0.0 <= tank <= 1e-12 * m0
    np.testing.assert_allclose(r["adsorbed"][0, wet, 0], m0 / A[wet].sum(), rtol=1e-12, atol=0.0)
    assert flow2.supply.pool_taken_sp.numpy()[0, 0, 0] == pytest.approx(m0, rel=1e-12, abs=0.0)
    assert 0.0 <= r["supply_species_refused"][0, 0] <= 1e-12 * m0       # the 64 scaled takes sum to m0 within ulps


def test_an_adsorbing_species_the_network_lacks_warns_once(caplog):
    """WP15's should-fix: a network that carries no soap would take the walls' soap into nothing (species away): the
    build warns once and names the species."""
    import logging  # noqa: PLC0415

    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": V_TANK, "area": 0.04, "species": (0.0,)}],
           "edges": [], "species": ("salt",), "pools": [Pool("sink", floor=(0.0, 0.0, -0.1), half=(0.2, 0.2), flood=True)]}
    flow = SurfaceFlow(charts=(PlaneChart("dry", cells=(NX, NY)), PlaneChart("plate", cells=(NX, NY))),
                       material="water", max_stride=1, network=net, species=("soap",), adsorption=ADS)
    with caplog.at_level(logging.WARNING, logger="alloy.plugins.surface_flow"):
        build_simulation(TestScene(build=_plate, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1, dt=DT,
                                   control_dt=DT), [flow], "cpu")
    assert sum("not carried by the supply's network" in m and "soap" in m for m in caplog.messages) == 1
