# SPDX-License-Identifier: Apache-2.0
"""Surface-flow process states carried by the film (design §4.8, §5; WP9): drying (F-12), laps, absorption (F-13), the
sanitizer clock and soil joining the film's species. CPU; flat plates with gravity along the normal, so the film
stands still and each cell can be checked against its closed form."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.process import Drying, Sanitizer, Soil, Substrate

G = 9.81
DT = 0.01


def _scene(init=None, periods: int = 100) -> TestScene:
    return TestScene(build=sl.plate_scene(0.1, 0.02), init=init, ground=False, gravity=(0.0, 0.0, -G),
                     periods=periods, worlds=1, dt=DT, control_dt=DT)


def _run(flow: SurfaceFlow, periods: int, init=None) -> None:
    run = run_scene(_scene(init, periods), [flow], "cpu")
    assert run.error is None, run.error


def _paint(process: dict, **config) -> SurfaceFlow:
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material="latex_paint", max_depth=1.0e-3,
                       species=("solids",), process=process, **config)


@pytest.mark.compile_heavy
def test_paint_dries_to_its_solids_f12():
    """F-12: only solvent leaves, so the dry film thickness is WFT·phi_0 (1e-6) and the film stops at solids/phi_m.
    Mid-way phi_s = phi_0·WFT/(WFT - E·t). hide where DFT ≥ hide_dft (a 100 µm coat hides, a 50 µm one does not)."""
    dr = Drying(k_m=0.6)                                              # E = 5.2 µm/s: dries in seconds
    wft = np.where(np.arange(20) < 10, 100.0e-6, 50.0e-6)[:, None] * np.ones((1, 4))

    def make():
        flow = _paint({"drying": dr})

        def init(sim):
            flow.set_depth(wft, chart=0)
            flow.set_species(dr.phi_0 * wft, "solids", chart=0)
        return flow, init
    flow, init = make()
    _run(flow, 300, init)                                             # 3 s
    p = flow.process.read()
    t = (300 // flow.stride) * flow.stride * DT                       # a slow film steps every `stride` exchanges
    expect = dr.phi_0 * wft / (wft - dr.rate * t)
    assert np.allclose(p["phi_s"][0].reshape(20, 4), expect, rtol=1e-5, atol=0.0)
    flow, init = make()
    _run(flow, 2000, init)                                            # 20 s: dry
    p = flow.process.read()
    dft = p["dft"][0].reshape(20, 4)
    assert np.allclose(dft, dr.phi_0 * wft, rtol=1e-6, atol=0.0)
    assert np.allclose(flow.depth(0)[0], dr.phi_0 * wft / dr.phi_m, rtol=1e-6, atol=0.0)
    assert p["hide"][0].reshape(20, 4)[:10].all() and not p["hide"][0].reshape(20, 4)[10:].any()


@pytest.mark.compile_heavy
def test_a_second_coat_past_the_wet_edge_is_a_lap():
    """Laps (WP15 review; SOFT-3's visible threshold). 100 µm coats at E = 5.2 µm/s: rows 0-4 repainted at 1 s, still
    wet (phi_s 0.44 < phi_we 0.57): no lap. Rows 10-14 repainted at 6 s, past the wet edge but not dry (phi_s 0.61 <
    phi_m 0.64): one lap. Rows 15-17 got a 5 µm fringe (2.1 µm solids, under lap_visible 4 µm) and are repainted at
    6 s: no lap. Rows 5-9 painted once and rows 18-19 first painted at 6 s: none."""
    dr = Drying(k_m=0.6)                                              # the wet edge at 5.0 s, dry at 6.6 s
    flow = _paint({"drying": dr})
    sim = build_simulation(_scene(), [flow], "cpu")
    src = np.zeros((20, 4))

    def paint(rates: dict, periods: int):                             # 2 periods at 5 mm/s: 100 µm
        src[:] = 0.0
        for rows, rate in rates.items():
            src[list(rows)] = rate
        flow.set_source(src, chart=0, species={"solids": dr.phi_0})
        for _ in range(periods):
            sim.step_period()
    paint({range(15): 5.0e-3, range(15, 18): 2.5e-4}, 2)
    paint({}, 98)                                                     # to 1 s
    paint({range(5): 5.0e-3}, 2)
    paint({}, 498)                                                    # to 6 s
    paint({range(10, 20): 5.0e-3}, 2)
    paint({}, 40)                                                     # longer than any film stride
    laps = flow.process.read()["laps"][0].reshape(20, 4)
    assert (laps[:10] == 0).all() and (laps[10:15] == 1).all() and (laps[15:] == 0).all()


@pytest.mark.compile_heavy
def test_mist_dust_then_a_wet_coat_is_no_lap():
    """SOFT-3's review: mist dust (6 µm of pure solids, over lap_visible) on dry rows 0-4 was read as an earlier coat
    dried to phi_m, so the next coat counted a lap. Solids that were never a wet film seed no lap: a 100 µm coat
    (phi_0) over the dust counts none, and its solids fraction is (coat solids + dust)/(coat + dust) exactly (no
    evaporation here)."""
    dr = Drying(k_m=0.0)
    flow = _paint({"drying": dr})
    dust = np.zeros((20, 4))
    dust[:5] = 6.0e-6

    def init(sim) -> None:
        flow.set_depth(dust.ravel())
        flow.set_species(dust.ravel(), "solids")
    sim = build_simulation(_scene(init), [flow], "cpu")
    for _ in range(5):
        sim.step_period()
    src = np.zeros((20, 4))
    src[:5] = 5.0e-3
    flow.set_source(src, chart=0, species={"solids": dr.phi_0})
    for _ in range(2):                                                # 100 µm
        sim.step_period()
    flow.set_source(0.0, chart=0)
    for _ in range(5):
        sim.step_period()
    d = flow.process.read()
    r = flow.read()
    h, sol = r["h"][0].reshape(20, 4)[:5], r["species"][0, :, 0].reshape(20, 4)[:5]
    assert (d["laps"][0] == 0).all()
    assert np.allclose(sol / h, (dr.phi_0 * 100.0e-6 + 6.0e-6) / (100.0e-6 + 6.0e-6), rtol=1e-9, atol=0.0)


def _rows(flow: SurfaceFlow, rows: slice, h: float, solids: float) -> None:
    """Set rows ``rows`` of the plate to depth ``h`` holding ``solids`` (m), the other rows as they are."""
    r = flow.read()
    depth, sol = r["h"][0].reshape(20, 4).copy(), r["species"][0, :, 0].reshape(20, 4).copy()
    depth[rows], sol[rows] = h, solids
    flow.set_depth(depth.ravel())
    flow.set_species(sol.ravel(), "solids")


@pytest.mark.compile_heavy
def test_a_coat_past_its_wet_edge_seeds_a_lap_and_a_bare_cell_forgets():
    """WP15's review of the dust rule. Rows 0-4: a coat set by the host past its wet edge, still holding solvent (phi
    0.598 > phi_we 0.57, under phi_m 0.64; paint_roll_strip's prepainted strip at ~500 s), then a new coat: one lap
    (a seed only below phi_we gave none). Rows 10-14: the same coat, cleaned bare, then 6 µm of mist dust (pure
    solids), then a new coat: no lap (a seed never cleared gave one). No evaporation."""
    dr = Drying(k_m=0.0)
    flow = _paint({"drying": dr})
    sim = build_simulation(_scene(), [flow], "cpu")

    def steps(n: int) -> None:
        for _ in range(n):
            sim.step_period()
    for rows in (slice(0, 5), slice(10, 15)):
        _rows(flow, rows, 50.0e-6, 0.598 * 50.0e-6)
    steps(20)                                                         # longer than any film stride
    _rows(flow, slice(10, 15), 0.0, 0.0)                              # cleaned bare
    steps(20)
    _rows(flow, slice(10, 15), 6.0e-6, 6.0e-6)                        # dust
    steps(20)
    src = np.zeros((20, 4))
    src[:15] = 5.0e-3
    flow.set_source(src, chart=0, species={"solids": dr.phi_0})
    steps(2)                                                          # 100 µm
    flow.set_source(0.0, chart=0)
    steps(5)
    laps = flow.process.read()["laps"][0].reshape(20, 4)
    assert (laps[:5] == 1).all() and (laps[5:] == 0).all()


@pytest.mark.compile_heavy
def test_paint_on_paper_stops_at_its_solids():
    """WP15 review must-fix: absorption takes solvent too, so paint on a fast paper (S = 3e-5) stops at solids/phi_m,
    and what left (evaporated + absorbed) is WFT·(1 - phi_0/phi_m) to 1e-9."""
    dr = Drying(k_m=0.6)
    wft = 100.0e-6
    flow = _paint({"drying": dr, "substrates": {"plate": Substrate(sorptivity=3.0e-5, capacity=1.0e-3)}})

    def init(sim):
        flow.set_depth(wft, chart=0)
        flow.set_species(dr.phi_0 * wft, "solids", chart=0)
    _run(flow, 400, init)                                              # 4 s
    area = flow.tables.area_np
    r = flow.read()
    gone = (r["evaporated_volume"][0] + r["absorbed_volume"][0]) / area.sum()
    assert np.allclose(flow.depth(0)[0], dr.phi_0 * wft / dr.phi_m, rtol=1e-6, atol=0.0)
    assert gone == pytest.approx(wft * (1.0 - dr.phi_0 / dr.phi_m), rel=1e-9, abs=0.0)
    assert r["absorbed_volume"][0] > r["evaporated_volume"][0] > 0.0                 # the paper took most


@pytest.mark.compile_heavy
def test_absorption_follows_lucas_washburn_to_capacity_f13():
    """F-13: under a standing film a porous substrate takes absorbed = S·sqrt(t) (exact increments), up to its
    capacity; the film's ledger books it (film + absorbed = the start)."""
    S = 3.0e-6
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material="water", max_depth=2.0e-3,
                       process={"substrates": {0: Substrate(sorptivity=S, capacity=5.0e-6)}})
    _run(flow, 200, lambda sim: flow.set_depth(1.0e-3, chart=0))    # 2 s
    a = flow.process.read()["absorbed_depth"][0]
    t = (200 // flow.stride) * flow.stride * DT
    assert t > 1.5 and np.allclose(a, S * np.sqrt(t), rtol=1e-4, atol=0.0)
    r = flow.read()
    start = 1.0e-3 * flow.tables.area_np.sum()                       # the cells' own areas (float32-derived)
    assert r["film_volume"][0] + r["absorbed_volume"][0] == pytest.approx(start, rel=1e-12, abs=0.0)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material="water", max_depth=2.0e-3,
                       process={"substrates": {0: Substrate(sorptivity=S, capacity=5.0e-6)}})
    _run(flow, 400, lambda sim: flow.set_depth(1.0e-3, chart=0))    # 4 s: S·sqrt(4) = 6 µm > capacity
    assert np.allclose(flow.process.read()["absorbed_depth"][0], 5.0e-6, rtol=1e-6, atol=0.0)


@pytest.mark.compile_heavy
def test_the_sanitizer_clock_runs_only_above_c_min():
    """Contact clock: a film at 0.1 kg/m³ (c_min 0.05) counts the whole run; at 0.02 kg/m³ it never starts."""
    def run(conc: float) -> np.ndarray:
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material="water", max_depth=2.0e-3,
                           species=("sanitizer",), process={"sanitizer": Sanitizer(c_min=0.05)})

        def init(sim):
            flow.set_depth(1.0e-3, chart=0)
            flow.set_species(conc * 1.0e-3, "sanitizer", chart=0)
        _run(flow, 100, init)
        return flow.process.read()["expo_best"][0], (100 // flow.stride) * flow.stride * DT
    best, t = run(0.1)
    assert t > 0.5 and np.allclose(best, t, rtol=1e-12, atol=0.0)
    assert np.all(run(0.02)[0] == 0.0)


@pytest.mark.compile_heavy
def test_the_sanitizer_clock_restarts_when_diluted():
    """expo restarts at 0 when the film drops below c_min; expo_best keeps the longest unbroken contact."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material="water", max_depth=2.0e-3,
                       species=("sanitizer",), process={"sanitizer": Sanitizer(c_min=0.05)})

    def init(sim):
        flow.set_depth(1.0e-3, chart=0)
        flow.set_species(0.1 * 1.0e-3, "sanitizer", chart=0)
    sim = build_simulation(_scene(init), [flow], "cpu")
    for _ in range(60):
        sim.step_period()
    flow.set_species(0.02 * 1.0e-3, "sanitizer", chart=0)              # diluted below c_min
    for _ in range(30):
        sim.step_period()
    k = flow.stride
    p = flow.process.read()
    assert np.all(p["expo"][0] == 0.0)
    assert np.allclose(p["expo_best"][0], (60 // k) * k * DT, rtol=1e-12, atol=0.0)


@pytest.mark.compile_heavy
def test_eroded_soil_joins_the_film_and_is_conserved():
    """Soil removed by jet impact becomes the film's soil species: surface soil + film soil (carried, pending, out)
    is the initial soil to 1e-12."""
    soil = Soil(initial=0.5, sigma_dry=1.0e3, sigma_wet=10.0, t_soak=1.0, k_e=0.0, k_i=0.02, k_wipe=0.0)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material="water", max_depth=2.0e-3,
                       species=("soil",), process={"soil": soil})

    def init(sim):
        flow.set_depth(1.0e-3, chart=0)
        flow.process.hyd.fill_(1.0)
        p = np.zeros((1, 80), np.float32)
        p[0, :8] = 100.0
        flow.impact_p.assign(p)
        flow.impact_share.assign((p > 0.0).astype(np.float32))           # a jet covering those cells
    _run(flow, 50, init)
    area = flow.tables.area_np
    d, r = flow.process.read(), flow.read()
    film = ((flow.sp.mass.numpy()[0, :, 0] + flow.sp.src.numpy()[0, :, 0]) * area).sum()
    film += flow.sp.out.numpy()[0, :, 0].sum()
    assert d["soil_removed_mass"][0] > 0.0
    assert film == pytest.approx(d["soil_removed_mass"][0], rel=1e-12, abs=0.0)
    assert d["soil_mass"][0] + film == pytest.approx(0.5 * area.sum(), rel=1e-12, abs=0.0)
