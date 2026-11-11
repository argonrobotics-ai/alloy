# SPDX-License-Identifier: Apache-2.0
"""Surface-flow process states (design §4.8, §5; WP9): setting. CPU, through the harness runner."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.film import FilmMaterial
from alloy.plugins.surface_flow.process import Setting, Soil

G = 9.81
PASTE = FilmMaterial("paste", density=2350.0, viscosity=5.0, yield_stress=0.0, surface_tension=0.0)


def _run(flow: SurfaceFlow, periods: int, gravity=(0.0, 0.0, -G), init=None, worlds: int = 1, dt: float = 0.01):
    scene = TestScene(build=sl.plate_scene(0.1, 0.02), init=init, ground=False, gravity=gravity, periods=periods,
                      worlds=worlds, dt=dt, control_dt=dt)
    run = run_scene(scene, [flow], "cpu")
    assert run.error is None, run.error
    return run


def _flow(setting: Setting, **config) -> SurfaceFlow:
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 4)),), material=PASTE, max_depth=4.0e-3,
                       per_cell_rheology=True, process={"setting": setting}, **config)


@pytest.mark.compile_heavy
def test_the_yield_follows_the_setting_law_from_first_wetting():
    """Wet cells are placed at their first film step and the film's yield is ``film_factor``·τ(age) at the start of
    each film step (C:424-436); dry cells are not placed (age < 0) and keep τ_0. The law's anchors: τ(t_IS) - τ_0 - A_thix·t_IS = τ_IS (initial set) and τ(1.5 t_IS) ≈ 8 τ_IS (final set)."""
    st = Setting(film_factor=0.25)
    flow = _flow(st)

    def init(sim):
        h = np.zeros((20, 4))
        h[:10] = 1.0e-3
        flow.set_depth(h, chart=0)
    _run(flow, 40, init=init)
    k = flow.stride                                   # a slow paste steps every k exchanges with the summed step
    age = flow.state_arrays()["age"].numpy()[0].reshape(20, 4)
    ty = flow.fluid.yield_cell.numpy()[0].reshape(20, 4)
    assert np.allclose(age[:10], (40 // k) * k * 0.01, rtol=1e-12, atol=0.0) and (age[10:] < 0.0).all()
    at_step = age[:10] - k * 0.01                     # a film step's yield is the law at the age it starts with
    assert np.allclose(ty[:10], np.float32(0.25 * st.yield_stress(at_step)), rtol=1e-6, atol=0.0)
    assert np.allclose(ty[10:], np.float32(0.25 * st.tau_0), rtol=1e-6, atol=0.0)
    setting = lambda a: st.yield_stress(a) - st.tau_0 - st.a_thix * a      # noqa: E731  the power-law term
    assert setting(4.0 * 3600.0) == pytest.approx(3.45e6 / 9.0, rel=2e-3, abs=0.0)   # ASTM C403 initial set, punch factor 9
    assert setting(6.0 * 3600.0) == pytest.approx(27.6e6 / 9.0, rel=2e-3, abs=0.0)   # final set


@pytest.mark.compile_heavy
def test_a_set_age_and_a_forming_bulk():
    """With a forming bulk every cell is placed at the start (``initial_age``) and ages wet or dry; ``bulk_yield`` is
    τ(age) and the film's yield ``film_factor``·τ(age). ``set_input("age")`` re-ages the paste (half the cells here)."""
    st = Setting(initial_age=3.0 * 3600.0, film_factor=0.2)
    flow = _flow(st, bulk={"yield_stress": 1566.0, "resistance": 1.0e6})

    def init(sim):
        a = np.full((1, 80), 3.0 * 3600.0)
        a[0, 40:] = 5.0 * 3600.0
        flow.set_input("age", a)
    _run(flow, 40, init=init)
    k = flow.stride
    age = flow.state_arrays()["age"].numpy()[0]
    step = (40 // k) * k * 0.01
    assert step > 0.0 and np.allclose(age[:40], 3.0 * 3600.0 + step, rtol=1e-12, atol=0.0)
    assert np.allclose(age[40:], 5.0 * 3600.0 + step, rtol=1e-12, atol=0.0)
    at_step = age - k * 0.01                          # the last film step's starting age
    at_exchange = age - (k * 0.01 if 40 % k == 0 else 0.0)   # the bulk's yield is the last exchange's (every one)
    assert np.allclose(flow.bulk_yield.numpy()[0], np.float32(st.yield_stress(at_exchange)), rtol=1e-6, atol=0.0)
    assert np.allclose(flow.fluid.yield_cell.numpy()[0], np.float32(0.2 * st.yield_stress(at_step)), rtol=1e-6, atol=0.0)


@pytest.mark.compile_heavy
def test_a_bulk_follows_a_set_age_at_the_next_exchange():
    """WP15 review: forming reads bulk_yield every exchange, so a set age reaches it at the next exchange (not at the
    next film step): after one period at one exchange per period, bulk_yield is τ(the set age)."""
    st = Setting(initial_age=3.0 * 3600.0)
    flow = _flow(st, bulk={"yield_stress": 1566.0, "resistance": 1.0e6})
    scene = TestScene(build=sl.plate_scene(0.1, 0.02), ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=0.01,
                      control_dt=0.01)
    sim = build_simulation(scene, [flow], "cpu")
    for _ in range(3):
        sim.step_period()
    flow.set_input("age", 5.0 * 3600.0)
    sim.step_period()
    assert np.allclose(flow.bulk_yield.numpy()[0], np.float32(st.yield_stress(5.0 * 3600.0)), rtol=1e-6, atol=0.0)


@pytest.mark.compile_heavy
def test_a_draining_paste_stops_when_it_has_set():
    """F-8 with setting: paste 2 mm deep on a plate with gravity along it drains while τ(age) < ρ·g·h, and stands from
    the age where τ(age) = ρ·g·h (here τ = 50 Pa/s · age, so about 0.9 s)."""
    h0 = 2.0e-3
    t_stop = 2350.0 * G * h0 / 50.0

    def depth(st: Setting, periods: int) -> np.ndarray:                # one fresh run per mark
        flow = _flow(st)
        _run(flow, periods, gravity=(-G, 0.0, -0.0), init=lambda sim, f=flow: f.set_depth(h0, chart=0))
        return flow.depth(0)[0].copy()
    sets, never = Setting(tau_0=1.0e-3, a_thix=50.0, tau_is=0.0), Setting(tau_0=1.0e-3, a_thix=0.0, tau_is=0.0)
    assert 1.5 > t_stop > 0.5
    assert np.abs(depth(sets, 30) - depth(sets, 10)).max() > 1.0e-5           # 0.1-0.3 s: flowing
    assert np.array_equal(depth(sets, 200), depth(sets, 240))                # 2.0 s and 2.4 s: set, at rest
    assert np.abs(depth(never, 240) - depth(never, 200)).max() > 1.0e-6      # the null case still moves then


# --- soil ---------------------------------------------------------------------------------------------------------

THETA, SRC = np.radians(30.0), 1.0e-3                 # incline; a source of 1 mm/s on the top 4 cells


def _incline(soil: Soil, periods: int, prefill: float):
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                       material="water", max_depth=2.0e-3, process={"soil": soil})

    def init(sim):
        src = np.zeros((40, 1))
        src[36:] = SRC
        flow.set_source(src, chart=0)
        flow.set_depth(prefill, chart=0)
    scene = TestScene(build=sl.plate_scene(0.1, 0.05), init=init, ground=False, periods=periods,
                      gravity=(-G * np.sin(THETA), 0.0, -G * np.cos(THETA)))
    run = run_scene(scene, [flow], "cpu")
    assert run.error is None, run.error
    return flow


@pytest.mark.compile_heavy
def test_soil_erodes_under_a_nusselt_film_at_the_shear_law():
    """F-15: on a steady Nusselt film (Q' = 1e-5 m²/s down a 30° plate, h_N = (3νQ'/(g sin θ))^(1/3)) the wall shear is
    τ_w = ρ g h_N sin θ; soaked soil (σ_c = σ_dry e^{-2t/t_soak} + σ_wet) falls at k_e (τ_w - σ_c)+ (3 %); soaking
    hyd = 1 - exp(-t/t_soak) exactly; soil + removed is the initial soil per world."""
    rho, nu = 998.2, 1.002e-3 / 998.2
    q = SRC * 4 * (0.1 / 40)                                            # m²/s per width
    h_n = (3.0 * nu * q / (G * np.sin(THETA))) ** (1.0 / 3.0)
    tau_w = rho * G * h_n * np.sin(THETA)
    soil = Soil(initial=0.02, sigma_dry=50.0, sigma_wet=0.2, t_soak=1.0, k_e=0.01, k_i=0.0, k_wipe=0.0)
    flow = _incline(soil, 200, h_n)
    T = 200 * 0.02                                                      # periods x the scene's control period
    d = flow.process.read()
    t_star = -0.5 * soil.t_soak * np.log((tau_w - soil.sigma_wet) / soil.sigma_dry)
    expect = soil.k_e * ((tau_w - soil.sigma_wet) * (T - t_star)
                         - 0.5 * soil.sigma_dry * soil.t_soak * (np.exp(-2 * t_star / soil.t_soak) - np.exp(-2 * T / soil.t_soak)))
    mid = d["soil_removed"][0][5:30]                                    # away from the open edge and the source
    assert 0.5 * soil.initial > expect > 0.0
    assert np.allclose(mid, expect, rtol=0.03, atol=0.0), (mid.min(), mid.max(), expect)
    assert np.allclose(d["soil_hyd"][0][5:30], 1.0 - np.exp(-T / soil.t_soak), rtol=1e-12, atol=0.0)
    total = soil.initial * flow.tables.area_np.sum()
    assert d["soil_mass"][0] + d["soil_removed_mass"][0] == pytest.approx(total, rel=1e-12, abs=0.0)


def _faucet(process=None, chart=None) -> tuple[SurfaceFlow, object]:
    """DC5's faucet (about 0.1 L/s, 0.28 m drop) on the slab, every exchange a film step, after one period (the jet is
    steady by then, to float64 rounding)."""
    from tests.plugins.test_surface_flow_supply import CHART, FAUCET, faucet_network, slab_scene  # noqa: PLC0415
    flow = SurfaceFlow(charts=(chart or CHART,), material="water", nozzles=(FAUCET,), network=faucet_network(),
                       max_depth=3.0e-3, max_stride=1, process=process)
    run = run_scene(TestScene(build=slab_scene(), ground=True, worlds=1, periods=1), [flow], "cpu")
    assert run.error is None, run.error
    return flow, run.sim


@pytest.mark.compile_heavy
@pytest.mark.parametrize("cells", [40, 160])
def test_a_real_jet_erodes_dry_soil_from_the_first_hit(cells):
    """The supply's impact_p is the jet's stagnation pressure times its share of each cell (``impact_share``: the
    footprint over the area of the cells it covers), so the soil loses k_i·max(impact_p/σ_c - share, 0) = GA's
    k_i·share·max(p_stag/σ_c - 1, 0). On 1 cm cells the 3.6 mm jet sits on its hit cell (share = footprint/A_cell;
    σ_c lies between the cell-averaged pressure and p_stag, so the share counted twice would erode nothing); on
    2.5 mm cells it spreads over the cells its footprint covers. The jet sweeps its sheet dry (h = 0), yet the soil
    under it soaks, and the sheet's shear erodes it at k_e·(jet_tau - σ_c) (the sheet's shear, not the film's: the
    film under it is the sweep's thin refill)."""
    from tests.plugins.test_surface_flow_supply import CHART  # noqa: PLC0415
    soil = Soil(initial=0.0, sigma_dry=1500.0, sigma_wet=1.0, t_soak=1.0e4, k_i=1.0e-3, k_e=1.0e-3, k_front=0.0,
                k_wipe=0.0)
    flow, sim = _faucet({"soil": soil}, chart=PlaneChart("slab", cells=(cells, cells), edges=CHART.edges))
    jet = flow.supply.jet.numpy()[0]                                     # the supply's own jet (one nozzle)
    A = flow.tables.area_np
    w_n = -float(np.dot(jet["wl"], jet["n_hit"]))
    footprint = np.pi * float(jet["a"]) ** 2 * float(jet["speed"]) / w_n
    p_stag = 0.5 * flow.supply.nozzles[0].density * w_n ** 2
    sheet = flow.jet_cover.numpy()[0] > 0.0
    ip, share, tau = flow.impact_p.numpy()[0], flow.impact_share.numpy()[0], flow.jet_tau.numpy()[0]
    hit = share > 0.0
    assert sheet[hit].all() and float((share * A).sum()) == pytest.approx(footprint, rel=1e-5)
    assert np.allclose(ip[hit] / share[hit], p_stag, rtol=1e-5)
    sig0 = soil.sigma_dry + soil.sigma_wet
    assert hit.sum() == 1 and ip[hit][0] < sig0 < p_stag if cells == 40 else hit.sum() >= 4
    flow.process.set_soil(0.5, chart=0)
    flow.process.hyd.assign(np.where(sheet & ~hit, 1.0, 0.0)[None, :])   # dry under the impact, soaked under the sheet
    periods, dt = 10, 0.002                                              # TestScene: control 0.02 s, exchange 2 ms
    for _ in range(periods):
        sim.step_period()
    n = periods * 10                                                     # film steps
    soak = np.exp(-dt / soil.t_soak)
    hy = 1.0 - soak ** np.arange(n)
    sig = soil.sigma_dry * (1.0 - hy) ** 2 + soil.sigma_wet
    expect = (soil.k_i * np.maximum(ip[hit][:, None] / sig[None, :] - share[hit][:, None], 0.0) * dt).sum(axis=1)
    d = flow.process.read()
    by = d["soil_removed_by"][0]
    assert np.allclose(by[hit, 0], expect, rtol=1.0e-6, atol=0.0)
    ga = (soil.k_i * share[hit][:, None] * np.maximum(p_stag / sig[None, :] - 1.0, 0.0) * dt).sum(axis=1)
    assert np.allclose(by[hit, 0], ga, rtol=1.0e-5, atol=0.0) and (ga > 0.0).all()
    assert np.count_nonzero(by[:, 0]) == hit.sum()                      # the footprint's cells only
    assert (flow.read()["h"][0][hit] == 0.0).all()
    assert np.allclose(d["soil_hyd"][0][hit], 1.0 - soak ** n, rtol=1.0e-6, atol=0.0)
    under = sheet & ~hit
    shear = soil.k_e * (tau[under] - soil.sigma_wet) * n * dt            # τ_w = jet_tau under the sheet
    assert (tau[under] > soil.sigma_wet).all() and np.allclose(by[under, 1], shear, rtol=1.0e-6, atol=0.0)
    assert d["soil_mass"][0] + d["soil_removed_mass"][0] == pytest.approx(d["soil_start_mass"][0], rel=1e-14, abs=0.0)


@pytest.mark.compile_heavy
def test_a_scrubbing_pad_abrades_dry_soil_at_archards_law():
    """WP8's G4: a guided pad 1 mm into dry soil (press = stiffness·1 mm = 1 kPa) sliding at 5 cm/s removes
    k_abr·cover·press·|v|·σ_ref/σ_c per exchange (Archard, GA's sponge law; σ_c 200 kPa dry) from the cells under it,
    exactly; the soil does not soak and the viscous wipe adds nothing (no film); booked as the tool's; ledger exact."""
    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from tests.plugins.test_surface_flow_tools import blade, scene  # noqa: PLC0415
    soil = Soil()
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 4)),), material="water", max_depth=2.0e-3,
                       max_stride=1, tools=(blade(stiffness=1.0e6),), process={"soil": soil})

    def init(sim) -> None:
        sim.state.joint_qd.assign(np.full(1, 0.05))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    dt = 2.0e-3
    sim = build_simulation(TestScene(build=scene(-1.0e-3, 0.0, (0.0, 0.0, 0.0), density=2.0e6, guided=True),
                                     init=init, ground=False, gravity=(0.0, 0.0, 0.0), worlds=1, dt=dt,
                                     control_dt=dt), [flow], "cpu")
    sim.step_period()                                                    # wakes the tiles under the pad
    sig = soil.sigma_dry + soil.sigma_wet
    for _ in range(20):
        before = flow.process.read()["soil_removed_by"][0][:, 3].copy()
        sim.step_period()
        tf = flow.tf
        cov, press = tf.cover.numpy()[0].astype(np.float64), tf.press.numpy()[0].astype(np.float64)
        v = np.linalg.norm(tf.vel.numpy()[0].astype(np.float64), axis=1)
        expect = soil.k_abr * cov * press * v * soil.sigma_ref / sig * dt
        assert (expect > 0.0).sum() >= 8 and abs(press.max() - 1.0e3) < 1.0
        assert np.allclose(flow.process.read()["soil_removed_by"][0][:, 3] - before, expect, rtol=1e-6, atol=0.0)
    d = flow.process.read()
    assert np.all(d["soil_hyd"][0] == 0.0) and np.all(d["soil_removed_by"][0][:, :3] == 0.0)
    assert d["soil_mass"][0] + d["soil_removed_mass"][0] == pytest.approx(d["soil_start_mass"][0], rel=1e-14, abs=0.0)


@pytest.mark.compile_heavy
def test_harness_quick_check_with_every_process_state_on():
    """The harness's quick checks (contract, float atomics, crash, hook-level determinism, identical and isolated
    worlds, graph capture) on its tilted plate with setting, soil, drying, a substrate and the sanitizer clock on."""
    import alloy.plugins as ap  # noqa: PLC0415
    from tests.plugins.test_surface_flow_process_gpu import every_part  # noqa: PLC0415
    report = ap.check(every_part(), quick=True, laws=False)
    assert report.passed, report.summary()


def _two_plates(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.05, hy=0.01, hz=0.01, label="a")
    b.add_shape_box(-1, xform=body_xform(0.0, 0.1, -0.01), hx=0.05, hy=0.01, hz=0.01, label="b")
    dummy = b.add_body(xform=body_xform(0.0, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(dummy, radius=0.01)


@pytest.mark.compile_heavy
def test_charts_by_name_and_set_soil_on_one_chart():
    """WP15 review: soil charts and substrates by chart name; an unknown name is a PluginError; set_soil(chart=1)
    resets the soaking of that chart's cells only."""
    from alloy.plugins.errors import PluginError  # noqa: PLC0415
    from alloy.plugins.surface_flow.process import Substrate  # noqa: PLC0415
    charts = (PlaneChart("a", cells=(10, 2)), PlaneChart("b", cells=(10, 2)))
    flow = SurfaceFlow(charts=charts, material="water", max_depth=2.0e-3,
                       process={"soil": Soil(initial=0.1, charts=("b",)), "substrates": {"a": Substrate()}})
    scene = TestScene(build=_two_plates, ground=False, gravity=(0.0, 0.0, -G), worlds=1, dt=0.01, control_dt=0.01)
    sim = build_simulation(scene, [flow], "cpu")
    flow.set_depth(1.0e-3)
    soil = flow.process.read()["soil"][0]
    assert np.all(soil[:20] == 0.0) and np.all(soil[20:] == 0.1)                     # chart "b" only
    assert np.all(flow.process.sorp.numpy()[:20] > 0.0) and np.all(flow.process.sorp.numpy()[20:] == 0.0)
    flow.process.set_soil(0.2, chart=0)                                            # soil chart "a" as well
    for _ in range(5):
        sim.step_period()
    hyd = flow.process.read()["soil_hyd"][0]
    assert np.all(hyd > 0.0)
    flow.process.set_soil(0.3, chart=1)
    hyd2 = flow.process.read()["soil_hyd"][0]
    assert np.array_equal(hyd2[:20], hyd[:20]) and np.all(hyd2[20:] == 0.0)
    before = {k: flow.process.read()[k][0].copy() for k in ("soil", "soil_hyd")}
    for kw in ({"hyd": 1.5}, {"hyd": -0.1}, {"hyd": np.zeros(3)}):                 # WP15: checked before writing
        with pytest.raises((PluginError, ValueError)):
            flow.process.set_soil(0.7, chart=1, **kw)
        after = flow.process.read()
        assert all(np.array_equal(after[k][0], before[k]) for k in before), kw
    bad = SurfaceFlow(charts=charts, material="water", process={"soil": Soil(charts=("nope",))})
    with pytest.raises(PluginError, match="no chart 'nope'"):
        build_simulation(scene, [bad], "cpu")


# --- soil, WP15's additions for WP8 (the jet sheet's shear, Wilson's front) ---------------------------------------

NX, NY, DX = 60, 4, 2.0e-3


def _slab(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.005), hx=0.5 * NX * DX, hy=0.5 * NY * DX, hz=0.005, label="slab")
    ball = b.add_body(xform=body_xform(0.0, 0.0, 0.5), label="ball")
    b.add_shape_sphere(ball, radius=0.01)


def _prescribed(soil: Soil, h: float, u: float):
    """A uniform film of depth h moving at u along +x (test only: the stage reads the film, it never writes it), every
    tile on; the stage's after_film is then called by hand."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(NX, NY)),), material="water", skip_dry=False,
                       process={"soil": soil})
    scene = TestScene(build=_slab, worlds=1, ground=False, control_dt=0.01, dt=1.25e-3, gravity=(0.0, 0.0, -G))
    sim = build_simulation(scene, [flow], "cpu")
    flow.h.assign(np.full(flow.h.shape, h))
    uf = np.zeros(flow.u.shape, np.float32)
    uf[:, :(NX + 1) * NY] = u                          # the chart's i-faces come first
    flow.u.assign(uf)
    flow.q.assign(uf * np.float32(h))
    return flow, sim


@pytest.mark.compile_heavy
@pytest.mark.parametrize("w", [1.0, 0.5])
def test_wilson_front_runs_at_k_w_m(w):
    """WP8's crossing-time law: a strip soaked to hyd = w (held: t_soak 1e9 s) behind a clean inlet under a uniform
    film; the front advances at v = k'·w·β·ρ·u²·h (Wilson et al. 2014) to within a cell at 10/w, 20/w and 30/w s (the
    crossing time per cell Δx/v doubles at w = 0.5), and nothing else removes soil (τ_w = 3μu/h = 1.5 Pa < σ_wet)."""
    h, u, dt = 1.0e-3, 0.5, 0.01
    flow, sim = _prescribed(Soil(t_soak=1.0e9), h, u)
    load = np.zeros((NX, NY))
    load[10:] = 0.2
    flow.process.set_soil(load, chart=0, hyd=w)                          # pre-soaked (WP8's API ask)
    v = 2.0e-3 * w * 1.2 * flow.mat.density * u * u * h                   # m/s
    k = flow.stride
    for t in (10.0 / w, 20.0 / w, 30.0 / w):
        for _ in range(int(round(10.0 / w / (dt * k)))):
            flow.process.after_film(sim.state, dt)
        clean = flow.process.read()["soil"][0].reshape(NX, NY)[:, 0] < 1.0e-6
        front = DX * (np.argmin(clean[10:]) if not clean[10:].all() else NX - 10)
        assert front == pytest.approx(v * t, abs=DX), t
    d = flow.process.read()
    assert d["soil_mass"][0] + d["soil_removed_mass"][0] == pytest.approx(d["soil_start_mass"][0], rel=1e-14, abs=0.0)
    assert d["soil_removed_by_mass"][0][[0, 1, 3]].tolist() == [0.0, 0.0, 0.0]          # the front only


@pytest.mark.compile_heavy
def test_the_jet_sheet_shear_removes_soil_the_film_alone_cannot():
    """τ_w is the larger of the jet sheet's shear (jet_tau) and the film's own: under a still film, a jet sheet of
    400 Pa on soaked tomato paste (σ_wet 150 Pa) removes k_e·(400 - 150) per second (k_e 1e-4 here, so the 0.2 kg/m²
    load lasts); without it nothing goes."""
    soil = Soil(k_i=0.0, k_front=0.0, k_e=1.0e-4)
    flow, sim = _prescribed(soil, 1.0e-3, 0.0)
    flow.process.hyd.fill_(1.0)
    tau = np.zeros(flow.jet_tau.shape, np.float32)
    tau[0, :4 * NY] = 400.0                                                  # the first 4 columns
    flow.jet_tau.assign(tau)
    n = 100
    for _ in range(n):
        flow.process.after_film(sim.state, 0.01)
    removed = flow.process.read()["soil_removed"][0]
    T = n * 0.01 * flow.stride
    assert 0.2 > 1.0e-4 * 250.0 * T and np.allclose(removed[:4 * NY], 1.0e-4 * (400.0 - 150.0) * T, rtol=1e-9,
                                                     atol=0.0)
    assert np.all(removed[4 * NY:] == 0.0)
