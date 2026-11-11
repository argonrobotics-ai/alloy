# SPDX-License-Identifier: Apache-2.0
"""Surface flow, structure and solids in the rheology (design §3.2, rheology.py): Moore's rebuild and breakdown laws,
a rolled coat that sags until its structure rebuilds, a roller's nip breaking it, and Krieger-Dougherty drainage
(CPU, strict comparisons)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.rheology import Thixotropy
from tests.plugins.test_surface_flow import _plate

G = 9.81


def _run(flow: SurfaceFlow, build, init, gravity, periods: int, **scene):
    run = run_scene(TestScene(build=build, init=init, ground=False, gravity=gravity, periods=periods, worlds=1,
                              **scene), [flow], "cpu")
    assert run.error is None, run.error
    return run


def test_structure_rebuilds_at_rest_as_moores_law():
    """lam from 0 at rest: 1 - exp(-t/t_r) (no shear, no breakdown)."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material="water", max_depth=1.0e-3,
                       thixotropy=Thixotropy(rebuild=2.0))

    def init(sim) -> None:
        flow.set_depth(1.0e-4)
        flow.fluid.lam.zero_()
    run = _run(flow, sl.plate_scene(0.1, 0.1), init, (0.0, 0.0, -G), periods=25)      # 0.5 s
    tick = int(flow.tick.numpy()[0])
    t = (tick // flow.stride) * flow.stride * flow._exchange_dt        # the film's own clock (its last step)
    assert 0.45 < t <= float(run.trace.t[-1])
    assert np.allclose(flow.read()["lam"][0], 1.0 - math.exp(-t / 2.0), rtol=1e-5, atol=0.0)


def test_a_falling_film_breaks_its_structure_to_moores_steady_state():
    """A Nusselt film (water, Re 10 on 30°) sheared at γ̇_N = 3Q'/h_N²: lam = 1/(1 + k_b·γ̇_N·t_r), here 1/2."""
    q, th = sl.NUSSELT_Q, sl.NUSSELT_TH
    nu = sl.NUSSELT["material"].nu
    h_n = (3.0 * nu * q / (G * math.sin(th))) ** (1.0 / 3.0)
    rate = 3.0 * q / h_n**2                                         # about 900 /s
    t_r = 0.2
    flow = SurfaceFlow(**sl.NUSSELT, thixotropy=Thixotropy(rebuild=t_r, breakdown=1.0 / (rate * t_r)))
    _run(flow, sl.plate_scene(0.2, 0.01), sl._nusselt_init, (-G * math.sin(th), 0.0, -G * math.cos(th)),
         periods=150)
    lam = flow.read()["lam"][0].reshape(100, 1)[30:70, 0]
    assert np.abs(lam / 0.5 - 1.0).max() < 0.02, lam


def _sag(thixo: bool, periods: int) -> np.ndarray:
    """A 100 µm coat on a wall (h_c = τ_y/ρg = 157 µm), just rolled (lam = 0): its face speeds after ``periods``."""
    paint = FilmMaterial("thixo_paint", density=1300.0, viscosity=5.0, yield_stress=2.0, surface_tension=0.0)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1)),), material=paint, inertia=False,
                       max_depth=3.0e-4, max_stride=1, thixotropy=Thixotropy(rebuild=0.5) if thixo else None)

    def init(sim) -> None:
        flow.set_depth(1.0e-4)
        flow.fluid.lam.zero_()
    _run(flow, _plate(False), init, (-G, 0.0, 0.0), periods=periods)
    return np.abs(flow.u.numpy()[0])


def test_a_rolled_coat_sags_until_its_structure_rebuilds():
    """Rolled, the coat's yield is gone and it sags; it stops when lam·τ_y = ρ·g·h, at t* = -t_r·ln(1 - ρgh/τ_y)
    (0.51 s). Rested (no thixotropy) it never moves."""
    t_star = -0.5 * math.log(1.0 - 1300.0 * G * 1.0e-4 / 2.0)
    assert 0.40 < t_star < 0.62
    mid = slice(15, 26)                                             # faces mid-wall, far from the walls' waves
    assert _sag(True, 20)[mid].min() > 0.0                          # 0.40 s: still sagging
    assert _sag(True, 31)[mid].max() == 0.0                         # 0.62 s: the plug stands again
    assert _sag(False, 20).max() == 0.0


def test_a_roller_nip_breaks_the_structure_it_rolls_over():
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415
    from tests.plugins.test_surface_flow_stores import HW, NX, R, U, VISCOUS, roller_scene  # noqa: PLC0415
    from alloy._vendor import newton  # noqa: PLC0415

    roller = Applicator(body="cover", shape="roller", size=(R, HW), axis=(0.0, 1.0, 0.0), stiffness=1.0e3,
                        store=Store(capacity=1.0e-5))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, inertia=False,
                       max_depth=1.0e-3, tools=(roller,), thixotropy=Thixotropy(rebuild=100.0))

    def init(sim) -> None:
        flow.set_depth(1.0e-4)
        qd = sim.state.joint_qd.numpy().reshape(-1, 2)
        qd[:] = (U, U / R)
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    sim = build_simulation(TestScene(build=roller_scene(R + 0.5e-4, -0.1), init=init, ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(40):
        sim.step_period()
    x = flow.cell_centres(0)[:, 0, 0]
    x1 = float(sim.state.body_q.numpy()[0][0])
    lam = flow.read()["lam"][0].reshape(NX, 4).mean(axis=1)
    rolled = (x > -0.1 + 0.005) & (x < x1 - 0.005)
    assert rolled.sum() >= 10 and lam[rolled].max() < 0.02          # broken, rebuilding over 100 s
    assert lam[x > x1 + 0.05].min() == 1.0                         # ahead: untouched


@pytest.mark.parametrize("phi, phi_ref", [(0.3, 0.0), (0.45, 0.3)])
def test_solids_raise_the_viscosity_by_krieger_dougherty(phi, phi_ref):
    """Jeffreys drainage of oil with solids at phi_s (phi_m 0.64): h = sqrt(ν_eff·x/(g·t)) with
    ν_eff = ν·((1 - phi_s/phi_m)/(1 - phi_ref/phi_m))^(-2.5·phi_m) (2.75 ν and 2.54 ν), to 3 %; the solids stay
    uniform."""
    phi_m = 0.64
    nu_eff = sl.OIL.nu * ((1.0 - phi / phi_m) / (1.0 - phi_ref / phi_m)) ** (-2.5 * phi_m)
    flow = SurfaceFlow(**sl.DRAIN, species=("solids",), krieger=phi_m, krieger_ref=phi_ref)

    def init(sim) -> None:
        flow.set_depth(sl.DRAIN_H0)
        flow.set_species(phi * sl.DRAIN_H0, "solids")
    run = _run(flow, sl.plate_scene(0.2, 0.01), init, (-G, 0.0, 0.0), periods=275, control_dt=0.2,
               dt=0.2)                                              # 55 s: the slower film drains as far
    t = float(run.trace.t[-1])
    x_top = 0.1 - flow.cell_centres(0)[:, 0, 0].astype(np.float64)
    h, m = flow.depth(0)[0][:, 0], flow.species("solids", 0)[0][:, 0]
    want = np.sqrt(nu_eff * x_top / (G * t))
    sel = (want < 0.8 * sl.DRAIN_H0) & (x_top > 0.04)
    assert sel.sum() > 20
    assert np.abs(h[sel] / want[sel] - 1.0).max() < 0.03
    assert np.abs(m[h > 1e-6] / h[h > 1e-6] / phi - 1.0).max() < 1e-6


def test_krieger_needs_a_solids_species():
    with pytest.raises(PluginError, match="solids"):
        flow = SurfaceFlow(**sl.DRAIN, krieger=0.64)
        build_simulation(TestScene(build=sl.plate_scene(0.2, 0.01), ground=False, worlds=1), [flow], "cpu")


def test_krieger_at_its_reference_is_the_card():
    """A paint laid at phi_ref (its card's as-applied fraction) drains as the card, bit for bit (SOFT-3's latex stood
    5.5x thicker at application with the absolute factor); a reference at or above phi_m is refused."""
    def drained(**kw) -> np.ndarray:
        flow = SurfaceFlow(**sl.DRAIN, species=("solids",), **kw)

        def init(sim) -> None:
            flow.set_depth(sl.DRAIN_H0)
            flow.set_species(0.4 * sl.DRAIN_H0, "solids")
        _run(flow, sl.plate_scene(0.2, 0.01), init, (-G, 0.0, 0.0), periods=20, control_dt=0.2, dt=0.2)
        return flow.depth(0)[0]
    card = drained()
    assert np.array_equal(drained(krieger=0.64, krieger_ref=0.4), card)
    assert not np.array_equal(drained(krieger=0.64), card)               # absolute: thicker, drains less
    with pytest.raises(PluginError, match="krieger_ref"):
        flow = SurfaceFlow(**sl.DRAIN, species=("solids",), krieger=0.64, krieger_ref=0.64)
        build_simulation(TestScene(build=sl.plate_scene(0.2, 0.01), ground=False, worlds=1), [flow], "cpu")
