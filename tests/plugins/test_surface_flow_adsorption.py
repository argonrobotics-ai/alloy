# SPDX-License-Identifier: Apache-2.0
"""Surface flow, species adsorbed on the walls (WP8's G6: surfactant on pans, cans, glassware): Langmuir kinetics
between the film's species and each cell's load, mass exact, a dry wall keeping its load (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.adsorption import Adsorption

G = 9.81
WATER = FilmMaterial("water_flat", surface_tension=0.0)
K, KD, GMAX = 20.0, 0.05, 1.0e-7            # m³/kg, 1/s, kg/m² (WP8: surfactant on glass)
H = 1.0e-3


def _flow(k_d: float = KD, **kw) -> SurfaceFlow:
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material=WATER, max_depth=2.0e-3,
                       species=("soap",), adsorption=Adsorption("soap", K=K, k_d=k_d, capacity=GMAX), **kw)


def _sim(flow: SurfaceFlow, init):
    return build_simulation(TestScene(build=sl.plate_scene(0.1, 0.1), init=init, ground=False, worlds=1,
                                      gravity=(0.0, 0.0, -G), dt=1.0e-2), [flow], "cpu")


def _film_time(flow: SurfaceFlow) -> float:
    tick = int(flow.tick.numpy()[0])
    return (tick // flow.stride) * flow.stride * flow._exchange_dt    # the film's own clock (its last step)


def _langmuir(m0: float, g0: float, t: float, n: int = 20000) -> tuple[float, float]:
    """The film's mass per area and the load after ``t`` [s]: the nonlinear pair, RK4 at t/n."""
    def rate(m: float, g: float) -> float:
        return KD * (K * (m / H) * (GMAX - g) - g)
    m, g, dt = m0, g0, t / n
    for _ in range(n):
        k1 = rate(m, g)
        k2 = rate(m - 0.5 * dt * k1, g + 0.5 * dt * k1)
        k3 = rate(m - 0.5 * dt * k2, g + 0.5 * dt * k2)
        k4 = rate(m - dt * k3, g + dt * k3)
        dg = dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        m, g = m - dg, g + dg
    return m, g


@pytest.mark.parametrize("c0, g0", [(0.5, 0.0), (0.0, GMAX)])
def test_soap_adsorbs_and_rinses_off_by_langmuir_kinetics(c0, g0):
    """Soap at 0.5 kg/m³ in a still 1 mm film on a clean wall loads it toward Γ_max·K·C/(1 + K·C) at
    k_d·(1 + K·C) (0.55 /s); clean water on a loaded wall takes it off at k_d. Either way the load follows the
    nonlinear pair to 0.2 % at 1 and 3 s (film steps of one 10 ms exchange), and film + load stays fixed."""
    flow = _flow(max_stride=1)

    def init(sim) -> None:
        flow.set_depth(H)
        flow.set_species(c0 * H, "soap")
        flow.set_adsorbed(g0, "soap")
    sim = _sim(flow, init)
    start = None
    for n in range(300):
        sim.step_period()
        if n == 0:
            start = float(flow.read()["species_ledger"][0, 0])
        if n + 1 in (100, 300):
            r = flow.read()
            m_ref, g_ref = _langmuir(c0 * H, g0, _film_time(flow))
            assert np.allclose(r["adsorbed"][0, :, 0], g_ref, rtol=2e-3, atol=0.0), (r["adsorbed"][0, 0, 0], g_ref)
            assert np.allclose(r["species"][0, :, 0], m_ref, rtol=2e-3, atol=0.0)
            assert abs(float(r["species_ledger"][0, 0]) - start) <= 1e-14 * abs(start) + 1e-24


def test_the_stepped_rest_is_langmuirs_and_a_dry_wall_keeps_its_load():
    """Strided film steps reach Γ_max·K·C/(1 + K·C) exactly (to 1e-9; C from the film's mass left; a fast k_d of
    5 /s so 2 s is 110 relaxation times). A dry wall's load does not move, bit for bit, and the dense and skipped
    runs agree bit for bit."""
    def run(skip: bool):
        flow = _flow(k_d=5.0, skip_dry=skip)

        def init(sim) -> None:
            h = np.zeros((4, 4))
            h[:2] = H                                                    # the wet half; the dry half is loaded
            g = np.zeros((4, 4))
            g[2:] = 0.5 * GMAX
            flow.set_regions((g > 0.0).astype(int).reshape(-1))         # the halves exchange no film
            flow.set_depth(h.reshape(-1))
            flow.set_species(0.5 * h.reshape(-1), "soap")
            flow.set_adsorbed(g.reshape(-1), "soap")
        sim = _sim(flow, init)
        for _ in range(200):
            sim.step_period()
        return flow.read()
    r = run(True)
    g, m = r["adsorbed"][0, :, 0].reshape(4, 4), r["species"][0, :, 0].reshape(4, 4)
    c = m[:2] / r["h"][0].reshape(4, 4)[:2]
    assert np.allclose(g[:2], GMAX * K * c / (1.0 + K * c), rtol=1e-9, atol=0.0)
    assert np.array_equal(g[2:], np.full((2, 4), 0.5 * GMAX))
    dense = run(False)
    assert np.array_equal(dense["adsorbed"], r["adsorbed"]) and np.array_equal(dense["species"], r["species"])


@pytest.mark.parametrize("spec, match", [(Adsorption("salt", K=K, k_d=KD, capacity=GMAX), "no such species"),
                                         (Adsorption("soap", K=0.0, k_d=KD, capacity=GMAX), "K"),
                                         (Adsorption("soap", K=K, k_d=KD, capacity=(GMAX, GMAX)), "per chart")])
def test_bad_adsorption_is_refused(spec, match):
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material=WATER, max_depth=2.0e-3,
                       species=("soap",), adsorption=spec)
    with pytest.raises(PluginError, match=match):
        _sim(flow, None)
