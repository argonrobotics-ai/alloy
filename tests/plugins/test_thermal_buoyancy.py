# SPDX-License-Identifier: Apache-2.0
"""Thermal buoyancy end to end (``scene.set_heat(buoyancy=True)``, gaps-now #8, 2026-11-08), CPU.

Water at 20 C fills a tank. After it settles, a region is set to 80 C, and two runs from that state are compared,
buoyancy on and buoyancy off (a twin). The hot slots' centroid rises against the twin as 1/2 a t^2. The Boussinesq
estimate for a body of fluid starting from rest is a = g (ρ_cold - ρ_hot) / ρ̄ / (1 + C_a), where C_a is the added
mass of its shape: 1/2 for a sphere (a = 2/3 g'), 1 for a long horizontal cylinder (a = 1/2 g'). ρ(T) is NIST's
water table (phase_materials.WATER_DENSITY).
- The start of rise, fitted over the first 30 ms, matches the estimate. Calibration ([internal note]
  notes/wp9-runs/sph/buoy/blob_probe.py): sphere R 16 mm, 269 hot particles, 0.95; cylinder R 12 mm spanning the
  tank, 0.92 (a wall-bounded cylinder carries more added mass). These are default-accuracy numbers: at
  liquid_accuracy="super" the cylinder reads 0.78. Later the rise falls behind (0.65 of the free estimate at 150 ms
  at both accuracies; cause not pinned, see the gap page).
- Energy books: the buoyancy force's work goes into health's external work (W_ext), all of it. The test's own sum
  of F . dx over every slot and step, with F rebuilt from the temperatures and NIST's table, equals W_ext's change
  within 1e-3. The liquid's own kinetic plus potential energy gains at most that work, the twin books none, and
  ``sim.check()`` is ok in both runs. (In this window W_ext is small and can be negative, because the liquid is
  still settling against the forces.)
- Without ``buoyancy=True`` the plug-in is the plain ParticleThermal (default False: heat as before, bitwise); with
  buoyancy and a time_scale other than 1 the build warns once (convection is that much weaker than conduction).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api._heat import thermal_of
from alloy.plugins.thermal.buoyancy import BuoyantParticleThermal
from alloy.plugins.thermal.particle import ParticleThermal
from alloy.plugins.thermal.phase_materials import WATER_DENSITY, material
from alloy.solvers.sph.buoyancy import density

T_COLD, T_HOT, DEPTH, G, WINDOW = 293.15, 353.15, 0.06, 9.81, 0.03


def _sim(tank: float, buoyancy: bool) -> alloy.Sim:
    sc = alloy.Scene(dt=0.005)
    sc.add_ground()
    sc.add_open_box("tank", inner=(tank, tank, 0.08), wall=0.004, pos=(0, 0, 0))
    sc.add_liquid("water", size=(tank - 0.004, tank - 0.004, DEPTH), pos=(0, 0, 0.004 + DEPTH / 2),
                  temperature=T_COLD)
    if buoyancy:
        sc.set_heat(buoyancy=True)
    sim = alloy.build(sc, device="cpu")
    sim.settle(0.1)
    return sim


def _heat_region(sim, hot: np.ndarray) -> None:
    th = thermal_of(sim.engine)
    h, t = th.enthalpy.numpy(), th.temperature.numpy()
    h[0, hot], t[0, hot] = material("water").enthalpy(T_HOT), T_HOT
    th.enthalpy.assign(h)
    th.temperature.assign(t)


def _liquid_energy(sim, gidx: np.ndarray) -> float:
    st, m = sim.engine.state, sim.engine.model
    q, v, mass = st.particle_q.numpy()[gidx], st.particle_qd.numpy()[gidx], m.particle_mass.numpy()[gidx]
    return float((0.5 * mass * (v.astype(np.float64) ** 2).sum(1) + mass * G * q[:, 2]).sum())


def _forces(th, gidx: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """The buoyancy force [N] of each slot from its temperature now (one world, one water group)."""
    rho = density(WATER_DENSITY, th.temperature.numpy()[0])
    phi = th.liquid.numpy()[0]
    wm = mass * phi
    rho_bar = float((wm * rho).sum() / wm.sum())
    return -wm * (rho - rho_bar) / rho_bar * G                  # z component; g = (0, 0, -G)


def _w_ext(sim) -> float:
    hm = sim.engine.health
    hm = getattr(hm, "health", hm) if not hasattr(hm, "_ref") else hm
    return float(hm._ref.w_cum.numpy()[0])


@pytest.mark.law
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("shape,radius,tank,added_mass", [("sphere", 0.016, 0.09, 0.5), ("cylinder", 0.012, 0.07, 1.0)])
def test_a_hot_region_starts_to_rise_at_the_boussinesq_rate_and_books_its_work(shape, radius, tank, added_mass):
    on, off = _sim(tank, True), _sim(tank, False)
    assert type(thermal_of(on.engine)) is BuoyantParticleThermal and type(thermal_of(off.engine)) is ParticleThermal
    gidx = thermal_of(on.engine)._gidx.numpy()
    rel = on.engine.state.particle_q.numpy()[gidx] - np.array([0.0, 0.0, 0.004 + DEPTH / 2])
    hot = (np.linalg.norm(rel, axis=1) if shape == "sphere" else np.hypot(rel[:, 0], rel[:, 2])) < radius
    for s in (on, off):
        _heat_region(s, hot)
    z0 = [float(s.engine.state.particle_q.numpy()[gidx][hot, 2].mean()) for s in (on, off)]
    e0, w0 = [_liquid_energy(s, gidx) for s in (on, off)], [_w_ext(s) for s in (on, off)]
    t, dz, w_test = [], [], 0.0
    mass, th = on.engine.model.particle_mass.numpy()[gidx].astype(np.float64), thermal_of(on.engine)
    for k in range(int(round(WINDOW / on.dt))):
        fz, x0 = _forces(th, gidx, mass), on.engine.state.particle_q.numpy()[gidx, 2].astype(np.float64)
        on.step(1)
        off.step(1)
        w_test += float(fz @ (on.engine.state.particle_q.numpy()[gidx, 2].astype(np.float64) - x0))
        z = [float(s.engine.state.particle_q.numpy()[gidx][hot, 2].mean()) for s in (on, off)]
        t.append((k + 1) * on.dt)
        dz.append((z[0] - z0[0]) - (z[1] - z0[1]))
    t, dz = np.asarray(t), np.asarray(dz)
    rho_c, rho_h = float(density(WATER_DENSITY, T_COLD)), float(density(WATER_DENSITY, T_HOT))
    share = hot.sum() / hot.size
    a_est = G * (rho_c - rho_h) / ((1 - share) * rho_c + share * rho_h) / (1.0 + added_mass)
    a_fit = float(2.0 * (t ** 2 @ dz) / (t ** 4).sum())
    work, gain = _w_ext(on) - w0[0], _liquid_energy(on, gidx) - e0[0]
    print(f"{shape}: {int(hot.sum())} hot; a {a_fit:.4f} vs {a_est:.4f} m/s^2 ({a_fit / a_est:.3f}); "
          f"buoyancy work {work:.4e} J (own sum {w_test:.4e}), liquid energy gain {gain:.3e} J; "
          f"twin work {_w_ext(off) - w0[1]:.1e} J")
    assert a_fit == pytest.approx(a_est, rel=0.12, abs=0)
    assert work == pytest.approx(w_test, rel=1e-3, abs=0) and gain <= work
    assert _w_ext(off) - w0[1] == 0.0
    assert on.check()["ok"] and off.check()["ok"]


def test_heat_without_buoyancy_is_the_plain_plugin():
    assert ParticleThermal.Config().buoyancy is False
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("cup", inner=(0.03, 0.03, 0.03), wall=0.004)
    sc.add_liquid("water", size=(0.026, 0.026, 0.01), pos=(0, 0, 0.009), temperature=T_COLD)
    sim = alloy.build(sc, device="cpu")
    assert type(thermal_of(sim.engine)) is ParticleThermal


def test_buoyancy_with_a_time_scale_warns_once(caplog):
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("cup", inner=(0.03, 0.03, 0.03), wall=0.004)
    sc.add_liquid("water", size=(0.026, 0.026, 0.01), pos=(0, 0, 0.009), temperature=T_COLD)
    sc.set_heat(buoyancy=True, time_scale=60.0)
    with caplog.at_level("WARNING", logger="alloy.plugins"):
        sim = alloy.build(sc, device="cpu")
    assert type(thermal_of(sim.engine)) is BuoyantParticleThermal
    assert sum("time_scale 60" in r.getMessage() for r in caplog.records) == 1
    # the documented label with buoyancy on too (it was "particle_thermal_buoyancy", so heat.md's lines raised)
    assert sim.plugins["particle_thermal"] is thermal_of(sim.engine)
    read = sim.plugins.read()
    assert list(read) == ["particle_thermal"] and "heater_area_m2" in read["particle_thermal"]
