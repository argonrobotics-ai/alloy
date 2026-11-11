# SPDX-License-Identifier: Apache-2.0
"""Heat v2 steps 1-3 (docs/notes/heat-v2-design.md): phase materials and flow laws, T(H) on host and device,
ParticleThermal config and the Thermal link, the particle laws H-1, H-2, H-5 (closed, temperature, power and body
heaters), H-7, and the phase laws H-3, H-4, H-6, H-8 (flow arrays, melt spreads), H-9 on CPU."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import _law, check, fresh
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.thermal import particle_kernels as K
from alloy.plugins.thermal.particle import ParticleThermal
from alloy.plugins.thermal.particle_laws import BUSY, _pan_scene
from alloy.plugins.thermal.sensor import Thermal
from alloy.plugins.thermal.phase_materials import MATERIALS, PhaseMaterial, material

MUSHY = PhaseMaterial("mushy_test", density=900.0, c_solid=2000.0, c_liquid=2500.0, k_solid=0.2, k_liquid=0.15,
                      t_solidus=300.0, t_liquidus=310.0, latent=1.5e5, source="test fixture")


def test_water_enthalpy_round_trip_and_plateau():
    m = material("water")
    for t in (250.0, 273.15, 293.15, 353.15, 372.0):
        assert m.temperature(m.enthalpy(t))[0] == pytest.approx(t, abs=1e-9)
    # the fusion plateau: H from 0 to L sits at 273.15 K with φ rising from 0 to 1
    t, phi = m.temperature(np.array([0.0, 0.5 * m.latent, m.latent]))
    np.testing.assert_allclose(t, 273.15)
    np.testing.assert_allclose(phi, [0.0, 0.5, 1.0])
    assert m.h_boil == pytest.approx(m.latent + m.c_liquid * 100.0)
    assert m.diffusivity(liquid=True) == pytest.approx(0.607 / (997.05 * 4180.0))


def test_mushy_zone_is_linear_between_solidus_and_liquidus():
    m = MUSHY
    hl = m.h_liquidus
    assert hl == pytest.approx(1.5e5 + 2250.0 * 10.0)
    t, phi = m.temperature(np.array([-2000.0, 0.25 * hl, hl, hl + 2500.0]))
    np.testing.assert_allclose(t, [299.0, 302.5, 310.0, 311.0])
    np.testing.assert_allclose(phi, [0.0, 0.25, 1.0, 1.0])


def test_bad_materials_and_configs_raise():
    with pytest.raises(ValueError):
        PhaseMaterial("x", density=0.0, c_solid=1.0, c_liquid=1.0, k_solid=1.0, k_liquid=1.0, t_solidus=1.0,
                      t_liquidus=1.0, latent=0.0)
    with pytest.raises(ValueError):
        PhaseMaterial("x", density=1.0, c_solid=1.0, c_liquid=1.0, k_solid=1.0, k_liquid=1.0, t_solidus=2.0,
                      t_liquidus=1.0, latent=0.0)
    with pytest.raises(KeyError):
        material("unobtainium")
    with pytest.raises(ValueError):
        ParticleThermal.Config(materials=("water",), ranges=((0, -1), (0, 2)), initial=(300.0,))
    with pytest.raises(ValueError):
        ParticleThermal.Config(heaters=("plate",), heater_modes=("hot",), heater_values=(400.0,))
    with pytest.raises(ValueError):
        ParticleThermal.Config(feedback="sometimes")
    with pytest.raises(ValueError):  # a start above boiling would flash to vapour on the first step
        ParticleThermal.Config(materials=("water",), initial=(380.0,))
    with pytest.raises(ValueError):  # a Thermal body runs in real time
        ParticleThermal.Config(heaters=("plate",), heater_modes=("body",), heater_values=(0.0,), time_scale=10.0)
    with pytest.raises(ValueError):
        PhaseMaterial("x", density=1.0, c_solid=1.0, c_liquid=1.0, k_solid=1.0, k_liquid=1.0, t_solidus=1.0,
                      t_liquidus=1.0, latent=0.0, young_solid=1e6, young_liquid=0.0)
    assert all(m.source for m in MATERIALS.values())  # every table value says where it comes from


def test_flow_law_ends_and_arrhenius():
    m = material("law_paste")
    assert m.flows and not material("water").flows
    assert m.viscosity(290.0, 0.0) == pytest.approx(m.mu_solid)  # solid: the cap, whatever T
    assert m.viscosity(m.t_liquidus, 1.0) == pytest.approx(m.mu_liquid)
    want = m.mu_liquid * math.exp(m.mu_energy / 8.314462618 * (1.0 / 310.0 - 1.0 / m.t_liquidus))
    assert m.viscosity(310.0, 1.0) == pytest.approx(want)
    assert m.viscosity(300.0, 0.5) == pytest.approx(math.sqrt(m.mu_solid * m.mu_liquid))  # log-linear in φ
    np.testing.assert_allclose(m.yield_stress([0.0, 0.5, 1.0]), [m.yield_solid, 0.25 * m.yield_solid, 0.0])
    np.testing.assert_allclose(m.friction([0.0, 1.0]), [m.friction_solid, 0.0])
    sauce = material("starch_sauce")                                 # thickens as it cooks: log-linear in doneness
    assert sauce.viscosity(sauce.t_liquidus, 1.0, 0.0) == pytest.approx(sauce.mu_liquid)
    assert sauce.viscosity(sauce.t_liquidus, 1.0, 1.0) == pytest.approx(sauce.mu_done)
    assert sauce.viscosity(sauce.t_liquidus, 1.0, 0.5) == pytest.approx(math.sqrt(sauce.mu_liquid * sauce.mu_done))
    assert m.viscosity(310.0, 1.0, 0.7) == pytest.approx(want)      # mu_done = 0: doneness leaves μ alone
    np.testing.assert_allclose(m.young([0.0, 1.0]), [m.young_solid, m.young_liquid])


def _pan_sim(order: str):
    """The H-5d pan scene without its auto-link; ``order`` "thermal_first" or "thermal_last"."""
    scene = dataclasses.replace(_pan_scene(), init=None, companions=None)
    therm = ParticleThermal(materials=("water",), initial=(293.15,), heaters=("plate",), heater_modes=("body",),
                            heater_values=(0.0,), heater_gap=2.0e-4, support=2.2e-3)
    thermal = Thermal(bodies=("pan",), initial=(360.0,), specific_heat=(500.0,), convection=0.0, friction_heat=False)
    plugins = [thermal, therm] if order == "thermal_first" else [therm, thermal]
    return build_simulation(scene, plugins, "cpu"), therm


def test_body_heater_needs_a_link_and_thermal_first():
    sim, therm = _pan_sim("thermal_last")
    with pytest.raises(PluginError, match="before"):
        therm.link_thermal(sim.plugins)
    sim, therm = _pan_sim("thermal_first")
    with pytest.raises(PluginError, match="link_thermal"):
        sim.step_period()


def test_flow_values_follow_a_reset_before_the_first_step():
    """WP13 review of step 3: the flow arrays are model arrays, not state, so after sim.reset() the solver's next step
    must still see the values of the restored temperature (apply rewrites them before every step)."""
    from alloy.plugins.harness.scene import TestScene  # noqa: PLC0415
    from alloy.plugins.thermal.phase_laws import PASTE, SPREAD_S, SPREAD_VOXEL, _columns  # noqa: PLC0415

    therm = ParticleThermal(materials=(PASTE,), initial=(290.0,), support=2.2 * SPREAD_S)
    mpm = {"voxel_size": SPREAD_VOXEL, "grid_type": "dense"}
    sim = build_simulation(TestScene(build=_columns, worlds=1, periods=4, solver="mpm", solver_kwargs=mpm), [therm],
                           "cpu")
    sim.step_period()
    therm.set_temperature(310.0)                                  # melt it: μ 0.34 Pa·s, no yield
    sim.step_period()
    mpm = sim.model.mpm
    assert mpm.yield_stress.numpy().max() == 0.0
    sim.reset()                                                   # T back to 290 K; the model arrays are not state
    seen = {}
    for solver in sim.stepper.solvers():
        def step(*args, _orig=solver.step, **kw):
            seen.setdefault("viscosity", mpm.viscosity.numpy().copy())
            seen.setdefault("yield", mpm.yield_stress.numpy().copy())
            return _orig(*args, **kw)
        solver.step = step
    sim.step_period()
    m = material(PASTE)
    assert seen["viscosity"] == pytest.approx(m.mu_solid) and seen["yield"] == pytest.approx(m.yield_solid), seen


def test_device_temperature_matches_host():
    mats = [material("water"), MUSHY]
    h = np.concatenate([np.linspace(-5e4, 8e5, 41), np.linspace(-5e4, 3e5, 41)])
    mat = np.repeat([0, 1], 41).astype(np.int32)
    mt = K.MatTable()
    for field, attr in (("density", "density"), ("c_s", "c_solid"), ("c_l", "c_liquid"), ("k_s", "k_solid"),
                        ("k_l", "k_liquid"), ("t_s", "t_solidus"), ("t_l", "t_liquidus")):
        setattr(mt, field, wp.array([getattr(m, attr) for m in mats], dtype=wp.float64, device="cpu"))
    mt.h_l = wp.array([m.h_liquidus for m in mats], dtype=wp.float64, device="cpu")
    t, phi = (wp.zeros(h.size, dtype=wp.float64, device="cpu") for _ in range(2))
    wp.launch(K.derive, dim=h.size, device="cpu", inputs=[wp.array(h, dtype=wp.float64, device="cpu"),
                                                          wp.array(mat, dtype=wp.int32, device="cpu"), mt],
              outputs=[t, phi])
    for k, m in enumerate(mats):
        sel = mat == k
        want_t, want_phi = m.temperature(h[sel])
        np.testing.assert_allclose(t.numpy()[sel], want_t, rtol=0, atol=1e-9)
        np.testing.assert_allclose(phi.numpy()[sel], want_phi, rtol=0, atol=1e-12)


@pytest.mark.law
@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", ["null", "energy_closed", "energy_heater", "power_heater", "power_dead",
                                  "body_heater", "body_rate", "convection_area", "plate_vs_slab", "film_contact",
                                  "contact_erf", "diffusion_mode",
                                  "latent_plateau", "stefan_front",
                                  "boil_rate", "boil_off", "flow_arrays", "melt_spreads", "doneness",
                                  "doneness_ramp", "gelatinise_ramp", "flow_doneness"])
def test_particle_law(name):
    plugin = ParticleThermal()
    law = next(x for x in plugin.laws() if x.name == name)
    res = _law(law, None, [fresh(plugin, law.config)], "cpu")
    assert res.passed, f"{name}: {res.value} > {res.bound}: {res.detail} {res.fix}"
    assert math.isfinite(res.value)


@pytest.mark.timeout(600)
def test_harness_repeat_checks_on_a_busy_scene():
    """Determinism, world independence and graph replay where heat moves: a power plate, convection and a source that
    boils the bar off near the end (WP13 review of step 2: the old test scene never moved heat)."""
    report = check(ParticleThermal(**BUSY), laws=False, speed=False)
    assert report.passed, report.summary()

