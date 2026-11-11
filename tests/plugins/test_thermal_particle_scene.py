# SPDX-License-Identifier: Apache-2.0
"""Heat v2 by object name on a public-API scene (alloy/plugins/thermal/particle_scene.py; light CPU tests)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.plugins.errors import PluginError
from alloy.plugins.thermal import particle_scene as PS

pytestmark = pytest.mark.law

FLOOR = 0.08 * 0.08   # m², the pan's inner floor
T0, T_PAN = 293.15, 353.15


def _pan():
    """Water 14 mm deep in a pan (the user guide's first quick start; the guide now uses the Scene API's heat)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0.0, 0.0, 0.02))   # floor top at 0.025
    scene.add_liquid("water", size=(0.07, 0.07, 0.02), pos=(0.0, 0.0, 0.0355))
    return alloy.build(scene, device="cpu").built


def _two_liquids():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0.0, 0.0, 0.02))
    scene.add_liquid("water", size=(0.07, 0.07, 0.02), pos=(0.0, 0.0, 0.0355), material="water")
    scene.add_open_box("cup", inner=(0.04, 0.04, 0.04), wall=0.005, pos=(0.2, 0.0, 0.0))
    scene.add_liquid("tea", size=(0.03, 0.03, 0.02), pos=(0.2, 0.0, 0.0155), material="water")
    return alloy.build(scene, device="cpu").built


@pytest.mark.timeout(300)   # ~55 s on CPU: build plus 150 periods
def test_water_on_a_hot_pan_floor_keeps_heater_contact_and_closes_the_ledger():
    """for_scene on an 80 °C pan floor, 90 s of heating (time scale 60)."""
    built = _pan()
    therm = PS.for_scene(built, objects={"water": ("water", T0)}, heaters={"pan_floor": T_PAN}, time_scale=60.0)
    sim = alloy.plugins.simulation(built, [therm])
    for _ in range(150):
        sim.step_period()
    r = therm.read()
    s = PS.object_summary(sim, "water")
    # SPH rests its bottom layer ~0.1 mm beyond a radius off the floor: with no gap the touching area fell from 28 % to
    # 4 % of the floor in 1.5 s; the default gap (a quarter spacing) keeps the whole layer touching (1.05-1.10 floors)
    spacing = (float(built.model.particle_mass.numpy()[0]) / therm.materials()[0].density) ** (1.0 / 3.0)
    assert therm.heater_gap == pytest.approx(0.25 * spacing, rel=1e-6)
    assert r["heater_area_m2"][0, 0] > 0.9 * FLOOR, r["heater_area_m2"]
    assert s["particles"][0] == built.model.particle_count and r["neighbor_overflow"][0] == 0
    assert T0 < s["mean_K"][0] < s["max_K"][0] <= T_PAN and s["min_K"][0] >= T0 - 1e-9, s
    m, h0 = built.model.particle_mass.numpy().astype(np.float64), therm.materials()[0].enthalpy(T0)
    start = r["stored_J"][0] - r["heat_in_J"][0] + r["heat_out_J"][0]
    assert start == pytest.approx(float(m.sum() * h0), rel=1e-9, abs=0.0)
    assert np.array_equal(PS.object_temperature(sim, "water"), r["temperature"])


def test_names_map_to_their_columns_and_bad_names_say_what_exists():
    built = _two_liquids()
    therm = PS.for_scene(built, objects={"tea": ("water", 300.0), "water": ("water", T0)},
                         heaters={"pan_floor": ("power", 50.0)})
    sim = alloy.plugins.simulation(built, [therm])
    tea, water = PS.columns(therm, built, "tea"), PS.columns(therm, built, "water")
    assert tea.start == 0 and water.start == tea.stop and water.stop == therm.P
    assert np.allclose(PS.object_temperature(sim, "tea"), 300.0)
    assert np.allclose(PS.object_temperature(sim, "water"), T0)
    assert therm.config.heater_modes == ("power",) and therm.config.heater_values == (50.0,)
    with pytest.raises(PluginError, match="pan_floor"):
        PS.for_scene(built, objects={"soup": ("water", T0)})
    with pytest.raises(PluginError, match="no particles"):
        PS.for_scene(built, objects={"pan_floor": ("water", T0)})


def test_heated_mpm_paste_gets_its_voxel_as_the_heater_gap():
    """MPM rests particles anywhere within a voxel of a collider (WP13 combo sweep F5: beside a coupled robot a paste
    rested 1.5 mm higher and took no heat at a quarter-spacing gap), so for_scene defaults the gap to the voxel."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.1, 0.1, 0.02), pos=(0.0, 0.0, 0.01), static=True)
    scene.add_box("paste", size=(0.04, 0.04, 0.02), pos=(0.0, 0.0, 0.031), material="peanut_butter")
    built = alloy.build(scene, device="cpu").built
    voxel = next(e.solver_kwargs["voxel_size"] for e in built.entries if e.capability == "mpm")
    therm = PS.for_scene(built, objects={"paste": ("porridge_oat", T0)}, heaters={"plate": T_PAN})
    assert PS.mpm_voxel(built, ["paste"]) == voxel and therm.config.heater_gap == voxel
    assert PS.for_scene(built, objects={"paste": ("porridge_oat", T0)}, heater_gap=1.0e-3).config.heater_gap == 1.0e-3
    assert PS.mpm_voxel(_two_liquids(), ["water", "tea"]) is None          # SPH keeps the quarter-spacing default
