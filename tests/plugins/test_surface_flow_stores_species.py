# SPDX-License-Identifier: Apache-2.0
"""Surface flow, species in stores and trays (stores_species.py): a well's pigment goes into the brush with the paint,
a store's bins mix what they pick up and take back, and lay at their own concentration (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.stores import Store, Tray
from tests.plugins.test_surface_flow_stores import NX, R, VISCOUS, roll, roller_scene

PIGMENT = 400.0                                                   # kg/m³ in each well and the start load


def test_a_pad_dipped_in_a_blue_well_mixes_blue_into_its_red_load():
    """A pad half full of red paint dips in a blue well: the well's blue comes in with the paint it picks up, the red
    stays. Every kilogram is booked: the store's and the well's species ledgers close exactly."""
    cap, vol = 2.0e-6, 2.0e-4
    pad = Applicator(body="cover", shape="disc", size=(0.02, 0.02), center=(0.0, 0.0, -R), stiffness=1.0e3,
                     store=Store(capacity=cap, pickup=2.0, initial=0.5, species={"red": PIGMENT}))
    well = Tray(center=(-0.1, 0.0, -0.05), half=(0.05, 0.05), volume=vol, species={"blue": PIGMENT})
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, inertia=False,
                       max_depth=1.0e-3, tools=(pad,), trays=(well,), species=("red", "blue"))
    sim = build_simulation(TestScene(build=roller_scene(-0.05 + R + 0.01, -0.1), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(25):
        sim.step_period()
    r = flow.read()
    load = float(r["tool_load"][0].sum())
    red, blue = r["tool_load_species"][0, 0, 0], r["tool_load_species"][0, 0, 1]
    assert load > 0.6 * cap
    assert red == pytest.approx(PIGMENT * 0.5 * cap, rel=1e-12, abs=0.0)              # the red stays
    assert blue == pytest.approx(PIGMENT * (load - 0.5 * cap), rel=1e-12, abs=0.0)    # the blue it picked up
    assert r["tray_species"][0, 0, 1] == pytest.approx(PIGMENT * float(r["tray_volume"][0, 0]), rel=1e-12, abs=0.0)
    assert np.allclose(r["store_species_ledger"][0], [PIGMENT * 0.5 * cap, 0.0], rtol=1e-12, atol=0.0)
    assert np.allclose(r["tray_species_ledger"][0], [0.0, PIGMENT * vol], rtol=1e-12, atol=0.0)


def test_a_well_without_species_holds_the_stores_paint():
    """``Tray(species=None)`` (every well before p36) holds the stores' own paint: a red pad dipping in it stays at
    400 kg/m³ red, and the well's red is 400 x its volume. Two stores of different paints make it ambiguous."""
    cap, vol = 2.0e-6, 2.0e-4
    pad = Applicator(body="cover", shape="disc", size=(0.02, 0.02), center=(0.0, 0.0, -R), stiffness=1.0e3,
                     store=Store(capacity=cap, pickup=2.0, initial=0.5, species={"red": PIGMENT}))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, inertia=False,
                       max_depth=1.0e-3, tools=(pad,), species=("red",),
                       trays=(Tray(center=(-0.1, 0.0, -0.05), half=(0.05, 0.05), volume=vol),))
    sim = build_simulation(TestScene(build=roller_scene(-0.05 + R + 0.01, -0.1), ground=False,
                                     gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3), [flow], "cpu")
    for _ in range(10):
        sim.step_period()
    r = flow.read()
    load = float(r["tool_load"][0].sum())
    assert load > 0.55 * cap
    assert r["tool_load_species"][0, 0, 0] == pytest.approx(PIGMENT * load, rel=1e-12, abs=0.0)
    assert r["tray_species"][0, 0, 0] == pytest.approx(PIGMENT * float(r["tray_volume"][0, 0]), rel=1e-12, abs=0.0)
    two = (pad, Applicator(body="cover", shape="disc", size=(0.02, 0.02), store=Store(capacity=cap)))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, tools=two, species=("red",),
                       trays=(Tray(center=(0.0, 0.0, 0.0), half=(0.01, 0.01), volume=vol),))
    with pytest.raises(PluginError, match="different ones"):
        build_simulation(TestScene(build=roller_scene(0.05, -0.1), ground=False, gravity=(0.0, 0.0, 0.0), worlds=1,
                                   dt=2.0e-3), [flow], "cpu")


def test_a_pad_lays_the_dye_it_took_back_from_the_film():
    """An empty pad sliding from a dyed film (7 kg/m³) onto a dry plate takes the dye with the film and lays it at
    that concentration on the dry part (v1 laid the store's own, here none). The books close."""
    h0, c0 = 1.0e-4, 7.0
    x_dry = -0.085

    def depth(flow) -> np.ndarray:
        x = np.repeat(flow.cell_centres(0)[:, 0, 0], 4)
        return np.where(x < x_dry, h0, 0.0).reshape(1, -1)

    def setup(flow) -> None:
        flow.set_species(depth(flow) * c0, "dye")
    store = Store(capacity=1.0e-5, retained=0.0, held=1.0, split=0.5, pickup=0.0, drip=0.0)
    flow, sim = roll(store, depth, 0.05, periods=40, blade_gap=0.0, flow_kw={"species": ("dye",)}, setup=setup)
    x = np.repeat(flow.cell_centres(0)[:, 0, 0], 4)
    h, m = flow.h.numpy()[0], flow.sp.mass.numpy()[0, :, 0]
    laid = (x > x_dry + 0.01) & (h > 1.0e-9)
    assert laid.sum() >= 8
    assert np.allclose(m[laid] / h[laid], c0, rtol=1e-12, atol=0.0)                  # the dye it took back
    r = flow.read()
    area = flow.tables.area_np
    start = float((depth(flow)[0] * c0 * area).sum())
    assert abs(float(r["species_ledger"][0, 0]) - start) <= 1e-12 * start
    assert abs(float(r["store_species_ledger"][0, 0])) <= 1e-12 * start
    held = float(r["tool_load_species"][0, 0, 0])
    assert held == pytest.approx(c0 * float(r["tool_load"][0, 0]), rel=1e-12, abs=0.0)


def test_tray_species_are_checked():
    with pytest.raises(PluginError, match="must be finite and >= 0"):
        Tray(center=(0.0, 0.0, 0.0), half=(0.01, 0.01), volume=1.0e-6, species={"blue": -1.0})
    pad = Applicator(body="cover", shape="disc", size=(0.02, 0.02), store=Store(capacity=1.0e-6))
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, tools=(pad,), species=("red",),
                       trays=(Tray(center=(0.0, 0.0, 0.0), half=(0.01, 0.01), volume=1.0e-6, species={"blue": 1.0}),))
    with pytest.raises(PluginError, match="no species 'blue'"):
        build_simulation(TestScene(build=roller_scene(0.05, -0.1), ground=False, gravity=(0.0, 0.0, 0.0), worlds=1,
                                   dt=2.0e-3), [flow], "cpu")
