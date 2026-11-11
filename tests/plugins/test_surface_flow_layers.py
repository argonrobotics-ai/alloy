# SPDX-License-Identifier: Apache-2.0
"""Surface flow, what process layers and the supply's pools use (phase 2): per-cell rheology inputs, the bulk's yield
per cell, the tools' work fields, flooded cells that give their film to a pool, and the film's travel (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.forming import Bulk
from tests.plugins.test_surface_flow import _plate
from tests.plugins.test_surface_flow_tools import punch_scene

G = 9.81
PASTE = FilmMaterial("paste_test", density=2000.0, viscosity=2.0, yield_stress=0.0, surface_tension=0.0)


def _drain(set_inputs, periods: int = 10, **config):
    """A 1 mm film on a vertical plate (40 x 1 cells, gravity along -x); ``set_inputs(flow)`` before the run."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1)),), material=PASTE, inertia=False,
                       max_depth=2.0e-3, **config)

    def init(sim) -> None:
        flow.set_depth(1.0e-3)
        set_inputs(flow)
    run = run_scene(TestScene(build=_plate(False), init=init, ground=False, gravity=(-G, 0.0, 0.0), periods=periods,
                              worlds=1), [flow], "cpu")
    assert run.error is None, run.error
    return flow


def test_per_cell_yield_stress_holds_one_half_and_lets_the_other_flow():
    """A setting layer's yield stress: 10 kPa on the upper half (x > 0) holds that film; the lower half flows."""
    def stiffen(flow):
        x = flow.cell_centres(0)[:, 0, 0]
        flow.set_input("yield_stress", np.where(x > 0.0, 1.0e4, 0.0).astype(np.float32))
    flow = _drain(stiffen, per_cell_rheology=True)
    h, x = flow.depth(0)[0, :, 0], flow.cell_centres(0)[:, 0, 0]
    assert np.abs(h[x > 0.01] - 1.0e-3).max() < 1.0e-9                   # held (the cell at the boundary drains)
    assert np.abs(h[x < -0.01] - 1.0e-3).max() > 1.0e-5                   # flowed


def test_per_cell_viscosity_never_falls_below_the_materials():
    """The step bound assumes the material's viscosity: a lower input is clamped, a higher one slows the film."""
    base = _drain(lambda f: None, per_cell_rheology=True).depth(0)
    low = _drain(lambda f: f.set_input("viscosity", 0.0), per_cell_rheology=True).depth(0)
    high = _drain(lambda f: f.set_input("viscosity", 20.0), per_cell_rheology=True).depth(0)
    assert np.array_equal(base, low)
    assert np.abs(high - 1.0e-3).max() < 0.5 * np.abs(base - 1.0e-3).max()


def test_travel_counts_how_far_the_film_moved():
    still = _drain(lambda f: f.set_input("yield_stress", 1.0e4), per_cell_rheology=True)
    moving = _drain(lambda f: None)
    assert still.read()["travel"].max() == 0.0
    assert moving.read()["travel"].max() > 1.0e-4


def test_flooded_cells_give_their_film_to_the_pool_with_an_exact_ledger():
    """The supply marks the cells under a pool; their film (and what flows in) goes to ``pooled``."""
    def flood(flow):
        x = flow.cell_centres(0)[:, 0, 0]
        flow.flooded.assign(np.where(x < -0.05, 0, -1).astype(np.int32)[None, :])
        flow.set_depth(1.0e-3)                                              # refreshes the wet cells
    flow = _drain(flood)
    r = flow.read()
    x = flow.cell_centres(0)[:, 0, 0]
    assert flow.depth(0)[0, x < -0.05, 0].max() == 0.0
    pooled = r["pooled_total_volume"][0]
    assert pooled > (x < -0.05).sum() * 1.0e-3 * flow.tables.area_np[0]   # its own film and more ran in
    assert r["pooled"][0].sum() > 0.0                                       # for the supply to collect
    start = 1.0e-3 * flow.tables.area_np.sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start


def test_bulk_yield_per_world_and_the_tools_pressure_field():
    """Two worlds, one punch load: the paste set to twice the yield holds; the soft one sinks. The tool's pressure
    field carries its weight."""
    bulk, k, radius, drop, load = Bulk(yield_stress=200.0, resistance=2.5e4), 5.0e6, 0.02, 0.02, 1.2
    density = load * bulk.bearing * np.pi * radius**2 / G / (0.05 * 0.05 * 0.02)
    punch = Applicator(body="tool", shape="disc", size=(radius, radius), center=(0.0, 0.0, -0.01 - drop),
                       stiffness=k, damping=0.004)
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(48, 48)),), material="concrete_paste", inertia=False,
                       max_depth=1.0e-3, tools=(punch,), bulk=bulk)

    def init(sim) -> None:
        flow.set_input("bulk_yield", np.asarray([[200.0], [400.0]], np.float32))
    sim = build_simulation(TestScene(build=punch_scene(density, -load * bulk.bearing / k, drop), init=init,
                                     ground=False, worlds=2, dt=2.0e-3), [flow], "cpu")
    for _ in range(10):
        sim.step_period()
    r = flow.read()
    moved = np.abs(r["bulk"]).sum(axis=1)
    assert moved[0] > 0.0 and moved[1] == 0.0
    weight = float(sim.model.body_mass.numpy()[1]) * G
    carried = (r["tool_pressure"][1] * r["tool_cover"][1] * flow.tables.area_np).sum()
    assert carried == pytest.approx(weight, rel=0.02, abs=0.0)


def test_pending_momentum_pushes_the_film_as_a_surface_traction():
    """src_mom (pending momentum per area: writers add, the film applies it over its step and zeroes it) is a surface
    traction T = M/dt on the film: T·dt added every exchange drives a level 1 mm water film as the model's
    ū' = (ū/dt_s + T/(ρh))/(1/dt_s + 3ν/h²) per substep (to 1e-4 on the faces away from the walls)."""
    T, h, dt = 0.05, 1.0e-3, 0.02                                   # Pa along x, m, s
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 4)),), material="water", max_depth=3.0e-3,
                       max_stride=1)

    def init(sim) -> None:
        flow.set_depth(h)
    sim = build_simulation(TestScene(build=_plate(False), init=init, ground=False, gravity=(0.0, 0.0, -G),
                                     worlds=1, dt=dt), [flow], "cpu")
    for _ in range(8):
        mom = flow.src_mom.numpy()
        mom[:, :, 0] += T * dt                                      # one exchange per 20 ms period
        flow.src_mom.assign(mom)
        sim.step_period()
    nu, rho = flow.mat.nu, flow.mat.density
    dt_s, u = dt / flow.n_sub, 0.0
    for _ in range(8 * flow.n_sub):
        u = (u / dt_s + T / (rho * h)) / (1.0 / dt_s + 3.0 * nu / (h * h))
    faces = np.arange(41 * 4).reshape(41, 4)[12:29].reshape(-1)     # i-faces (+x) in the middle 40 % of the plate
    got = flow.u.numpy()[0, faces]
    assert u > 1.0e-3 and np.abs(got / u - 1.0).max() < 1.0e-4, (got.min(), got.max(), u)
    assert not flow.src_mom.numpy().any()                           # consumed by the film step


def _stencil(regions: bool) -> SurfaceFlow:
    """Enamel (134 µm, with surface tension) on a plate beside a 0.3 mm stencil sheet (base step) over x > 0."""
    enamel = FilmMaterial("enamel_thin", viscosity=0.02, surface_tension=0.03)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 4)),), material=enamel, inertia=False,
                       max_depth=1.0e-3)

    def init(sim) -> None:
        x = flow.cell_centres(0)[..., 0]
        sheet = x > 0.0
        flow.set_input("base", np.where(sheet, 3.0e-4, 0.0).reshape(1, -1).astype(np.float32))
        flow.set_depth(np.where(sheet, 2.0e-5, 1.34e-4), chart=0)          # a little paint on the sheet too
        if regions:
            flow.set_regions(sheet.astype(np.int32), chart=0)
    run = run_scene(TestScene(build=_plate(False), init=init, ground=False, gravity=(0.0, 0.0, -G), periods=10,
                              worlds=1), [flow], "cpu")
    assert run.error is None, run.error
    return flow


def test_regions_wall_off_a_stencil_sheet_with_no_capillary_climb():
    """A stencil sheet as its own region: no film crosses its edge, the film beside it stays flat (no meniscus up the
    sheet's step, as GA's sealed stencil), and the paint on the sheet stays there. Control: the base step alone draws
    a meniscus."""
    walled, stepped = _stencil(True), _stencil(False)
    x = walled.cell_centres(0)[:, 0, 0]
    h = walled.depth(0)[0]
    assert np.abs(h[x < 0.0] - 1.34e-4).max() < 1.0e-9 * 1.34e-4 + 1.0e-12       # flat to the edge
    assert np.abs(h[x > 0.0] - 2.0e-5).max() < 1.0e-12
    assert np.abs(stepped.depth(0)[0][x < 0.0] - 1.34e-4).max() > 1.0e-8            # the step alone draws film
    r = walled.read()
    start = (np.where(x > 0.0, 2.0e-5, 1.34e-4)[:, None] * walled.tables.area_np.reshape(40, 4)).sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start


def _drain_floor(angle) -> float:
    """SC9's flush-floor in miniature: water on a floor sloped to a drain at its low end; the share left after 2 s."""
    from alloy.plugins.surface_flow.supply import Drain  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply_stub import slab_scene  # noqa: PLC0415

    water = FilmMaterial("water_60deg" if angle else "water_spread", surface_tension=0.0728, contact_angle=angle)
    chart = PlaneChart("slab", cells=(40, 40), edges=("wall",) * 4)
    sink = {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16}], "edges": []}
    flow = SurfaceFlow(charts=(chart,), material=water, max_depth=6.0e-3, network=sink,
                       drains=(Drain("hole", chart=0, center=(-0.17, 0.0, 0.0), radius=0.02, to="sink"),))

    def init(sim) -> None:
        flow.set_depth(2.0e-3)
    slope = math.atan(1.0 / 20.0)
    _run = run_scene(TestScene(build=slab_scene(), init=init, ground=True, worlds=1, periods=100,
                               gravity=(-G * math.sin(slope), 0.0, -G * math.cos(slope))), [flow], "cpu")
    assert _run.error is None, _run.error
    r = flow.read()
    start = 2.0e-3 * flow.tables.area_np.sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start
    return float(r["film_volume"][0] / start)


def test_a_drain_is_a_hole_not_dry_ground_for_a_contact_line():
    """SC9's finding: a drain empties its cells each step, and the contact-line condition took them as dry substrate,
    so a 60° film stood as a puddle at the drain's rim. With drains as holes it drains as a spreading film does."""
    left_angle, left_spread = _drain_floor(60.0), _drain_floor(None)
    assert left_spread < 0.8, left_spread                            # the drain takes the floor's film
    assert left_angle < left_spread + 0.05, (left_angle, left_spread)
