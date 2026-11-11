# SPDX-License-Identifier: Apache-2.0
"""Surface flow, species the film carries (phase 2: surfactant, soil, sanitizer): they ride the film's fluxes without
spurious mixing, leave with it, stay behind when it evaporates, and keep exact ledgers (CPU)."""

from __future__ import annotations

import math

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from tests.plugins.test_surface_flow import _plate

G = 9.81
WATER = FilmMaterial("water_flat", surface_tension=0.0)


def test_a_uniform_tracer_stays_uniform_and_leaves_with_the_film():
    """A dye at 2 kg/m³ in a film draining off an open edge: its concentration stays 2 everywhere, and what leaves is
    exactly the spilled volume times 2; the species ledger closes."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                       material=WATER, max_depth=2.0e-3, species=("dye",))

    def init(sim) -> None:
        x = flow.cell_centres(0)[:, 0, 0]
        h = (1.0e-3 + 5.0e-4 * np.cos(math.pi * x / 0.1))[:, None]
        flow.set_depth(h, chart=0)
        flow.set_species(2.0 * h, "dye", chart=0)
    run = run_scene(TestScene(build=_plate(False), init=init, ground=False, gravity=(-G, 0.0, 0.0), periods=10,
                              worlds=1), [flow], "cpu")
    assert run.error is None, run.error
    r = flow.read()
    h, m = flow.depth(0)[0], flow.species("dye", 0)[0]
    wet = h > 1.0e-6
    assert np.abs(m[wet] / h[wet] - 2.0).max() < 1.0e-9
    spilled = r["spilled_volume"][0]
    assert spilled > 0.0
    assert abs(r["species_out"][0].sum() - 2.0 * spilled) < 1.0e-9 * 2.0 * spilled
    start = 2.0 * r["film_volume"][0] + 2.0 * spilled
    assert abs(r["species_ledger"][0, 0] - start) < 1.0e-12 * start


def test_evaporation_concentrates_a_species_and_leaves_its_residue():
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),),
                       material=FilmMaterial("drying", surface_tension=0.0, evaporation=2.0e-5), max_depth=1.0e-3,
                       species=("salt",))

    def init(sim) -> None:
        flow.set_depth(4.0e-4)
        flow.set_species(4.0e-4 * 10.0, "salt")                   # 10 kg/m³
    run = run_scene(TestScene(build=sl.plate_scene(0.1, 0.1), init=init, ground=False, gravity=(0.0, 0.0, -G),
                              periods=25, worlds=1), [flow], "cpu")            # 0.5 s: 10 µm evaporates
    assert run.error is None, run.error
    h, m = flow.depth(0)[0], flow.species("salt", 0)[0]
    assert np.allclose(h, 3.9e-4, rtol=1e-3, atol=0.0)                        # the float32 depth, over many substeps
    assert np.allclose(m, 4.0e-3, rtol=1e-12, atol=0.0)                       # the salt stays: 10 · 0.4/0.39 kg/m³ now
    r = flow.read()
    assert r["species_out"][0].sum() == 0.0
    dry = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),),
                      material=FilmMaterial("drying", surface_tension=0.0, evaporation=1.0e-3), max_depth=1.0e-3,
                      species=("salt",))

    def init_dry(sim) -> None:
        dry.set_depth(1.0e-4)
        dry.set_species(1.0e-3, "salt")
    run_scene(TestScene(build=sl.plate_scene(0.1, 0.1), init=init_dry, ground=False, gravity=(0.0, 0.0, -G),
                        periods=10, worlds=1), [dry], "cpu")
    assert dry.depth(0)[0].max() == 0.0                                                  # dried: the salt is residue
    assert np.allclose(dry.species("salt", 0)[0], 1.0e-3, rtol=1e-12, atol=0.0)


def test_the_dry_tile_skip_stays_exact_with_species():
    def run(skip: bool) -> tuple[np.ndarray, np.ndarray]:
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(96, 96)),), material="water", max_depth=2.0e-3,
                           skip_dry=skip, species=("soil", "soap"))

        def init(sim) -> None:
            xy = flow.cell_centres(0)[..., :2]
            r2 = ((xy[..., 0] - 0.035) ** 2 + (xy[..., 1] + 0.035) ** 2) / 0.006 ** 2
            h = 1.5e-3 * np.clip(1.0 - r2, 0.0, None)
            flow.set_depth(h, chart=0)
            flow.set_species(3.0 * h, "soil", chart=0)
            flow.set_species(1.0e-4 * (xy[..., 0] > 0.0), "soap", chart=0)    # some soap on dry cells: residue
        sim = build_simulation(TestScene(build=sl.plate_scene(0.096, 0.096), init=init, ground=False,
                                         gravity=(-2.0, 0.0, -G), worlds=1, dt=2.0e-3), [flow], "cpu")
        for _ in range(8):
            sim.step_period()
        return flow.h.numpy(), flow.sp.mass.numpy()
    dense, skip = run(False), run(True)
    assert np.array_equal(dense[0], skip[0]) and np.array_equal(dense[1], skip[1])



def test_a_drain_takes_soap_with_the_rinse_water():
    """Clean water rinses a soapy slab into a grate: the soap the drain takes (``drained_species``) is part of what
    left, the soap ledger closes, and the clean water dilutes what stays (no cell above the start's concentration).
    The rinse water is a host source: the faucet jet's sweep does not carry species yet (DC5, with the transfers)."""
    from alloy.plugins.surface_flow.supply import Drain  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply_stub import CHART, slab_scene  # noqa: PLC0415

    grate = Drain("grate", chart=0, center=(0.12, 0.0, 0.0), radius=0.03, to="sink")
    sink = {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16}], "edges": []}
    flow = SurfaceFlow(charts=(CHART,), material="water", drains=(grate,), network=sink, max_depth=3.0e-3,
                       species=("soap",))

    def init(sim) -> None:
        flow.set_depth(5.0e-4)
        flow.set_species(5.0e-4 * 20.0, "soap")                          # 20 kg/m³
        x = flow.cell_centres(0)[..., 0]
        flow.set_source(np.where((x > 0.05) & (x < 0.09), 2.0e-3, 0.0), chart=0)    # 2 mm/s of clean water
    run = run_scene(TestScene(build=slab_scene(), init=init, ground=True, worlds=1, periods=25), [flow], "cpu")
    assert run.error is None, run.error
    r = flow.read()
    area = flow.tables.area_np
    start = 5.0e-4 * 20.0 * area.sum()
    drained = r["supply_species_tanks"][0, 0] + r["supply_species_away"][0, 0]   # the supply takes drained_species
    assert drained > 0.01 * start and drained <= r["species_out"][0, :, 0].sum() * (1.0 + 1e-12)
    assert abs(r["species_ledger"][0, 0] - start) < 1e-12 * start
    h, m = flow.depth(0)[0], flow.species("soap", 0)[0]
    wet = h > 1.0e-5
    assert (m[wet] / h[wet]).max() < 20.0 * (1.0 + 1e-9) and (m[wet] / h[wet]).min() < 19.0    # diluted


def test_a_source_brings_its_species_at_its_concentration():
    """A dye solution (3 kg/m³) poured on a dry plate: every wet cell holds dye at 3 kg/m³, and the dye supplied is
    exactly the volume supplied times 3."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material=WATER, max_depth=2.0e-3,
                       species=("dye",))
    def init(sim) -> None:
        flow.set_source(1.0e-4, species={"dye": 3.0})
    run = run_scene(TestScene(build=sl.plate_scene(0.1, 0.1), init=init, ground=False, gravity=(0.0, 0.0, -G),
                              periods=5, worlds=1), [flow], "cpu")
    assert run.error is None, run.error
    r = flow.read()
    h, m = flow.depth(0)[0], flow.species("dye", 0)[0]
    assert (h > 0.0).all() and np.abs(m / h - 3.0).max() < 1.0e-9
    dye = r["species_supplied"][0, :, 0].sum()
    assert abs(dye - 3.0 * r["supplied_volume"][0]) < 1e-12 * dye


def test_a_strided_source_changing_rate_and_species_every_period_supplies_exactly():
    """Paint whose source moves and whose solids change every period, on a film striding 16 exchanges (4 periods):
    the solids supplied equal the sum of rate x concentration x period, as when stepping every exchange."""
    def run(max_stride: int):
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("wall",) * 4),), material="latex_paint",
                           inertia=False, max_depth=3.0e-4, max_stride=max_stride, species=("solids",))
        sim = build_simulation(TestScene(build=_plate(False), init=lambda s: flow.set_depth(1.0e-4), ground=False,
                                         gravity=(-G, 0.0, 0.0), worlds=1, dt=5.0e-3), [flow], "cpu")
        want = 0.0
        for p in range(24):
            rate = np.zeros((40, 1))
            rate[5 + p // 2: 8 + p // 2] = 2.0e-5 * (1 + p % 3)
            conc = 400.0 + 50.0 * (p % 4)
            flow.set_source(rate, chart=0, species={"solids": conc})
            want += (rate.reshape(-1) * flow.tables.area_np).sum() * 0.02 * conc
            sim.step_period()
        return flow, want
    for k in (1, 16):
        flow, want = run(k)
        assert flow.stride == k
        r = flow.read()
        assert abs(r["species_supplied"][0, :, 0].sum() - want) < 1e-12 * want
        assert abs(r["species_ledger"][0, 0] - 0.0) < 1e-12 * want


def test_a_roller_lays_its_store_species_and_takes_the_films():
    """A loaded roller with solids at 500 kg/m³ lays them at that concentration on a dry plate; an empty roller over
    a dyed film takes dye with the film, leaving the concentration unchanged. Species ledgers exact."""
    from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415
    from alloy.plugins.surface_flow.stores import Store  # noqa: PLC0415
    from tests.plugins.test_surface_flow_stores import NX, R, U, VISCOUS, HW, roller_scene  # noqa: PLC0415

    def roll(store: Store, depth: float, axis_z: float, species: str):
        roller = Applicator(body="cover", shape="roller", size=(R, HW), axis=(0.0, 1.0, 0.0), stiffness=1.0e3,
                            store=store)
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(NX, 4)),), material=VISCOUS, inertia=False,
                           max_depth=1.0e-3, tools=(roller,), species=(species,))

        def init(sim) -> None:
            flow.set_depth(depth)
            flow.set_species(depth * 7.0, species)                     # 7 kg/m³ in the film at the start
            qd = sim.state.joint_qd.numpy().reshape(-1, 2)
            qd[:] = (U, U / R)
            sim.state.joint_qd.assign(qd.reshape(-1))
            newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
        sim = build_simulation(TestScene(build=roller_scene(axis_z, -0.1), init=init, ground=False,
                                         gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3), [flow], "cpu")
        for _ in range(40):
            sim.step_period()
        return flow
    lay = roll(Store(capacity=6.0e-6, initial=1.0, drip=0.0, species={"solids": 500.0}), 0.0, R, "solids")
    h, m = lay.depth(0)[0], lay.species("solids", 0)[0]
    wet = h > 1.0e-6
    assert wet.sum() >= 20 and np.abs(m[wet] / h[wet] - 500.0).max() < 1.0e-6 * 500.0
    r = lay.read()
    assert abs(r["species_ledger"][0, 0]) < 1e-12 * r["species_supplied"].sum()
    take = roll(Store(capacity=1.0e-5, split=0.5), 1.0e-4, R + 0.5e-4, "dye")
    h, m = take.depth(0)[0], take.species("dye", 0)[0]
    wet = h > 1.0e-6
    assert np.abs(m[wet] / h[wet] - 7.0).max() < 1.0e-6 * 7.0
    r = take.read()
    start = 7.0e-4 * take.tables.area_np.sum()
    assert r["species_out"][0].sum() > 0.03 * start                 # half the film on the ~4 cm rolled over
    assert abs(r["species_ledger"][0, 0] - start) < 1e-12 * start
