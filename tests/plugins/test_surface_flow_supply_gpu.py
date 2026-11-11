# SPDX-License-Identifier: Apache-2.0
"""Surface-flow supply on the GPU (light-test rule: one smoke, 64 worlds or fewer). SurfaceFlow with a faucet, a drain
and a tank through the full harness check on CUDA graphs (determinism, identical and isolated worlds, graph replay);
the supply's ledgers on the GPU; and a cost row: a faucet over a wet plate at 64 worlds, with and without the supply."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import warp as wp

import alloy.plugins as ap
from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.supply import Drain, Nozzle

G = 9.81


def _network() -> dict:
    """Mains -> valve -> line -> spout (~0.1 L/s); the sink tank drains through its waste orifice."""
    return {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4},
                      {"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16}],
            "edges": [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + 2e5, "cd_area": 5e-6,
                       "dp_lam": 1.0},
                      {"name": "spout", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 1e-4,
                       "dp_lam": 1.0},
                      {"name": "waste", "kind": "orifice", "a": "sink", "reservoir": P_ATM, "cd_area": 2e-5,
                       "dp_lam": 1.0}]}


def _flow(cells: tuple[int, int] = (50, 50), length: float = 0.2) -> SurfaceFlow:
    jet = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=1e-4)
    grate = Drain("grate", chart=0, center=(0.35 * length, 0.0, 0.0), radius=0.1 * length, to="sink")
    return SurfaceFlow(charts=(PlaneChart("plate", cells=cells, edges=("open",) * 4),), material="water",
                       nozzles=(jet,), drains=(grate,), network=_network())


@pytest.mark.gpu
def test_supply_harness_full_check_on_gpu():
    """The film's own laws run without a supply (WP15's harness); here the hooks, determinism, world independence and
    graph replay with the jets, the drain and the network in."""
    report = ap.check(_flow(), device="cuda:0", laws=False)
    print(report.summary())
    assert report.passed, report.summary()


@pytest.mark.gpu
def test_supply_ledgers_close_on_gpu():
    flow = _flow()
    run = run_scene(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=8, periods=25), [flow], "cuda:0",
                    graph=True)
    assert run.error is None, run.error
    p = run.sim.plugins[flow.label]
    r, s = p.read(), p.supply.read()
    print("sent", s["sent"][0], "swept", s["swept"][0], "drained", s["drained"][0], "film ledger", r["ledger"][0])
    assert np.all(s["sent"] > 0) and np.allclose(r["ledger"], 0.0, atol=1e-15)
    assert np.allclose(s["inflow_total"][:, 1], r["drained_total_volume"], rtol=1e-12)
    assert np.all(s["sent"] == s["sent"][0])                                   # identical worlds stay identical


def _ms(scene: TestScene, plugins: list, periods: int = 10) -> float:
    sim = build_simulation(scene, plugins, "cuda:0", graph=True)
    sim.step_period()
    wp.synchronize_device("cuda:0")
    best = math.inf
    for _ in range(2):
        s0, t0 = sim.steps, time.perf_counter()
        for _ in range(periods):
            sim.step_period()
        wp.synchronize_device("cuda:0")
        best = min(best, (time.perf_counter() - t0) / max(sim.steps - s0, 1))
    return 1e3 * best


@pytest.mark.gpu
def test_supply_cost_row_64_worlds():
    """A faucet over a 0.3 x 0.3 m plate at 3 mm (100 x 100 cells) with a drain into a draining tank, 64 worlds:
    ms per exchange with the supply, the film alone (no nozzles), and neither."""
    def scene(flow: SurfaceFlow | None) -> TestScene:
        def init(sim) -> None:
            if flow is not None and "surface_flow" in sim.plugins.plugins:
                flow.set_depth(2.0e-4)
        return TestScene(build=sl.plate_scene(0.3, 0.3), init=init, ground=False, worlds=64, periods=10, dt=2.0e-3)

    with_supply = _flow((100, 100), 0.3)
    film = SurfaceFlow(charts=(PlaneChart("plate", cells=(100, 100), edges=("open",) * 4),), material="water")
    t_supply = _ms(scene(with_supply), [with_supply])
    t_film = _ms(scene(film), [film])
    t_none = _ms(scene(None), [])
    print(f"faucet + drain + tank over a 100 x 100 film, 64 worlds: {t_supply:.3f} ms/exchange with the supply, "
          f"{t_film:.3f} film alone, {t_none:.3f} without ({t_supply / t_none:.2f}x; supply "
          f"{t_supply - t_film:.3f} ms)")
    assert t_supply > 0.0


@pytest.mark.gpu
def test_spray_cost_row_64_worlds():
    """A 64-ray aerosol (30° cone) over the same plate, 64 worlds: ms per exchange with the spray vs the film alone."""
    can = {"nodes": [{"name": "can", "volume": 1e-4, "pressure": 4.0e5}],
           "edges": [{"name": "valve", "kind": "expel", "a": "can", "cd_area": 8.2e-8, "density": 1000.0,
                      "dp_lam": 1.0, "capacity": 3e-4}]}
    spray = Nozzle("can", edge="valve", pos=(0.0, 0.0, 0.2), exit_area=8.2e-8, pattern="spray",
                   angle=math.radians(30), rays=64, efficiency=0.7, density=1000.0)
    chart = PlaneChart("plate", cells=(100, 100), edges=("open",) * 4)
    with_spray = SurfaceFlow(charts=(chart,), material="water", nozzles=(spray,), network=can)
    film = SurfaceFlow(charts=(chart,), material="water")

    def scene(flow: SurfaceFlow) -> TestScene:
        def init(sim) -> None:
            if "surface_flow" in sim.plugins.plugins:
                flow.set_depth(2.0e-4)
        return TestScene(build=sl.plate_scene(0.3, 0.3), init=init, ground=False, worlds=64, periods=10, dt=2.0e-3)

    t_spray, t_film = _ms(scene(with_spray), [with_spray]), _ms(scene(film), [film])
    print(f"64-ray aerosol over a 100 x 100 film, 64 worlds: {t_spray:.3f} ms/exchange with the spray, {t_film:.3f} "
          f"film alone (spray {t_spray - t_film:.3f} ms)")
    assert t_spray > 0.0
