# SPDX-License-Identifier: Apache-2.0
"""Surface-flow film core on the GPU (light-test rule: one smoke). The full harness check with CUDA graphs, and cost
rows at 64 worlds: a painted wall and a rinsed plate, with and without the film."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import warp as wp

import alloy.plugins as ap
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl

G = 9.81


@pytest.mark.gpu
def test_harness_full_check_on_gpu():
    report = ap.check(SurfaceFlow(), device="cuda:0")
    print(report.summary())
    assert report.passed, report.summary()


def _ms_per_exchange(scene: TestScene, plugins: list, periods: int = 10) -> float:
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


def _row(name: str, length: float, width: float, cells: tuple[int, int], gravity: tuple[float, float, float],
         depth: float, wet: float = 1.0, **config) -> dict[str, float]:
    """ms per exchange step at 64 worlds: the bare scene, the film stepped densely and with the dry-tile skip. ``wet``:
    the share of the plate's length the film covers (a strip from x = -length/2), 1 = all wet (the skip's worst case)."""
    def init_for(flow: SurfaceFlow):
        def init(sim) -> None:
            if "surface_flow" not in sim.plugins.plugins:
                return
            xyz = flow.cell_centres(0)
            x, y = xyz[..., 0], xyz[..., 1]
            h = depth * (1.0 + 0.5 * np.cos(20.0 * x) * np.cos(20.0 * y))
            flow.set_depth(np.where(x < -0.5 * length + wet * length, h, 0.0), chart=0)
        return init

    def scene(flow: SurfaceFlow | None) -> TestScene:
        return TestScene(build=sl.plate_scene(length, width), init=init_for(flow) if flow else None, ground=False,
                         gravity=gravity, worlds=64, periods=10, dt=2.0e-3)

    row = {"bare": _ms_per_exchange(scene(None), [])}
    for mode, skip in (("dense", False), ("skip", True)):
        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=cells),), skip_dry=skip, **config)
        row[mode] = _ms_per_exchange(scene(flow), [flow])
        if skip:
            row["tiles"] = float(flow.read()["tiles_active"].mean())
            row["n_sub"], row["stride"] = flow.n_sub, flow.stride
    print(f"{name}: {cells[0]}x{cells[1]} cells x 64 worlds, wet {wet:.0%}: dense {row['dense']:.3f} ms/exchange, "
          f"skip {row['skip']:.3f} ({row['tiles']:.0%} of tiles active), bare {row['bare']:.3f}; n_sub {row['n_sub']}, "
          f"stride {row['stride']}")
    return row


@pytest.mark.gpu
def test_cost_rows_64_worlds():
    paint = {"material": "latex_paint", "inertia": False, "max_depth": 4.0e-4}
    water = {"material": "water", "max_depth": 2.0e-3}
    wall = ("paint wall 0.6 x 1.2 m at 4 mm", 0.6, 1.2, (150, 300), (0.0, -G, 0.0), 1.5e-4)
    plate = ("water plate 0.3 x 0.3 m at 3 mm", 0.3, 0.3, (100, 100), (-1.0, 0.0, -G), 5.0e-4)
    rows = [_row(*wall, **paint), _row(*wall, wet=0.15, **paint), _row(*plate, **water), _row(*plate, wet=0.15, **water)]
    assert all(r["skip"] > 0.0 for r in rows)
    assert rows[3]["skip"] < rows[3]["dense"], "the skip did not pay on a mostly dry plate"


@pytest.mark.gpu
def test_species_and_a_moving_source_in_cuda_graphs():
    """Phase-2 additions under graph capture: a dye stays at its concentration in a draining film with its ledger
    exact, and a source moved every period on a strided paint film supplies exactly what was set."""
    from tests.plugins.test_surface_flow import _plate  # noqa: PLC0415

    dye = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                      material="water", max_depth=2.0e-3, species=("dye",))

    def init(sim) -> None:
        x = dye.cell_centres(0)[:, 0, 0]
        h = (1.0e-3 + 5.0e-4 * np.cos(math.pi * x / 0.1))[:, None]
        dye.set_depth(h, chart=0)
        dye.set_species(2.0 * h, "dye", chart=0)
    sim = build_simulation(TestScene(build=_plate(False), init=init, ground=False, gravity=(-G, 0.0, 0.0), worlds=4),
                           [dye], "cuda:0", graph=True)
    for _ in range(10):
        sim.step_period()
    r = dye.read()
    h, m = dye.depth(0), dye.species("dye", 0)
    wet = h > 1.0e-6
    assert np.abs(m[wet] / h[wet] - 2.0).max() < 1.0e-6
    start = 2.0 * (r["film_volume"] + r["spilled_volume"])
    assert np.abs(r["species_ledger"][:, 0] - start).max() < 1.0e-9 * start.max()

    paint = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("wall",) * 4),), material="latex_paint",
                        inertia=False, max_depth=3.0e-4)
    sim = build_simulation(TestScene(build=_plate(False), init=lambda s: paint.set_depth(1.0e-4), ground=False,
                                     gravity=(-G, 0.0, 0.0), worlds=4, dt=5.0e-3), [paint], "cuda:0", graph=True)
    want = 0.0
    for p in range(24):
        rate = np.zeros((40, 1))
        rate[5 + p // 2: 8 + p // 2] = 2.0e-5 * (1 + p % 3)
        paint.set_source(rate, chart=0)
        want += (rate.reshape(-1) * paint.tables.area_np).sum() * 0.02
        sim.step_period()
    assert paint.stride == 16
    assert np.abs(paint.read()["supplied_volume"] - want).max() < 1e-9 * want


@pytest.mark.gpu
def test_structure_and_solids_in_cuda_graphs():
    """rheology.py under graph capture: Moore's rebuild at rest is exact, and a solids film drains (Krieger)."""
    from alloy.plugins.surface_flow.rheology import Thixotropy  # noqa: PLC0415

    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material="water", max_depth=1.0e-3,
                       thixotropy=Thixotropy(rebuild=2.0), species=("solids",), krieger=0.64)

    def init(sim) -> None:
        flow.set_depth(1.0e-4)
        flow.set_species(3.0e-5, "solids")
        flow.fluid.lam.zero_()
    sim = build_simulation(TestScene(build=sl.plate_scene(0.1, 0.1), init=init, ground=False,
                                     gravity=(0.0, 0.0, -G), worlds=4), [flow], "cuda:0", graph=True)
    for _ in range(25):
        sim.step_period()
    t = (int(flow.tick.numpy()[0]) // flow.stride) * flow.stride * flow._exchange_dt
    assert np.allclose(flow.read()["lam"], 1.0 - math.exp(-t / 2.0), rtol=1e-5, atol=0.0)
    assert np.allclose(flow.fluid.vfac.numpy(), (1.0 - 0.3 / 0.64) ** (-2.5 * 0.64), rtol=1e-5, atol=0.0)
