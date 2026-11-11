# SPDX-License-Identifier: Apache-2.0
"""Surface flow, what the process stage (WP9, design §4.8) writes and is called from: per-cell evaporation and
absorption rates with exact books, and the stage's hooks in the exchange step (CPU)."""

from __future__ import annotations

import sys
import types

import numpy as np

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl

G = 9.81
DRYING = FilmMaterial("drying_10Pas", viscosity=10.0, surface_tension=0.0, evaporation=1.0e-5)   # bands stay put


def test_per_cell_sinks_replace_the_material_rate_and_book_absorption():
    """Three bands: the material's evaporation (untouched), none (rate 0) and absorption 2e-5 m/s into the substrate.
    Each band loses exactly its rate; absorption is booked apart, the ledger closes, and the salt stays."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(9, 3)),), material=DRYING, inertia=False, max_depth=1.0e-3,
                       per_cell_sinks=True, species=("salt",))

    def init(sim) -> None:
        flow.set_depth(4.0e-4)
        flow.set_species(4.0e-3, "salt")
        x = flow.cell_centres(0)[..., 0].reshape(1, -1)
        flow.set_input("evaporation", np.where(x < -0.0167, 1.0e-5, 0.0).astype(np.float32))
        flow.set_input("absorption", np.where(x > 0.0167, 2.0e-5, 0.0).astype(np.float32))
    run = run_scene(TestScene(build=sl.plate_scene(0.1, 0.1), init=init, ground=False, gravity=(0.0, 0.0, -G),
                              periods=25, worlds=1), [flow], "cpu")                 # 0.5 s
    assert run.error is None, run.error
    h, m = flow.depth(0)[0], flow.species("salt", 0)[0]
    assert np.allclose(h[:3], 4.0e-4 - 0.5e-5, rtol=1e-5, atol=0.0)     # the bands trade ~1e-6 at their faces
    assert np.allclose(h[3:6], 4.0e-4, rtol=1e-5, atol=0.0)
    assert np.allclose(h[6:], 4.0e-4 - 1.0e-5, rtol=1e-5, atol=0.0)
    assert np.allclose(m, 4.0e-3, rtol=1e-5, atol=0.0)                     # solvent only: the salt stays
    r = flow.read()
    area = flow.tables.area_np.reshape(9, 3)
    assert abs(r["absorbed_volume"][0] - 1.0e-5 * area[6:].sum()) < 1e-4 * 1.0e-5 * area[6:].sum()
    start = 4.0e-4 * area.sum()
    assert abs(r["ledger"][0] - start) < 1e-12 * start
    salt = 4.0e-3 * area.sum()
    assert abs(r["species_mass"][0, 0] - salt) < 1e-12 * salt                     # none left with the solvent


def test_the_process_stage_is_called_around_and_inside_the_film_step(monkeypatch):
    """A stand-in stage records its calls: before_film and after_film once per exchange, rates once per exchange
    after the tile kernels (the first substep), in that order."""
    calls: list[str] = []

    class Stage:
        def __init__(self, spec) -> None:
            self.spec = spec

        def setup(self, ctx, tables, flow) -> None:
            calls.append("setup")
            assert flow.fluid.per_sink == 1                          # the rate inputs are allocated

        def before_film(self, state, dt) -> None:
            calls.append("before")

        def rates(self, dt) -> None:
            calls.append("rates")

        def after_film(self, state, dt) -> None:
            calls.append("after")

    module = types.ModuleType("alloy.plugins.surface_flow.process")
    module.ProcessStage = Stage
    monkeypatch.setitem(sys.modules, "alloy.plugins.surface_flow.process", module)
    import alloy.plugins.surface_flow as surface_flow_pkg  # noqa: PLC0415
    monkeypatch.setattr(surface_flow_pkg, "process", module, raising=False)   # wins over an imported real module
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material="water", max_depth=1.0e-3,
                       process={"drying": {}})
    run = run_scene(TestScene(build=sl.plate_scene(0.1, 0.1), init=lambda s: flow.set_depth(1.0e-4), ground=False,
                              gravity=(0.0, 0.0, -G), periods=2, worlds=1), [flow], "cpu")
    assert run.error is None, run.error
    assert calls[0] == "setup"
    steps = calls[1:]
    assert len(steps) % 3 == 0 and len(steps) >= 6
    assert steps == ["before", "rates", "after"] * (len(steps) // 3)



def test_read_carries_the_process_stage_maps(monkeypatch):
    """SOFT-3: the stage's derived maps (dft, phi_s, laps) are in the film's read(), beside its own keys."""
    import alloy.plugins.surface_flow as sf  # noqa: PLC0415

    class Stage:
        def __init__(self, spec) -> None:
            pass

        def setup(self, ctx, tables, flow) -> None:
            self.n = tables.cell_count

        def before_film(self, state, dt) -> None:
            pass

        def rates(self, dt) -> None:
            pass

        def after_film(self, state, dt) -> None:
            pass

        def read(self):
            return {"dft": np.full((1, self.n), 3.0e-5), "h": np.zeros((1, self.n))}     # "h" must not override

    module = types.ModuleType("alloy.plugins.surface_flow.process")
    module.ProcessStage = Stage
    monkeypatch.setitem(sys.modules, "alloy.plugins.surface_flow.process", module)
    monkeypatch.setattr(sf, "process", module, raising=False)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", max_depth=1.0e-3,
                       process={"drying": {}})
    run = run_scene(TestScene(build=sl.plate_scene(0.1, 0.1), init=lambda s: flow.set_depth(1.0e-4), ground=False,
                              gravity=(0.0, 0.0, -G), periods=1, worlds=1), [flow], "cpu")
    assert run.error is None, run.error
    r = flow.read()
    assert np.array_equal(r["dft"], np.full((1, 16), 3.0e-5)) and r["h"].max() > 0.0
