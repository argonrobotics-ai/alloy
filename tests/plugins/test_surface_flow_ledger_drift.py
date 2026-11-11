# SPDX-License-Identifier: Apache-2.0
"""Surface flow, the film's books over a long dry (WP11's behind-radiator settle): what evaporation takes is booked
exactly as the cell lost it, so a uniform film's ledger doesn't drift (CPU)."""

from __future__ import annotations

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl

DRY = FilmMaterial("drying_test", density=1300.0, viscosity=5.0, surface_tension=0.0, evaporation=1.0e-6)


def test_a_uniform_drying_film_keeps_its_books():
    """95 um of film evaporating over 2,000 film steps: every cell rounded hn - e the same way while the book took e,
    so the ledger drifted ~6e-17 of the film per step (1.2e-13 at 2,000; behind-radiator's 55k exchanges failed
    1e-12; 1.13e-13 here). Booking hn - (hn - e) leaves the accumulators' own rounding."""
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material=DRY, inertia=False, max_stride=1,
                       max_depth=1.0e-3)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.04, 0.04), init=lambda sim: flow.set_depth(9.5e-5),
                                     ground=False, gravity=(0.0, 0.0, -9.81), worlds=1, dt=1.0e-2), [flow], "cpu")
    start = 9.5e-5 * float(flow.tables.area_np.sum())
    for _ in range(1000):
        sim.step_period()
    r = flow.read()
    assert int(flow.tick.numpy()[0]) * flow.n_sub >= 2000
    assert float(r["evaporated_volume"][0]) > 0.02 * start
    # left: the books' own accumulation (4.7e-15 here; ~1e-13 over behind-radiator's 55k exchanges)
    assert abs(float(r["ledger"][0]) - start) <= 1.0e-14 * start, (float(r["ledger"][0]) - start) / start
