# SPDX-License-Identifier: Apache-2.0
"""The latex card on the surface-flow film core (CPU), as a sag law: a Bingham paint film draining down a vertical wall
follows the yield-stress Jeffreys profile and stops at h_c = τ_y/(ρg); a film thinner than h_c does not move. The
volume ledger closes."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.paint_materials import film_material
from alloy.tasks.paint_metrics import ledger

G = 9.81
WIDTH, HEIGHT, CELL = 0.004, 0.2, 0.001
CONTROL_DT = 0.05


def bingham_drainage(x: np.ndarray, t: float, rho: float, mu: float, tau: float) -> np.ndarray:
    """Depth x below the top of a vertical wall, t after a uniform Bingham film started draining (Jeffreys 1930 with
    a yield stress). The plug-flow discharge q = ρg(h - h_c)²(2h + h_c)/(6μ) carries depth h at the kinematic speed
    q'(h) = ρg·h(h - h_c)/μ, so x = q'(h)·t inside the fan:  h = (h_c + sqrt(h_c² + 4μx/(ρgt)))/2,  h_c = τ/(ρg).
    With τ = 0 this is Jeffreys' sqrt(μx/(ρgt))."""
    hc = tau / (rho * G)
    return 0.5 * (hc + np.sqrt(hc * hc + 4.0 * mu * x / (rho * G * t)))


def _drain(h0: float, seconds: float) -> tuple[np.ndarray, dict, float]:
    """A uniform film h0 thick on a WIDTH x HEIGHT wall chart whose bottom edge is open (the film leaves there, booked
    as spilled), after ``seconds``: the depth [nx, ny] (j up), the plug-in's read, and the volume set."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.tasks.paint_scene import chart_scene

    cells = (int(round(WIDTH / CELL)), int(round(HEIGHT / CELL)))
    chart = PlaneChart("wall", cells=cells, edges=("wall", "wall", "open", "wall"))
    flow = SurfaceFlow(charts=(chart,), material=film_material("latex_eggshell"), inertia=False, max_depth=5e-4)
    s = float(np.sin(np.pi / 4))
    wall = ("wall", (0.0, 0.01, 0.5), (s, 0.0, 0.0, s), (0.5 * WIDTH, 0.5 * HEIGHT, 0.01))     # face -y, j up
    sim = build_simulation(chart_scene([wall], lambda sim: flow.set_depth(h0), CONTROL_DT), [flow], "cpu")
    for _ in range(int(round(seconds / CONTROL_DT))):
        sim.step_period()
    return flow.depth(0)[0], flow.read(), h0 * WIDTH * HEIGHT


@pytest.mark.compile_heavy
def test_bingham_film_drains_as_the_yield_stress_jeffreys_profile() -> None:
    mat = film_material("latex_eggshell")
    h0, t = 400e-6, 600.0
    h, read, set_volume = _drain(h0, t)
    profile = h.mean(axis=0)[::-1]                                        # from the top edge down
    x = (np.arange(profile.size) + 0.5) * CELL
    want = bingham_drainage(x, t, mat.density, mat.viscosity, mat.yield_stress)
    sel = (x > 0.04) & (want < 0.8 * h0)        # as the plug-in's Jeffreys law: 40 mm from the singular top corner
    assert sel.sum() >= 50                      # 40-101 mm: h 206-320 um
    assert np.max(np.abs(profile[sel] / want[sel] - 1.0)) < 0.02         # 1.6 %; Newtonian 6.8 %
    # the yield stress shows: the film stands about h_c/2 (10 um) above Jeffreys' Newtonian profile
    newtonian = bingham_drainage(x[sel], t, mat.density, mat.viscosity, 0.0)
    lift = float(np.mean(profile[sel] - newtonian)) / float(np.mean(want[sel] - newtonian))
    assert lift == pytest.approx(1.0, abs=0.3)
    assert ledger(read, set_volume=set_volume).ok


@pytest.mark.compile_heavy
def test_film_thinner_than_its_yield_depth_does_not_move() -> None:
    mat = film_material("latex_eggshell")
    h0 = 0.8 * mat.yield_stress / (mat.density * G)                       # 16 um < h_c = 19.7 um
    h, read, set_volume = _drain(h0, 60.0)
    assert np.max(np.abs(h - h0)) < 1e-3 * h0
    assert ledger(read, set_volume=set_volume).ok
