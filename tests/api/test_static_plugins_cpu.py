# SPDX-License-Identifier: Apache-2.0
"""A scene whose only dynamics are plug-ins builds and steps (WP13's combination sweep, F6; CPU).

A surface film on a static plate failed with "scene has no dynamic entry to step" and needed a dummy free body. Now
the plug-ins step alone, on the rigid step a dummy would get: the film flows exactly as with the dummy (bit for bit).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.api as alloy
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow

WATER = FilmMaterial("water_flat", surface_tension=0.0)


def film_on_plate(dummy: bool) -> tuple[np.ndarray, np.ndarray, alloy.Sim]:
    scene = alloy.Scene(dt=0.01)
    scene.add_box("plate", size=(0.2, 0.05, 0.02), pos=(0, 0, 0.5), rpy=(0.0, 0.2, 0.0), static=True)  # tilted
    if dummy:
        scene.add_sphere("dummy", 0.01, pos=(1.0, 0.0, 1.0), mass=0.01)          # the old workaround
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 1), edges=("open", "wall", "wall", "wall")),),
                       material=WATER, max_depth=2.0e-3)
    sim = alloy.build(scene, device="cpu", plugins=[flow])
    x = flow.cell_centres(0)[:, 0, 0]
    flow.set_depth((1.0e-3 + 5.0e-4 * np.cos(math.pi * x / 0.1))[:, None], chart=0)
    start = flow.depth(0)[0].copy()
    sim.run(0.2)
    return start, flow.depth(0)[0].copy(), sim


@pytest.mark.compile_heavy
def test_film_alone_on_a_static_plate_flows_as_with_a_dummy_body():
    start, alone, sim = film_on_plate(dummy=False)
    assert float(np.abs(alone - start).max()) > 1e-4 and sim.check()["ok"]      # the film flowed down the tilt
    assert sim.engine.parts.stepper.report().notes["stepper"].startswith("static")
    sim.close()
    _, with_dummy, sim = film_on_plate(dummy=True)
    assert np.array_equal(alone, with_dummy)
    sim.close()


def test_a_static_scene_without_plugins_says_how_to_step():
    scene = alloy.Scene(dt=0.01)
    scene.add_box("plate", size=(0.2, 0.05, 0.02), pos=(0, 0, 0.5), static=True)
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(scene, device="cpu")
    assert "dynamic object" in err.value.fix and "plugins=" in err.value.fix
