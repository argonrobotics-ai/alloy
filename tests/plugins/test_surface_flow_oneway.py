# SPDX-License-Identifier: Apache-2.0
"""Surface flow, the one-way gap made explicit (docs/known_gaps/surface-film-one-way.md; WP13's sweep F3): setup
warns once, naming the colliding bodies the film does not couple to and the particles it does not see; a body
declared as a tool and the body a chart rides stay quiet (CPU)."""

from __future__ import annotations

import logging

from alloy._vendor import newton
import warp as wp

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.oneway import GAP_DOC

WATER = FilmMaterial("water_flat", surface_tension=0.0)


def _scene(block: bool, particles: int = 0, carrier: bool = False):
    """A plate's top face (the chart; on a free body with ``carrier``), a free block above it, and loose particles."""
    def build(b: newton.ModelBuilder) -> None:
        xf = body_xform(0.0, 0.0, 0.5)
        if carrier:
            b.add_shape_box(b.add_body(xform=xf, label="carrier"), hx=0.1, hy=0.1, hz=0.01, label="plate")
        else:
            b.add_shape_box(-1, xform=xf, hx=0.1, hy=0.1, hz=0.01, label="plate")
        if block:
            body = b.add_body(xform=body_xform(0.0, 0.0, 0.56), label="block")
            b.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, label="block_box")
        for k in range(particles):
            b.add_particle(wp.vec3(0.01 * k, 0.0, 0.6), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.002)
    return build


def _flow(scene, caplog, **config) -> SurfaceFlow:
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(8, 8)),), material=WATER, max_depth=2.0e-3, **config)
    with caplog.at_level(logging.WARNING, logger="alloy.plugins.surface_flow"):
        build_simulation(TestScene(build=scene, ground=False, gravity=(0.0, 0.0, 0.0), worlds=1, dt=2.0e-3), [flow],
                         "cpu")
    return flow


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "one-way" in r.getMessage()]


def test_a_free_block_and_loose_particles_are_named_once(caplog):
    flow = _flow(_scene(True, particles=3), caplog)
    notes = _warnings(caplog)
    assert len(notes) == 1 and notes[0] == flow.one_way
    assert "1 bodies not declared as its tools (block)" in notes[0]
    assert "3 particles (cloth, paper, paste, liquid or soft bodies)" in notes[0] and GAP_DOC in notes[0]


def test_a_block_declared_as_a_tool_stays_quiet(caplog):
    tool = Applicator(body="block", shape="rect", size=(0.02, 0.02), stiffness=1.0e3)
    flow = _flow(_scene(True), caplog, tools=(tool,))
    assert flow.one_way is None and _warnings(caplog) == []


def test_a_film_on_its_own_carrier_stays_quiet(caplog):
    flow = _flow(_scene(False, carrier=True), caplog)
    assert flow.one_way is None and _warnings(caplog) == []
