# SPDX-License-Identifier: Apache-2.0
"""GA's contact episodes from real contacts (SCR-WP8-12 F5, WP13): a box slid into a rack upright reads its approach
speed before the impact, along a normal pointing at the box. Every criterion test elsewhere injects synthetic
episodes; this pins the sign and the timing on the solver's own contacts."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.vehicle import laws_grip as lg
from alloy.tasks.forklift_ga_contacts import ContactWatch

pytestmark = pytest.mark.law
SPEED, HALF, GAP = 0.3, 0.2, 0.15


def _build(b: newton.ModelBuilder) -> None:
    slick = newton.ModelBuilder.ShapeConfig(mu=0.0)
    b.add_shape_box(-1, xform=body_xform(HALF + GAP + 0.05, 0.0, 0.5), hx=0.05, hy=0.3, hz=0.5, cfg=slick,
                    label="A0_up0_0.50")
    box = b.add_body(xform=body_xform(0.0, 0.0, HALF + 2.0e-4), mass=50.0, label="box")
    b.add_shape_box(box, hx=HALF, hy=HALF, hz=HALF, cfg=slick, label="box")


def _push(sim) -> None:
    """The box slides toward the upright (the solver reads the free joint's velocity)."""
    qd = sim.state.joint_qd.numpy()
    qd[0] = SPEED
    sim.state.joint_qd.assign(qd)


def test_a_box_meeting_an_upright_reads_its_approach_speed_before_the_impact():
    scene = TestScene(build=_build, ground=True, ground_mu=0.0, worlds=1, dt=lg.STEP, control_dt=lg.PERIOD,
                      init=_push)
    sim = build_simulation(scene, [], "cpu", worlds=1)
    watch = ContactWatch(sim, {"A0_": "rack"}, loads={"x": "box"})
    rows, t = [], []
    for k in range(int(1.5 / lg.PERIOD)):
        sim.step_period()
        rows.append({"contacts": watch.read(), "touch": watch.touch, "speed": 0.0,
                     "vel:x": sim.state.body_qd.numpy()[0, :3].astype(np.float64), "vel_body": {}})
        t.append((k + 1) * lg.PERIOD)
    episodes = [e for e in watch.episodes(np.asarray(t), rows, None) if e.mover == "load:x"]
    assert episodes, "the box never met the upright"
    first = episodes[0]
    onset = int(np.searchsorted(t, first.t))
    normal = rows[onset]["contacts"]["load:x"]["rack"][3]
    assert first.t == pytest.approx(GAP / SPEED, abs=2.5 * lg.PERIOD)
    assert first.onset_normal == pytest.approx(SPEED, rel=0.05) and first.onset_speed == pytest.approx(SPEED, rel=0.05)
    assert normal[0] < -0.99                                  # from the upright toward the box (the mover)
    assert float(rows[onset]["vel:x"][0]) < 0.5 * SPEED       # by that sample the impact has already stopped it
