# SPDX-License-Identifier: Apache-2.0
"""A proxy whose source body has no colliding shape never touches the destination, so it sets no exchange-clock bound
and no strong-pair mass (``alloy.coupling.endpoints.clock_masses``).

WP9 2026-11-08: a hinged wrist with an inertial only (0.3 kg, centre of mass on the axis) carrying two 0.2 kg pads on
slides got a 0.85 g proxy on the MuJoCo Warp route (Newton's effective mass at its endpoint) and set dt_x 16x finer
than its pads (j = 10 instead of 6). The same rig here, through the lagged MuJoCo -> VBD Scheduler on a small sheet.
A visual-only body (VISIBLE flag only) reads 1 g. The wrist read 1.6 g, from MuJoCo Warp's invweight0 fallback; since
``axis_mass.ProxyMassMixin`` it reads its own 0.3 kg (test_proxy_mass_route). Neither sets a bound. The pair mass is the
pads' 0.2 kg axis mass, with the contact gate (lever #2) too."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling.endpoints import clock_masses, colliding_bodies
from alloy.coupling.types import ClockPolicy, SchedulerPolicy

WRIST = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="wrist" pos="0 0 0.3"><joint name="roll" type="hinge" axis="1 0 0" range="-3 3"/>
    <inertial pos="0 0 0" mass="0.3" diaginertia="1e-4 1e-4 1e-4"/>
    <body name="pad_a" pos="0 0.03 -0.05"><joint type="slide" axis="0 1 0" range="-0.03 0.03"/>
      <geom type="box" size="0.01 0.01 0.02" mass="0.2"/></body>
    <body name="pad_b" pos="0 -0.03 -0.05"><joint type="slide" axis="0 -1 0" range="-0.03 0.03"/>
      <geom type="box" size="0.01 0.01 0.02" mass="0.2"/></body>
  </body>
  <body name="ghost" pos="0.3 0 0.3"><joint type="slide" axis="0 0 1"/>
    <geom type="box" size="0.01 0.01 0.01" mass="0.001" contype="0" conaffinity="0" group="3"/></body>
</worldbody></mujoco>"""
PADS = 0.2                     # kg: each pad's mass along its slide (its axis mass)


def _model() -> newton.Model:
    from alloy.coupling.pinch_rigs import _finalize

    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(WRIST)
    b.add_cloth_grid(pos=wp.vec3(-0.05, -0.05, 0.1), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=4, dim_y=4,
                     cell_x=0.025, cell_y=0.025, mass=1.0e-3)
    return _finalize(b)


def _by_label(model: newton.Model, values) -> dict[str, object]:
    return dict(zip((str(x).split("/")[-1] for x in model.body_label), values))


def test_colliding_bodies_reads_the_shape_flags():
    m = _model()
    assert _by_label(m, colliding_bodies(m)) == {"wrist": False, "pad_a": True, "pad_b": True, "ghost": False}


@pytest.mark.parametrize("gate", [False, True], ids=["clock", "contact_gate"])
def test_a_proxy_that_cannot_touch_sets_no_clock_bound(gate):
    """The pads set the pair mass and the bound; the shapeless wrist (its own 0.3 kg) and the ghost (1 g) do not."""
    from alloy.coupling.pinch_rigs import _cloth_scheduler

    model = _model()
    sched = _cloth_scheduler(model, False, policy=SchedulerPolicy(clock=ClockPolicy(contact_gate=gate)))
    es = next(e for e in sched._sets if e.kind == "body")
    labels = [str(model.body_label[g]).split("/")[-1] for g in es.src_global.numpy()]
    raw = dict(zip(labels, np.asarray(es.m_axis, dtype=np.float64)))
    assert raw["ghost"] < 0.005 and raw["wrist"] == pytest.approx(0.3, rel=1e-6)    # the proxies are there
    assert dict(zip(labels, clock_masses(es))) == pytest.approx({"wrist": 0.0, "pad_a": PADS, "pad_b": PADS,
                                                                 "ghost": 0.0}, rel=1e-5, abs=0)
    assert sched._pair_mass[("mjc", "vbd")] == pytest.approx(PADS, rel=1e-5)
    pads_bound = 0.5 * math.sqrt(PADS / sched._pair_k_c(sched._alloy_pairs[0]))      # safety 0.5, omega_dt 1
    if gate:
        bounds = dict(zip(labels, sched._gate.ends.bound_s))
        assert math.isinf(bounds["wrist"]) and math.isinf(bounds["ghost"])
        assert bounds["pad_a"] == pytest.approx(pads_bound, rel=1e-4)
    else:
        assert sched._interface_bound() == pytest.approx(pads_bound, rel=1e-4)
