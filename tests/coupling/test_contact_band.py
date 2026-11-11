# SPDX-License-Identifier: Apache-2.0
"""The exchange-step bound from the proxy contact band (alloy/coupling/contact_band.py, R2-E3), CPU.

A proxy pipeline re-collides once per exchange, so the clock keeps each contact within 1/6 of its pipeline's band
per exchange, from the contact's speed relative to the body point it touches. These fast tests use stand-in contact
records; the slow law is the trial's B3 fold world ([internal note] notes, R2-E3)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.coupling import contact_band as CB
from alloy.coupling.types import ClockPolicy

GAP = 0.003                    # m: the full-surface band (FULL_SURFACE_GAP)


def _route(qd, indices, body_vel, count, gap=GAP, mass=None):
    """One pipeline that built one Contacts (``count`` valid records of ``len(indices)``) for destination "cloth",
    whose particles have ``mass`` (default 0: only the records count)."""
    contacts = SimpleNamespace(
        soft_contact_max=len(indices), soft_contact_count=wp.array([count], dtype=wp.int32, device="cpu"),
        soft_contact_indices=wp.array(np.asarray(indices, np.int32), dtype=wp.vec3i, device="cpu"),
        soft_contact_body_vel=wp.array(np.asarray(body_vel, np.float32), dtype=wp.vec3, device="cpu"))
    factory = SimpleNamespace(built=[(None, SimpleNamespace(soft_contact_gap=gap), contacts, None)])
    mass = np.zeros(len(qd), np.float32) if mass is None else np.asarray(mass, np.float32)
    entries = {"cloth": SimpleNamespace(
        state_1=SimpleNamespace(particle_qd=wp.array(np.asarray(qd, np.float32), dtype=wp.vec3, device="cpu")),
        solver=SimpleNamespace(model=SimpleNamespace(particle_mass=wp.array(mass, dtype=float, device="cpu"))))}
    return factory, entries


def test_the_bound_keeps_the_fastest_contact_within_a_sixth_of_its_band():
    """Records alone (massless particles): vertex, edge and face records, the speed is the fastest of the record's
    particles relative to the body point. Records past the count are ignored, however fast. The bound is band/6 over
    that speed, and the period clears."""
    qd = [(3.0, 0, 0), (0.5, 0, 0), (0, 2.0, 0), (0, 0, 1.0), (0.2, 0, 0), (0, 0, 0), (50.0, 0, 0)]
    indices = [(0, -1, -1), (1, 2, -1), (3, 4, 5), (6, -1, -1)]          # vertex, edge, face, past the count
    body_vel = [(2.0, 0, 0), (0, 0, 0), (0, 0, 1.0), (0, 0, 0)]          # vertex 0: 1 m/s relative to its point
    factory, entries = _route(qd, indices, body_vel, count=3)
    band = CB.ContactBandBound([(factory, "cloth"), (factory, "cloth")], "cpu")   # one factory, counted once
    assert len(band.routes) == 1
    band.accumulate(entries)
    bound = band.end_period()
    assert band.last_speed[0] == pytest.approx(2.0, rel=1e-6, abs=0)       # the edge's second particle
    assert bound == pytest.approx(GAP / 6.0 / 2.0, rel=1e-6, abs=0)
    assert band.end_period() == math.inf                                    # cleared: nothing this period
    band.accumulate(entries)
    assert band.end_period(travel=None) == math.inf                         # the policy can turn it off
    assert band.end_period() == math.inf                                    # and the maxima still clear


def test_the_scheduler_clock_takes_the_band_bound():
    """The exchange-step bound the clock reads (``SchedulerMixin._interface_bound``) includes the last period's band
    bound, alongside the pairs' stiffness bounds (none here)."""
    from alloy.coupling.mixin import SchedulerMixin
    from alloy.coupling.types import SchedulerPolicy

    class Clocked(SchedulerMixin):
        def __init__(self, band_bound: float) -> None:
            self._alloy_policy, self._band_bound = SchedulerPolicy(), band_bound

    assert Clocked(1.25e-4)._interface_bound() == 1.25e-4
    assert Clocked(math.inf)._interface_bound() == math.inf


def test_a_sheet_in_contact_is_bounded_by_its_fastest_particle():
    """B3 world 7's second whip: the corner that blew up was not yet in a record. A sheet whose pipeline holds a record
    is bounded by its fastest particle with mass (massless, disabled particles do not count), even with every record
    moving with its pad."""
    qd = [(0.4, 0.0, 0.3), (0.4, 0.0, 0.3), (0.0, 3.0, 0.0), (40.0, 0.0, 0.0)]
    factory, entries = _route(qd, [(0, -1, -1), (1, -1, -1)], [(0.4, 0.0, 0.3)] * 2, count=2, mass=[1, 1, 1, 0])
    band = CB.ContactBandBound([(factory, "cloth")], "cpu")
    band.accumulate(entries)
    assert band.end_period() == pytest.approx(GAP / 6.0 / 3.0, rel=1e-6, abs=0)


def test_a_sheet_with_nothing_in_its_band_or_at_rest_sets_no_bound():
    factory, entries = _route([(0.0, 0.0, -5.0)] * 2, [(0, -1, -1)], [(0, 0, 0)], count=0, mass=[1, 1])   # falling
    band = CB.ContactBandBound([(factory, "cloth")], "cpu")
    band.accumulate(entries)
    assert band.end_period() == math.inf
    factory, entries = _route([(0.0, 0.0, 0.0)] * 2, [(0, -1, -1)], [(0, 0, 0)], count=1, mass=[1, 1])    # at rest
    band = CB.ContactBandBound([(factory, "cloth")], "cpu")
    band.accumulate(entries)
    assert band.end_period() == math.inf
    assert CB.band_bound(GAP, 0.0) == math.inf and CB.band_bound(0.0, 3.0) == math.inf


def test_the_period_keeps_its_fastest_exchange():
    """Exchanges fold into one maximum per period (atomic max), so a whip mid-period sets the next period's dt_x."""
    factory, slow = _route([(0.1, 0, 0)], [(0, -1, -1)], [(0, 0, 0)], count=1)
    _, fast = _route([(3.0, 0, 0)], [(0, -1, -1)], [(0, 0, 0)], count=1)
    band = CB.ContactBandBound([(factory, "cloth")], "cpu")
    for entries in (slow, fast, slow):
        band.accumulate(entries)
    assert band.end_period() == pytest.approx(GAP / 6.0 / 3.0, rel=1e-6, abs=0)
    band.accumulate(fast)
    assert band.end_period(travel=1.0 / 3.0) == pytest.approx(GAP / 3.0 / 3.0, rel=1e-6, abs=0)   # the policy's


def test_the_b3_whip_moves_the_clock_to_the_step_that_held():
    """B3's numbers: the corner at 2.5-3 m/s relative to the pad needs dt_x <= 0.167-0.2 ms, which the grid (10 ms
    control period) puts at 0.156 ms, the step that kept world 7 healthy; at its 0.3 m/s carry the bound (1.67 ms)
    is above the 0.3125 ms the pads' clock already uses."""
    from alloy.coupling.clocks import grid_level
    control_dt = 0.01
    j_fast, _ = grid_level(CB.band_bound(GAP, 3.0), control_dt, ClockPolicy().j_max)
    j_slow, _ = grid_level(min(CB.band_bound(GAP, 0.3), 0.3125e-3), control_dt, ClockPolicy().j_max)
    assert control_dt / 2 ** j_fast == pytest.approx(0.15625e-3, rel=1e-12)
    assert control_dt / 2 ** j_slow == pytest.approx(0.3125e-3, rel=1e-12)
    assert ClockPolicy().band_travel == pytest.approx(CB.BAND_TRAVEL, rel=0, abs=0)
