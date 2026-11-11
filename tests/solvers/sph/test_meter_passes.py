# SPDX-License-Identifier: Apache-2.0
"""SolverDFSPH's shape meters under a coupler's restart passes (WP9 (a1)): a staggered pair re-steps the liquid from
the exchange's start several times; ``shape_impulse`` must count the pass that sticks, not every pass (it added up to
8.7x the walls' impulse in a floating-cup scene). CPU."""

from __future__ import annotations

import numpy as np
from alloy._vendor.newton import StateFlags

from alloy.testing.sph import water_column

DT = 1.0e-3
FLAGS = int(StateFlags.PARTICLE_Q) | int(StateFlags.PARTICLE_QD)


def test_restart_passes_leave_the_meters_at_one_pass():
    """A resting column: an exchange window's first pass meters J1 on the floor; two restart passes replaying the same
    step leave the meters at J1 bit for bit (they read 3 J1 before); the next window adds its own J2."""
    sc = water_column(4, 4, 6)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    s.reset_shape_meters()
    s.coupling_notify_input_state_update(a, FLAGS, dt=DT)
    s.step(a, b, None, None, DT)
    j1 = s.shape_impulse().copy()
    for _ in range(2):
        s.coupling_notify_input_state_update(a, FLAGS, iteration_restart=True, dt=DT)
        s.step(a, b, None, None, DT)
    assert np.abs(j1).max() > 0.0 and np.array_equal(s.shape_impulse(), j1)
    s.coupling_notify_input_state_update(b, FLAGS, dt=DT)
    c = sc.model.state()
    s.step(b, c, None, None, DT)
    assert np.abs(s.shape_impulse() - j1).max() > 0.0
