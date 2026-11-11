# SPDX-License-Identifier: Apache-2.0
"""R1 bitwise replay after snapshot restore, per-world independence, seeded blow-up (SPEC §7 WP9 CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.defaults import SPEED_FAIL_MPS
from alloy.core.snapshot import WorldSnapshot
from alloy.testing.sph import default_config, run, water_column

DT = 1.0e-3


def _bits(state) -> bytes:
    return state.particle_q.numpy().tobytes() + state.particle_qd.numpy().tobytes()


@pytest.mark.det
def test_r1_bitwise_after_snapshot_restore():
    sc = water_column(4, 4, 8, worlds=2, fills=[8, 5])
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    a, b = run(s, a, b, 10, DT)  # non-trivial warm starts and velocities
    snap = WorldSnapshot(sc.model, a, s.alloy_state_arrays(a), extra_world=s.alloy_state_worlds())
    snap.save(a)
    x, y = run(s, a, b, 15, DT)
    first = _bits(x)
    snap.restore(x)
    x2, _ = run(s, x, y, 15, DT)
    assert _bits(x2) == first


@pytest.mark.det
def test_per_world_independence_bitwise():
    """Two worlds with different fill levels match their 1-world runs bitwise (fixed dt)."""
    fills = [8, 5]
    packed = water_column(4, 4, 8, worlds=2, fills=fills, headroom=4)
    singles = [water_column(4, 4, 8, worlds=1, fills=[f], headroom=max(fills) + 4 - f) for f in fills]  # same tank
    sp = packed.make_solver()
    a, b = packed.model.state(), packed.model.state()
    a, _ = run(sp, a, b, 25, DT)
    q = a.particle_q.numpy()
    w = packed.model.particle_world.numpy()
    for k, sc in enumerate(singles):
        s1 = sc.make_solver()
        c, d = sc.model.state(), sc.model.state()
        c, _ = run(s1, c, d, 25, DT)
        assert q[w == k].tobytes() == c.particle_q.numpy().tobytes(), f"world {k}"
        assert sp.report().iters_density[k] == s1.report().iters_density[0]


def test_seeded_blowup_reaches_speed_fail():
    """An over-relaxed Jacobi (omega = 1, rejected by validate) blows the column up; the report shows an
    active-particle speed above SPEED_FAIL (20 m/s, D12) instead of waiting for NaN."""
    sc = water_column(4, 4, 8, cfg=default_config(pressure_solver="jacobi"))
    s = sc.make_solver()
    s.config.jacobi_omega = 1.0  # seeded instability (bypasses validate on purpose)
    a, b = sc.model.state(), sc.model.state()
    peak = []
    run(s, a, b, 150, DT, record=lambda k, st, r: peak.append(float(r.max_speed[0])))
    assert max(peak) > SPEED_FAIL_MPS
