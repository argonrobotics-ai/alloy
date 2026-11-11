# SPDX-License-Identifier: Apache-2.0
"""HealthMonitor x GapMonitor protocol (D11): the GapMonitor's bits and per-step pair values reach the record, the
inbox resets every step, value-based pair checks use the loosest pair, and with every > 1 a latch never restores an
unchecked state."""

from __future__ import annotations

import numpy as np

from alloy.contact.types import PrecisionPairSpec
from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.testing.precision import HealthLoop, set_body, world_bodies
from alloy.testing.scenes import box_on_plane

TIGHT = PrecisionPairSpec(0, 1, clearance=2e-4)   # S2 = S1 = 0.1 mm
LOOSE = PrecisionPairSpec(2, 3, clearance=2e-3)   # S2 = 0.1 mm, S1 = 0.5 mm


def test_gap_monitor_bits_reach_the_record(device: str) -> None:
    loop = HealthLoop(box_on_plane(3, device=device), precision_pairs=[TIGHT])
    loop.gap.set_bits(1, int(HealthFlags.PAIR_PEN_FAIL))  # a tunnel: FAIL with zero sampled penetration
    loop.step(1)
    rep = loop.monitor.poll()
    assert rep.invalid_worlds == (1,) and rep.flags[1] & int(HealthFlags.PAIR_PEN_FAIL)


def test_inbox_resets_every_step(device: str) -> None:
    loop = HealthLoop(box_on_plane(2, device=device), precision_pairs=[TIGHT, LOOSE])
    loop.gap.set(0, 2e-4)  # above the loosest S2 (0.1 mm), below the loosest S1 (0.5 mm): DEGRADED
    loop.step(1)
    rep = loop.monitor.poll()
    assert rep.severity[0] == Severity.DEGRADED and rep.flags[0] & int(HealthFlags.PAIR_PEN_WARN)
    assert not rep.flags[0] & int(HealthFlags.PAIR_PEN_FAIL)  # loosest pair: no false latch
    assert rep.max_pair_pen_m[0] == np.float32(2e-4)
    loop.gap.set(0, 0.0)
    loop.step(1)
    rep = loop.monitor.poll()
    assert rep.severity[0] == Severity.OK and rep.max_pair_pen_m[0] == 0.0
    assert np.isinf(loop.monitor.health.numpy()["min_pair_gap_m"]).all()  # inbox neutral after update


def test_every_n_never_restores_an_unchecked_state(device: str) -> None:
    loop = HealthLoop(box_on_plane(2, device=device), checks=CheckPolicy(level="cheap", every=5))
    loop.step(5)  # steps 0-4; step 0 was checked
    b = int(world_bodies(loop.scene.model, 1)[0])

    def nan(step: int, state) -> None:
        if step == 6:
            q = state.body_q.numpy()[b].copy()
            q[2] = np.nan
            set_body(state.body_q, b, q)

    loop.step(6, inject=nan)  # steps 5-10: checked at 5 and 10
    rep = loop.monitor.poll()
    assert rep.invalid_worlds == (1,) and rep.first_bad_step[1] == 10
    assert np.isfinite(loop.state.body_q.numpy()[b]).all()  # restored to the step-5 output, not a NaN state


def test_retired_bodies_leave_the_energy_sums(device: str) -> None:
    """SCR-WP10-2: a body with body_active == 0 (crumbled) is out of KE/PE, so moving it is no energy jump."""
    import warp as wp

    loop = HealthLoop(box_on_plane(2, device=device), checks=CheckPolicy(level="full", full_every=1))
    active = np.ones(2, dtype=np.int8)
    active[int(world_bodies(loop.scene.model, 1)[0])] = 0
    loop.monitor.set_body_active(wp.array(active, dtype=wp.int8, device=device))
    loop.step(2)

    def kick(step: int, state) -> None:
        qd = state.body_qd.numpy()
        qd[:, 2] = 6.0  # both boxes thrown up at 6 m/s: 18 J each, E_scale ~3.4 J
        state.body_qd.assign(qd)

    loop.step(1, inject=kick)
    rep = loop.monitor.poll()
    assert rep.flags[0] & int(HealthFlags.ENERGY_JUMP) and rep.severity[0] == Severity.INVALID
    assert not rep.flags[1] & int(HealthFlags.ENERGY_JUMP)


def test_state_arrays_are_per_world(device: str) -> None:
    from alloy.core.snapshot import WorldSnapshot, mask_from_worlds

    loop = HealthLoop(box_on_plane(3, device=device))
    arrays = loop.monitor.state_arrays()
    assert {"health.record", "health.latched", "health.energy_e0"} <= set(arrays)
    assert all(a.shape[0] == 3 for a in arrays.values())
    snap = WorldSnapshot(loop.scene.model, loop.state, arrays)  # world-major: per-world save/restore works
    snap.save(loop.state)
    loop.step(2)
    snap.restore(loop.state, mask_from_worlds([1], 3, device))


def test_first_bad_step_counts_update_calls(device: str) -> None:
    loop = HealthLoop(box_on_plane(2, device=device))
    loop.monitor._step.assign(np.array([99], dtype=np.int64))  # as if start_step = 100
    b = int(world_bodies(loop.scene.model, 0)[0])

    def nan(step: int, state) -> None:
        q = state.body_q.numpy()[b].copy()
        q[0] = np.nan
        set_body(state.body_q, b, q)

    loop.step(3)
    loop.step(1, inject=nan)
    assert loop.monitor.poll().first_bad_step[0] == 103
