# SPDX-License-Identifier: Apache-2.0
"""Kernel-level checks: per-world Aitken (bitwise batch-invariant, CPL-9 on stats), residual freeze, watch states,
D20 energy (datum, E_scale, conservation of a free fall)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.coupling import aitken as ak
from alloy.coupling import residuals as rs
from alloy.coupling.watchdog import WatchParams, watch_kernel

DEV = "cpu"


def _aitken_stats(raw, used, prev, worlds, world_count, omega0=0.5):
    """Run aitken_update_kernel on endpoint data; returns (stats (W,2), omega (W,))."""
    csr = ak.build_world_csr(worlds, world_count, DEV)
    sv = wp.spatial_vector
    stats = wp.zeros((world_count, 2), dtype=wp.float64, device=DEV)
    omega = wp.full(world_count, omega0, dtype=float, device=DEV)
    has_prev = wp.ones(world_count, dtype=wp.int32, device=DEV)
    conv = wp.zeros(world_count, dtype=wp.int32, device=DEV)
    wp.launch(ak.aitken_update_kernel, dim=world_count,
              inputs=[csr.start, csr.index, conv, wp.array(raw, dtype=sv, device=DEV),
                      wp.array(used, dtype=sv, device=DEV), wp.array(prev, dtype=sv, device=DEV), 0.05, 1.0,
                      stats, omega, has_prev], device=DEV)
    return stats.numpy(), omega.numpy()


def test_aitken_stats_bitwise_batch_invariant_and_correct():
    rng = np.random.default_rng(7)
    w, per = 16, 3
    worlds = np.repeat(np.arange(w), per)
    raw, used, prev = (rng.normal(size=(w * per, 6)).astype(np.float32) for _ in range(3))
    stats, omega = _aitken_stats(raw, used, prev, worlds, w)
    for k in range(w):
        sl = slice(k * per, (k + 1) * per)
        s1, o1 = _aitken_stats(raw[sl], used[sl], prev[sl], np.zeros(per, int), 1)
        assert stats[k].tobytes() == s1[0].tobytes()            # bitwise (D9, CPL-9)
        assert omega[k].tobytes() == o1[0].tobytes()
        r = (raw[sl] - used[sl]).astype(np.float64)
        d = r - prev[sl]
        s0_ref, s1_ref = float(np.sum(prev[sl] * d)), float(np.sum(d * d))
        assert stats[k] == pytest.approx([s0_ref, s1_ref], rel=1e-5, abs=1e-6)
        assert omega[k] == pytest.approx(np.clip(-0.5 * s0_ref / s1_ref, 0.05, 1.0), rel=1e-5)


def test_world_csr_orders_by_world_and_drops_globals():
    csr = ak.build_world_csr(np.array([2, 0, -1, 2, 1, 0]), 3, DEV)
    assert csr.start.numpy().tolist() == [0, 2, 3, 5]
    assert csr.index.numpy().tolist() == [1, 5, 4, 0, 3]


def test_freeze_pins_feedback_and_restores_carried_value():
    sv = wp.spatial_vector
    pg = wp.array([0, 1], dtype=int, device=DEV)
    world = wp.array([0, 1], dtype=int, device=DEV)
    conv = wp.array([rs.JUST_CONVERGED, rs.ACTIVE], dtype=wp.int32, device=DEV)
    used = wp.array([[1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], dtype=sv, device=DEV)
    cf = wp.array([[5, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0]], dtype=sv, device=DEV)
    frozen = wp.zeros(2, dtype=sv, device=DEV)
    wp.launch(rs.freeze_new_worlds_kernel, dim=2, inputs=[pg, world, conv, used, cf, frozen], device=DEV)
    assert cf.numpy()[:, 0].tolist() == [1, 6] and frozen.numpy()[0, 0] == 5
    wp.launch(rs.settle_new_worlds_kernel, dim=2, inputs=[conv], device=DEV)
    wp.launch(rs.scatter_frozen_kernel, dim=2, inputs=[pg, world, conv, frozen, cf], device=DEV)
    assert cf.numpy()[:, 0].tolist() == [5, 6]


def _watch(interface_work, window_base, res_pos, speed, energy=0.0, elapsed_s=2.0, initial=None,
           work_before=None, onset_limit=np.inf, armed=None, ladder=None, out=None):
    n = len(interface_work)
    f64 = lambda v: wp.array(np.asarray(v, np.float64), dtype=wp.float64, device=DEV)  # noqa: E731
    p = WatchParams()
    p.warn_per_s, p.warn_floor_s, p.escalate = wp.float64(0.01 / 60), wp.float64(1.0), wp.float64(0.05)
    p.abort_speed, p.res_pos_escalate = 20.0, 1.0e-2
    state = wp.array(np.asarray(initial if initial is not None else [0] * n, np.int32), dtype=wp.int32, device=DEV)
    gain = wp.zeros(n, dtype=wp.float64, device=DEV)
    peak, hit = wp.zeros(n, dtype=wp.float64, device=DEV), wp.zeros(n, dtype=wp.int32, device=DEV)
    lad = wp.array(np.asarray(ladder if ladder is not None else [0] * n, np.int32), dtype=wp.int32, device=DEV)
    arm = wp.array(np.asarray(armed if armed is not None else [1] * n, np.int32), dtype=wp.int32, device=DEV)
    wp.launch(watch_kernel, dim=n,
              inputs=[p, wp.array([int(elapsed_s * 1e9)], dtype=wp.int64, device=DEV), f64([energy] * n),
                      f64([0.0] * n), f64([0.0] * n), f64(interface_work), f64(window_base), f64([1.0] * n),
                      wp.array(np.asarray(speed, np.float32), dtype=float, device=DEV),
                      wp.zeros(n, dtype=wp.int32, device=DEV),
                      wp.array(np.asarray(res_pos, np.float32), dtype=float, device=DEV),
                      f64(work_before if work_before is not None else interface_work), f64([onset_limit]), peak, hit,
                      lad, arm, gain, state],
              device=DEV)
    if out is not None:
        out.update(peak=peak.numpy().tolist(), hit=hit.numpy().tolist(), ladder=lad.numpy().tolist())
    return state.numpy().tolist()


def test_watch_levels():
    # E_scale = 1 J; warn allowance at t = 2 s: 0.01/60·2 = 3.3e-4; escalate: 5 % within the window
    assert _watch([0.0, 1e-3, 0.06, 0.06, 0.0, 0.0], [0, 0, 0, 0.02, 0, 0],
                  [0, 0, 0, 0, 0.02, 0], [0, 0, 0, 0, 0, 25.0]) == [0, 1, 2, 1, 2, 4]


def test_watch_rerun_of_rolled_back_world_escalates_again():
    # a world restored from a rollback starts its re-run at ROLLED_BACK (3); a new escalation must show as 2 so the
    # stepper asks for the next ladder rung, a warning keeps 3, and FAILED (4) stays
    assert _watch([0.06, 1e-3, 0.0, 0.06], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                  initial=[3, 3, 3, 4]) == [2, 3, 3, 4]


def test_watch_ignores_negative_energy():
    # D20: a world below its datum has E < 0; with no injection nothing fires
    assert _watch([0.0], [0.0], [0.0], [0.1], energy=-3.0) == [0]


def test_watch_force_jump_escalates_and_flags_onset():
    # SCR-WP11-4: one exchange that injects more than the limit (E_scale 1 J) escalates and flags a force jump; under
    # the limit, or as a loss, it does not; an escalation for another reason is not flagged as one
    out: dict = {}
    states = _watch([2e-4, 5e-5, 0.0, 0.06], [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [0, 0, 0, 0],
                    work_before=[0.0, 0.0, 1e-3, 0.0], onset_limit=1e-4, out=out)
    assert states == [2, 0, 0, 2]
    assert out["hit"] == [1, 0, 0, 0]
    np.testing.assert_allclose(out["peak"], [2e-4, 5e-5, 0.0, 0.06])


def test_watch_cooldown_and_the_sticky_ladder_flag():
    # a world in its force-jump cooldown (not armed) is not flagged (WP4's clamped towel hold); a ladder escalation
    # sets the sticky ladder flag, also over an earlier force jump of the same attempt
    out: dict = {}
    states = _watch([2e-4, 2e-4, 0.06], [0.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0], work_before=[0.0, 0.0, 0.0],
                    onset_limit=1e-4, armed=[0, 1, 1], ladder=[0, 0, 0], out=out)
    assert states == [0, 2, 2] and out["hit"] == [0, 1, 0] and out["ladder"] == [0, 0, 1]
    out = {}
    _watch([0.06], [0.0], [0], [0], work_before=[0.0], onset_limit=1e-4, ladder=[0], out=out)
    assert out["ladder"] == [1]
