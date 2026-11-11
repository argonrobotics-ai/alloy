# SPDX-License-Identifier: Apache-2.0
"""PRC-12 (SPEC §5.2): time error 0 after 10 sim-min with dt changes (rung changes, watchdog halving) on the tick
clock; a body at 40 m is flagged FAR_FROM_ORIGIN."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import warp as wp

from alloy.core.timebase import SimClock
from alloy.precision.types import CheckPolicy, FramePolicy, HealthFlags, PrecisionConfig, Severity
from alloy.testing.precision import HealthLoop, set_body, world_bodies
from alloy.testing.scenes import box_on_plane

CONTROL_DT = 0.02
SIM_SECONDS = 600


def _schedule(rng: np.random.Generator):
    """Per period: a rung j in [2, 6] (dt 5 ms .. 0.3125 ms), sometimes a watchdog halving after k steps."""
    for _ in range(int(SIM_SECONDS / CONTROL_DT)):
        j = int(rng.integers(2, 7))
        halve_at = int(rng.integers(1, 2**j)) if rng.random() < 0.1 and j < 10 else None
        yield j, halve_at


def test_prc12_tick_clock_is_exact(rng: np.random.Generator) -> None:
    clock = SimClock(CONTROL_DT, 4)
    exact = Fraction(0)
    naive64, naive32 = 0.0, np.float32(0.0)
    changes = 0
    for j, halve_at in _schedule(rng):
        clock.set_dt(CONTROL_DT / 2**j)
        remaining = Fraction(1, 2**0)  # one period, in units of control_dt
        k = 0
        while remaining > 0:
            if halve_at is not None and k == halve_at:  # watchdog halves dt for the rest of the period
                clock.set_dt(clock.dt / 2.0)
                changes += 1
            step = Fraction(1, 2 ** (clock.j))
            clock.advance(1)
            exact += step * Fraction(CONTROL_DT)
            naive64 += clock.dt
            naive32 = np.float32(naive32 + np.float32(clock.dt))
            remaining -= step
            k += 1
        changes += 1
    assert remaining == 0 and clock.ticks == SIM_SECONDS / CONTROL_DT * 2**10
    assert clock.t == float(exact)  # bitwise: the PRC-12 time error is 0
    err64, err32 = abs(naive64 - float(exact)), abs(float(naive32) - float(exact))
    assert err32 > 1e-3  # what float seconds would have cost (reported in wp5-results)
    print(f"PRC-12: tick clock error 0 after {SIM_SECONDS} s, {changes} dt changes; "
          f"naive float64 sum error {err64:.3e} s, float32 {err32:.3e} s")


def _far_box_loop(config: PrecisionConfig | None, device: str) -> HealthLoop:
    loop = HealthLoop(box_on_plane(3, device=device), checks=CheckPolicy(), config=config)
    b = int(world_bodies(loop.scene.model, 1)[0])
    q = loop.state.body_q.numpy()[b].copy()
    q[0] = 40.0  # 40 m from the origin, still resting on the (infinite) ground plane
    set_body(loop.state.body_q, b, q)
    loop.step(2)
    return loop


def test_prc12_body_at_40m_flagged(device: str) -> None:
    rep = _far_box_loop(None, device).monitor.poll()
    assert rep.flags[1] & int(HealthFlags.FAR_FROM_ORIGIN)
    assert rep.severity[1] == Severity.OK  # a notice under the standard frames policy (TRAIN-FIX 4c)
    assert not rep.flags[0] & int(HealthFlags.FAR_FROM_ORIGIN)


def test_far_is_invalid_under_precision_frames(device: str) -> None:
    from alloy.precision.config import preset
    from alloy.precision.types import GpuTier

    assert preset("precision", GpuTier.B).frames.far_invalid and not preset("standard", GpuTier.B).frames.far_invalid
    cfg = PrecisionConfig(frames=FramePolicy(far_invalid=True))
    rep = _far_box_loop(cfg, device).monitor.poll()
    assert rep.severity[1] == Severity.INVALID and rep.flags[1] & int(HealthFlags.FAR_FROM_ORIGIN)
