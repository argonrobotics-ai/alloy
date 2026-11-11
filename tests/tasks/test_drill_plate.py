# SPDX-License-Identifier: Apache-2.0
"""task_drill_plate (engine item 10, drilling): through holes at two feeds, short CPU run."""

from __future__ import annotations

import math

import pytest

from alloy.tasks.drill_plate import DrillPlate, run

pytestmark = pytest.mark.law


@pytest.fixture(scope="module")
def drilled():
    task = DrillPlate(feeds=(0.2e-3, 0.16e-3))
    res, read = run(task, worlds=2, seconds=1.7, timing=False)
    return task, res, read


def test_both_holes_break_through_and_remove_exactly_the_hole(drilled):
    _, res, _ = drilled
    for h in res.holes:
        assert math.isfinite(h.t_through), h
        assert h.removed_mg == pytest.approx(h.exact_mg, rel=1e-9)
        assert h.ledger_rel < 1e-12
        assert h.captured_mg == pytest.approx(0.8 * h.removed_mg, rel=1e-7)     # capture is a float32 fraction
        assert h.position_mm < 0.01


def test_steady_thrust_and_torque_follow_shaw_oxford_and_rise_with_feed(drilled):
    _, res, _ = drilled
    fast, slow = res.holes
    for h in res.holes:
        assert h.thrust_n == pytest.approx(h.thrust_law_n, rel=0.01), h
        assert h.torque_nm == pytest.approx(h.torque_law_nm, rel=0.01), h
    assert fast.feed_mm_rev > slow.feed_mm_rev
    assert fast.thrust_n > slow.thrust_n and fast.torque_nm > slow.torque_nm
    assert fast.t_through < slow.t_through


@pytest.mark.gpu
def test_gpu_smoke_and_cost_per_step_at_64_worlds():
    """GPU smoke (light-test rule: <= 64 worlds, short): the task on 64 worlds for 0.5 s, and the cost per exchange
    step with and without the drill (stated in docs/notes/drilling-v1.md)."""
    res, read = run(DrillPlate(), worlds=64, device="cuda:0", seconds=0.5)
    assert (read["removed"] > 0.0).all() and (read["depth"] > 0.0).all()
    print(f"GPU 64 worlds: {res.ms_per_step:.3f} ms/step with the drill, {res.ms_per_step_without:.3f} without")
