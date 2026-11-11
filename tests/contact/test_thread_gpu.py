# SPDX-License-Identifier: Apache-2.0
"""Law A9 and rig twin R2 on GPU: the M10 bolt/nut on SolverMuJoCo with the analytic thread pair (run_law, quick).

A9: advance per revolution (p ± 0.5 %) and run-down torque (power-screw formula ± 5 %) for μ 0.14 and 0.10, one
turn after settling. R2: the twin fitted on the calib split of the synthetic dataset (seed 0), judged on valid:
advance per rev, run-down torque, seating slope. Rows go to bench/results/wp3/thread.jsonl.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from alloy.bench.runner import RunContext, plan_worlds, run_law, step_twin
from alloy.contact.thread_rig import R2_LAW, ThreadR2Twin, synthetic_r2_dataset
from alloy.contact.thread_twins import A9_LAW, ThreadA9Twin

from peg_harness import record

pytestmark = pytest.mark.gpu


def _row(res, law, wall: float) -> dict:
    return dict(law=law.id, verdict=res.verdict, wall_s=wall, dt=law.dt, quick=True, device=res.device,
                verdicts=[dict(name=v.name, verdict=v.verdict, error=v.error, bound=v.bound, tur=v.tur,
                               sim=v.sim.mean, real=None if v.real is None else v.real.mean) for v in res.verdicts],
                per_world=res.monitors.get("per_world"), fit=res.monitors.get("fit_calib"),
                error=res.monitors.get("error"), steps=res.monitors.get("steps"))


@pytest.mark.timeout(300)
def test_thread_step_timing(device):
    """Smoke: 0.2 s of the A9 twin; records the wall time per step so the long runs can be sized."""
    spec = dataclasses.replace(A9_LAW, duration_s=0.2, quick={"worlds": 2, "duration_s": 0.2})
    twin = ThreadA9Twin()
    ctx = RunContext(spec, device, 0, "mujoco", True)
    t0 = time.time()
    run = step_twin(spec, twin, plan_worlds(spec, twin.conditions(True), ctx), ctx)
    wall = time.time() - t0
    steps = run.monitors["steps"]
    t1 = time.time()
    step_twin(spec, ThreadA9Twin(), plan_worlds(spec, twin.conditions(True), ctx), ctx)
    warm = time.time() - t1
    record("thread", dict(law="timing", device=device, steps=steps, first_wall_s=wall, warm_wall_s=warm,
                          ms_per_step=1e3 * warm / steps, finite=bool(np.all(np.isfinite(run.log.series("z"))))))
    assert np.all(np.isfinite(run.log.series("z")))


@pytest.mark.timeout(480)
def test_a9_thread_rundown(device):
    t0 = time.time()
    res = run_law(A9_LAW, ThreadA9Twin(), None, device=device, quick=True)
    row = _row(res, A9_LAW, time.time() - t0)
    record("thread", row)
    assert res.verdict == "pass", row


@pytest.mark.timeout(480)
def test_r2_thread_rig_quick(device):
    t0 = time.time()
    res = run_law(R2_LAW, ThreadR2Twin(), synthetic_r2_dataset(0), device=device, quick=True)
    row = _row(res, R2_LAW, time.time() - t0)
    record("thread", row)
    assert res.verdict == "pass", row
