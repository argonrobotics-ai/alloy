# SPDX-License-Identifier: Apache-2.0
"""CPL-7: mean passes per exchange and the coupler's share of step time (Scheduler vs stock proxy, CPU)."""

from __future__ import annotations

import time

import numpy as np

from alloy._vendor import newton
from alloy.testing.coupling import (ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth, record_metric,
                                    stock_proxy_solver)

SPEC = ClothSceneSpec(box_mass=0.2, cells=8)
DT = 1.0 / 480.0
WARM, STEPS, REPEATS = 5, 30, 5


def _timed(model, solver, iters=None):
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    passes, t0 = [], 0.0
    for k in range(WARM + STEPS):
        if k == WARM:
            t0 = time.perf_counter()
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        if iters is not None:
            passes.append(iters.numpy().copy())
    return (time.perf_counter() - t0) / STEPS, passes


def test_cpl7_iterations_and_overhead():
    """Step-time share of the coupler: median over interleaved repeats (the CPU host is shared and noisy)."""
    model = coupling_box_on_cloth(4, SPEC, device="cpu")
    stock = stock_proxy_solver(model, SPEC, iterations=2)
    sched = cloth_scheduler(model, SPEC, vbd_adaptive=False)
    stock_t, sched_t = [], []
    for _ in range(REPEATS):
        stock_t.append(_timed(model, stock)[0])
        t, passes = _timed(model, sched, sched.iteration_loop.iters_used)
        sched_t.append(t)
    t_stock, t_sched = float(np.median(stock_t)), float(np.median(sched_t))
    stag = cloth_scheduler(model, SPEC, mode="staggered", vbd_adaptive=False)
    _, passes_stag = _timed(model, stag, stag.iteration_loop.iters_used)
    mean_lagged = float(np.mean(passes))
    mean_stag = float(np.mean(passes_stag))
    share = max(0.0, (t_sched - t_stock) / t_sched)
    record_metric("CPL-7", mean_passes_lagged=mean_lagged, mean_passes_staggered=mean_stag,
                  coupler_share=share, ms_per_step_scheduler=t_sched * 1e3, ms_per_step_stock=t_stock * 1e3,
                  device="cpu", worlds=4)
    print(f"CPL-7: passes lagged {mean_lagged:.3f}, staggered {mean_stag:.3f}; step {t_sched * 1e3:.2f} ms vs "
          f"stock {t_stock * 1e3:.2f} ms -> coupler share {share:.2%}")
    assert mean_lagged <= 2.2
    assert share <= 0.10
