# SPDX-License-Identifier: Apache-2.0
"""SchedulerADMM (SPEC §3.6): residual-gated Jacobi ADMM, 2 <= k <= 20 with Boyd-style stop, per-world freeze, so a
world alone equals the same world in a batch bitwise on CPU (D9); escalation raises ρ."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton
from alloy.testing.coupling import admm_model, admm_scheduler, record_metric

DT = 1.0 / 480.0
STEPS = 30
GAMMA = 0.0          # bitwise batch invariance holds at γ = 0; γ > 0 is measured in the second test (open finding)


def _run(offsets, gamma=GAMMA):
    model = admm_model(offsets, device="cpu")
    sched = admm_scheduler(model, gamma=gamma)
    loop = sched.iteration_loop
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    rec = {"body_q": [], "iters": [], "primal": [], "dual": []}
    nb = model.body_count // len(offsets)
    for _ in range(STEPS):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        sched.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        rec["body_q"].append(s0.body_q.numpy().reshape(len(offsets), nb, 7).copy())
        rec["iters"].append(loop.iters_used.numpy().copy())
        rec["primal"].append(loop.primal.numpy().copy())
        rec["dual"].append(loop.dual.numpy().copy())
    return sched, {k: np.array(v) for k, v in rec.items()}


def test_admm_residual_stop_and_batch_invariance():
    offsets = [0.0, 0.01, 0.02, 0.03]
    sched, batch = _run(offsets)
    iters = batch["iters"]
    assert iters.min() >= 2 and iters.max() <= 20
    assert len(np.unique(iters)) > 1
    assert np.all(np.isfinite(batch["body_q"]))
    for i in (0, 3):
        _, alone = _run([offsets[i]])
        for key in batch:
            assert batch[key][:, i].tobytes() == alone[key][:, 0].tobytes(), (i, key)
    rho0 = float(sched._coupling.rho)
    assert sched._escalate_mode() and float(sched._coupling.rho) == 4.0 * rho0
    record_metric("ADMM", mean_iters=float(iters.mean()), min_iters=int(iters.min()), max_iters=int(iters.max()),
                  batch_invariant_cpu=True, worlds=len(offsets), steps=STEPS)
    print(f"SchedulerADMM: iterations mean {iters.mean():.2f} (min {iters.min()}, max {iters.max()})")


def test_admm_proximal_batch_deviation_reported():
    """γ = 0.1 (Newton's example): world alone vs in batch is not bitwise after a batch-mate converges later; the
    MuJoCo entry re-step sees identical per-world inputs but returns a different velocity (cause open, see
    docs/notes/wp4-results.md). Reported, bounded."""
    offsets = [0.0, 0.01, 0.02, 0.03]
    _, batch = _run(offsets, gamma=0.1)
    _, alone = _run([offsets[3]], gamma=0.1)
    dev = float(np.max(np.abs(batch["body_q"][:, 3, :, :3] - alone["body_q"][:, 0, :, :3])))
    record_metric("ADMM-gamma-batch-deviation", gamma=0.1, max_dq_m=dev, steps=STEPS)
    print(f"SchedulerADMM gamma 0.1: world 3 alone vs batch max |dq| {dev:.3e} m over {STEPS} steps")
    assert dev < 1.0e-3
