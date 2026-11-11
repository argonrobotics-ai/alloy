# SPDX-License-Identifier: Apache-2.0
"""GPU (SPEC §7 WP4): the graph-captured residual loop (``wp.capture_while`` -> CUDA conditional node) equals the eager
host loop, and CPL-9 on GPU (world alone vs in 1,024 worlds), on the NATIVE FakeSolver population (D9). The
MuJoCo + VBD box on cloth is compared with a tolerance (float atomics in both solvers make it non-NATIVE)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.testing.coupling import (ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth, fake_drop_model,
                                    fake_scheduler, record_metric)

DEV = "cuda:0"
DT = 1.0 / 480.0
STATE_FIELDS = ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "joint_qd")


def _copy_state(dst, src):
    for name in STATE_FIELDS:
        a, b = getattr(dst, name), getattr(src, name)
        if a is not None and a.size:
            wp.copy(a, b)


def _runner(model, sched, contacts_fn=None):
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    contacts = contacts_fn(model) if contacts_fn else None

    def one():
        s0.clear_forces()
        if contacts is not None:
            contacts[0].collide(s0, contacts[1])
        sched.step(s0, s1, control, None if contacts is None else contacts[1], DT)
        _copy_state(s0, s1)
    return s0, one


def _record(s0, sched):
    loop = sched.iteration_loop
    return {"body_q": s0.body_q.numpy().copy(), "body_qd": s0.body_qd.numpy().copy(),
            "iters": loop.iters_used.numpy().copy(), "res_force": loop.res_force.numpy().copy(),
            "omega": sched.endpoint_sets[0].aitken_omega.numpy().copy(),
            "feedback": sched.endpoint_sets[0].mapping.coupling_forces.numpy().copy()}


def _run(make, steps, captured):
    model, sched, contacts_fn = make()
    s0, one = _runner(model, sched, contacts_fn)
    one()                                                      # eager: loads modules, sets the energy datum
    if captured:
        with wp.ScopedCapture(device=DEV) as cap:
            one()
        for _ in range(steps - 1):
            wp.capture_launch(cap.graph)
    else:
        for _ in range(steps - 1):
            one()
    wp.synchronize_device(DEV)
    return _record(s0, sched)


def _fake(n):
    heights = [0.2 + 0.0137 * k for k in range(n)]
    def make():
        model = fake_drop_model(heights, device=DEV)
        return model, fake_scheduler(model), None
    return make


@pytest.mark.gpu
@pytest.mark.smoke
def test_capture_while_equals_host_loop_native():
    steps = 120
    eager = _run(_fake(64), steps, captured=False)
    graph = _run(_fake(64), steps, captured=True)
    assert len(set(eager["iters"].tolist())) > 1 or eager["iters"].max() > 1
    for key in eager:
        assert eager[key].tobytes() == graph[key].tobytes(), key
    record_metric("capture_while==host_loop", population="FakeSolver x2 (NATIVE)", worlds=64, steps=steps,
                  bitwise=True, device=DEV)


@pytest.mark.gpu
@pytest.mark.smoke
def test_cpl9_gpu_native_world_alone_vs_1024():
    steps, n = 100, 1024
    batch = _run(_fake(n), steps, captured=True)
    worst = []
    for i in (0, 511, n - 1):
        heights = [0.2 + 0.0137 * i]
        def make(h=heights):
            model = fake_drop_model(h, device=DEV)
            return model, fake_scheduler(model), None
        alone = _run(make, steps, captured=True)
        for key in ("iters", "res_force", "omega"):
            assert batch[key][i].tobytes() == alone[key][0].tobytes(), (i, key)
        assert batch["body_q"][2 * i].tobytes() == alone["body_q"][0].tobytes(), (i, "body_q")
        worst.append(i)
    record_metric("CPL-9", device=DEV, population="FakeSolver x2 (NATIVE)", worlds=n, checked=worst, bitwise=True,
                  steps=steps)


@pytest.mark.gpu
def test_capture_while_box_on_cloth_close():
    """MuJoCo + VBD (non-NATIVE): captured vs eager, staggered + residual stop, reported with max difference."""
    spec = ClothSceneSpec(box_mass=0.2, cells=8)

    def make():
        model = coupling_box_on_cloth(8, spec, drop_offsets=[0.002 * k for k in range(8)], device=DEV)
        pipe = newton.CollisionPipeline(model)
        return model, cloth_scheduler(model, spec, mode="staggered", vbd_adaptive=False), \
            (lambda m, p=pipe: (p, p.contacts()))
    steps = 60
    eager = _run(make, steps, captured=False)
    graph = _run(make, steps, captured=True)
    dq = float(np.max(np.abs(eager["body_q"][:, :3] - graph["body_q"][:, :3])))
    same_iters = float(np.mean(eager["iters"] == graph["iters"]))
    record_metric("capture_while~host_loop", population="MuJoCo+VBD (float atomics)", worlds=8, steps=steps,
                  max_dq_m=dq, iters_agree=same_iters, device=DEV)
    print(f"box on cloth captured vs eager: max |dq| {dq:.3e} m, iteration agreement {same_iters:.2f}")
    assert dq <= 1.0e-4
