# SPDX-License-Identifier: Apache-2.0
"""CPL-9 on CPU (SPEC §5.2, D9): world i alone == world i inside a 16-world batch, bitwise, with staggered coupling,
per-world Aitken and residual-gated passes, worlds converging at different pass counts (different drop heights)."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton
from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth

SPEC = ClothSceneSpec(box_mass=0.2, cells=6)
N = 16
STEPS = 40
DT = 1.0 / 480.0


def _run(offsets):
    model = coupling_box_on_cloth(len(offsets), SPEC, drop_offsets=offsets, device="cpu")
    sched = cloth_scheduler(model, SPEC, mode="staggered", vbd_adaptive=False)
    es = sched.endpoint_sets[0]
    loop = sched.iteration_loop
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    rec = {"body_q": [], "particle_q": [], "iters": [], "omega": [], "stats": [], "res_vel": []}
    npart = model.particle_count // len(offsets)
    for _ in range(STEPS):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        sched.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        rec["body_q"].append(s0.body_q.numpy().copy())
        rec["particle_q"].append(s0.particle_q.numpy().reshape(len(offsets), npart, 3).copy())
        rec["iters"].append(loop.iters_used.numpy().copy())
        rec["omega"].append(es.aitken_omega.numpy().copy())
        rec["stats"].append(es.aitken_stats.numpy().copy())
        rec["res_vel"].append(loop.res_vel.numpy().copy())
    return {k: np.array(v) for k, v in rec.items()}


def test_world_alone_equals_world_in_batch():
    offsets = [0.002 * k for k in range(N)]
    batch = _run(offsets)
    iters = batch["iters"]
    assert len({tuple(col) for col in iters.T}) > 1, "worlds should converge at different pass counts"
    for i in (0, 7, N - 1):
        alone = _run([offsets[i]])
        for key in batch:
            got = batch[key][:, i]
            want = alone[key][:, 0]
            assert got.tobytes() == want.tobytes(), f"world {i}: {key} differs"
    print("CPL-9 CPU: mean passes per exchange per world", iters.mean(axis=0).round(2))
