# SPDX-License-Identifier: Apache-2.0
"""CT-11: with ``deterministic=True`` the precision contacts and the states are bitwise equal over 3 runs.

Each run builds a fresh scene, layer and solver (SolverMuJoCo with Warp RUN_TO_RUN determinism) and runs 0.15 s of
the CT-1 scenario. Contacts compared: count, shapes, points, normals, sidecar class/pair/weight; states: body_q,
body_qd. The verdict is reported per (native, quantity) so a nondeterministic solver path shows as such (D9).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.testing.contact import peg_hole_scene

from peg_harness import advance, l2_dt, make_run, perturb, ramp, record

pytestmark = pytest.mark.gpu
WORLDS = 8


def _one_run(native: str) -> dict[str, np.ndarray]:
    scene = peg_hole_scene(WORLDS, device="cuda:0", mode="hydroelastic")
    run = make_run(scene, native, l2_dt(scene))
    run.solver = newton.solvers.SolverMuJoCo(
        scene.model, use_mujoco_contacts=(native == "mujoco"), use_speculative_contacts=(native == "newton"),
        integrator="euler", cone="elliptic", impratio=scene.spec.mj_impratio, nconmax=256, njmax=1024,
        iterations=100, ls_iterations=50, deterministic=wp.DeterministicMode.RUN_TO_RUN)
    if native == "mujoco":
        run.layer._binding = None
        run.layer.bind_mujoco(run.solver).install()
    perturb(run, np.random.default_rng(11))
    advance(run, int(0.05 / run.dt))
    ramp(run, np.array([0.0, 0.0, -1.0]), 0.0, 20.0, 0.1, n_seg=5)
    c, side = run.layer.contacts(), run.layer.sidecar()
    n = int(c.rigid_contact_count.numpy()[0])
    return {"n": np.array([n]), "shapes": np.stack([c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]]),
            "points": np.concatenate([c.rigid_contact_point0.numpy()[:n], c.rigid_contact_point1.numpy()[:n]]),
            "normal": c.rigid_contact_normal.numpy()[:n], "pair": side.pair.numpy()[:n],
            "weight": side.weight.numpy()[:n], "body_q": run.s0.body_q.numpy(), "body_qd": run.s0.body_qd.numpy()}


@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_ct11_bitwise(native):
    runs = [_one_run(native) for _ in range(3)]
    contact_keys = ("n", "shapes", "points", "normal", "pair", "weight")
    same = {k: all(r[k].shape == runs[0][k].shape and r[k].tobytes() == runs[0][k].tobytes() for r in runs[1:])
            for k in runs[0]}
    row = dict(metric="CT-11", native=native, contacts_bitwise=all(same[k] for k in contact_keys),
               states_bitwise=same["body_q"] and same["body_qd"], detail=same)
    record("ct11", row)
    assert row["contacts_bitwise"], row
    assert row["states_bitwise"], row
