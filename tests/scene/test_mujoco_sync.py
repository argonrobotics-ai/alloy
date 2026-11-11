# SPDX-License-Identifier: Apache-2.0
"""SyncedMuJoCo (rk4 entries): Newton coordinates reach MuJoCo Warp only for worlds that changed since the last pull,
so a tumbling body keeps MuJoCo's own state (no float32 round trip each step) and every external write still lands."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.scene.mujoco_sync import SyncedMuJoCo, _index_worlds

W0 = np.array([3.5, -2.5, 4.5])          # WP6 A4 tumble: box half extents 0.05 / 0.1 / 0.15 m
DT = 1.25e-3


def _rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _boxes(worlds: int, synced: bool):
    sub = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(sub)
    body = sub.add_body()
    sub.add_shape_box(body, hx=0.05, hy=0.1, hz=0.15)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(sub, copy_count=worlds)
    model = b.finalize(device="cpu")
    qd = model.joint_qd.numpy().reshape(worlds, 6)
    qd[:, 3:6] = W0
    model.joint_qd.assign(qd.reshape(-1))
    cls = SyncedMuJoCo if synced else newton.solvers.SolverMuJoCo
    solver = cls(model, solver="newton", integrator="rk4", cone="elliptic", disable_contacts=True)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return model, solver, s0, s1


def _momentum(model, s, w: int = 0) -> np.ndarray:
    q = s.body_q.numpy()[w].astype(np.float64)
    r = _rot(q[3:])
    inertia = model.body_inertia.numpy()[w].astype(np.float64)
    return r @ inertia @ r.T @ s.body_qd.numpy()[w, 3:].astype(np.float64)


def _run(model, solver, s0, s1, steps: int, edit=None):
    for k in range(steps):
        if edit is not None:
            edit(k, s0)
        solver.step(s0, s1, None, None, DT)
        s0, s1 = s1, s0
    return s0, s1


def test_index_worlds():
    np.testing.assert_array_equal(_index_worlds(np.array([0, 7, 14]), 14, np.array([0, 1])),
                                  [0] * 7 + [1] * 7)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_tumble_keeps_mujoco_state():
    """2 s of A4's tumble: stock push each step drifts ~2.6e-4, the masked push like MuJoCo Warp alone (~2e-6)."""
    out = {}
    for synced in (False, True):
        model, solver, s0, s1 = _boxes(1, synced)
        l0 = _momentum(model, s0)
        s0, _ = _run(model, solver, s0, s1, round(2.0 / DT))
        out[synced] = float(np.linalg.norm(_momentum(model, s0) - l0) / np.linalg.norm(l0))
    assert out[True] < 2e-5 and out[False] > 5.0 * out[True], out


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_external_writes_reach_mujoco_per_world():
    """A velocity written into world 1's Newton state mid-run (as a reset, rollback, latch or test would) takes effect
    on the next step; world 0, untouched, matches a run without the write."""
    v = 0.3

    def kick(k, s):
        if k == 40:
            qd = s.joint_qd.numpy().reshape(2, 6)
            qd[1, :3] = (v, 0.0, 0.0)
            s.joint_qd.assign(qd.reshape(-1))

    model, solver, s0, s1 = _boxes(2, True)
    s_kick, _ = _run(model, solver, s0, s1, 80, kick)
    model2, solver2, r0, r1 = _boxes(2, True)
    s_ref, _ = _run(model2, solver2, r0, r1, 80)
    q_kick, q_ref = s_kick.body_q.numpy(), s_ref.body_q.numpy()
    np.testing.assert_array_equal(q_kick[0], q_ref[0])                      # world 0 never pushed differently
    assert q_kick[1, 0] == pytest.approx(v * 40 * DT, rel=1e-3)             # world 1 moved at the written velocity
    assert abs(q_ref[1, 0]) < 1e-6
