# SPDX-License-Identifier: Apache-2.0
"""CUDA: the controller's pre_step/apply can be captured in a graph (no allocation, no host sync) and its kernels
are NATIVE. The same inputs give bitwise-equal outputs, eager or replayed (D9). A full captured loop
(VBD + controller) tears the strip and stays finite."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.cards.types import Failure
from alloy.core.quantity import Q
from alloy.testing.topology import clamp_particles, cloth_kwargs, grid_tri_mesh
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_cloth
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget

pytestmark = pytest.mark.gpu


def _strip(device: str, worlds: int = 2):
    t = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, tris = grid_tri_mesh(12, 6, 5e-3)
    add_presplit_cloth(t, v, tris, SplitPlan(spacing=1.5e-2, axes=(0, 1)), cloth_kwargs(),
                       budget=TopologyBudget(spare_vertices=1000),
                       failure=Failure(fracture_strain=Q(0.05, "model", note="test")))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.replicate(t, worlds)
    b.color()
    return b.finalize(device=device)


def _cuda(device: str) -> None:
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")


def test_controller_capture_matches_eager_bitwise(device):
    _cuda(device)
    m = _strip(device)
    state = m.state()
    q = state.particle_q.numpy()
    q[:, 0] *= 1.08                                     # stretched past fracture: many bonds want to break
    state.particle_q.assign(q)
    outs = []
    for mode in ("eager", "graph"):
        bonds = bonds_from_model(m)
        ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=3))
        if mode == "graph":
            with wp.ScopedCapture(device=device) as cap:
                ctl.pre_step(state, None, 1e-3)
                ctl.apply(state)
            for _ in range(4):
                wp.capture_launch(cap.graph)
        else:
            for _ in range(4):
                ctl.pre_step(state, None, 1e-3)
                ctl.apply(state)
        outs.append((bonds.intact.numpy().copy(), ctl.strain.numpy().copy(), ctl.ledger().events.copy()))
        m.spring_stiffness.assign(np.where(m.spring_stiffness.numpy() == 0.0, 1e5, m.spring_stiffness.numpy()))
    for a, b in zip(outs[0], outs[1]):
        np.testing.assert_array_equal(a, b)
    assert outs[0][2].sum() == 2 * 4 * 3                # capped at 3 per world per step, 4 steps, 2 worlds


def test_captured_loop_tears_and_stays_finite(device):
    _cuda(device)
    m = _strip(device)
    x0 = m.particle_q.numpy()
    right = np.flatnonzero(np.isclose(x0[:, 0], 0.06))
    clamp_particles(m, np.concatenate([np.flatnonzero(np.isclose(x0[:, 0], 0.0)), right]))
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=8))
    solver = newton.solvers.SolverVBD(m, iterations=10)
    s0, s1, c = m.state(), m.state(), m.control()
    target = wp.array(x0[right], dtype=wp.vec3, device=device)
    ids = wp.array(right.astype(np.int32), dtype=wp.int32, device=device)

    @wp.kernel
    def pull(ids: wp.array[wp.int32], x0: wp.array[wp.vec3], s: float, q: wp.array[wp.vec3]):
        i = wp.tid()
        q[ids[i]] = wp.vec3(x0[i][0] * s, x0[i][1], x0[i][2])

    def period() -> None:
        for _ in range(10):
            wp.launch(pull, dim=len(right), inputs=[ids, target, 1.0, s0.particle_q], device=device)
            ctl.pre_step(s0, None, 1e-3)
            solver.step(s0, s1, c, None, 1e-3)
            ctl.apply(s1)
            wp.copy(s0.particle_q, s1.particle_q)
            wp.copy(s0.particle_qd, s1.particle_qd)

    with wp.ScopedCapture(device=device) as cap:         # one capture, replayed with a host-updated pull target
        period()
    for k in range(40):
        target.assign((x0[right] * np.array([1.0 + 2.5e-3 * (k + 1), 1.0, 1.0])).astype(np.float32))
        wp.capture_launch(cap.graph)
    assert (bonds.intact.numpy() == 0).sum() > 0
    assert np.isfinite(s0.particle_q.numpy()).all()
