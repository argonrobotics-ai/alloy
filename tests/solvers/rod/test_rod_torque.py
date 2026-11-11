# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD R1: applied torques on rod chain bodies enter SolverVBDRod's Gauss-Newton step as a potential, not
SolverVBD's inertial target (alloy.solvers.rod.solver, Applied torques). CPU.

- A clamped rod with its physical axial inertia twists T L / (G J) under an end torque, on both paths. On the old
  path the forward step's first-order target read 2 atan(x / 2) / x of it, x = dt^2 T / I_axis.
- The user's body_f is unchanged by a step.
- With no torque on a chain body, the step is bit for bit the old path's, also with an applied force (a force stays
  in the target). Compared with R3's exact rotation target off, since the old path has Newton's first-order one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.rod import SolverVBDRod

L, R, RHO, N, DT = 0.5, 0.003, 1200.0, 12, 1.25e-3
EI = 0.05
GJ = EI / 1.3


class _OldTorquePath(SolverVBDRod):
    """SolverVBDRod before R1: applied torques go into VBD's inertial target."""

    _initialize_rigid_bodies = newton.solvers.SolverVBD._initialize_rigid_bodies


def _rod(gravity: bool):
    h = L / (N - 1)
    ls = np.full(N, h)
    ls[0] = ls[-1] = 0.5 * h
    x = np.concatenate([[0.0], np.cumsum(ls)])
    E, area = EI / (math.pi * R**4 / 4.0), math.pi * R**2
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if gravity else 0.0))
    pts = np.stack([x, np.zeros_like(x), np.full_like(x, 1.0)], axis=1)
    bodies, _ = b.add_rod(rod=newton.Rod(pts, radius=R), cfg=newton.ModelBuilder.ShapeConfig(density=RHO),
                          stretch_stiffness=E * area / h, shear_stiffness=0.35 * E * area / h, bend_stiffness=EI / h,
                          twist_stiffness=GJ / h, bend_damping=1e-3, twist_damping=1e-3, label="rod")
    for i, bi in enumerate(bodies):
        m = RHO * area * ls[i]
        It, Ia = m * (3 * R**2 + ls[i] ** 2) / 12.0, 0.5 * m * R**2
        b.body_mass[bi], b.body_inv_mass[bi] = m, 1.0 / m
        b.body_inertia[bi] = wp.mat33(It, 0, 0, 0, It, 0, 0, 0, Ia)
        b.body_inv_inertia[bi] = wp.mat33(1 / It, 0, 0, 0, 1 / It, 0, 0, 0, 1 / Ia)
    b.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
    b.color()
    return b.finalize(device="cpu"), bodies


def _twist(q0: np.ndarray, q1: np.ndarray) -> float:
    """Rotation of body 1 relative to body 0 about the segments' local z (the rod axis)."""
    rel = wp.mul(wp.quat_inverse(wp.quat(*q0[3:7])), wp.quat(*q1[3:7]))
    return 2.0 * math.atan2(float(rel[2]), float(rel[3]))


def _run(cls, model, bodies, wrench, steps: int, split: bool = False, **kw):
    """Step with `wrench` (force x y z, torque x y z, world) on the tip each step; returns the final state and the
    body_f the solver saw after its last step."""
    solver = cls(model, iterations=1, rod_split=split, **kw)
    s0, s1, c = model.state(), model.state(), model.control()
    f = np.zeros((model.body_count, 6), np.float32)
    f[bodies[-1]] = wrench
    for _ in range(steps):
        s0.clear_forces()
        s0.body_f.assign(f)
        solver.step(s0, s1, c, None, DT)
        after = s0.body_f.numpy().copy()
        s0, s1 = s1, s0
    return s0, after, f


@pytest.mark.law
@pytest.mark.parametrize("split", [False, True])
def test_end_torque_twists_t_l_over_gj_at_physical_inertia(split):
    model, bodies = _rod(gravity=False)
    T = 0.03                                                       # N m: x = dt^2 T / I_tip = 13.5 on the old path
    s, _, _ = _run(SolverVBDRod, model, bodies, (0, 0, 0, T, 0, 0), 1600, split)
    q = s.body_q.numpy()
    assert _twist(q[bodies[0]], q[bodies[-1]]) == pytest.approx(T * L / GJ, rel=1e-4, abs=0)
    assert np.abs(q[bodies, 1]).max() < 1e-6 and np.abs(q[bodies, 2] - 1.0).max() < 1e-6     # no bend from a twist


@pytest.mark.law
def test_the_old_torque_path_misses_the_twist():
    """The mutant (R1 undone) twists a fraction of T L / (G J), so the law above discriminates."""
    model, bodies = _rod(gravity=False)
    T = 0.03
    s, _, _ = _run(_OldTorquePath, model, bodies, (0, 0, 0, T, 0, 0), 1600)
    q = s.body_q.numpy()
    assert _twist(q[bodies[0]], q[bodies[-1]]) < 0.5 * T * L / GJ


def test_a_step_leaves_the_users_body_f_unchanged():
    model, bodies = _rod(gravity=True)
    _, after, given = _run(SolverVBDRod, model, bodies, (0, 0.01, 0, 0.02, -0.01, 0.005), 3)
    assert after.tobytes() == given.tobytes()


@pytest.mark.parametrize("split", [False, True])
def test_no_torque_steps_bit_for_bit_as_before(split):
    """Gravity and a lateral end force, no torque: R1 changes nothing (coordinator 06:46Z condition 1)."""
    model, bodies = _rod(gravity=True)
    new, _, _ = _run(SolverVBDRod, model, bodies, (0, 0.02, 0, 0, 0, 0), 300, split, exact_rotation_target=False)
    old, _, _ = _run(_OldTorquePath, model, bodies, (0, 0.02, 0, 0, 0, 0), 300, split)
    assert new.body_q.numpy().tobytes() == old.body_q.numpy().tobytes()
    assert new.body_qd.numpy().tobytes() == old.body_qd.numpy().tobytes()
