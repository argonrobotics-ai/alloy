# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD R3: a free rod rolls down an incline at its own rolling acceleration, CPU.

A 12-segment rod (r 5 mm, 0.3 m) lies across a plane tilted by alpha (gravity tilted in x), mu 0.8, 2 iterations at
the production 1.25 ms.
- Without slip, the COM accelerates at g sin(alpha) / (1 + I / (m rho^2)). Here m, I_axis and the rolling radius rho
  (the centre height at the end) are the model's own.
- Bounds: 3 % on the acceleration and 1 % on v / (omega rho).
- Newton's first-order rotation target (`exact_rotation_target=False`) loses (omega dt)^2 / 12 of the spin per step
  and reads 0.948 at 10 deg. That run must miss the bound, so the law discriminates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.rod import SolverVBDRod

L, R, RHO, N, DT, G, EI = 0.3, 0.005, 1200.0, 12, 1.25e-3, 9.81, 0.05


def _rod_on_incline(alpha: float, mu: float):
    h = L / (N - 1)
    ls = np.full(N, h)
    ls[0] = ls[-1] = 0.5 * h
    y = np.concatenate([[0.0], np.cumsum(ls)]) - L / 2
    E, area = EI / (math.pi * R**4 / 4.0), math.pi * R**2
    b = newton.ModelBuilder(gravity=(G * math.sin(alpha), 0.0, -G * math.cos(alpha)))
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    pts = np.stack([np.zeros_like(y), y, np.full_like(y, R)], axis=1)
    bodies, _ = b.add_rod(rod=newton.Rod(pts, radius=R),
                          cfg=newton.ModelBuilder.ShapeConfig(density=RHO, gap=0.002, mu=mu),
                          stretch_stiffness=E * area / h, shear_stiffness=0.35 * E * area / h, bend_stiffness=EI / h,
                          twist_stiffness=EI / 1.3 / h, bend_damping=1e-3, twist_damping=1e-3, label="rod")
    for i, bi in enumerate(bodies):
        m = RHO * area * ls[i]
        It, Ia = m * (3 * R**2 + ls[i] ** 2) / 12.0, 0.5 * m * R**2
        b.body_mass[bi], b.body_inv_mass[bi] = m, 1.0 / m
        b.body_inertia[bi] = wp.mat33(It, 0, 0, 0, It, 0, 0, 0, Ia)
        b.body_inv_inertia[bi] = wp.mat33(1 / It, 0, 0, 0, 1 / It, 0, 0, 0, 1 / Ia)
    b.color()
    return b.finalize(device="cpu"), bodies


def _roll(alpha_deg: float, exact: bool, seconds: float = 0.4) -> tuple[float, float]:
    """(acceleration / the model's rolling acceleration, v / (omega rho)) over the second half of the run."""
    a = math.radians(alpha_deg)
    model, bodies = _rod_on_incline(a, 0.8)
    solver = SolverVBDRod(model, iterations=2, exact_rotation_target=exact)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    s0, s1, c = model.state(), model.state(), model.control()
    m = model.body_mass.numpy()[bodies].astype(np.float64)
    ia = model.body_inertia.numpy()[bodies][:, 2, 2].astype(np.float64)
    xs = []
    for _ in range(int(round(seconds / DT))):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, c, contacts, DT)
        s0, s1 = s1, s0
        xs.append(float(np.sum(m * s0.body_q.numpy()[bodies, 0]) / m.sum()))
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    assert np.isfinite(q).all()
    t = DT * np.arange(1, len(xs) + 1)
    half = len(t) // 2
    acc = 2.0 * np.polyfit(t[half:], np.asarray(xs[half:]), 2)[0]
    rho = float(np.mean(q[bodies, 2]))
    model_acc = G * math.sin(a) / (1.0 + ia.sum() / (m.sum() * rho**2))
    v = float(np.sum(m * qd[bodies, 0]) / m.sum())
    return acc / model_acc, v / (abs(float(np.mean(qd[bodies, 4]))) * rho)


@pytest.mark.law
@pytest.mark.parametrize("alpha", [10.0, 20.0])
def test_a_free_rod_rolls_at_its_own_rolling_acceleration(alpha):
    acc, slip = _roll(alpha, exact=True)
    assert acc == pytest.approx(1.0, abs=0.03)
    assert slip == pytest.approx(1.0, abs=0.01)


@pytest.mark.law
def test_the_first_order_rotation_target_misses_it():
    acc, _ = _roll(10.0, exact=False)
    assert acc < 0.97
