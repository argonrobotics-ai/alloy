# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD Phase 2: J2 rod joints (alloy.solvers.rod.plastic) on SolverVBDRod, CPU, gravity off.

A clamped copper wire (0.2 m, R 1 mm, 20 segments; E 117 GPa, σ_y0 70 MPa, perfectly plastic) under a held end moment
about a bend axis: ramped over 0.2 s, held, released. The moment is uniform along the span, so every joint bends to
the same curvature, read from the end segment's rotation.
- 1.5 M_y: the loaded curvature matches the closed-form M(κ) of a round section, and the wire keeps
  κ_res = κ_max - M / EI after release (alloy.solvers.plastic.closed_forms).
- 0.8 M_y: elastic, so no fibre yields and the rest rotations stay the authored ones, bit for bit.
- An end torque of 1.25 T_y (0.94 T_p, T_p = 4 T_y / 3) about the wire's axis: the twist rate matches the closed-form
  T(φ') of a round section (τ_y = σ_y0 / sqrt 3) within 1 % (measured +0.74 %), and the wire keeps φ'_max - T / GJ
  after release within 1 % of φ'_max (measured 0.73 %). In torsion most of the loaded twist is elastic, so the
  residual is judged against the loaded twist: as a fraction of itself it carries the loaded error several times
  over (+3.5 % here, -6.6 % at 1.15 T_y).
- The bend residual is judged the same way.
- A model change (`notify_model_changed`) re-reads the authored rest; a yielded joint keeps its plastic rest
  bit for bit, so the set wire holds its shape (WP7's review).
- A yielding joint bends alike about both axes: SolverVBD refuses one that does not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.scene.analytic_rods import rod_card
from alloy.scene.analytic_rods_vbd import vbd_card
from alloy.scene.emitters.rod_vbd import build_rod_vbd
from alloy.solvers.plastic import closed_forms as C
from alloy.solvers.rod import SolverVBDRod
from alloy.solvers.rod.plastic import mark_joints

L, R, N, E, SY, RHO = 0.2, 1.0e-3, 20, 117e9, 70e6, 8960.0
EI = E * math.pi * R**4 / 4.0
M_Y = SY * math.pi * R**3 / 4.0
DT, RAMP, HOLD = 1.25e-3, 0.2, 0.4


def _kappa_of(moment: float) -> float:
    """The closed form's curvature at `moment` (bisection on round_moment)."""
    lo, hi = 0.0, 50.0 * M_Y / EI
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if C.round_moment(mid, R, E, SY) < moment else (lo, mid)
    return 0.5 * (lo + hi)


def _phi_of(torque: float, G: float) -> float:
    """The closed form's twist rate at `torque` (bisection on round_torque)."""
    lo, hi = 0.0, 50.0 * torque / (G * math.pi * R**4 / 2.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if C.round_torque(mid, R, G, SY) < torque else (lo, mid)
    return 0.5 * (lo + hi)


def _about(axis: int, q: np.ndarray, q0: np.ndarray) -> float:
    """Rotation about world axis `axis` (0 x, 2 z) of xyzw q relative to q0 (swing-twist)."""
    w = q[3] * q0[3] + q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2]
    if axis == 0:
        c = -q[3] * q0[0] + q[0] * q0[3] - q[1] * q0[2] + q[2] * q0[1]
    else:
        c = -q[3] * q0[2] + q[2] * q0[3] - q[0] * q0[1] + q[1] * q0[0]
    return 2.0 * math.atan2(c, w)


def _about_z(q: np.ndarray, q0: np.ndarray) -> float:
    w = q[3] * q0[3] + q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2]
    z = -q[3] * q0[2] + q[2] * q0[3] - q[0] * q0[1] + q[1] * q0[0]
    return 2.0 * math.atan2(z, w)


def _bend(moment: float, axis: int = 2):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rod = vbd_card(rod_card(L, EI, N, radius=R, density=RHO, damping_ratio=1.0))
    h = build_rod_vbd(b, rod, [(0.0, 0.0, 0.3), (L, 0.0, 0.3)], clamp="start", label="wire")
    mark_joints(b, h.joints, SY, 0.0, R, L / (N - 1))               # the emitter's joint spacing (half-length ends)
    b.color()
    model = b.finalize(device="cpu")
    solver = SolverVBDRod(model, iterations=2)
    assert solver.rod_plastic is not None and solver.rod_plastic.joints.size == N - 1
    s0, s1, c = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    end = int(h.bodies[-1])
    q0 = s0.body_q.numpy()[end, 3:].copy()
    rest0 = solver._rest_rel.numpy().copy()

    def run(seconds: float, m_of_t) -> float:
        nonlocal s0, s1
        for k in range(int(round(seconds / DT))):
            f = np.zeros((model.body_count, 6), np.float32)
            f[end, 3 + axis] = m_of_t((k + 1) * DT)
            s0.body_f.assign(f)
            solver.step(s0, s1, c, None, DT)
            s0, s1 = s1, s0
        return _about(axis, s0.body_q.numpy()[end, 3:], q0) / L             # N - 1 joints, L / (N - 1) apart

    run(RAMP, lambda t: moment * min(t / RAMP, 1.0))
    loaded = run(HOLD, lambda t: moment)
    released = run(HOLD, lambda t: 0.0)
    return loaded, released, solver, rest0, rod, (s0, s1, c, end, q0)


@pytest.mark.law
def test_a_wire_bent_past_yield_keeps_the_closed_form_set():
    m = 1.5 * M_Y
    loaded, released, solver, _, _, _ = _bend(m)
    k_max = _kappa_of(m)
    assert loaded == pytest.approx(k_max, rel=0.01, abs=0)
    assert abs(released - C.springback_curvature(k_max, m, EI)) < 0.01 * k_max
    assert float(solver.rod_plastic.ebar_max.numpy().max()) > 0.0


@pytest.mark.law
def test_below_yield_the_wire_springs_back_and_its_rest_is_untouched():
    loaded, released, solver, rest0, _, _ = _bend(0.8 * M_Y)
    assert loaded == pytest.approx(0.8 * M_Y / EI, rel=0.01, abs=0)
    assert abs(released) < 1e-4 * loaded
    assert not solver.rod_plastic.ebar.numpy().any()
    assert np.array_equal(solver._rest_rel.numpy(), rest0)


@pytest.mark.law
def test_a_wire_twisted_past_yield_keeps_the_closed_form_set():
    G = E / (2.0 * 1.3)                                  # the card's GJ = EI / 1.3 (rod_card: nu 0.3)
    t_y = math.pi * R**3 * (SY / math.sqrt(3.0)) / 2.0
    t = 1.25 * t_y                                       # 0.94 T_p (T_p = 4 T_y / 3)
    loaded, released, solver, _, rod, _ = _bend(t, axis=0)
    gj = rod.GJ
    assert gj == pytest.approx(G * math.pi * R**4 / 2.0, rel=1e-9, abs=0)
    phi = _phi_of(t, G)
    assert loaded == pytest.approx(phi, rel=0.01, abs=0)
    assert abs(released - (phi - t / gj)) < 0.01 * phi


def test_a_model_change_keeps_a_yielded_wires_set():
    """WP7's review: notify_model_changed re-reads the authored rest; a yielded joint keeps its plastic rest, so the
    set wire does not spring back for a step."""
    _, released, solver, rest0, _, (s0, s1, c, end, q0) = _bend(1.5 * M_Y)
    plastic_rest = solver._rest_rel.numpy().copy()
    assert not np.array_equal(plastic_rest, rest0)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    assert np.array_equal(solver._rest_rel.numpy(), plastic_rest)
    assert np.array_equal(solver.rod_plastic.rest0.numpy(), rest0)
    for _ in range(40):
        solver.step(s0, s1, c, None, DT)
        s0, s1 = s1, s0
    assert _about(2, s0.body_q.numpy()[end, 3:], q0) / L == pytest.approx(released, rel=1e-3, abs=0)


def test_a_yielding_joint_must_bend_alike_both_ways():
    """The J2 section is round (plastic.py takes E from the first bend stiffness); SolverVBD refuses a ROD joint whose
    two bend stiffnesses differ, which that relies on."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    h = build_rod_vbd(b, vbd_card(rod_card(L, EI, N, radius=R, density=RHO)), [(0, 0, 0.3), (L, 0, 0.3)],
                      clamp="start", label="wire", plastic=(SY, 0.0))
    j = h.joints[3]
    b.joint_target_ke[b.joint_qd_start[j] + 4] *= 1.5
    b.color()
    with pytest.raises(ValueError, match="isotropic ROD"):
        SolverVBDRod(b.finalize(device="cpu"), iterations=1)
