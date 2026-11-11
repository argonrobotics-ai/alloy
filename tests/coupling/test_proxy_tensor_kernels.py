# SPDX-License-Identifier: Apache-2.0
"""Kernel side of the anisotropic proxy mass (GAP-10A): SolverVBD's rigid solve with a per-body 3x3 mass tensor
(``SolverVBD.set_proxy_mass_tensors``, ``forward_step_rigid_bodies_tensor``, the tensor variant of
``solve_rigid_body``).

One free rigid body of mass m at the origin, turned by R (not the identity), stepped by a plain SolverVBD on the CPU.
The slider-like tensor has, in the body frame, the eigenvalues (1/m, 1/(kappa m), 1/(kappa m)) with the light axis
a_body (not a coordinate axis); Minv_world = R Minv_body R^T, M = inverse(Minv).

Laws:
- L1 identity: Minv = I/m, M = I m is the scalar body, so poses and velocities match the scalar path to float32
  round-off; with the tensors switched off again the run is bitwise the run of a solver that never had them.
- L2 forward step: dv = dt Minv_world F (Newton's second law with the proxy's W), for F along a, across a, oblique.
- L3 the solve keeps it: with no constraint the solve minimises (x - x*)^T M (x - x*) / (2 dt^2), whose minimiser
  is the forward step's target x* for any positive definite M, so the end velocity is L2's.
- L4 a spring through the solve: a spring k to an anchor, body at rest at offset p. One implicit step solves
  (M / dt^2 + k I) dx = -k p. For p along an eigen-direction of mass mu: dx = -p k dt^2 / (mu + k dt^2). The heavy
  direction (mu = kappa m) answers less than the light one (mu = m); with the scalar mass both answer alike.
- L5 gravity: it enters as the force m g through the tensor, so the coupler's -m g on a proxy cancels it exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.solvers.vbd.rigid_vbd_kernels import (create_solve_rigid_body,
                                                                    forward_step_rigid_bodies_tensor)

M = 0.8                     # kg
KAPPA = 100.0               # heavy / light mass
DT = 0.005                  # s
G = 9.81
SPRING_K = 2000.0           # N/m: k dt^2 = 0.05 kg, small beside m
INERTIA = 2.0e-3            # kg m^2, isotropic


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _rot(axis, angle: float) -> np.ndarray:
    """Rotation matrix of ``angle`` about ``axis`` (Rodrigues)."""
    k = _unit(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * kx @ kx


ROT_AXIS, ROT_ANGLE = _unit([0.3, -0.5, 0.8]), 0.9
R = _rot(ROT_AXIS, ROT_ANGLE)                       # body orientation
A_BODY = _unit([1.0, 2.0, -1.0])                    # light axis, body frame
A_WORLD = R @ A_BODY
B_WORLD = _unit(np.cross(A_WORLD, [0.0, 0.0, 1.0]))  # a heavy direction, world frame


def _quat() -> wp.quat:
    s = np.sin(0.5 * ROT_ANGLE)
    return wp.quat(*(ROT_AXIS * s), np.cos(0.5 * ROT_ANGLE))


def slider_tensors() -> tuple[np.ndarray, np.ndarray]:
    """(Minv_body, M_body) as (1, 3, 3) float32: 1/m along A_BODY, 1/(kappa m) across it."""
    aa = np.outer(A_BODY, A_BODY)
    minv = aa / M + (np.eye(3) - aa) / (KAPPA * M)
    return minv[None].astype(np.float32), np.linalg.inv(minv)[None].astype(np.float32)


def identity_tensors() -> tuple[np.ndarray, np.ndarray]:
    return (np.eye(3) / M)[None].astype(np.float32), (np.eye(3) * M)[None].astype(np.float32)


def minv_world() -> np.ndarray:
    return R @ slider_tensors()[0][0].astype(np.float64) @ R.T


def build(gravity: float = 0.0, spring: float = 0.0, pos=(0.0, 0.0, 0.0)):
    """One body of mass M at ``pos`` turned by R; with ``spring`` a D6 joint to the world whose three linear axes
    are springs of that stiffness to the origin (anchor at the body's centre of mass, rotation locked)."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    xform = wp.transform(wp.vec3(*pos), _quat())
    inertia = wp.mat33(*(np.eye(3) * INERTIA).ravel())
    if spring > 0.0:
        dof = newton.ModelBuilder.JointDofConfig
        body = b.add_link(xform=xform, mass=M, inertia=inertia)
        axes = [dof(axis=ax, target_ke=spring, target_kd=0.0, target_q=0.0)
                for ax in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)]
        joint = b.add_joint_d6(-1, body, linear_axes=axes, angular_axes=[],
                               parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), _quat()),
                               child_xform=wp.transform_identity())
        b.add_articulation([joint])
    else:
        b.add_body(xform=xform, mass=M, inertia=inertia)
    b.color()
    return b.finalize(device="cpu")


def run(model, solver, steps: int, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)):
    """Step ``steps`` times with a constant wrench at the centre of mass; returns (body_q, body_qd) of the body."""
    s0, s1, control = model.state(), model.state(), model.control()
    wrench = np.array([[*force, *torque]], dtype=np.float32)
    for _ in range(steps):
        s0.body_f.assign(wrench)
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
    return s0.body_q.numpy()[0].copy(), s0.body_qd.numpy()[0].copy()


def _solver(model, tensors=None, iterations: int = 5):
    solver = newton.solvers.SolverVBD(model, iterations=iterations)
    if tensors is not None:
        solver.set_proxy_mass_tensors(*tensors)
    return solver


FORCES = {"along": 30.0 * A_WORLD, "across": 30.0 * B_WORLD, "oblique": np.array([12.0, -20.0, 9.0])}
WRENCH = dict(force=(12.0, -20.0, 9.0), torque=(0.02, 0.01, -0.03))


def test_l1_identity_tensors_match_the_scalar_path():
    """L1: I/m and I m give the scalar body's motion to float32 round-off (1e-5 of the largest entry), after 1 and
    20 steps, with gravity, a force and a torque."""
    for steps in (1, 20):
        model = build(gravity=G)
        q_ref, qd_ref = run(model, _solver(model), steps, **WRENCH)
        model = build(gravity=G)
        q, qd = run(model, _solver(model, identity_tensors()), steps, **WRENCH)
        assert np.abs(qd_ref[:3]).max() > 0.1
        np.testing.assert_allclose(q, q_ref, rtol=0.0, atol=1e-5 * np.abs(q_ref).max())
        np.testing.assert_allclose(qd, qd_ref, rtol=0.0, atol=1e-5 * np.abs(qd_ref).max())


def test_l1_switched_off_is_bitwise_the_untouched_solver():
    """L1: a solver whose tensors were set and switched off again launches the scalar kernels: bit for bit the run
    of a solver that never had them. The slider tensors would move it elsewhere (checked)."""
    model = build(gravity=G)
    q_ref, qd_ref = run(model, _solver(model), 20, **WRENCH)
    model = build(gravity=G)
    solver = _solver(model, slider_tensors())
    assert solver.proxy_mass_tensor_enabled
    solver.set_proxy_mass_tensors(None, None)
    assert not solver.proxy_mass_tensor_enabled
    q, qd = run(model, solver, 20, **WRENCH)
    assert np.array_equal(q, q_ref) and np.array_equal(qd, qd_ref)
    model = build(gravity=G)
    _, qd_on = run(model, _solver(model, slider_tensors()), 20, **WRENCH)
    assert np.abs(qd_on[:3] - qd_ref[:3]).max() > 0.1


def test_the_default_solve_kernel_is_the_one_object_and_keeps_its_types():
    """The default variant is the cached kernel SolverVBD always used, with a float body_mass; the tensor variant
    is another kernel in a module of its own with a mat33 body_mass and the same argument names."""
    plain, tensor = create_solve_rigid_body(False), create_solve_rigid_body(False, True)
    assert plain is create_solve_rigid_body(False) and plain is not tensor and plain.module is not tensor.module
    args = {a.label: getattr(a.type, "dtype", a.type) for a in plain.adj.args}
    args_t = {a.label: getattr(a.type, "dtype", a.type) for a in tensor.adj.args}
    assert list(args) == list(args_t)
    assert args["body_mass"] is wp.float32 and args_t["body_mass"] is wp.mat33
    assert all(args[k] is args_t[k] for k in args if k != "body_mass")
    model = build()
    assert _solver(model)._solve_rigid_body_kernel is plain


def test_set_proxy_mass_tensors_checks_shapes_and_keeps_its_arrays():
    model = build()
    solver = _solver(model)
    assert solver.body_mass_tensor is None and not solver.proxy_mass_tensor_enabled
    minv, mass = slider_tensors()
    with pytest.raises(ValueError):
        solver.set_proxy_mass_tensors(np.zeros((2, 3, 3), np.float32), np.zeros((2, 3, 3), np.float32))
    with pytest.raises(ValueError):
        solver.set_proxy_mass_tensors(minv, None)
    solver.set_proxy_mass_tensors(minv, mass)
    arrays = (solver.body_inv_mass_tensor, solver.body_mass_tensor)
    ptrs = [a.ptr for a in arrays]
    solver.set_proxy_mass_tensors(*identity_tensors())
    assert (solver.body_inv_mass_tensor, solver.body_mass_tensor) == arrays and [a.ptr for a in arrays] == ptrs
    np.testing.assert_array_equal(solver.body_mass_tensor.numpy(), identity_tensors()[1])


@pytest.mark.parametrize("name", list(FORCES))
def test_l2_forward_step_gives_dt_minv_world_f(name):
    """L2: the forward-step kernel alone, from rest: dv = dt R Minv_body R^T F within 1e-4 of |dv|. Along a the body
    answers with 1/m, across it with 1/(kappa m); an oblique force is not parallel to its dv."""
    model = build()
    force = FORCES[name]
    state = model.state()
    body_f = wp.array(np.array([[*force, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.spatial_vector, device="cpu")
    q_prev, q_target = wp.clone(state.body_q), wp.clone(state.body_q)
    inv_inertia = wp.array((np.eye(3) / INERTIA)[None].astype(np.float32), dtype=wp.mat33, device="cpu")
    wp.launch(forward_step_rigid_bodies_tensor, dim=1, device="cpu",
              inputs=[DT, model.gravity, model.body_world, wp.zeros(2, dtype=wp.bool, device="cpu"), body_f,
                      model.body_com, model.body_inertia, model.body_inv_mass, inv_inertia, state.body_q,
                      state.body_qd, wp.array(slider_tensors()[0], dtype=wp.mat33, device="cpu")],
              outputs=[q_prev, q_target])
    dv = state.body_qd.numpy()[0, :3].astype(np.float64)
    expected = DT * minv_world() @ force
    assert np.linalg.norm(dv - expected) <= 1e-4 * np.linalg.norm(expected)
    scalar = DT * np.linalg.norm(force) / M
    ratio = {"along": 1.0, "across": 1.0 / KAPPA}.get(name)
    if ratio is not None:
        assert np.linalg.norm(dv) == pytest.approx(ratio * scalar, rel=1e-4)
    else:
        assert np.linalg.norm(np.cross(_unit(dv), _unit(force))) > 0.1
    np.testing.assert_allclose(q_target.numpy()[0, :3], DT * expected, rtol=0.0,
                               atol=1e-4 * DT * np.linalg.norm(expected))


@pytest.mark.parametrize("name", list(FORCES))
def test_l3_the_solve_keeps_the_forward_step(name):
    """L3: a full step (forward step and 5 iterations, no constraint) ends at L2's velocity, within 1e-4 of |dv|:
    the inertial term's minimiser is the forward-step target for any positive definite M."""
    model = build()
    force = FORCES[name]
    q, qd = run(model, _solver(model, slider_tensors()), 1, force=tuple(force))
    expected = DT * minv_world() @ force
    assert np.linalg.norm(qd[:3] - expected) <= 1e-4 * np.linalg.norm(expected)
    assert np.linalg.norm(q[:3] - DT * expected) <= 1e-4 * DT * np.linalg.norm(expected)


def _spring_step(direction: np.ndarray, tensors) -> float:
    """The body's move along ``direction`` after one step from rest at 10 mm from the spring's anchor, in m."""
    p = 0.01 * direction
    model = build(spring=SPRING_K, pos=tuple(p))
    q, _ = run(model, _solver(model, tensors, iterations=20), 1)
    return float((q[:3] - p) @ direction)


def test_l4_a_spring_moves_the_heavy_direction_less():
    """L4: dx = -p k dt^2 / (mu + k dt^2) along an eigen-direction of mass mu (docstring above). Tensor: the light
    axis answers with mu = m, the heavy one with kappa m, a ratio (kappa m + k dt^2) / (m + k dt^2) = 94.2; the
    scalar mass answers alike along both."""
    kdt2 = SPRING_K * DT * DT

    def law(mu: float) -> float:
        return -0.01 * kdt2 / (mu + kdt2)

    light, heavy = _spring_step(A_WORLD, slider_tensors()), _spring_step(B_WORLD, slider_tensors())
    assert light == pytest.approx(law(M), rel=2e-2)
    assert heavy == pytest.approx(law(KAPPA * M), rel=2e-2)
    assert light / heavy == pytest.approx((KAPPA * M + kdt2) / (M + kdt2), rel=3e-2)
    scalar_a, scalar_b = _spring_step(A_WORLD, None), _spring_step(B_WORLD, None)
    assert scalar_a == pytest.approx(law(M), rel=2e-2)
    assert scalar_a == pytest.approx(scalar_b, rel=1e-3)


def test_l5_gravity_is_the_weight_through_the_tensor():
    """L5: gravity enters as the force m g through the tensor, dv = dt Minv_world (m g). An identity tensor falls
    with g (L1). The coupler writes -f - m g into a proxy's force input (it takes VBD's own gravity back), so a
    proxy carrying that input alone gains no velocity along any direction (within 1e-4 g dt), where a scalar
    gravity term beside the tensor would leave g dt (1 - m Minv_world) behind."""
    model = build(gravity=G)
    weight = np.array([0.0, 0.0, -M * G])
    _, qd = run(model, _solver(model, slider_tensors()), 1)
    assert np.linalg.norm(qd[:3] - DT * minv_world() @ weight) <= 1e-4 * G * DT
    _, qd = run(model, _solver(model, slider_tensors()), 1, force=tuple(-weight))
    assert np.linalg.norm(qd[:3]) <= 1e-4 * G * DT


def test_a_model_with_tendons_keeps_the_scalar_kernels():
    """The tendon variant of the tensor solve has never been stepped: the solver refuses the tensors and warns."""
    model = build()
    solver = _solver(model)
    model.tendon_count = 1                                  # after construction: only the refusal reads it here
    with pytest.warns(UserWarning, match="tendons"):
        solver.set_proxy_mass_tensors(*slider_tensors())
    assert not solver.proxy_mass_tensor_enabled and solver.body_mass_tensor is None
