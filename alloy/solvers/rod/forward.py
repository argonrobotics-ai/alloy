# SPDX-License-Identifier: Apache-2.0
"""SolverVBDRod's forward-step kernels (SCR-WP9-ROD R1, R3a; the method is in `solver`'s docstring, The forward
step): an applied torque on a chain body leaves the inertial target for the chain's Gauss-Newton step, and the
target rotates by exp(omega dt)."""

from __future__ import annotations

import warp as wp


@wp.kernel
def rod_before_forward(body: wp.array(dtype=int), body_f: wp.array(dtype=wp.spatial_vector),
                       body_q: wp.array(dtype=wp.transform), applied_t: wp.array(dtype=wp.vec3),
                       pose0: wp.array(dtype=wp.transform)):
    """Per chain slot, before SolverVBD's forward step: keep the start pose, and move a nonzero applied torque from
    `body_f` (Newton order: force, torque) to `applied_t`, so the inertial target leaves it out. A zero torque stays
    where it is."""
    k = wp.tid()
    b = body[k]
    pose0[k] = body_q[b]
    f = body_f[b]
    tau = wp.spatial_bottom(f)
    applied_t[k] = wp.vec3(0.0)
    if wp.length_sq(tau) > 0.0:
        applied_t[k] = tau
        body_f[b] = wp.spatial_vector(wp.spatial_top(f), wp.vec3(0.0))


@wp.kernel
def rod_after_forward(body: wp.array(dtype=int), applied_t: wp.array(dtype=wp.vec3),
                      pose0: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                      inv_mass: wp.array(dtype=float), body_qd: wp.array(dtype=wp.spatial_vector), dt: float,
                      exact: int, body_f: wp.array(dtype=wp.spatial_vector), body_q: wp.array(dtype=wp.transform),
                      body_inertia_q: wp.array(dtype=wp.transform)):
    """Per chain slot, after the forward step: put the torque back into the user's `body_f`. With `exact`, rotate the
    inertial target (and the initial guess) by exp(omega dt) from the start pose instead of Newton's first-order
    normalize(q + dt/2 omega q), keeping the forward step's COM."""
    k = wp.tid()
    b = body[k]
    tau = applied_t[k]
    if wp.length_sq(tau) > 0.0:
        body_f[b] = wp.spatial_vector(wp.spatial_top(body_f[b]), tau)
    if exact == 0 or inv_mass[b] == 0.0:
        return
    w = wp.spatial_bottom(body_qd[b])
    ang = wp.length(w) * dt
    dq = wp.quat_identity()
    if ang > 0.0:
        dq = wp.quat_from_axis_angle(wp.normalize(w), ang)
    rot = wp.normalize(wp.mul(dq, wp.transform_get_rotation(pose0[k])))
    c = body_com[b]
    target = wp.transform(wp.transform_point(body_q[b], c) - wp.quat_rotate(rot, c), rot)
    body_q[b] = target
    body_inertia_q[b] = target


__all__ = ["rod_after_forward", "rod_before_forward"]
