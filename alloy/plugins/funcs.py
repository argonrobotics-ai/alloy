# SPDX-License-Identifier: Apache-2.0
"""Warp helper functions for plug-in kernels (``@wp.func``; call them inside ``@wp.kernel``).

Conventions (Newton): ``body_q[b]`` is the body frame's world transform; ``model.body_com[b]`` the centre of mass in
the body frame; ``body_qd[b]`` = (COM linear velocity, angular velocity), world frame; ``body_f[b]`` = (force at the
COM, torque), world frame.

Example::

    @wp.kernel
    def push(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], body: wp.array[wp.int32],
             tip: wp.vec3, force: wp.vec3, body_f: wp.array[wp.spatial_vector]):
        w = wp.tid()
        b = body[w]
        p = body_point(body_q, b, tip)
        add_wrench(body_f, b, wrench_at(force, p, com_world(body_q, body_com, b)))
"""

from __future__ import annotations

import warp as wp


@wp.func
def body_point(body_q: wp.array[wp.transform], b: wp.int32, p_local: wp.vec3) -> wp.vec3:
    """World position of point ``p_local`` (body frame) of body ``b``."""
    return wp.transform_point(body_q[b], p_local)


@wp.func
def body_vector(body_q: wp.array[wp.transform], b: wp.int32, v_local: wp.vec3) -> wp.vec3:
    """World direction of vector ``v_local`` (body frame) of body ``b``."""
    return wp.transform_vector(body_q[b], v_local)


@wp.func
def com_world(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], b: wp.int32) -> wp.vec3:
    """World position of body ``b``'s centre of mass."""
    return wp.transform_point(body_q[b], body_com[b])


@wp.func
def point_velocity(body_qd: wp.array[wp.spatial_vector], com: wp.vec3, b: wp.int32, p: wp.vec3) -> wp.vec3:
    """World velocity of world point ``p`` moving with body ``b`` (``com`` = its world COM)."""
    qd = body_qd[b]
    v = wp.vec3(qd[0], qd[1], qd[2])
    omega = wp.vec3(qd[3], qd[4], qd[5])
    return v + wp.cross(omega, p - com)


@wp.func
def wrench_at(force: wp.vec3, point: wp.vec3, com: wp.vec3) -> wp.spatial_vector:
    """The COM wrench of ``force`` applied at world ``point``: (force, (point - com) x force)."""
    tau = wp.cross(point - com, force)
    return wp.spatial_vector(force[0], force[1], force[2], tau[0], tau[1], tau[2])


@wp.func
def torque_wrench(torque: wp.vec3) -> wp.spatial_vector:
    """A pure torque as a wrench."""
    return wp.spatial_vector(0.0, 0.0, 0.0, torque[0], torque[1], torque[2])


@wp.func
def add_wrench(body_f: wp.array[wp.spatial_vector], b: wp.int32, w: wp.spatial_vector):
    """``body_f[b] += w`` without an atomic: call it from the one thread that owns ``b`` (e.g. its world's)."""
    body_f[b] = body_f[b] + w


__all__ = ["add_wrench", "body_point", "body_vector", "com_world", "point_velocity", "torque_wrench", "wrench_at"]
