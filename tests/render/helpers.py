# SPDX-License-Identifier: Apache-2.0
"""Tiny Newton models and state helpers for the render recorder tests (CPU)."""
from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import warp as wp

IDENT = wp.quat_identity()


def tetra_mesh() -> newton.Mesh:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32) * 0.1
    f = np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 0, 3, 2], np.int32)
    return newton.Mesh(v, f)


def rigid_world() -> newton.ModelBuilder:
    """Ground box, free box, sphere, capsule, mesh, plus a hidden site and an invisible shape."""
    b = newton.ModelBuilder()
    b.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05), IDENT), label="ground")
    box = b.add_body(xform=wp.transform((0, 0, 1), IDENT), label="box_body")
    b.add_shape_box(box, hx=0.1, hy=0.2, hz=0.3, label="box_shape")
    sph = b.add_body(xform=wp.transform((1, 0, 1), IDENT), label="sphere_body")
    b.add_shape_sphere(sph, radius=0.15, label="sphere_shape")
    cap = b.add_body(xform=wp.transform((2, 0, 1), IDENT), label="capsule_body")
    b.add_shape_capsule(cap, radius=0.05, half_height=0.2, label="capsule_shape")
    msh = b.add_body(xform=wp.transform((3, 0, 1), IDENT), label="mesh_body")
    b.add_shape_mesh(msh, mesh=tetra_mesh(), scale=(2.0, 2.0, 2.0), label="mesh_shape")
    b.add_site(msh, label="a_site")
    b.add_shape_box(msh, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(is_visible=False),
                    label="hidden_proxy")
    return b


def move_bodies(state, dz: float) -> None:
    q = state.body_q.numpy().copy()
    q[:, 2] += dz
    state.body_q.assign(q)


def shift_particles(state, dx: float) -> None:
    q = state.particle_q.numpy().copy()
    q[:, 0] += dx
    state.particle_q.assign(q)
