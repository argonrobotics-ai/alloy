# SPDX-License-Identifier: Apache-2.0
"""Host-side measure helpers shared by scenarios (numpy only, called outside the timed window).

Conventions (Newton ac7f7830, docs/concepts/conventions.rst:105-135):
- body_q rows are (px, py, pz, qx, qy, qz, qw), the body-frame origin in world coordinates.
- body_qd rows are (vx, vy, vz, wx, wy, wz): linear = COM velocity, angular, both in the world frame.
- rigid contact point0/point1 are body-frame points; normal points from shape 0 to shape 1;
  margin0/margin1 are the surface thickness (effective radius + margin) of each shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) stored xyzw."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def transform_points(body_q: np.ndarray, body: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    """World position of body-frame points; body index -1 means the static world frame (identity)."""
    out = p_local.astype(np.float64).copy()
    moving = body >= 0
    if np.any(moving):
        x = body_q[body[moving]].astype(np.float64)
        out[moving] = x[:, :3] + quat_rotate(x[:, 3:7], out[moving])
    return out


def per_world(values: np.ndarray, world_of_item: np.ndarray, nworld: int, reduce=np.max) -> np.ndarray:
    """Reduce per-item values into one value per world (worlds without items get NaN)."""
    res = np.full(nworld, np.nan)
    for w in range(nworld):
        sel = values[world_of_item == w]
        if sel.size:
            res[w] = reduce(sel)
    return res


def rigid_energy(model: Any, state: Any, z_ref: float = 0.0) -> float:
    """Total kinetic + gravitational energy [J] of all rigid bodies, gravity taken from model.gravity.

    Potential energy is m * |g| * (h - z_ref) along the gravity direction, using the COM position.
    """
    if model.body_count == 0:
        return 0.0
    q = state.body_q.numpy().astype(np.float64)
    qd = state.body_qd.numpy().astype(np.float64)
    mass = model.body_mass.numpy().astype(np.float64)
    inertia = model.body_inertia.numpy().astype(np.float64)  # body frame, about the COM
    com = transform_points(q, np.arange(len(q)), model.body_com.numpy())
    g = np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)
    g_body = g[model.body_world.numpy().clip(min=0)] if g.shape[0] > 1 else np.broadcast_to(g[0], com.shape)
    ke_lin = 0.5 * mass * np.sum(qd[:, :3] ** 2, axis=1)
    q_inv = q[:, 3:7] * np.array([-1.0, -1.0, -1.0, 1.0])
    w_body = quat_rotate(q_inv, qd[:, 3:6])
    ke_ang = 0.5 * np.einsum("bi,bij,bj->b", w_body, inertia, w_body)
    pe = -mass * np.sum(g_body * com, axis=1) - mass * np.linalg.norm(g_body, axis=1) * z_ref
    return float(np.sum(ke_lin + ke_ang + pe))


def particle_energy(model: Any, state: Any, z_ref: float = 0.0) -> float:
    """Kinetic + gravitational energy [J] of all particles (elastic energy is solver specific, not included)."""
    if model.particle_count == 0:
        return 0.0
    x = state.particle_q.numpy().astype(np.float64)
    v = state.particle_qd.numpy().astype(np.float64)
    m = model.particle_mass.numpy().astype(np.float64)
    g = np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)[0]
    return float(np.sum(0.5 * m * np.sum(v**2, axis=1) - m * (x @ g)) - np.sum(m) * np.linalg.norm(g) * z_ref)


def rigid_penetration_mm(model: Any, state: Any, contacts: Any) -> np.ndarray:
    """Penetration depth [mm] (positive = overlap) of every active rigid contact.

    depth = (margin0 + margin1) - dot(n, X1 p1 - X0 p0), the signed gap convention of Newton's XPBD contact kernel.
    Returns an empty array when there are no contacts. Hydroelastic and SDF contacts use the same rows.
    """
    if contacts is None or getattr(contacts, "rigid_contact_count", None) is None:
        return np.zeros(0)
    n = int(contacts.rigid_contact_count.numpy()[0])
    n = min(n, int(contacts.rigid_contact_max))
    if n == 0:
        return np.zeros(0)
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    ok = (s0 >= 0) & (s1 >= 0)
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy() if model.body_count else np.zeros((0, 7), np.float32)
    p0 = transform_points(body_q, shape_body[s0[ok]], contacts.rigid_contact_point0.numpy()[:n][ok])
    p1 = transform_points(body_q, shape_body[s1[ok]], contacts.rigid_contact_point1.numpy()[:n][ok])
    nrm = contacts.rigid_contact_normal.numpy()[:n][ok].astype(np.float64)
    thick = contacts.rigid_contact_margin0.numpy()[:n][ok] + contacts.rigid_contact_margin1.numpy()[:n][ok]
    gap = np.sum(nrm * (p1 - p0), axis=1) - thick
    return np.clip(-gap, 0.0, None) * 1e3


def finite_state(arrays: dict[str, np.ndarray]) -> bool:
    """True when every state array is finite."""
    return all(bool(np.isfinite(a).all()) for a in arrays.values())


def arrays_identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    """Bitwise equality of two state snapshots (NaN == NaN)."""
    if a.keys() != b.keys():
        return False
    return all(x.shape == b[k].shape and np.array_equal(x.view(np.uint8), b[k].view(np.uint8)) for k, x in a.items())
