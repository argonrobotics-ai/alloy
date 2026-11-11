# SPDX-License-Identifier: Apache-2.0
"""Per-world physical measures on the device: rigid energy (D20 datum) and linear/angular momentum.

One thread per world loops over that world's bodies (`WorldRanges`) and accumulates in float64, so the sums are
deterministic (no atomics) and precise enough for the 1e-6 relative energy checks of A3.
Conventions (newton@ac7f7830:newton/_src/sim/state.py:135-138): `body_q` is the body-origin transform, `body_qd` is
(COM linear velocity, angular velocity), both in the world frame; `body_com` is the COM in the body frame and
`body_inertia` is about the COM in the body frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.bench.twin import DeviceTwinLog, WorldRanges


@wp.func
def _body_energy(X: wp.transform, qd: wp.spatial_vector, m: float, inertia: wp.mat33, com: wp.vec3, up: wp.vec3d,
                 gmag: wp.float64, z_ref: wp.float64) -> wp.float64:
    x = wp.vec3d(wp.transform_point(X, com))
    v = wp.vec3d(wp.spatial_top(qd))
    omega = wp.spatial_bottom(qd)
    w_body = wp.vec3d(wp.quat_rotate_inv(wp.transform_get_rotation(X), omega))
    i_d = wp.mat33d(inertia)
    md = wp.float64(m)
    ke = wp.float64(0.5) * md * wp.dot(v, v) + wp.float64(0.5) * wp.dot(w_body, i_d * w_body)
    return ke + md * gmag * (wp.dot(x, up) - z_ref)


@wp.kernel
def _world_energy(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                  mass: wp.array(dtype=float), inertia: wp.array(dtype=wp.mat33), com: wp.array(dtype=wp.vec3),
                  gravity: wp.array(dtype=wp.vec3), start: wp.array(dtype=wp.int32), end: wp.array(dtype=wp.int32),
                  z_ref: wp.array(dtype=wp.float64), col: int, update_gain: int,
                  out: wp.array2d(dtype=wp.float64), prev: wp.array(dtype=wp.float64),
                  max_gain: wp.array(dtype=wp.float64)):
    w = wp.tid()
    g = wp.vec3d(gravity[w])
    gmag = wp.length(g)
    up = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    if gmag > wp.float64(0.0):
        up = -g / gmag
    e = wp.float64(0.0)
    for b in range(start[w], end[w]):
        e += _body_energy(body_q[b], body_qd[b], mass[b], inertia[b], com[b], up, gmag, z_ref[w])
    if col >= 0:
        out[w, col] = e
    if update_gain != 0:
        max_gain[w] = wp.max(max_gain[w], e - prev[w])
    prev[w] = e


@wp.kernel
def _world_momentum(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                    mass: wp.array(dtype=float), inertia: wp.array(dtype=wp.mat33), com: wp.array(dtype=wp.vec3),
                    start: wp.array(dtype=wp.int32), end: wp.array(dtype=wp.int32), col: int,
                    p_out: wp.array2d(dtype=wp.vec3d), l_out: wp.array2d(dtype=wp.vec3d)):
    w = wp.tid()
    p = wp.vec3d()
    ang = wp.vec3d()
    for b in range(start[w], end[w]):
        X = body_q[b]
        md = wp.float64(mass[b])
        x = wp.vec3d(wp.transform_point(X, com[b]))
        v = wp.vec3d(wp.spatial_top(body_qd[b]))
        q = wp.transform_get_rotation(X)
        w_body = wp.vec3d(wp.quat_rotate_inv(q, wp.spatial_bottom(body_qd[b])))
        l_body = wp.mat33d(inertia[b]) * w_body
        rot = wp.mat33d(wp.quat_to_matrix(q))
        p += md * v
        ang += wp.cross(x, md * v) + rot * l_body
    p_out[w, col] = p
    l_out[w, col] = ang


def reset_heights(model, state) -> np.ndarray:
    """D20 datum per world: the lowest body COM height (along -g) in the given state, float64 `[W]`."""
    q = state.body_q.numpy().astype(np.float64)
    com = model.body_com.numpy().astype(np.float64)
    world = model.body_world.numpy()
    g = model.gravity.numpy().astype(np.float64).reshape(-1, 3)
    out = np.zeros(model.world_count)
    for w in range(model.world_count):
        idx = np.flatnonzero(world == w)
        if not idx.size:
            continue
        gw = g[w] if g.shape[0] > w else g[-1]
        up = -gw / np.linalg.norm(gw) if np.linalg.norm(gw) > 0 else np.array([0.0, 0.0, 1.0])
        x = q[idx, :3] + _rotate(q[idx, 3:7], com[idx])
        out[w] = float(np.min(x @ up))
    return out


def _rotate(qxyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, s = qxyzw[:, :3], qxyzw[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + s * t + np.cross(u, t)


class EnergyMeter:
    """Records per-world rigid energy into channel `name` and tracks the max per-step gain (`acc:<name>_gain`)."""

    def __init__(self, model, state, log: DeviceTwinLog, name: str = "energy") -> None:
        self.model = model
        self.log = log
        self.name = name
        self.ranges = WorldRanges.bodies(model)
        self.z_ref = wp.array(reset_heights(model, state), dtype=wp.float64, device=model.device)
        self.prev = log.accumulator(f"{name}_prev")
        self.gain = log.accumulator(f"{name}_gain", fill=-np.inf)

    def update(self, state, step: int) -> None:
        """Measure after outer step `step` (step 0 = initial state, which only seeds the gain tracker)."""
        m = self.model
        wp.launch(_world_energy, dim=self.ranges.n_worlds,
                  inputs=[state.body_q, state.body_qd, m.body_mass, m.body_inertia, m.body_com, m.gravity,
                          self.ranges.start, self.ranges.end, self.z_ref, self.log.column(step), int(step > 0)],
                  outputs=[self.log.channel(self.name, wp.float64), self.prev, self.gain], device=m.device)


class MomentumMeter:
    """Records per-world linear momentum `<prefix>p` and angular momentum about the origin `<prefix>L` (vec3d)."""

    def __init__(self, model, log: DeviceTwinLog, prefix: str = "") -> None:
        self.model = model
        self.log = log
        self.prefix = prefix
        self.ranges = WorldRanges.bodies(model)

    def update(self, state, step: int) -> None:
        col = self.log.column(step)
        if col < 0:
            return
        m = self.model
        outs = [self.log.channel(self.prefix + "p", wp.vec3d), self.log.channel(self.prefix + "L", wp.vec3d)]
        wp.launch(_world_momentum, dim=self.ranges.n_worlds,
                  inputs=[state.body_q, state.body_qd, m.body_mass, m.body_inertia, m.body_com,
                          self.ranges.start, self.ranges.end, col], outputs=outs, device=m.device)


__all__ = ["EnergyMeter", "MomentumMeter", "reset_heights"]
