# SPDX-License-Identifier: Apache-2.0
"""Soft-soft interpenetration in health (SCR-SOFT-3-1; coordinator ruling 2026-11-06 18:07Z).

Two VBD soft bodies in one world that do not collide pass through each other: SOFT-3's skin pad fell through the
fat pad onto the ground while ``check()`` said ok. Health now flags it as PEN_FAIL (INVALID, S1). Contact itself is
WP2's.

``SoftPen.build(model)`` splits the model's tets into connected components, one per soft body: tets that share a
particle belong to the same body, so no assembler data is needed. WP10's virtual-node copies (``topology:copy_hub``)
join their hub's body: their fragment tets sit on the copies alone, and as a body of their own a split cut piece
read as inside its parent (PEN_FAIL in the first cut). It returns None unless some world holds two or
more bodies; single bodies, cloth and rigid scenes cost nothing. Each check step it runs three launches, all graph
capturable:
- reset the per-body bounds;
- the bounds, with atomic min / max, which are order-independent;
- per soft particle, the other bodies of its world whose bounds hold it, and their tets (CSR by body), with a
  barycentric test. A particle strictly inside another body's live tet (nonzero material: WP10's dormant cut
  fragments are skipped), every barycentric coordinate >= ``INSIDE_BARY``, ORs PEN_FAIL into its world.
In a resting stack the particles of the upper body stay outside the lower body's volume, so the bounds test skips
nearly all of them. A particle on a face (coordinate ~0) is contact, not interpenetration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.precision.health_kernels import Accum

INSIDE_BARY = 0.1


@wp.kernel
def reset_bounds(lo: wp.array2d(dtype=wp.float32), hi: wp.array2d(dtype=wp.float32)):
    c = wp.tid()
    for k in range(3):
        lo[c, k] = 1.0e30
        hi[c, k] = -1.0e30


@wp.kernel
def body_bounds(parts: wp.array(dtype=wp.int32), part_body: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3),
                lo: wp.array2d(dtype=wp.float32), hi: wp.array2d(dtype=wp.float32)):
    i = wp.tid()
    p = parts[i]
    b = part_body[p]
    for k in range(3):
        wp.atomic_min(lo, b, k, x[p][k])
        wp.atomic_max(hi, b, k, x[p][k])


@wp.func
def strictly_inside(xp: wp.vec3, x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, beta: float) -> bool:
    m = wp.mat33(x1[0] - x0[0], x2[0] - x0[0], x3[0] - x0[0],
                 x1[1] - x0[1], x2[1] - x0[1], x3[1] - x0[1],
                 x1[2] - x0[2], x2[2] - x0[2], x3[2] - x0[2])
    if wp.abs(wp.determinant(m)) < 1.0e-18:
        return False
    l = wp.inverse(m) * (xp - x0)
    l0 = 1.0 - l[0] - l[1] - l[2]
    return wp.min(wp.min(l0, l[0]), wp.min(l[1], l[2])) >= beta


@wp.kernel
def inside_other_body(parts: wp.array(dtype=wp.int32), part_body: wp.array(dtype=wp.int32),
                      body_world: wp.array(dtype=wp.int32), world_start: wp.array(dtype=wp.int32),
                      world_bodies: wp.array(dtype=wp.int32), tet_start: wp.array(dtype=wp.int32),
                      body_tets: wp.array(dtype=wp.int32), tets: wp.array2d(dtype=wp.int32),
                      materials: wp.array2d(dtype=wp.float32), x: wp.array(dtype=wp.vec3), lo: wp.array2d(dtype=wp.float32), hi: wp.array2d(dtype=wp.float32),
                      beta: float, bit: wp.uint32, acc: Accum):
    i = wp.tid()
    p = parts[i]
    a = part_body[p]
    w = body_world[a]
    xp = x[p]
    for k in range(world_start[w], world_start[w + 1]):
        b = world_bodies[k]
        if b == a:
            continue
        if xp[0] < lo[b, 0] or xp[1] < lo[b, 1] or xp[2] < lo[b, 2]:
            continue
        if xp[0] > hi[b, 0] or xp[1] > hi[b, 1] or xp[2] > hi[b, 2]:
            continue
        for e in range(tet_start[b], tet_start[b + 1]):
            t = body_tets[e]
            if materials[t, 0] == 0.0 and materials[t, 1] == 0.0:     # dormant (no material): not volume
                continue
            if strictly_inside(xp, x[tets[t, 0]], x[tets[t, 1]], x[tets[t, 2]], x[tets[t, 3]], beta):
                wp.atomic_or(acc.overflow, w, bit)
                return


def _csr(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(start [n + 1], items sorted by key, stable)."""
    order = np.argsort(keys, kind="stable").astype(np.int32)
    return np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=n))]).astype(np.int32), order


class SoftPen:
    """Device tables for the soft-soft check of one model (see the module docstring)."""

    def __init__(self, model: Any, part_body: np.ndarray, body_world: np.ndarray, parts: np.ndarray, worlds: int):
        dev = model.device
        i32 = lambda a: wp.array(np.ascontiguousarray(a, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
        tets = model.tet_indices.numpy()
        bodies = int(body_world.shape[0])
        tet_body = part_body[tets[:, 0]]
        self.world_start, world_bodies = _csr(body_world, worlds)
        self.tet_start, body_tets = _csr(tet_body, bodies)
        self.arrays = [i32(parts), i32(part_body), i32(body_world), i32(self.world_start), i32(world_bodies),
                       i32(self.tet_start), i32(body_tets), model.tet_indices, model.tet_materials]
        self.lo = wp.zeros((bodies, 3), dtype=wp.float32, device=dev)
        self.hi = wp.zeros((bodies, 3), dtype=wp.float32, device=dev)
        self.model, self.bodies, self.count = model, bodies, int(parts.shape[0])

    @staticmethod
    def build(model: Any) -> SoftPen | None:
        """The check for ``model``, or None when no world holds two or more soft tet bodies."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if not int(getattr(model, "tet_count", 0) or 0):
            return None
        tets = model.tet_indices.numpy().astype(np.int64)
        n = int(model.particle_count)
        rows, cols = np.repeat(tets[:, :1], 3, axis=1).ravel(), tets[:, 1:].ravel()
        copy_hub = getattr(getattr(model, "topology", None), "copy_hub", None)   # WP10 virtual-node copies: a
        if copy_hub is not None and copy_hub.shape[0] == n:                     # copy is its hub's body
            hub = copy_hub.numpy().astype(np.int64)
            copies = np.flatnonzero(hub >= 0)
            rows, cols = np.concatenate([rows, copies]), np.concatenate([cols, hub[copies]])
        graph = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        in_tet = np.zeros(n, bool)
        in_tet[tets.ravel()] = True
        uniq, body = np.unique(label[in_tet], return_inverse=True)
        part_body = np.full(n, -1, np.int64)
        part_body[in_tet] = body
        first = np.zeros(len(uniq), np.int64)
        first[body[::-1]] = np.flatnonzero(in_tet)[::-1]         # a particle of each body
        body_world = np.maximum(model.particle_world.numpy()[first], 0)
        worlds = max(int(model.world_count), 1)
        crowded = np.bincount(body_world, minlength=worlds) >= 2
        if not crowded.any():
            return None
        parts = np.flatnonzero(in_tet & crowded[body_world[np.maximum(part_body, 0)]])
        return SoftPen(model, part_body, body_world, parts, worlds)

    def launch(self, x: wp.array, acc: Accum, bit: int, beta: float = INSIDE_BARY) -> None:
        dev = self.model.device
        parts, part_body = self.arrays[0], self.arrays[1]
        wp.launch(reset_bounds, dim=self.bodies, inputs=[self.lo, self.hi], device=dev)
        wp.launch(body_bounds, dim=self.count, inputs=[parts, part_body, x, self.lo, self.hi], device=dev)
        wp.launch(inside_other_body, dim=self.count,
                  inputs=[*self.arrays, x, self.lo, self.hi, float(beta), wp.uint32(bit), acc], device=dev)
