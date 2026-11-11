# SPDX-License-Identifier: Apache-2.0
"""Device kernels of the VBD coarse translation step (SCR-WP11-2; maths in solver.py).

Per particle iteration: zero `blocked` -> mark components that exchange momentum with anything else this iteration
(body/shape contact records, enabled body attachments, self-contact pairs between two components) -> chunk sums of
m (x - y) and m, with pinned/inactive particles flagged -> one translation per component -> apply it to the positions
and VBD's accumulated displacements. Every array is sized at construction; float64 sums in a fixed order, and the
marks are idempotent writes of 1, so the result does not depend on thread scheduling.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton import ParticleFlags
from alloy._vendor.newton._src.geometry.tri_mesh_collision import (
    TriMeshCollisionInfo,
    get_edge_collision_buffer_edge_index,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
    get_vertex_collision_buffer_vertex_index,
)

ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.func
def _mark(comp_of: wp.array[wp.int32], particle: int, blocked: wp.array[wp.int32]):
    if particle >= 0:
        c = comp_of[particle]
        if c >= 0:
            blocked[c] = 1


@wp.kernel(enable_backward=False)
def mark_soft_contacts(count: wp.array[wp.int32], indices: wp.array[wp.vec3i], comp_of: wp.array[wp.int32],
                       blocked: wp.array[wp.int32]):
    """Every particle of a body/shape contact record (particle, edge or face record) blocks its component."""
    k = wp.tid()
    if k >= count[0]:
        return
    ids = indices[k]
    for a in range(3):
        _mark(comp_of, ids[a], blocked)


@wp.kernel(enable_backward=False)
def mark_attachments(particle: wp.array[wp.int32], enabled: wp.array[wp.bool], comp_of: wp.array[wp.int32],
                     blocked: wp.array[wp.int32]):
    """An enabled particle-body attachment exchanges momentum with its body."""
    k = wp.tid()
    if enabled[k]:
        _mark(comp_of, particle[k], blocked)


@wp.kernel(enable_backward=False)
def mark_vertex_triangle_pairs(info: TriMeshCollisionInfo, tri_indices: wp.array2d[wp.int32],
                               comp_of: wp.array[wp.int32], blocked: wp.array[wp.int32]):
    """A self-contact pair between two components blocks both (within one component it is internal)."""
    row = wp.tid()
    for j in range(get_vertex_colliding_triangles_count(info, row)):
        v = get_vertex_collision_buffer_vertex_index(info, row, j)
        t = get_vertex_colliding_triangles(info, row, j)
        if v < 0 or t < 0:
            continue
        u = tri_indices[t, 0]
        if comp_of[v] != comp_of[u]:
            _mark(comp_of, v, blocked)
            _mark(comp_of, u, blocked)


@wp.kernel(enable_backward=False)
def mark_edge_edge_pairs(info: TriMeshCollisionInfo, edge_indices: wp.array2d[wp.int32],
                         comp_of: wp.array[wp.int32], blocked: wp.array[wp.int32]):
    """Edge-edge self-contact between two components blocks both (column 2 is always a real vertex)."""
    row = wp.tid()
    for j in range(get_edge_colliding_edges_count(info, row)):
        e1 = get_edge_collision_buffer_edge_index(info, row, j)
        e2 = get_edge_colliding_edges(info, row, j)
        if e1 < 0 or e2 < 0:
            continue
        a = edge_indices[e1, 2]
        b = edge_indices[e2, 2]
        if comp_of[a] != comp_of[b]:
            _mark(comp_of, a, blocked)
            _mark(comp_of, b, blocked)


@wp.kernel(enable_backward=False)
def chunk_sums(chunk_start: wp.array[wp.int32], members: wp.array[wp.int32], mass: wp.array[float],
               flags: wp.array[wp.int32], base: wp.array[wp.vec3], displacement: wp.array[wp.vec3],
               pos_prev: wp.array[wp.vec3], inertia_disp: wp.array[wp.vec3], g_out: wp.array[wp.vec3d],
               m_out: wp.array[wp.float64], pinned_out: wp.array[wp.int32]):
    """Per chunk, in member order: sum m (x - y) with x = base + displacement (the iterate VBD's truncation is about
    to write), sum m, and whether a particle is pinned (m = 0) or inactive. x - y is taken in displacements from the
    step's start, small numbers (SCR-WP11-F32). Unlike the solve, x is base + displacement, not particle_q: this
    runs between the last colour's solve and the truncation that writes it."""
    k = wp.tid()
    g = wp.vec3d(0.0, 0.0, 0.0)
    msum = wp.float64(0.0)
    pinned = int(0)
    for j in range(chunk_start[k], chunk_start[k + 1]):
        i = members[j]
        m = mass[i]
        if m == 0.0 or (flags[i] & ACTIVE) == 0:
            pinned = 1
        g = g + wp.float64(m) * wp.vec3d(((base[i] - pos_prev[i]) + displacement[i]) - inertia_disp[i])
        msum = msum + wp.float64(m)
    g_out[k] = g
    m_out[k] = msum
    pinned_out[k] = pinned


@wp.kernel(enable_backward=False)
def component_delta(comp_chunk_start: wp.array[wp.int32], g_chunk: wp.array[wp.vec3d],
                    m_chunk: wp.array[wp.float64], pinned_chunk: wp.array[wp.int32], blocked: wp.array[wp.int32],
                    delta: wp.array[wp.vec3]):
    """delta_c = -sum m (x - y) / sum m over component c (0 when blocked, pinned or massless)."""
    c = wp.tid()
    g = wp.vec3d(0.0, 0.0, 0.0)
    msum = wp.float64(0.0)
    stop = int(blocked[c])
    for k in range(comp_chunk_start[c], comp_chunk_start[c + 1]):
        g = g + g_chunk[k]
        msum = msum + m_chunk[k]
        stop = stop + pinned_chunk[k]
    if stop != 0 or msum <= wp.float64(0.0):
        delta[c] = wp.vec3(0.0)
        return
    delta[c] = wp.vec3(-g / msum)


@wp.kernel(enable_backward=False)
def apply_delta(comp_of: wp.array[wp.int32], delta: wp.array[wp.vec3], displacement: wp.array[wp.vec3]):
    """Shift every particle of a component by its translation in VBD's accumulated displacement (the positions are
    written from it by the truncation that follows)."""
    i = wp.tid()
    c = comp_of[i]
    if c >= 0:
        displacement[i] = displacement[i] + delta[c]


__all__ = ["apply_delta", "chunk_sums", "component_delta", "mark_attachments", "mark_edge_edge_pairs",
           "mark_soft_contacts", "mark_vertex_triangle_pairs"]

