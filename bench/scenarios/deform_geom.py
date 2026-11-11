# SPDX-License-Identifier: Apache-2.0
"""Mesh generation and host-side geometric measures for the `deform` family.

- grid / disc / strip triangle meshes (vertices in a local plane, triangles as an (M, 3) int array)
- drape silhouette area by rasterising projected triangles at a fixed pixel size (Cusick / ISO 9073-9)
- cloth self-intersection count: Warp BVH over triangle AABBs + wp.intersect_tri_tri, same-world pairs only,
  topological neighbours skipped (the filter of newton@ac7f7830 geometry/kernels.py:1864-1870)
- membrane strain from edge lengths
"""
from __future__ import annotations

import numpy as np
import warp as wp


# ---- meshes -------------------------------------------------------------------------------------------------------
def grid_mesh(nx: int, ny: int, hx: float, hy: float) -> tuple[np.ndarray, np.ndarray]:
    """(nx+1)(ny+1) vertices in the local xy plane from the origin, x fastest; two triangles per cell."""
    xs, ys = np.meshgrid(np.arange(nx + 1) * hx, np.arange(ny + 1) * hy)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    tris = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1, v2, v3 = v0 + 1, v0 + nx + 2, v0 + nx + 1
            tris += [(v0, v1, v2), (v0, v2, v3)]
    return verts, np.asarray(tris, dtype=np.int32)


def disc_mesh(radius: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Square-grid disc centred at the origin: grid triangles whose three vertices lie inside the radius.

    The jagged rim is kept (no projection) so every interior edge keeps the grid stencil the bending-stiffness
    mapping assumes. The mesh area is reported separately (dc uses the true mesh area A_D).
    """
    n = int(np.ceil(2 * radius / h))
    verts, tris = grid_mesh(n, n, h, h)
    verts[:, :2] -= 0.5 * n * h
    inside = np.linalg.norm(verts[:, :2], axis=1) <= radius + 1e-9
    keep = inside[tris].all(axis=1)
    tris = tris[keep]
    used = np.unique(tris)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris].astype(np.int32)


def mesh_area(verts: np.ndarray, tris: np.ndarray) -> float:
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def mesh_edges(tris: np.ndarray) -> np.ndarray:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def max_edge_strain(rest: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    l0 = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    l1 = np.linalg.norm(cur[edges[:, 1]] - cur[edges[:, 0]], axis=1)
    return float(np.max(np.abs(l1 / l0 - 1.0)))


# ---- silhouette ---------------------------------------------------------------------------------------------------
def silhouette_area(xy: np.ndarray, tris: np.ndarray, pixel: float = 1e-3, max_extent: float = 2.0) -> float:
    """Area [m^2] covered by the union of projected triangles, rasterised at `pixel` (pixel centres).

    NaN when the footprint is wider than max_extent [m] or not finite (a blown-up cloth would otherwise ask for a
    raster of gigabytes; a D03 XPBD run was SIGKILLed that way on the Mac).
    """
    lo = xy.min(axis=0) - pixel
    hi = xy.max(axis=0) + pixel
    if not (np.all(np.isfinite(xy)) and np.all(hi - lo <= max_extent)):
        return float("nan")
    nx, ny = (np.ceil((hi - lo) / pixel)).astype(int)
    mask = np.zeros((ny, nx), dtype=bool)
    p = (xy - lo) / pixel - 0.5  # pixel-centre coordinates
    for t in tris:
        a, b, c = p[t[0]], p[t[1]], p[t[2]]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, nx - 1), min(y1, ny - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
        l3 = 1.0 - l1 - l2
        eps = -1e-9
        inside = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        mask[gy[inside], gx[inside]] = True
    return float(mask.sum()) * pixel * pixel


# ---- self-intersection --------------------------------------------------------------------------------------------
@wp.func
def _shares_vertex(a: int, b0: int, b1: int, b2: int) -> bool:
    return a == b0 or a == b1 or a == b2


@wp.kernel
def _tri_aabb(pos: wp.array(dtype=wp.vec3), tris: wp.array2d(dtype=wp.int32),
              lowers: wp.array(dtype=wp.vec3), uppers: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    a, b, c = pos[tris[t, 0]], pos[tris[t, 1]], pos[tris[t, 2]]
    lowers[t] = wp.min(wp.min(a, b), c)
    uppers[t] = wp.max(wp.max(a, b), c)


@wp.kernel
def _tri_tri_count(bvh: wp.uint64, pos: wp.array(dtype=wp.vec3), tris: wp.array2d(dtype=wp.int32),
                   tri_world: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    t = wp.tid()
    i0, i1, i2 = tris[t, 0], tris[t, 1], tris[t, 2]
    a, b, c = pos[i0], pos[i1], pos[i2]
    q = wp.bvh_query_aabb(bvh, wp.min(wp.min(a, b), c), wp.max(wp.max(a, b), c))
    s = int(0)
    n = int(0)
    while wp.bvh_query_next(q, s):
        if s <= t or tri_world[s] != tri_world[t]:
            continue
        j0, j1, j2 = tris[s, 0], tris[s, 1], tris[s, 2]
        if _shares_vertex(i0, j0, j1, j2) or _shares_vertex(i1, j0, j1, j2) or _shares_vertex(i2, j0, j1, j2):
            continue
        if wp.intersect_tri_tri(a, b, c, pos[j0], pos[j1], pos[j2]):
            n += 1
    count[t] = n


def self_intersections(model, state, tri_world: np.ndarray | None = None) -> np.ndarray:
    """Intersecting non-adjacent triangle pairs per triangle (each pair counted once) for all model triangles."""
    ntri = model.tri_count
    if ntri == 0:
        return np.zeros(0, dtype=np.int32)
    dev = model.device
    lowers = wp.empty(ntri, dtype=wp.vec3, device=dev)
    uppers = wp.empty(ntri, dtype=wp.vec3, device=dev)
    wp.launch(_tri_aabb, dim=ntri, inputs=[state.particle_q, model.tri_indices, lowers, uppers], device=dev)
    bvh = wp.Bvh(lowers, uppers)
    if tri_world is None:
        tri_world = model.particle_world.numpy()[model.tri_indices.numpy()[:, 0]]
    tw = wp.array(np.asarray(tri_world, np.int32), dtype=wp.int32, device=dev)
    cnt = wp.zeros(ntri, dtype=wp.int32, device=dev)
    wp.launch(_tri_tri_count, dim=ntri, inputs=[bvh.id, state.particle_q, model.tri_indices, tw, cnt], device=dev)
    return cnt.numpy()
