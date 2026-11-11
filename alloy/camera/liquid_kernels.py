# SPDX-License-Identifier: Apache-2.0
"""Device kernels of the batched liquid surface (:mod:`alloy.camera.liquid_gpu`): the particles of every world to one
closed triangle mesh per world, each step one launch over all worlds.

The method follows the liquid surface of GA's wine-pouring render worker (Yu and Turk 2013 smoothing, the Zhu and
Bridson 2005 implicit surface, marching cubes, Taubin passes, area-weighted normals):

1. ``mark_active``: the poured particles (ACTIVE flag) and each world's bounds.
2. ``compact``: the active particles packed into one array, world w shifted into its own tile of a shared space, so one
   hash grid serves every world (tiles lie farther apart than the kernel support).
3. ``smooth``: Yu-Turk smoothing of the centres, ``x' = (1 - lambda) x + lambda * mean(neighbours)`` with weights
   ``1 - (r / R)^3`` within ``R = 2 dx`` (lambda 0.9).
4. ``stretch``: stream kernels. A pour leaves its nozzle as layers of particles that drift apart as they fall (10 to
   16 mm at 4 mm particles), so round kernels draw a stack of plates. A sparse particle (fewer than ``n_eps``
   neighbours within R) that moves becomes a segment along its velocity, 0.6 x the gap to the nearest layer each way
   (at most ``l_max``); the field below then measures from the segment's closest point, so the layers join into one
   stream. (Yu-Turk's covariance kernels follow the plates, their long axes in the layer; a stretched round kernel
   does not close the gap either: Zhu-Bridson's weighted mean jumps from one layer to the next.)
5. ``implicit``: the Zhu-Bridson field ``phi(x) = |x - xbar(x)| - rbar(x)``, ``xbar`` and ``rbar`` the
   ``(1 - q^2)^3``-weighted means of the smoothed centres and radii within the (anisotropic) support, on a grid of
   ``dx / 2`` per world; stored as ``-phi`` (inside >= 0).
6. marching cubes in two scanned passes: one vertex per crossed grid edge (shared by its triangles), Warp's case
   tables (``warp._src.geometry.marching_cubes``, read at import).
7. ``taubin``: Taubin passes (lambda 0.5, mu -0.53) of the uniform Laplacian over each vertex's triangle neighbours,
   found through the four cells around its edge (no adjacency lists, no atomics: the same bits every run).
8. ``vertex_normals``: the area-weighted mean of the face normals around each vertex.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from warp._src.geometry import marching_cubes as _mc

ACTIVE = wp.constant(1)  # newton.ParticleFlags.ACTIVE
FAR = wp.constant(1.0e4)  # metres: particles beyond this are not drawn


@wp.struct
class Grid:
    """World w's grid: nodes ``lo[w] + (ix, iy, iz) * cell`` in world space, ``tile[w] + ...`` in the packed space."""

    nx: int
    ny: int
    nz: int
    nn: int  # nodes per world
    cell: float
    lo: wp.array(dtype=wp.vec3)
    tile: wp.array(dtype=wp.vec3)
    field: wp.array(dtype=float)  # -phi per node, world-major
    vbase: wp.array(dtype=wp.int32)  # exclusive scan of the crossed edges per node
    tbase: wp.array(dtype=wp.int32)  # exclusive scan of the triangles per cell (cell = its lowest node)
    vb0: wp.array(dtype=wp.int32)  # vbase / tbase of world w's first node
    tb0: wp.array(dtype=wp.int32)
    bshift: int  # blocks of 2^bshift nodes a side (liquid_blocks: 1 where a particle's support reaches)
    nbx: int  # blocks per axis
    nby: int
    nbz: int
    blocks: wp.array(dtype=wp.int32)  # [W * nbx * nby * nbz]


@wp.struct
class Tables:
    rng: wp.array(dtype=wp.int32)  # case -> first entry in inds (257)
    inds: wp.array(dtype=wp.int32)  # cell-local edge of each triangle corner
    corner: wp.array(dtype=wp.vec3i)  # cube corner offsets (8)
    eoff: wp.array2d(dtype=wp.int32)  # edge -> (dx, dy, dz, axis) of its node (12, 4)
    elook: wp.array(dtype=wp.int32)  # ((ox * 2 + oy) * 2 + oz) * 3 + axis -> edge (24; -1: none)


def make_tables(device: str) -> Tables:
    """Warp's marching-cubes tables on ``device``."""
    eoff = np.asarray(_mc._MC_EDGE_OFFSETS, np.int32)
    look = np.full(24, -1, np.int32)
    for e, (ox, oy, oz, a) in enumerate(eoff):
        look[((ox * 2 + oy) * 2 + oz) * 3 + a] = e
    tb = Tables()
    tb.rng = wp.array(np.asarray(_mc.MC_CASE_TO_TRI_RANGE, np.int32), dtype=wp.int32, device=device)
    tb.inds = wp.array(np.asarray(_mc.MC_TRI_LOCAL_INDICES, np.int32), dtype=wp.int32, device=device)
    tb.corner = wp.array(np.asarray(_mc.MC_CUBE_CORNER_OFFSETS, np.int32), dtype=wp.vec3i, device=device)
    tb.eoff = wp.array(eoff, dtype=wp.int32, device=device)
    tb.elook = wp.array(look, dtype=wp.int32, device=device)
    return tb


# ---------------------------------------------------------------- particles
# The hash grids are float64 (``to_f64`` copies of the float32 points): Warp's HashGrid.build copies its device
# descriptor from one static host variable per precision, and a CUDA graph that captured a build (the SPH solver's
# float32 neighbour grid) replays that copy; a float32 build here, between two graph replays, handed the solver this
# grid's descriptor (out-of-bounds reads, then CUDA error 700). Float64 grids use their own variable.
@wp.kernel(enable_backward=False)
def to_f64(src: wp.array(dtype=wp.vec3), dst: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    dst[i] = wp.vec3d(src[i])


@wp.kernel(enable_backward=False)
def mark_active(pq: wp.array(dtype=wp.vec3), flags: wp.array(dtype=wp.int32), start: int, stride: int, n: int,
                act: wp.array(dtype=wp.int32), lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3)):
    w, i = wp.tid()
    p = start + w * stride + i
    x = pq[p]
    a = int(0)
    if (flags[p] & ACTIVE) != 0 and wp.abs(x[0]) < FAR and wp.abs(x[1]) < FAR and wp.abs(x[2]) < FAR:
        a = 1
        wp.atomic_min(lo, w, x)  # min / max: exact in any order
        wp.atomic_max(hi, w, x)
    act[w * n + i] = a


@wp.kernel(enable_backward=False)
def compact(pq: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), act: wp.array(dtype=wp.int32),
            slot: wp.array(dtype=wp.int32), start: int, stride: int, n: int, g: Grid, out: wp.array(dtype=wp.vec3),
            out_v: wp.array(dtype=wp.vec3), out_w: wp.array(dtype=wp.int32)):
    w, i = wp.tid()
    if act[w * n + i] != 0:
        p = start + w * stride + i
        out[slot[w * n + i]] = pq[p] - g.lo[w] + g.tile[w]
        out_v[slot[w * n + i]] = qd[p]
        out_w[slot[w * n + i]] = w


@wp.kernel(enable_backward=False)
def smooth(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), radius: float, lam: float, out: wp.array(dtype=wp.vec3),
           count: wp.array(dtype=wp.int32)):
    s = wp.tid()
    x = pts[s]
    q = wp.hash_grid_query(grid, wp.vec3d(x), wp.float64(radius))
    j = int(0)
    n = int(0)
    sw = float(0.0)
    sx = wp.vec3(0.0)
    while wp.hash_grid_query_next(q, j):
        d = wp.length(pts[j] - x) / radius
        if d < 1.0:
            k = 1.0 - d * d * d
            sw += k
            sx += pts[j] * k
            n += 1
    out[s] = x * (1.0 - lam) + sx * (lam / sw)  # sw >= 1: the particle itself
    count[s] = n


@wp.func
def _layer_gap(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), s: int, a: wp.vec3, reach: float) -> float:
    """Distance along ``a`` to the nearest particle within a 60-degree cone around +-a (``reach`` when none)."""
    x = pts[s]
    q = wp.hash_grid_query(grid, wp.vec3d(x), wp.float64(reach))
    j = int(0)
    g = reach
    while wp.hash_grid_query_next(q, j):
        if j != s:
            d = pts[j] - x
            along = wp.abs(wp.dot(d, a))
            if along > 0.5 * wp.length(d) and along < g:
                g = along
    return g


@wp.kernel(enable_backward=False)
def stretch(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
            count: wp.array(dtype=wp.int32), n_eps: int, l_max: float, v_min: float, axis: wp.array(dtype=wp.vec3),
            half: wp.array(dtype=float), flag: wp.array(dtype=wp.int32)):
    """Velocity axis and half length of each stream segment; flag 1 = a segment."""
    s = wp.tid()
    flag[s] = 0
    sp = wp.length(vel[s])
    if count[s] >= n_eps or sp < v_min:
        return
    a = vel[s] / sp
    g = _layer_gap(grid, pts, s, a, l_max / 0.6)
    if g * 0.6 >= l_max:
        return
    axis[s] = a
    half[s] = 0.6 * g
    flag[s] = 1


@wp.kernel(enable_backward=False)
def gather_aniso(flag: wp.array(dtype=wp.int32), slot: wp.array(dtype=wp.int32), pts: wp.array(dtype=wp.vec3),
                 axis: wp.array(dtype=wp.vec3), half: wp.array(dtype=float), out_p: wp.array(dtype=wp.vec3),
                 out_a: wp.array(dtype=wp.vec3), out_h: wp.array(dtype=float)):
    s = wp.tid()
    if flag[s] != 0:
        a = slot[s]
        out_p[a] = pts[s]
        out_a[a] = axis[s]
        out_h[a] = half[s]


@wp.func
def unravel(g: Grid, i: int) -> wp.vec4i:
    """(w, ix, iy, iz) of global node i."""
    w = i // g.nn
    r = i - w * g.nn
    return wp.vec4i(w, r // (g.ny * g.nz), (r // g.nz) % g.ny, r % g.nz)


@wp.struct
class Aniso:
    """The stream segments (``gather_aniso``: centre, unit axis, half length) and their own hash grid."""

    grid: wp.uint64
    n: int
    pts: wp.array(dtype=wp.vec3)
    axis: wp.array(dtype=wp.vec3)
    half: wp.array(dtype=float)
    reach: float  # query radius: support + the longest half length


@wp.kernel(enable_backward=False)
def implicit(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), flag: wp.array(dtype=wp.int32), an: Aniso, g: Grid,
             radius: float, r0: float, field: wp.array(dtype=float)):
    i = wp.tid()
    c = unravel(g, i)
    b = ((c[0] * g.nbx + (c[1] >> g.bshift)) * g.nby + (c[2] >> g.bshift)) * g.nbz + (c[3] >> g.bshift)
    if g.blocks[b] == 0:  # no particle's support reaches this block: what the sums below give there (sw = 0)
        field[i] = -radius
        return
    x = g.tile[c[0]] + wp.vec3(float(c[1]), float(c[2]), float(c[3])) * g.cell
    q = wp.hash_grid_query(grid, wp.vec3d(x), wp.float64(radius))
    j = int(0)
    sw = float(0.0)
    sr = float(0.0)
    sx = wp.vec3(0.0)
    r2 = radius * radius
    while wp.hash_grid_query_next(q, j):
        d2 = wp.length_sq(pts[j] - x) / r2
        if d2 < 1.0:
            if flag[j] == 0:  # segments are summed below
                s = 1.0 - d2
                k = s * s * s
                sw += k
                sr += k * r0
                sx += pts[j] * k
    if an.n > 0:
        qa = wp.hash_grid_query(an.grid, wp.vec3d(x), wp.float64(an.reach))
        while wp.hash_grid_query_next(qa, j):
            a = an.axis[j]
            h = an.half[j]
            p = an.pts[j] + a * wp.clamp(wp.dot(x - an.pts[j], a), -h, h)  # closest point of the segment
            d2 = wp.length_sq(p - x) / r2
            if d2 < 1.0:
                s = 1.0 - d2
                k = s * s * s
                sw += k
                sr += k * r0
                sx += p * k
    phi = radius
    if sw > 0.0:
        phi = wp.length(x - sx / sw) - sr / sw
    field[i] = -phi


# ---------------------------------------------------------------- marching cubes
@wp.func
def node(g: Grid, w: int, ix: int, iy: int, iz: int) -> int:
    return w * g.nn + (ix * g.ny + iy) * g.nz + iz


@wp.func
def inside(g: Grid, w: int, ix: int, iy: int, iz: int) -> bool:
    return g.field[node(g, w, ix, iy, iz)] >= 0.0


@wp.func
def crosses(g: Grid, w: int, ix: int, iy: int, iz: int, a: int) -> bool:
    """True when the +a edge of node (ix, iy, iz) crosses the surface."""
    jx = ix + wp.where(a == 0, 1, 0)
    jy = iy + wp.where(a == 1, 1, 0)
    jz = iz + wp.where(a == 2, 1, 0)
    if jx >= g.nx or jy >= g.ny or jz >= g.nz:
        return False
    return inside(g, w, ix, iy, iz) != inside(g, w, jx, jy, jz)


@wp.func
def edge_vertex(g: Grid, w: int, ix: int, iy: int, iz: int, a: int) -> int:
    """World w's local id of the vertex on the +a edge of node (ix, iy, iz) (the edge crosses)."""
    r = int(0)
    for b in range(a):
        if crosses(g, w, ix, iy, iz, b):
            r += 1
    return g.vbase[node(g, w, ix, iy, iz)] + r - g.vb0[w]


@wp.func
def cell_vertex(g: Grid, tb: Tables, w: int, cx: int, cy: int, cz: int, e: int) -> int:
    return edge_vertex(g, w, cx + tb.eoff[e, 0], cy + tb.eoff[e, 1], cz + tb.eoff[e, 2], tb.eoff[e, 3])


@wp.func
def cell_case(g: Grid, tb: Tables, w: int, cx: int, cy: int, cz: int) -> int:
    k = int(0)
    for j in range(8):
        o = tb.corner[j]
        if inside(g, w, cx + o[0], cy + o[1], cz + o[2]):
            k += 1 << j
    return k


@wp.kernel(enable_backward=False)
def vert_count(g: Grid, out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    c = unravel(g, i)
    k = int(0)
    for a in range(3):
        if crosses(g, c[0], c[1], c[2], c[3], a):
            k += 1
    out[i] = k


@wp.kernel(enable_backward=False)
def tri_count(g: Grid, tb: Tables, out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    c = unravel(g, i)
    k = int(0)
    if c[1] < g.nx - 1 and c[2] < g.ny - 1 and c[3] < g.nz - 1:
        case = cell_case(g, tb, c[0], c[1], c[2], c[3])
        k = (tb.rng[case + 1] - tb.rng[case]) // 3
    out[i] = k


@wp.kernel(enable_backward=False)
def world_counts(g: Grid, nv: wp.array(dtype=wp.int32), nt: wp.array(dtype=wp.int32)):
    w = wp.tid()
    a = w * g.nn
    g.vb0[w] = g.vbase[a]
    g.tb0[w] = g.tbase[a]
    nv[w] = g.vbase[a + g.nn] - g.vbase[a]
    nt[w] = g.tbase[a + g.nn] - g.tbase[a]


@wp.kernel(enable_backward=False)
def emit_verts(g: Grid, vcap: int, pts: wp.array2d(dtype=wp.vec3), vkey: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    c = unravel(g, i)
    w = c[0]
    f0 = g.field[i]
    p0 = g.lo[w] + wp.vec3(float(c[1]), float(c[2]), float(c[3])) * g.cell
    r = g.vbase[i] - g.vb0[w]
    for a in range(3):
        if crosses(g, w, c[1], c[2], c[3], a):
            e = wp.vec3(wp.where(a == 0, 1.0, 0.0), wp.where(a == 1, 1.0, 0.0), wp.where(a == 2, 1.0, 0.0))
            f1 = g.field[node(g, w, c[1] + int(e[0]), c[2] + int(e[1]), c[3] + int(e[2]))]
            if r < vcap:
                t = wp.clamp(f0 / (f0 - f1), 0.0, 1.0)
                pts[w, r] = p0 + e * (t * g.cell)
                vkey[w, r] = (i - w * g.nn) * 3 + a
            r += 1


@wp.kernel(enable_backward=False)
def emit_tris(g: Grid, tb: Tables, vcap: int, tcap: int, tris: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    c = unravel(g, i)
    w = c[0]
    if c[1] >= g.nx - 1 or c[2] >= g.ny - 1 or c[3] >= g.nz - 1:
        return
    case = cell_case(g, tb, w, c[1], c[2], c[3])
    s = g.tbase[i] - g.tb0[w]
    for t in range(tb.rng[case], tb.rng[case + 1], 3):
        if s < tcap:
            v0 = cell_vertex(g, tb, w, c[1], c[2], c[3], tb.inds[t])
            v1 = cell_vertex(g, tb, w, c[1], c[2], c[3], tb.inds[t + 1])
            v2 = cell_vertex(g, tb, w, c[1], c[2], c[3], tb.inds[t + 2])
            if v0 >= vcap or v1 >= vcap or v2 >= vcap:
                v1 = 0  # a vertex past the cap (overflow, warned): a degenerate triangle
                v2 = 0
                v0 = 0
            tris[w, 3 * s] = v0  # Warp's tables wind clockwise seen from the inside (>= 0) corners: outward
            tris[w, 3 * s + 1] = v2
            tris[w, 3 * s + 2] = v1
        s += 1


@wp.kernel(enable_backward=False)
def finish_empty(nt: wp.array(dtype=wp.int32), g: Grid, pts: wp.array2d(dtype=wp.vec3),
                 tris: wp.array2d(dtype=wp.int32)):
    """A world without triangles keeps one degenerate triangle at its grid corner (meshes need one)."""
    w = wp.tid()
    if nt[w] == 0:
        pts[w, 0] = g.lo[w]
        tris[w, 0] = 0
        tris[w, 1] = 0
        tris[w, 2] = 0


# ---------------------------------------------------------------- smoothing and normals
@wp.func
def ring(g: Grid, tb: Tables, pts: wp.array2d(dtype=wp.vec3), vcap: int, w: int, key: int):
    """(sum of the corner positions, triangle count, area-weighted normal) of the triangles around vertex ``key``."""
    a = key % 3
    r = key // 3
    ix = r // (g.ny * g.nz)
    iy = (r // g.nz) % g.ny
    iz = r % g.nz
    b = (a + 1) % 3
    c = (a + 2) % 3
    s = wp.vec3(0.0)
    n = wp.vec3(0.0)
    k = float(0.0)
    for q in range(4):
        ox = wp.where(b == 0, q & 1, wp.where(c == 0, q >> 1, 0))
        oy = wp.where(b == 1, q & 1, wp.where(c == 1, q >> 1, 0))
        oz = wp.where(b == 2, q & 1, wp.where(c == 2, q >> 1, 0))
        cx = ix - ox
        cy = iy - oy
        cz = iz - oz
        if cx >= 0 and cy >= 0 and cz >= 0 and cx < g.nx - 1 and cy < g.ny - 1 and cz < g.nz - 1:
            e = tb.elook[((ox * 2 + oy) * 2 + oz) * 3 + a]
            case = cell_case(g, tb, w, cx, cy, cz)
            for t in range(tb.rng[case], tb.rng[case + 1], 3):
                if tb.inds[t] == e or tb.inds[t + 1] == e or tb.inds[t + 2] == e:
                    v0 = cell_vertex(g, tb, w, cx, cy, cz, tb.inds[t])
                    v1 = cell_vertex(g, tb, w, cx, cy, cz, tb.inds[t + 2])  # outward winding (emit_tris)
                    v2 = cell_vertex(g, tb, w, cx, cy, cz, tb.inds[t + 1])
                    if v0 < vcap and v1 < vcap and v2 < vcap:
                        p0 = pts[w, v0]
                        p1 = pts[w, v1]
                        p2 = pts[w, v2]
                        s += p0 + p1 + p2
                        n += wp.cross(p1 - p0, p2 - p0)
                        k += 1.0
    return s, k, n


@wp.kernel(enable_backward=False)
def taubin(g: Grid, tb: Tables, vkey: wp.array2d(dtype=wp.int32), nv: wp.array(dtype=wp.int32), vcap: int,
           weight: float, src: wp.array2d(dtype=wp.vec3), dst: wp.array2d(dtype=wp.vec3)):
    """One Laplacian step ``x + weight * (mean of the ring's other corners - x)``."""
    w, v = wp.tid()
    if v >= wp.min(nv[w], vcap):
        return
    x = src[w, v]
    s, k, n = ring(g, tb, src, vcap, w, vkey[w, v])
    if k > 0.0:
        mean = (s - x * k) / (2.0 * k)  # each triangle adds its two other corners
        x = x + (mean - x) * weight
    dst[w, v] = x


@wp.kernel(enable_backward=False)
def vertex_normals(g: Grid, tb: Tables, vkey: wp.array2d(dtype=wp.int32), nv: wp.array(dtype=wp.int32), vcap: int,
                   pts: wp.array2d(dtype=wp.vec3), out: wp.array2d(dtype=wp.vec3)):
    w, v = wp.tid()
    if v >= wp.min(nv[w], vcap):
        return
    s, k, n = ring(g, tb, pts, vcap, w, vkey[w, v])
    if wp.length(n) > 1.0e-30:
        n = wp.normalize(n)
    out[w, v] = n


@wp.kernel(enable_backward=False)
def corner_normals(vn: wp.array2d(dtype=wp.vec3), pts: wp.array2d(dtype=wp.vec3), tris: wp.array2d(dtype=wp.int32),
                   nt: wp.array(dtype=wp.int32), tcap: int, tri_off: wp.array(dtype=wp.int32), first: int,
                   out: wp.array(dtype=wp.vec3)):
    """Corner shading normals (ShadeData.tri_n) of world w's liquid mesh: its vertex normals (face normal where the
    vertex normal vanishes)."""
    w, t, c = wp.tid()
    if t >= wp.min(nt[w], tcap):
        return
    n = vn[w, tris[w, 3 * t + c]]
    if wp.length(n) <= 1.0e-30:
        a = pts[w, tris[w, 3 * t]]
        n = wp.cross(pts[w, tris[w, 3 * t + 1]] - a, pts[w, tris[w, 3 * t + 2]] - a)
    if wp.length(n) <= 1.0e-30:
        n = wp.vec3(0.0, 0.0, 1.0)
    out[(tri_off[first + w] + t) * 3 + c] = wp.normalize(n)


@wp.kernel(enable_backward=False)
def mesh_bounds(g: Grid, nt: wp.array(dtype=wp.int32), pts: wp.array2d(dtype=wp.vec3), first: int,
                lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3)):
    """Each world mesh's box: its grid (a point for a world without triangles)."""
    w = wp.tid()
    if nt[w] == 0:
        lo[first + w] = pts[w, 0]
        hi[first + w] = pts[w, 0]
    else:
        lo[first + w] = g.lo[w]
        hi[first + w] = g.lo[w] + wp.vec3(float(g.nx - 1), float(g.ny - 1), float(g.nz - 1)) * g.cell
