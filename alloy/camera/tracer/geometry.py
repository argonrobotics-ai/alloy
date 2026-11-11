# SPDX-License-Identifier: Apache-2.0
"""Device-side two-level scene: a grouped top-level wp.Bvh over instance bounds (one group per world, plus a global
group of shared static instances at index num_worlds) and one wp.Mesh per unique mesh, queried in its local frame.

Same pattern as Newton's ray-traced sensor: for each of the world root and the global root, one
wp.bvh_query_ray loop; each candidate maps the ray to local space and calls wp.mesh_query_ray. No nested
bvh_query_ray (Warp's BVH queries share stack state).
"""
from __future__ import annotations

import warp as wp

from .sampling import hash_hp_uint

T_INF = wp.constant(1.0e30)
# Distance ties between a transmissive and an opaque surface: hits closer than this window (the larger of an absolute
# and a relative part, as offset_point) are compared by their exact plane distances in float64; on an exact tie (a
# glass whose base lies in a table's plane) the opaque surface wins.
TIE_ABS = wp.constant(1.0 / 65536.0)
TIE_REL = wp.constant(256.0 / 8388608.0)
TIE_EXACT = wp.constant(wp.float64(1.0e-9))  # metres: plane distances closer than this are a tie
# On an exact tie Cycles z-fights. In C2 (a glass base in a table's plane, A/B against the glass lifted and sunk
# 20 um) the glass is hit first by 35 % of the rays that see the base from the camera and by 19 % of those lighting the
# table in front of it; a per-ray coin of 0.25 keeps both within about 2 % (always the table: 0.7 % and 6.6 % off).
TIE_P_TRANSMISSIVE = wp.constant(0.25)


@wp.struct
class Geo:
    bvh: wp.uint64
    group_roots: wp.array(dtype=wp.int32)  # (num_worlds + 1,), last entry = global group
    inst_mesh: wp.array(dtype=wp.uint64)
    inst_xf: wp.array(dtype=wp.transform)  # world from local (rigid)
    inst_inv: wp.array(dtype=wp.transform)  # local from world
    inst_mat: wp.array(dtype=wp.int32)
    inst_tie: wp.array(dtype=wp.int32)  # 1: transmissive, 2: opaque touching a transmissive one, 0: other
    num_worlds: wp.int32
    use_global: wp.int32


@wp.kernel
def instance_bounds(mesh_lo: wp.array(dtype=wp.vec3), mesh_hi: wp.array(dtype=wp.vec3),
                    inst_src: wp.array(dtype=wp.int32), xf: wp.array(dtype=wp.transform),
                    inv: wp.array(dtype=wp.transform), lowers: wp.array(dtype=wp.vec3),
                    uppers: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    m = inst_src[i]
    lo = mesh_lo[m]
    hi = mesh_hi[m]
    t = xf[i]
    inv[i] = wp.transform_inverse(t)
    blo = wp.vec3(1.0e30)
    bhi = wp.vec3(-1.0e30)
    for c in range(8):
        p = wp.vec3(wp.where((c & 1) != 0, hi[0], lo[0]), wp.where((c & 2) != 0, hi[1], lo[1]),
                    wp.where((c & 4) != 0, hi[2], lo[2]))
        q = wp.transform_point(t, p)
        blo = wp.min(blo, q)
        bhi = wp.max(bhi, q)
    pad = wp.vec3(1.0e-5) * (1.0 + wp.max(wp.max(wp.abs(bhi[0]), wp.abs(bhi[1])), wp.abs(bhi[2])))
    lowers[i] = blo - pad
    uppers[i] = bhi + pad


@wp.kernel
def group_roots_kernel(bvh: wp.uint64, out: wp.array(dtype=wp.int32)):
    g = wp.tid()
    out[g] = wp.bvh_get_group_root(bvh, g)


@wp.func
def _root(geo: Geo, world: wp.int32, g: wp.int32) -> wp.int32:
    if g == 0:
        return geo.group_roots[world]
    if geo.use_global == 0:
        return -1
    return geo.group_roots[geo.num_worlds]


@wp.func
def closest_hit(geo: Geo, world: wp.int32, o: wp.vec3, d: wp.vec3,
                tmax: float) -> tuple[wp.int32, float, wp.vec3, wp.int32]:
    """(instance or -1, t, world-space unit face normal by winding, face index)."""
    best_t = tmax
    best_i = int(-1)
    best_f = int(-1)
    best_n = wp.vec3(0.0)
    for g in range(2):
        root = _root(geo, world, g)
        if root >= 0:
            q = wp.bvh_query_ray(geo.bvh, o, d, root)
            i = int(0)
            while wp.bvh_query_next(q, i, best_t):
                inv = geo.inst_inv[i]
                h = wp.mesh_query_ray(geo.inst_mesh[i], wp.transform_point(inv, o), wp.transform_vector(inv, d), best_t)
                if h.result:
                    if h.t < best_t:
                        best_t = h.t
                        best_i = i
                        best_f = h.face
                        best_n = h.normal
    if best_i >= 0:
        best_n = wp.normalize(wp.transform_vector(geo.inst_xf[best_i], best_n))
    return best_i, best_t, best_n, best_f


@wp.func
def occluded(geo: Geo, world: wp.int32, o: wp.vec3, d: wp.vec3, tmax: float) -> bool:
    hit = bool(False)
    for g in range(2):
        root = _root(geo, world, g)
        if root >= 0 and not hit:
            q = wp.bvh_query_ray(geo.bvh, o, d, root)
            i = int(0)
            while wp.bvh_query_next(q, i, tmax):
                inv = geo.inst_inv[i]
                if wp.mesh_query_ray_anyhit(geo.inst_mesh[i], wp.transform_point(inv, o),
                                            wp.transform_vector(inv, d), tmax):
                    hit = True
                    break
    return hit


@wp.func
def tie_slack(p: wp.vec3) -> float:
    """The near-tie window at ``p`` (see TIE_ABS)."""
    return wp.max(TIE_ABS, TIE_REL * wp.max(wp.max(wp.abs(p[0]), wp.abs(p[1])), wp.abs(p[2])))


@wp.func
def _vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def plane_distance(geo: Geo, i: wp.int32, face: wp.int32, o: wp.vec3, d: wp.vec3) -> wp.float64:
    """Distance along the world ray (o, d) to the plane of triangle ``face`` of instance ``i``, in float64 from the
    mesh's vertices and the ray in the instance's frame."""
    inv = geo.inst_inv[i]
    lo = _vec3d(wp.transform_point(inv, o))
    ld = _vec3d(wp.transform_vector(inv, d))
    m = wp.mesh_get(geo.inst_mesh[i])
    a = _vec3d(m.points[m.indices[3 * face]])
    n = wp.cross(_vec3d(m.points[m.indices[3 * face + 1]]) - a, _vec3d(m.points[m.indices[3 * face + 2]]) - a)
    return wp.dot(n, a - lo) / wp.dot(n, ld)


@wp.func
def _ray_bits(x: float) -> wp.uint32:
    return wp.uint32(wp.int32(wp.clamp(x, -1024.0, 1024.0) * 1048576.0))


@wp.func
def tie_coin(o: wp.vec3, d: wp.vec3) -> float:
    """A uniform number in [0, 1) fixed by the ray (the same integer hash in the OptiX programs)."""
    h = wp.uint32(0)
    for k in range(3):
        h = hash_hp_uint(h ^ _ray_bits(o[k]))
        h = hash_hp_uint(h ^ _ray_bits(d[k]))
    return float(h >> wp.uint32(8)) * (1.0 / 16777216.0)


@wp.func
def wins_tie(geo: Geo, i: wp.int32, face: wp.int32, best_i: wp.int32, best_f: wp.int32, o: wp.vec3,
             d: wp.vec3) -> bool:
    """A near tie between a transmissive and an opaque hit: the truly closer plane wins (a glass floating a
    micrometre above a table is seen as floating, as in Cycles). On an exact tie (a glass base in the table's plane)
    the transmissive surface wins for TIE_P_TRANSMISSIVE of the rays (``tie_coin``), as Cycles' z-fighting."""
    dt = plane_distance(geo, i, face, o, d) - plane_distance(geo, best_i, best_f, o, d)
    if dt < -TIE_EXACT:
        return True
    if dt > TIE_EXACT:
        return False
    glass_wins = tie_coin(o, d) < TIE_P_TRANSMISSIVE
    return (geo.inst_tie[i] == 1) == glass_wins


@wp.func
def closest_hit_uv(geo: Geo, world: wp.int32, o: wp.vec3, d: wp.vec3,
                   tmax: float) -> tuple[wp.int32, float, wp.vec3, wp.int32, float, float]:
    """closest_hit plus the barycentrics (u, v) of wp.mesh_query_ray (p = u p0 + v p1 + (1 - u - v) p2).
    A transmissive and an opaque hit within ``tie_slack`` of each other are ordered by ``wins_tie``."""
    best_t = tmax
    best_key = tmax  # hits up to here are candidates: best_t plus the near-tie window
    best_i = int(-1)
    best_f = int(-1)
    best_u = float(0.0)
    best_v = float(0.0)
    best_n = wp.vec3(0.0)
    for g in range(2):
        root = _root(geo, world, g)
        if root >= 0:
            q = wp.bvh_query_ray(geo.bvh, o, d, root)
            i = int(0)
            while wp.bvh_query_next(q, i, best_key):
                inv = geo.inst_inv[i]
                h = wp.mesh_query_ray(geo.inst_mesh[i], wp.transform_point(inv, o), wp.transform_vector(inv, d),
                                      best_key)
                if h.result:
                    take = h.t < best_t
                    if best_i >= 0 and (geo.inst_tie[i] == 1) != (geo.inst_tie[best_i] == 1):
                        if wp.abs(h.t - best_t) < tie_slack(o + d * h.t):
                            take = wins_tie(geo, i, h.face, best_i, best_f, o, d)
                    if take:
                        best_key = wp.min(tmax, h.t + tie_slack(o + d * h.t))
                        best_t = h.t
                        best_i = i
                        best_f = h.face
                        best_n = h.normal
                        best_u = h.u
                        best_v = h.v
    if best_i >= 0:
        best_n = wp.normalize(wp.transform_vector(geo.inst_xf[best_i], best_n))
    return best_i, best_t, best_n, best_f, best_u, best_v
