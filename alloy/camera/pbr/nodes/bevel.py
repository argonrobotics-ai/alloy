# SPDX-License-Identifier: Apache-2.0
"""Bevel node (Cycles 4.5 svm/bevel.h)."""
import numpy as np
import warp as wp

from .noise import _U, hash_uint3
from .mathnodes import safe_normalize3
from .bump import make_orthonormals


# ---------------------------------------------------------------------------------------------------------
# Bevel (svm/bevel.h). World-space mesh; per-corner shading normals tri_n[face * 3 + corner].
# ---------------------------------------------------------------------------------------------------------
# first 8 points of the 2D Sobol sequence in units of 1/16 (top 4 bits), packed as nibbles
_SOBOL_X = wp.constant(wp.uint32(sum(v << (4 * i) for i, v in enumerate([0, 8, 4, 12, 2, 10, 6, 14]))))
_SOBOL_Y = wp.constant(wp.uint32(sum(v << (4 * i) for i, v in enumerate([0, 8, 12, 4, 6, 14, 10, 2]))))


@wp.func
def _bevel_cubic_eval(radius: float, r: float) -> float:
    if r >= radius:
        return 0.0
    Rm5 = (radius * radius) * (radius * radius) * radius
    f = radius - r
    return (10.0 * f * f * f) / (Rm5 * 3.14159265358979323846)


@wp.func
def _bevel_cubic_root(xi: float) -> float:
    x = float(0.25)
    for i in range(10):
        x2 = x * x
        x3 = x2 * x
        nx = 1.0 - x
        f = 10.0 * x2 - 20.0 * x3 + 15.0 * x2 * x2 - 4.0 * x2 * x3 - xi
        fp = 20.0 * (x * nx) * (nx * nx)
        if wp.abs(f) < 1e-6 or fp == 0.0:
            break
        x = wp.clamp(x - f / fp, 0.0, 1.0)
    return x


@wp.func
def _lcg_next(state: wp.uint32) -> wp.uint32:
    return _U(1103515245) * state + _U(12345)


@wp.func
def _bevel_disk_frame(Ng: wp.vec3, r1_in: float):
    """Pick the disk axis (normal 1/2, tangents 1/4 each) from r1; returns (N, T, B axes, pick pdfs, remapped r1)."""
    r1 = r1_in
    tN = Ng
    tT, tB = make_orthonormals(Ng)
    disk_N = tN
    disk_T = tT
    disk_B = tB
    pick_N = float(0.5)
    pick_T = float(0.25)
    pick_B = float(0.25)
    if r1 < 0.5:
        r1 = r1 * 2.0
    elif r1 < 0.75:
        disk_N = tT
        disk_T = tN
        pick_N = 0.25
        pick_T = 0.5
        r1 = (r1 - 0.5) * 4.0
    else:
        disk_N = tB
        disk_B = tN
        pick_N = 0.25
        pick_B = 0.5
        r1 = (r1 - 0.75) * 4.0
    return disk_N, disk_T, disk_B, pick_N, pick_T, pick_B, r1


@wp.func
def _bevel_collect_hits(mesh: wp.uint64, tri_obj: wp.array(dtype=int), obj_id: int, start: wp.vec3,
                        direction: wp.vec3, tmax: float, eps: float, state_in: wp.uint32):
    """March through all hits (closest first), keep a uniform random subset of at most 4 (LOCAL_MAX_HITS).
    Returns (nhits, hit_face, hit_u, hit_v, lcg state)."""
    state = state_in
    nhits = int(0)
    hit_face = wp.vec4i(-1, -1, -1, -1)
    hit_u = wp.vec4(0.0)
    hit_v = wp.vec4(0.0)
    t0 = float(0.0)
    last_face = int(-1)
    for k in range(24):
        q = wp.mesh_query_ray(mesh, start + direction * t0, direction, tmax - t0)
        if not q.result:
            break
        t0 = t0 + q.t
        if q.face == last_face or (obj_id >= 0 and tri_obj[q.face] != obj_id):
            t0 = t0 + eps
            continue
        last_face = q.face
        t0 = t0 + eps
        slot = int(-1)
        if nhits < 4:
            slot = nhits
        else:
            state = _lcg_next(state)
            rr = int(state % _U(nhits + 1))
            if rr < 4:
                slot = rr
        nhits += 1
        if slot >= 0:
            hit_face[slot] = q.face
            hit_u[slot] = q.u
            hit_v[slot] = q.v
    return nhits, hit_face, hit_u, hit_v, state


@wp.func
def _bevel_hit_contrib(mesh: wp.uint64, tri_n: wp.array(dtype=wp.vec3), f: int, u: float, v: float, P: wp.vec3,
                       radius: float, nhits: int, disk_N: wp.vec3, disk_T: wp.vec3, disk_B: wp.vec3,
                       pick_N: float, pick_T: float, pick_B: float, disk_r: float) -> wp.vec3:
    """Weighted shading normal of one kept hit (its term of the sum in svm_bevel)."""
    n0 = tri_n[f * 3]
    n1 = tri_n[f * 3 + 1]
    n2 = tri_n[f * 3 + 2]
    # Warp's (u, v) are the weights of vertices 0 and 1 (position = u*p0 + v*p1 + (1-u-v)*p2)
    Nh = wp.normalize(n0 * u + n1 * v + n2 * (1.0 - u - v))
    hit_P = wp.mesh_eval_position(mesh, f, u, v)
    e1 = wp.mesh_eval_position(mesh, f, 0.0, 1.0) - wp.mesh_eval_position(mesh, f, 1.0, 0.0)
    e2 = wp.mesh_eval_position(mesh, f, 0.0, 0.0) - wp.mesh_eval_position(mesh, f, 1.0, 0.0)
    hit_Ng = wp.normalize(wp.cross(e1, e2))
    pdf_N = pick_N * wp.abs(wp.dot(disk_N, hit_Ng))
    pdf_T = pick_T * wp.abs(wp.dot(disk_T, hit_Ng))
    pdf_B = pick_B * wp.abs(wp.dot(disk_B, hit_Ng))
    w = pdf_N / (pdf_N * pdf_N + pdf_T * pdf_T + pdf_B * pdf_B)
    if nhits > 4:
        w = w * float(nhits) / 4.0
    rdist = wp.length(hit_P - P)
    pdf = _bevel_cubic_eval(radius, rdist)
    disk_pdf = _bevel_cubic_eval(radius, disk_r)
    w = w * pdf / disk_pdf
    return Nh * w


@wp.func
def bevel_normal(mesh: wp.uint64, tri_n: wp.array(dtype=wp.vec3), tri_obj: wp.array(dtype=int),
                 obj_id: int, P: wp.vec3, Ng: wp.vec3, N: wp.vec3, backfacing: int,
                 radius: float, num_samples: int, min_ray_pdf: float, seed: wp.uint32) -> wp.vec3:
    """Bevel node output for a hit (svm_bevel). P, Ng, N world space (Ng/N as in ShaderData: facing the viewer
    side, Ng the geometric normal). `mesh`: wp.Mesh.id of the WORLD-space mesh containing the object;
    hits are restricted to triangles with tri_obj[face] == obj_id (pass obj_id = -1 to accept all, e.g. when the
    mesh is just this object). tri_n: shading normal per triangle corner (T*3,). seed: per-call RNG seed (hash of
    pixel, sample, bounce). min_ray_pdf: path min_ray_pdf (1e30 for camera rays); < 8 returns N (blurry rays)."""
    if radius <= 0.0 or num_samples < 1 or min_ray_pdf < 8.0:
        return N
    state = hash_uint3(seed, _U(0x64C6A40E), _U(num_samples))
    rng = wp.rand_init(int(seed))
    sum_n = wp.vec3(0.0, 0.0, 0.0)
    shx = _U(wp.randf(rng) * 16777215.0) << _U(8)
    shy = _U(wp.randf(rng) * 16777215.0) << _U(8)
    eps = radius * 1.0e-3
    for s in range(num_samples):
        # Cycles draws the samples of one bevel call from one block of a scrambled Sobol sequence (a (0,m,2)-net),
        # so they are jointly stratified in (axis/phi, radius); the node's output is the normalised sum over the
        # block, which is non-linear, so the block structure changes the converged mean. 8 samples: Sobol points
        # 0..7 digitally shifted by a random per-call value (num_samples != 8: first n points, still stratified).
        sidx = s % 8
        xi = (_SOBOL_Y >> _U(4 * sidx)) & _U(15)
        yi = (_SOBOL_X >> _U(4 * sidx)) & _U(15)
        r1 = float((xi << _U(28)) ^ shx) * (1.0 / 4294967296.0)
        r2 = float((yi << _U(28)) ^ shy) * (1.0 / 4294967296.0)
        disk_N, disk_T, disk_B, pick_N, pick_T, pick_B, r1 = _bevel_disk_frame(Ng, r1)
        phi = 6.28318530717958647692 * r1
        xr = _bevel_cubic_root(r2)
        disk_r = xr * radius
        disk_h = wp.sqrt(wp.max(radius * radius - disk_r * disk_r, 0.0))
        disk_P = disk_T * (disk_r * wp.cos(phi)) + disk_B * (disk_r * wp.sin(phi))
        start = P + disk_N * disk_h + disk_P
        direction = -disk_N
        tmax = 2.0 * disk_h
        # march through all hits (closest first), keep a uniform random subset of at most 4 (LOCAL_MAX_HITS)
        nhits, hit_face, hit_u, hit_v, state = _bevel_collect_hits(mesh, tri_obj, obj_id, start, direction, tmax,
                                                                   eps, state)
        nev = wp.min(nhits, 4)
        for h in range(nev):
            sum_n = sum_n + _bevel_hit_contrib(mesh, tri_n, hit_face[h], hit_u[h], hit_v[h], P, radius, nhits,
                                               disk_N, disk_T, disk_B, pick_N, pick_T, pick_B, disk_r)
    out = safe_normalize3(sum_n)
    if out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0:
        return N
    if backfacing != 0:
        return -out
    return out


@wp.func
def bevel_node(bevel_n: wp.vec3, sd_N: wp.vec3, ref_N: wp.vec3, has_ref: int) -> wp.vec3:
    """svm_node_bevel tail: when the node's Normal input is linked, keep it: normalize(ref + (bevel - sd.N))."""
    if has_ref != 0:
        return wp.normalize(ref_N + (bevel_n - sd_N))
    return bevel_n

