# SPDX-License-Identifier: Apache-2.0
"""Bevel node as a wavefront pass (nodes.bevel_normal split at its ray march).

    bevel_setup  (Warp, pre-pass): the call's sample rays, LCG state and march step -> BevelBuf
    trace_bevel  (OptiX ``__raygen__bevel_any`` / ``__raygen__bevel``): collect the hits in the hit object's own GAS,
                 keep <= 4 per sample with the same reservoir and LCG
    bevel_gather (Warp, in the shade kernel): bevel_normal's weighting of the kept hits -> the bevelled normal

Same random numbers, sample rays, reservoir and weights as ``bevel_normal``. One difference, by design: the megakernel
marches the combined mesh of all objects and skips other objects' triangles; the GAS of the object alone has none.
``warp_bevel_march`` is the Warp stand-in for the OptiX programs (``backend="wavefront-warp"``).
"""
from __future__ import annotations

import warp as wp

from ..pbr.nodes.bevel import _SOBOL_X, _SOBOL_Y, _bevel_cubic_root, _bevel_disk_frame, _bevel_hit_contrib
from ..pbr.nodes.mathnodes import safe_normalize3
from ..pbr.nodes.noise import _U, hash_uint3
from ..tracer.sampling import hash_hp_uint

MAX_SAMPLES = 8  # csrc/owf_params.h OWF_BEVEL_MAX_SAMPLES
MAX_HITS = 4  # OWF_BEVEL_MAX_HITS (Cycles LOCAL_MAX_HITS)


@wp.struct
class BevelBuf:
    ray_o: wp.array(dtype=wp.vec3)  # (paths * 8) sample rays, frame space of the hit object
    ray_d: wp.array(dtype=wp.vec3)
    ray_tmax: wp.array(dtype=float)
    mesh: wp.array(dtype=wp.int32)  # per path: scene mesh (GAS) of the hit object
    n: wp.array(dtype=wp.int32)  # per path: samples (-1: flat region, no rays)
    state: wp.array(dtype=wp.uint32)  # per path: reservoir LCG state
    eps: wp.array(dtype=float)  # per path: march step past a hit
    nhits: wp.array(dtype=wp.int32)  # (paths * 8) hits found per sample
    face: wp.array(dtype=wp.int32)  # (paths * 8 * 4) kept hits: mesh-local face, Warp barycentrics
    u: wp.array(dtype=float)
    v: wp.array(dtype=float)


def alloc_bevel(n_paths: int, device) -> BevelBuf:
    """A BevelBuf for ``n_paths`` paths (zero-filled)."""
    bb = BevelBuf()
    sizes = {"ray_o": MAX_SAMPLES, "ray_d": MAX_SAMPLES, "ray_tmax": MAX_SAMPLES, "nhits": MAX_SAMPLES,
             "face": MAX_SAMPLES * MAX_HITS, "u": MAX_SAMPLES * MAX_HITS, "v": MAX_SAMPLES * MAX_HITS}
    for name, var in BevelBuf.vars.items():
        setattr(bb, name, wp.zeros(n_paths * sizes.get(name, 1), dtype=var.type.dtype, device=device))
    return bb


@wp.struct
class BevelSample:
    start: wp.vec3
    direction: wp.vec3
    tmax: float
    disk_N: wp.vec3
    disk_T: wp.vec3
    disk_B: wp.vec3
    pick_N: float
    pick_T: float
    pick_B: float
    disk_r: float


@wp.func
def bevel_active(radius: float, num_samples: int, min_ray_pdf: float) -> bool:
    return not (radius <= 0.0 or num_samples < 1 or min_ray_pdf < 8.0)


@wp.func
def bevel_seed(rng_pixel: wp.uint32, sample: int, bounce: int) -> wp.uint32:
    """Per-call seed of a Bevel evaluation (hash of pixel, sample and bounce)."""
    return hash_hp_uint(rng_pixel ^ hash_hp_uint(wp.uint32(sample * 64 + bounce) ^ wp.uint32(0x7A3C11E5)))


@wp.func
def bevel_shift(seed: wp.uint32) -> wp.vec2ui:
    """bevel_normal's per-call digital shift of the 2D Sobol points."""
    rng = wp.rand_init(int(seed))
    shx = _U(wp.randf(rng) * 16777215.0) << _U(8)
    shy = _U(wp.randf(rng) * 16777215.0) << _U(8)
    return wp.vec2ui(shx, shy)


@wp.func
def bevel_sample(P: wp.vec3, Ng: wp.vec3, radius: float, s: int, sh: wp.vec2ui) -> BevelSample:
    """Sample s of a bevel call (bevel_normal's loop body up to the march), same arithmetic."""
    sidx = s % 8
    xi = (_SOBOL_Y >> _U(4 * sidx)) & _U(15)
    yi = (_SOBOL_X >> _U(4 * sidx)) & _U(15)
    r1 = float((xi << _U(28)) ^ sh[0]) * (1.0 / 4294967296.0)
    r2 = float((yi << _U(28)) ^ sh[1]) * (1.0 / 4294967296.0)
    disk_N, disk_T, disk_B, pick_N, pick_T, pick_B, r1 = _bevel_disk_frame(Ng, r1)
    phi = 6.28318530717958647692 * r1
    xr = _bevel_cubic_root(r2)
    disk_r = xr * radius
    disk_h = wp.sqrt(wp.max(radius * radius - disk_r * disk_r, 0.0))
    disk_P = disk_T * (disk_r * wp.cos(phi)) + disk_B * (disk_r * wp.sin(phi))
    out = BevelSample()
    out.start = P + disk_N * disk_h + disk_P
    out.direction = -disk_N
    out.tmax = 2.0 * disk_h
    out.disk_N = disk_N
    out.disk_T = disk_T
    out.disk_B = disk_B
    out.pick_N = pick_N
    out.pick_T = pick_T
    out.pick_B = pick_B
    out.disk_r = disk_r
    return out


@wp.func
def bevel_setup(bb: BevelBuf, p: int, mesh: int, P: wp.vec3, Ng: wp.vec3, radius: float, num_samples: int,
                seed: wp.uint32):
    """Rays and reservoir state of path p's bevel call (bevel_normal's arguments; call only if bevel_active)."""
    sh = bevel_shift(seed)
    ns = wp.min(num_samples, MAX_SAMPLES)
    for s in range(ns):
        b = bevel_sample(P, Ng, radius, s, sh)
        bb.ray_o[p * MAX_SAMPLES + s] = b.start
        bb.ray_d[p * MAX_SAMPLES + s] = b.direction
        bb.ray_tmax[p * MAX_SAMPLES + s] = b.tmax
    bb.mesh[p] = mesh
    bb.n[p] = ns
    bb.state[p] = hash_uint3(seed, _U(0x64C6A40E), _U(num_samples))
    bb.eps[p] = radius * 1.0e-3


@wp.func
def bevel_gather(bb: BevelBuf, p: int, mesh: wp.uint64, tri_n: wp.array(dtype=wp.vec3), tri_off: int, P: wp.vec3,
                 Ng: wp.vec3, N: wp.vec3, backfacing: int, radius: float, num_samples: int, min_ray_pdf: float,
                 seed: wp.uint32) -> wp.vec3:
    """bevel_normal's output from the traced hits. ``mesh``: the combined frame-space wp.Mesh (``sd.bevel_mesh``,
    global triangles = ``tri_off`` + mesh-local face)."""
    if not bevel_active(radius, num_samples, min_ray_pdf):
        return N
    sh = bevel_shift(seed)
    sum_n = wp.vec3(0.0, 0.0, 0.0)
    for s in range(wp.min(num_samples, MAX_SAMPLES)):
        b = bevel_sample(P, Ng, radius, s, sh)
        k = p * MAX_SAMPLES + s
        nhits = bb.nhits[k]
        for h in range(wp.min(nhits, MAX_HITS)):
            sum_n = sum_n + _bevel_hit_contrib(mesh, tri_n, tri_off + bb.face[k * MAX_HITS + h],
                                               bb.u[k * MAX_HITS + h], bb.v[k * MAX_HITS + h], P, radius, nhits,
                                               b.disk_N, b.disk_T, b.disk_B, b.pick_N, b.pick_T, b.pick_B, b.disk_r)
    out = safe_normalize3(sum_n)
    if out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0:
        return N
    if backfacing != 0:
        return -out
    return out


@wp.func
def bevel_n(bb: BevelBuf, p: int, bevel_mesh: wp.uint64, tri_n: wp.array(dtype=wp.vec3), tri_off: int, P: wp.vec3,
            Ng: wp.vec3, N: wp.vec3, backfacing: int, radius: float, samples: int, min_ray_pdf: float,
            seed: wp.uint32) -> wp.vec3:
    """The Bevel normal of a shaded vertex: N in a flat region (the pre-pass cast no rays), else the gathered hits."""
    if bb.n[p] < 0:
        return N
    return bevel_gather(bb, p, bevel_mesh, tri_n, tri_off, P, Ng, N, backfacing, radius, samples, min_ray_pdf, seed)


@wp.func
def reservoir_keep(state: wp.uint32, nhits: int) -> int:
    """Slot (0..3) a new hit takes in the reservoir of 4, or -1 to drop it (Cycles' uniform subset)."""
    if nhits < MAX_HITS:
        return nhits
    rr = int(state % _U(nhits + 1))
    if rr < MAX_HITS:
        return rr
    return -1


@wp.kernel(enable_backward=False)
def warp_bevel_march(mesh_ids: wp.array(dtype=wp.uint64), q: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                     bb: BevelBuf):
    """Warp stand-in for the OptiX Bevel programs (closest-hit march with the same reservoir and LCG)."""
    i = wp.tid()
    if i >= count[0]:
        return
    p = q[i]
    mesh = mesh_ids[bb.mesh[p]]
    state = bb.state[p]
    eps = bb.eps[p]
    for s in range(bb.n[p]):
        k = p * MAX_SAMPLES + s
        start = bb.ray_o[k]
        direction = bb.ray_d[k]
        tmax = bb.ray_tmax[k]
        nhits = int(0)
        t0 = float(0.0)
        last_face = int(-1)
        for step in range(24):
            hq = wp.mesh_query_ray(mesh, start + direction * t0, direction, tmax - t0)
            if not hq.result:
                break
            t0 = t0 + hq.t
            if hq.face == last_face:
                t0 = t0 + eps
                continue
            last_face = hq.face
            t0 = t0 + eps
            if nhits >= MAX_HITS:
                state = _U(1103515245) * state + _U(12345)
            slot = reservoir_keep(state, nhits)
            nhits += 1
            if slot >= 0:
                bb.face[k * MAX_HITS + slot] = hq.face
                bb.u[k * MAX_HITS + slot] = hq.u
                bb.v[k * MAX_HITS + slot] = hq.v
        bb.nhits[k] = nhits
