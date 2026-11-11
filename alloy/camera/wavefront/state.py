# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the wavefront: per-path state, work-item decoding, queues and the small kernels around the OptiX
casts (shadow add, per-slot accumulation, resolve).

One wave = up to ``max_paths`` (sample, pixel) work items. Work item ``g`` is sample ``g // npix`` of image-major pixel
``g % npix``; the image's world is ``img_world[img]`` (the renderer's ``cams.world``).
"""
from __future__ import annotations

import warp as wp

from ..tracer.integrator import Params
from ..tracer.sampling import Rng, hash_hp_uint, pixel_rng

F_DEN = wp.constant(1)  # denoising guides not finished yet
F_MIS_SKIP = wp.constant(2)  # previous bounce singular / camera ray: emission without MIS
F_CAMERA = wp.constant(4)  # current ray is the camera ray
F_PATH_DIFFUSE = wp.constant(8)  # last bounce was diffuse (Principled PATH_DIFFUSE)
F_TERM_NEXT = wp.constant(16)  # emission only at the next vertex
F_MIS_HT = wp.constant(32)  # the previous vertex transmits (light-tree MIS pdf of the next hit is two-sided)
F_SSS_EXIT = wp.constant(64)  # the next vertex is a random walk's exit (hit stored in x_*, not traced)
F_ANCESTOR = wp.constant(128)  # some bounce was diffuse (PATH_RAY_DIFFUSE_ANCESTOR)


@wp.struct
class PathState:
    o: wp.array(dtype=wp.vec3)  # current ray (read by OptiX)
    d: wp.array(dtype=wp.vec3)
    tmax: wp.array(dtype=float)
    world: wp.array(dtype=wp.int32)
    hit_t: wp.array(dtype=float)  # closest hit of the current ray (written by OptiX)
    hit_u: wp.array(dtype=float)
    hit_v: wp.array(dtype=float)
    hit_prim: wp.array(dtype=wp.int32)
    hit_inst: wp.array(dtype=wp.int32)
    thr: wp.array(dtype=wp.vec3)
    rad: wp.array(dtype=wp.vec3)
    alb: wp.array(dtype=wp.vec3)
    nrm: wp.array(dtype=wp.vec3)
    den_thr: wp.array(dtype=wp.vec3)
    mis_n: wp.array(dtype=wp.vec3)
    mis_pdf: wp.array(dtype=float)
    min_ray_pdf: wp.array(dtype=float)
    rdP: wp.array(dtype=float)
    rdD: wp.array(dtype=float)
    bounce: wp.array(dtype=wp.int32)
    dbounce: wp.array(dtype=wp.int32)
    gbounce: wp.array(dtype=wp.int32)
    tbounce: wp.array(dtype=wp.int32)
    vst_inst: wp.array(dtype=wp.vec4i)  # absorbing volumes the ray is in (pbr.path.Medium), -1 = empty
    vst_mid: wp.array(dtype=wp.vec4i)
    rng_extra: wp.array(dtype=wp.int32)  # extra PRNG dimension blocks used by subsurface walks
    x_t: wp.array(dtype=float)  # a random walk's exit hit (restored after the closest-hit cast)
    x_u: wp.array(dtype=float)
    x_v: wp.array(dtype=float)
    x_prim: wp.array(dtype=wp.int32)
    x_inst: wp.array(dtype=wp.int32)
    flags: wp.array(dtype=wp.int32)
    sh_o: wp.array(dtype=wp.vec3)  # NEE shadow ray (read by OptiX) and its contribution if unoccluded
    sh_d: wp.array(dtype=wp.vec3)
    sh_tmax: wp.array(dtype=float)
    sh_L: wp.array(dtype=wp.vec3)
    occ: wp.array(dtype=wp.int32)


@wp.struct
class Wave:
    g0: wp.int32  # first work item of the wave: item g = sample_local * npix + pixel
    npix: wp.int32  # images * W * H
    n_slots: wp.int32  # accumulator slots (samples per wave when a wave holds whole images)
    img_world: wp.array(dtype=wp.int32)  # image -> world


@wp.struct
class Item:
    sample: wp.int32
    pix: wp.int32  # image-major pixel index (img * W * H + row * W + col)
    img: wp.int32
    world: wp.int32
    px: wp.int32
    pyr: wp.int32  # raster y from the bottom row


def alloc_path_state(n: int, device) -> PathState:
    """A PathState with every array zero-filled to ``n`` paths."""
    ps = PathState()
    for name, var in PathState.vars.items():
        setattr(ps, name, wp.zeros(n, dtype=var.type.dtype, device=device))
    return ps


@wp.kernel(enable_backward=False)
def pixel_rng_kernel(prm: Params, out: wp.array(dtype=wp.uint32)):
    """Per-pixel sampler state, as the megakernels compute it."""
    tid = wp.tid()
    npix = prm.width * prm.height
    img = tid // npix
    pix = tid - img * npix
    row = pix // prm.width
    px = pix - row * prm.width
    pyr = prm.height - 1 - row
    kseed = prm.seed_hash
    if img != 0:
        kseed = kseed ^ hash_hp_uint(wp.uint32(img))
    out[tid] = pixel_rng(prm.sampler, px, pyr, kseed)


@wp.func
def item_of(wv: Wave, prm: Params, p: wp.int32) -> Item:
    g = wv.g0 + p
    s = g // wv.npix
    it = Item()
    it.sample = prm.sample_offset + s
    it.pix = g - s * wv.npix
    npix = prm.width * prm.height
    it.img = it.pix // npix
    q = it.pix - it.img * npix
    row = q // prm.width
    it.px = q - row * prm.width
    it.pyr = prm.height - 1 - row
    it.world = wv.img_world[it.img]
    return it


@wp.func
def item_rng(prm: Params, pix_rng: wp.array(dtype=wp.uint32), pix: wp.int32) -> Rng:
    rng = Rng()
    rng.kind = prm.sampler
    rng.pixel = pix_rng[pix]
    rng.seq_len = wp.uint32(prm.total_spp)
    rng.mask = prm.sobol_mask
    return rng


@wp.func
def path_of(q: wp.array(dtype=wp.int32), identity: wp.int32, i: wp.int32) -> wp.int32:
    if identity != 0:
        return i
    return q[i]


@wp.func
def enqueue(q: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), k: wp.int32, p: wp.int32):
    q[wp.atomic_add(counts, k, 1)] = p


@wp.kernel(enable_backward=False)
def iota(q: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q[i] = i


@wp.kernel(enable_backward=False)
def shadow_add(ps: PathState, q: wp.array(dtype=wp.int32)):
    p = q[wp.tid()]
    if ps.occ[p] == 0:
        ps.rad[p] = ps.rad[p] + ps.sh_L[p]


@wp.kernel(enable_backward=False)
def accumulate(ps: PathState, wv: Wave, acc_rgb: wp.array(dtype=wp.vec3), acc_alb: wp.array(dtype=wp.vec3),
               acc_nrm: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    g = wv.g0 + p
    s = g // wv.npix
    k = (s % wv.n_slots) * wv.npix + (g - s * wv.npix)
    acc_rgb[k] = acc_rgb[k] + ps.rad[p]
    acc_alb[k] = acc_alb[k] + ps.alb[p]
    acc_nrm[k] = acc_nrm[k] + ps.nrm[p]


@wp.kernel(enable_backward=False)
def resolve(acc_rgb: wp.array(dtype=wp.vec3), acc_alb: wp.array(dtype=wp.vec3), acc_nrm: wp.array(dtype=wp.vec3),
            n_slots: wp.int32, npix: wp.int32, inv_spp: float, out_rgb: wp.array(dtype=wp.vec3),
            out_alb: wp.array(dtype=wp.vec3), out_nrm: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    a = wp.vec3(0.0)
    b = wp.vec3(0.0)
    c = wp.vec3(0.0)
    for k in range(n_slots):
        a += acc_rgb[k * npix + i]
        b += acc_alb[k * npix + i]
        c += acc_nrm[k * npix + i]
    out_rgb[i] = a * inv_spp
    out_alb[i] = b * inv_spp
    out_nrm[i] = c * inv_spp
