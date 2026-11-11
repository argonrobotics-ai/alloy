# SPDX-License-Identifier: Apache-2.0
"""Random-walk subsurface scattering of Cycles 4.5 (kernel/integrator/subsurface.h, subsurface_random_walk.h, tag
v4.5.0): the entry bounce into the object and the volumetric walk inside it, with Dwivedi guiding, the similarity
relation after bounce 9, spectral (channel) MIS and the opposite-interface backward guiding.

The walk intersects only the object it entered (Cycles' scene_intersect_local); when it reaches the surface again the
path continues from the exit point lit as a white Lambertian surface (``integrator``). Random numbers come from the
dimension block after the entry vertex (Cycles advances rng_offset by PRNG_BOUNCE_NUM for the entry bounce) and from
a scrambled offset for the walk itself (path_state_rng_scramble with 0xdeadbeef).

Example (inside a Warp function)::

    ok, wo = entry_bounce(sss, wi, ng, rand2(rng, sample, rofs + PRNG_BOUNCE_NUM, sobol))
    w = random_walk(geo, inst, p, offset_point(p, -ng, prm), wo, ns, sss, thr, rng, sample, rofs + PRNG_BOUNCE_NUM,
                    sobol)
"""
import warp as wp

from ..tracer.geometry import Geo
from ..tracer.sampling import Rng, hash_hp_uint, rand1, rand2
from .bsdf.bssrdf import Bssrdf, walk_coefficients
from .bsdf.microfacet import ggx_sample_vndf
from .bsdf.util import make_orthonormals_a

BSSRDF_MAX_BOUNCES = 256
SIMILARITY_LEVEL = 9
VOLUME_THROUGHPUT_EPSILON = 1e-6
PRNG_SUBSURFACE_BSDF = 0
PRNG_SUBSURFACE_COLOR_CHANNEL = 1
PRNG_SUBSURFACE_SCATTER_DISTANCE = 2
PRNG_SUBSURFACE_GUIDE_STRATEGY = 3
PRNG_SUBSURFACE_GUIDE_DIRECTION = 4
WALK_SEED = wp.constant(wp.uint32(0x29D5BEEF))  # 0xdeadbeef ^ (0xdeadbeef << 19): hash_hp_seeded_uint's seed


@wp.struct
class WalkResult:
    ok: int          # 1: the walk left the object at the exit point below
    thr: wp.vec3     # path throughput after the walk
    p: wp.vec3       # exit point
    d: wp.vec3       # direction of the walk's last segment (it reached the surface from inside along d)
    t: float         # length of that segment
    ng: wp.vec3      # geometric normal at the exit (world space, not yet facing anything)
    face: wp.int32
    u: float         # barycentrics of the exit, as closest_hit_uv
    v: float


@wp.struct
class LocalHit:
    hit: int
    t: float
    ng: wp.vec3
    face: wp.int32
    u: float
    v: float


@wp.func
def local_hit(geo: Geo, inst: wp.int32, o: wp.vec3, d: wp.vec3, tmax: float) -> LocalHit:
    """Closest hit of the ray with instance ``inst`` only (scene_intersect_local, one hit)."""
    r = LocalHit()
    inv = geo.inst_inv[inst]
    h = wp.mesh_query_ray(geo.inst_mesh[inst], wp.transform_point(inv, o), wp.transform_vector(inv, d), tmax)
    r.hit = wp.where(h.result, 1, 0)
    if h.result:
        r.t = h.t
        r.ng = wp.normalize(wp.transform_vector(geo.inst_xf[inst], h.normal))
        r.face = h.face
        r.u = h.u
        r.v = h.v
    return r


@wp.func
def entry_bounce(b: Bssrdf, wi: wp.vec3, ng: wp.vec3, u: wp.vec2):
    """subsurface_entry_bounce (RANDOM_WALK): refract wi through a GGX microfacet of the BSSRDF's roughness into the
    object. Returns (ok, wo); fails when wi is below the closure normal or wo does not point into the surface."""
    cos_ni = wp.dot(b.n, wi)
    if cos_ni <= 0.0:
        return 0, wp.vec3(0.0)
    x = make_orthonormals_a(b.n)
    y = wp.cross(b.n, x)
    local_i = wp.vec3(wp.dot(x, wi), wp.dot(y, wi), cos_ni)
    lh = ggx_sample_vndf(local_i, b.alpha, b.alpha, u)
    h = lh[0] * x + lh[1] * y + lh[2] * b.n
    neta = 1.0 / b.ior
    cos_hi = wp.dot(h, wi)
    arg = 1.0 - neta * neta * (1.0 - cos_hi * cos_hi)
    dnp = wp.max(wp.sqrt(wp.max(arg, 0.0)), 1e-7)
    wo = -(neta * wi) + (neta * cos_hi - dnp) * h
    if wp.dot(ng, wo) >= 0.0:
        return 0, wo
    return 1, wo


@wp.func
def _hg(cos_theta: float, g: float) -> float:
    """phase_henyey_greenstein."""
    if wp.abs(g) < 1e-3:
        return 0.07957747154594767
    fac = 1.0 + g * (g - 2.0 * cos_theta)
    return (1.0 - g * g) / (12.566370614359172 * fac * wp.sqrt(wp.max(fac, 0.0)))


@wp.func
def _around(d: wp.vec3, cos_theta: float, rv: float) -> wp.vec3:
    """phase_sample_direction / direction_from_cosine: cos_theta around d, azimuth 2 pi rv."""
    phi = 6.283185307179586 * rv
    sin_theta = wp.sqrt(wp.max(1.0 - cos_theta * cos_theta, 0.0))
    t = make_orthonormals_a(d)
    b = wp.cross(d, t)
    return (sin_theta * wp.cos(phi)) * t + (sin_theta * wp.sin(phi)) * b + cos_theta * d


@wp.func
def _hg_sample(d: wp.vec3, g: float, r: wp.vec2):
    """phase_henyey_greenstein_sample: (direction, pdf)."""
    cos_theta = 1.0 - 2.0 * r[0]
    if wp.abs(g) >= 1e-3:
        k = (1.0 - g * g) / (1.0 - g * cos_theta)
        cos_theta = (1.0 + g * g - k * k) / (2.0 * g)
    return _around(d, cos_theta, r[1]), _hg(cos_theta, g)


@wp.func
def _channel(alpha: wp.vec3, thr: wp.vec3, r_in: float):
    """volume_sample_channel: (channel, channel pdfs) proportional to |throughput * alpha|."""
    w = wp.vec3(wp.abs(thr[0] * alpha[0]), wp.abs(thr[1] * alpha[1]), wp.abs(thr[2] * alpha[2]))
    s = w[0] + w[1] + w[2]
    pdf = wp.vec3(1.0 / 3.0)
    if (1.0 - s) < 1.0:
        pdf = w / s
    ch = int(2)
    acc = float(0.0)
    for c in range(3):
        if ch == 2 and c < 2 and r_in < acc + pdf[c]:
            ch = c
        acc = acc + pdf[c]
    return ch, pdf


@wp.func
def _seg_pdf(sigma_t: wp.vec3, t: float, hit: int) -> wp.vec3:
    """subsurface_random_walk_pdf: transmittance (a hit) or sigma_t * transmittance (a scatter)."""
    tr = wp.vec3(wp.exp(-sigma_t[0] * t), wp.exp(-sigma_t[1] * t), wp.exp(-sigma_t[2] * t))
    if hit != 0:
        return tr
    return wp.cw_mul(sigma_t, tr)


@wp.func
def _diffusion_length(alpha: float) -> float:
    return 1.0 / wp.sqrt(1.0 - wp.pow(alpha, 2.44294 - 0.0215813 * alpha + 0.578637 / alpha))


@wp.struct
class WalkDir:
    """One scattering direction of the walk and the terms its guided pdf needs (Cycles keeps them as locals)."""
    d: wp.vec3
    guided: int
    back: int          # guided away from the entry interface (backward guiding)
    back_frac: float   # probability of backward guiding
    fwd_pdf_f: float   # Dwivedi / phase pdf ratios, forward and backward
    back_pdf_f: float
    fwd_str: float     # distance-sampling stretch factors, forward and backward
    back_str: float


@wp.func
def _walk_direction(d: wp.vec3, n: wp.vec3, x: float, opp_dist: float, have_opp: int, g: float, gfrac: float,
                    dl: float, phase_log: float, rng: Rng, sample: wp.int32, ofs: wp.int32,
                    sobol: wp.array(dtype=wp.uint32)) -> WalkDir:
    """Scatter direction after bounce 0: Dwivedi-guided (forward or, with an opposite interface, backward) with
    probability ``gfrac``, else Henyey-Greenstein; ``x`` = the depth below the entry plane."""
    w = WalkDir()
    w.guided = wp.where(rand1(rng, sample, ofs + PRNG_SUBSURFACE_GUIDE_STRATEGY, sobol) < gfrac, 1, 0)
    w.back = 0
    w.back_frac = 0.0
    if have_opp != 0:
        w.back_frac = 1.0 / (1.0 + wp.exp((opp_dist - 2.0 * x) / dl))
        w.back = wp.where(rand1(rng, sample, ofs + PRNG_SUBSURFACE_GUIDE_DIRECTION, sobol) < w.back_frac, 1, 0)
    rs = rand2(rng, sample, ofs + PRNG_SUBSURFACE_BSDF, sobol)
    cos_theta = float(0.0)
    hg_pdf = float(0.0)
    if w.guided != 0:
        cos_theta = dl - (dl + 1.0) * wp.exp(-rs[0] * phase_log)
        if w.back != 0:
            cos_theta = -cos_theta
        w.d = _around(n, cos_theta, rs[1])
        hg_pdf = _hg(wp.dot(d, w.d), g)
    else:
        nd, hp = _hg_sample(d, g, rs)
        w.d = nd
        hg_pdf = hp
        cos_theta = wp.dot(nd, n)
    w.fwd_pdf_f = 0.15915494309189535 / ((dl - cos_theta) * phase_log) / hg_pdf
    w.back_pdf_f = 0.15915494309189535 / ((dl + cos_theta) * phase_log) / hg_pdf
    w.fwd_str = 1.0 - cos_theta / dl
    w.back_str = 1.0 + cos_theta / dl
    return w


@wp.func
def _walk_pdf(sigma_t: wp.vec3, t: float, hit: int, w: WalkDir, have_opp: int, gfrac: float) -> wp.vec3:
    """Per-channel pdf of a segment after bounce 0: classic and guided (forward / backward) sampling, MIS'd."""
    gpdf = _seg_pdf(sigma_t * w.fwd_str, t, hit)
    if have_opp != 0:
        bpdf = _seg_pdf(sigma_t * w.back_str, t, hit)
        gpdf = gpdf * w.fwd_pdf_f * (1.0 - w.back_frac) + bpdf * w.back_pdf_f * w.back_frac
    else:
        gpdf = gpdf * w.fwd_pdf_f
    return _seg_pdf(sigma_t, t, hit) * (1.0 - gfrac) + gpdf * gfrac


@wp.func
def _bounce_coefficients(bounce: int, g0: float, sigma_t0: wp.vec3, alpha: wp.vec3):
    """(g, guided fraction, sigma_t, sigma_s) of a bounce; after SIMILARITY_LEVEL the similarity relation swaps in
    isotropic scattering with the reduced coefficients."""
    sigma_s0 = wp.cw_mul(sigma_t0, alpha)
    if bounce > SIMILARITY_LEVEL:
        sigma_s_star = sigma_s0 * (1.0 - g0)
        return 0.0, 0.75, sigma_t0 - sigma_s0 + sigma_s_star, sigma_s_star
    return g0, 1.0 - wp.max(0.5, wp.pow(wp.abs(g0), 0.125)), sigma_t0, sigma_s0


@wp.func
def _segment_weight(sigma_t: wp.vec3, sigma_s: wp.vec3, t: float, hit: int, pdf: wp.vec3, ch_pdf: wp.vec3):
    """Throughput factor of a segment: transmittance (an exit) or sigma_s x transmittance, over the MIS'd pdf."""
    tr = wp.vec3(wp.exp(-sigma_t[0] * t), wp.exp(-sigma_t[1] * t), wp.exp(-sigma_t[2] * t))
    return wp.where(hit != 0, tr, wp.cw_mul(sigma_s, tr)) / wp.dot(ch_pdf, pdf)


@wp.func
def _walk_exit(lh: LocalHit, o: wp.vec3, d: wp.vec3, t: float, thr: wp.vec3) -> WalkResult:
    res = WalkResult()
    res.ok = 1
    res.thr = thr
    res.p = o
    res.d = d
    res.t = t
    res.ng = lh.ng
    res.face = lh.face
    res.u = lh.u
    res.v = lh.v
    return res


@wp.func
def random_walk(geo: Geo, inst: wp.int32, p0: wp.vec3, cast0: wp.vec3, d0: wp.vec3, n: wp.vec3, b: Bssrdf,
                thr_in: wp.vec3, rng: Rng, sample: wp.int32, rng_offset: wp.int32,
                sobol: wp.array(dtype=wp.uint32)) -> WalkResult:
    """subsurface_random_walk from the entry point p0 (on the surface, as Cycles) along d0; ``n`` = the entry's sd->N.
    Only the first ray-cast leaves from ``cast0`` (p0 pushed inside); a walk started there was 4 % dark on a 0.1 mm
    mean free path (the push was 0.2 of it). ``ok`` = 0 when the walk does not leave the object."""
    sigma_t0, alpha, thr = walk_coefficients(b.albedo, b.radius, b.anisotropy, thr_in)
    dl = _diffusion_length(wp.max(wp.max(alpha[0], alpha[1]), alpha[2]))
    if dl == 1.0:
        return WalkResult()
    phase_log = wp.log((dl + 1.0) / (dl - 1.0))
    ofs = wp.int32(hash_hp_uint(wp.uint32(rng_offset) ^ WALK_SEED) & wp.uint32(0xFFFFFFFC))
    o = p0
    d = d0
    have_opp = int(0)
    opp_dist = float(0.0)
    for bounce in range(BSSRDF_MAX_BOUNCES):
        ofs = ofs + 16
        g, gfrac, sigma_t, sigma_s = _bounce_coefficients(bounce, b.anisotropy, sigma_t0, alpha)
        ch, ch_pdf = _channel(alpha, thr, rand1(rng, sample, ofs + PRNG_SUBSURFACE_COLOR_CHANNEL, sobol))
        sample_sigma_t = sigma_t[ch]
        randt = rand1(rng, sample, ofs + PRNG_SUBSURFACE_SCATTER_DISTANCE, sobol)
        w = WalkDir()
        if bounce > 0:
            x = wp.clamp(wp.dot(o - p0, -n), 0.0, opp_dist)
            w = _walk_direction(d, n, x, opp_dist, have_opp, g, gfrac, dl, phase_log, rng, sample, ofs, sobol)
            d = w.d
            if w.guided != 0:
                sample_sigma_t = sample_sigma_t * wp.where(w.back != 0, w.back_str, w.fwd_str)
        t = -wp.log(1.0 - randt) / sample_sigma_t
        tmax = t
        if bounce == 0:
            tmax = wp.max(t, 10.0 / wp.min(wp.min(sigma_t[0], sigma_t[1]), sigma_t[2]))
        lh = local_hit(geo, inst, wp.where(bounce == 0, cast0, o), d, tmax)
        hit = lh.hit
        if bounce == 0:
            if hit != 0:
                have_opp = 1
                opp_dist = wp.dot(cast0 + lh.t * d - p0, -n)
            hit = wp.where(hit != 0 and lh.t < t, 1, 0)
        if hit != 0:
            t = lh.t
        o = wp.where(hit != 0 and bounce == 0, cast0 + t * d, o + t * d)  # a bounce-0 exit: on the far surface
        pdf = wp.where(bounce > 0, _walk_pdf(sigma_t, t, hit, w, have_opp, gfrac), _seg_pdf(sigma_t, t, hit))
        thr = wp.cw_mul(thr, _segment_weight(sigma_t, sigma_s, t, hit, pdf, ch_pdf))
        if hit != 0:
            return _walk_exit(lh, o, d, t, thr)
        if wp.max(wp.max(thr[0], thr[1]), thr[2]) < VOLUME_THROUGHPUT_EPSILON:
            return WalkResult()
    return WalkResult()
