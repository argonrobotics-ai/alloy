# SPDX-License-Identifier: Apache-2.0
"""Path-tracing megakernel for diffuse scenes: one thread per pixel of every (world, camera) image, looping over
samples and bounces with Blender Cycles 4.5 conventions (Blackman-Harris filter, Sobol-Burley sampler, next-event
estimation with power-heuristic MIS, Russian roulette from bounce 2, indirect clamp, diffuse bounce limit).

Materials are Lambert (Cycles' Diffuse BSDF, roughness 0) with one linear RGB colour each, or an image texture
looked up by the mesh's UVs at the hit (``textures.py``); lights are one sun with
an angular size plus a constant world colour (``sky.py``). Ported from Cycles 4.5 kernel/integrator (Apache-2.0).
"""
from __future__ import annotations

import warp as wp

from .geometry import T_INF, Geo, closest_hit, occluded
from .sampling import (PI, PRNG_BOUNCE_NUM, PRNG_FILTER, PRNG_LIGHT, PRNG_SURFACE_BSDF, PRNG_TERMINATE, Rng,
                       hash_hp_uint, lookup_table, pixel_rng, power_heuristic, rand1, rand2, rand3,
                       sample_cos_hemisphere)
from .sky import LIGHT_NONE, LIGHT_SUN, Lights, bg_pdf, light_sample, select_light, selection_probs, sun_hit


@wp.struct
class Mats:
    kind: wp.array(dtype=wp.int32)  # 0 = Lambert (Cycles Diffuse BSDF, roughness 0)
    color: wp.array(dtype=wp.vec3)
    tex: wp.array(dtype=wp.uint32)  # image textures (layout: tracer/textures.py); tex_w 0 = no texture
    uv: wp.array(dtype=wp.vec2)
    tex_w: wp.array(dtype=wp.int32)
    tex_h: wp.array(dtype=wp.int32)
    tex_off: wp.array(dtype=wp.int32)
    uv_off: wp.array(dtype=wp.int32)


@wp.struct
class Cams:
    world: wp.array(dtype=wp.int32)  # per image: the world it looks into
    eye: wp.array(dtype=wp.vec3)  # per image
    rot: wp.array(dtype=wp.mat33)  # columns = camera x (right), y (up), z (back) in world; looks along -z
    extent: wp.array(dtype=wp.vec2)  # half extents of the image plane at camera-space z = 1
    clip: wp.array(dtype=wp.vec2)  # (clip_start, clip_end)


@wp.struct
class Params:
    width: wp.int32
    height: wp.int32
    spp: wp.int32
    sample_offset: wp.int32
    total_spp: wp.int32
    max_bounce: wp.int32  # Cycles kernel values (settings + 1)
    max_diffuse: wp.int32
    min_bounce: wp.int32
    clamp_direct: float  # on the RGB sum
    clamp_indirect: float
    sampler: wp.int32
    seed_hash: wp.uint32
    sobol_mask: wp.uint32
    off_abs: float
    off_rel: float
    normal_camera: wp.int32
    first_centered: wp.int32


# ---------------------------------------------------------------- material hooks (Lambert)
@wp.func
def texel(mats: Mats, off: wp.int32, w: wp.int32, h: wp.int32, x: wp.int32, y: wp.int32) -> wp.vec3:
    """sRGB bytes (0..255, as floats) of texel (x, y) of the texture at ``off``, wrapped (repeat)."""
    xi = x - (x // w) * w
    yi = y - (y // h) * h
    p = mats.tex[off + yi * w + xi]
    return wp.vec3(float(p & wp.uint32(255)), float((p >> wp.uint32(8)) & wp.uint32(255)),
                   float((p >> wp.uint32(16)) & wp.uint32(255)))


@wp.func
def srgb_decode(c: float) -> float:
    """Linear value of an sRGB-encoded value in 0..1 (IEC 61966-2-1)."""
    if c <= 0.04045:
        return c / 12.92
    return wp.pow((c + 0.055) / 1.055, 2.4)


@wp.func
def texture_color(geo: Geo, mats: Mats, inst: wp.int32, m: wp.int32, face: wp.int32, p: wp.vec3) -> wp.vec3:
    """Linear texture colour at world point ``p`` on triangle ``face`` of instance ``inst``: the triangle's
    per-vertex UVs blended by the point's barycentric weights, a bilinear blend of the four sRGB texels, decoded."""
    mesh = wp.mesh_get(geo.inst_mesh[inst])
    i0 = mesh.indices[3 * face]
    i1 = mesh.indices[3 * face + 1]
    i2 = mesh.indices[3 * face + 2]
    p0 = mesh.points[i0]
    e1 = mesh.points[i1] - p0
    e2 = mesh.points[i2] - p0
    q = wp.transform_point(geo.inst_inv[inst], p) - p0
    d11 = wp.dot(e1, e1)
    d12 = wp.dot(e1, e2)
    d22 = wp.dot(e2, e2)
    den = d11 * d22 - d12 * d12
    b1 = float(0.0)
    b2 = float(0.0)
    if den > 0.0:
        q1 = wp.dot(q, e1)
        q2 = wp.dot(q, e2)
        b1 = wp.clamp((d22 * q1 - d12 * q2) / den, 0.0, 1.0)
        b2 = wp.clamp((d11 * q2 - d12 * q1) / den, 0.0, 1.0)
    uo = mats.uv_off[m]
    t = mats.uv[uo + i0] * (1.0 - b1 - b2) + mats.uv[uo + i1] * b1 + mats.uv[uo + i2] * b2
    w = mats.tex_w[m]
    h = mats.tex_h[m]
    x = (t[0] - wp.floor(t[0])) * float(w) - 0.5
    y = (t[1] - wp.floor(t[1])) * float(h) - 0.5
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    fx = x - float(x0)
    fy = y - float(y0)
    off = mats.tex_off[m]
    top = texel(mats, off, w, h, x0, y0) * (1.0 - fx) + texel(mats, off, w, h, x0 + 1, y0) * fx
    bot = texel(mats, off, w, h, x0, y0 + 1) * (1.0 - fx) + texel(mats, off, w, h, x0 + 1, y0 + 1) * fx
    c = (top * (1.0 - fy) + bot * fy) * (1.0 / 255.0)
    return wp.vec3(srgb_decode(c[0]), srgb_decode(c[1]), srgb_decode(c[2]))


@wp.func
def surface_color(geo: Geo, mats: Mats, inst: wp.int32, m: wp.int32, face: wp.int32, p: wp.vec3) -> wp.vec3:
    """The Lambert colour at a hit: the instance's flat colour, times its image texture when it has one."""
    if mats.tex_w[m] == 0:
        return mats.color[m]
    return wp.cw_mul(mats.color[m], texture_color(geo, mats, inst, m, face, p))


@wp.func
def bsdf_eval(color: wp.vec3, n: wp.vec3, ng: wp.vec3, wo: wp.vec3) -> tuple[wp.vec3, float]:
    """(eval = f * cos, pdf) toward wo (closure/bsdf_diffuse.h)."""
    c = wp.max(wp.dot(n, wo), 0.0) / PI
    return color * c, c


@wp.func
def bsdf_sample(color: wp.vec3, n: wp.vec3, ng: wp.vec3, r: wp.vec3) -> tuple[wp.vec3, wp.vec3, float]:
    """(wo, weight = eval / pdf, pdf); pdf 0 = rejected (sample below the geometric normal)."""
    wo, pdf = sample_cos_hemisphere(n, wp.vec2(r[0], r[1]))
    if wp.dot(ng, wo) <= 0.0:
        return wo, wp.vec3(0.0), 0.0
    return wo, color, pdf


# ---------------------------------------------------------------- helpers
@wp.func
def clamp_light(c: wp.vec3, bounce: wp.int32, prm: Params) -> wp.vec3:
    """film_clamp_light: scale so the RGB sum is at most the direct (bounce <= 0) or indirect limit."""
    lim = prm.clamp_direct
    if bounce > 0:
        lim = prm.clamp_indirect
    s = c[0] + c[1] + c[2]
    if s > lim:
        return c * (lim / s)
    return c


@wp.func
def offset_point(p: wp.vec3, ng: wp.vec3, prm: Params) -> wp.vec3:
    m = wp.max(wp.max(wp.abs(p[0]), wp.abs(p[1])), wp.abs(p[2]))
    return p + ng * wp.max(prm.off_abs, prm.off_rel * m)


@wp.func
def camera_ray(cams: Cams, img: wp.int32, rx: float, ry: float, prm: Params) -> tuple[wp.vec3, wp.vec3, float]:
    """camera_sample_perspective: raster (y up from the bottom row) -> world ray (origin at the near clip)."""
    ext = cams.extent[img]
    ndc_x = 2.0 * rx / float(prm.width) - 1.0
    ndc_y = 2.0 * ry / float(prm.height) - 1.0
    dc = wp.normalize(wp.vec3(ndc_x * ext[0], ndc_y * ext[1], 1.0))
    r = cams.rot[img]
    d = wp.normalize(wp.vec3(r[0, 0], r[1, 0], r[2, 0]) * dc[0] + wp.vec3(r[0, 1], r[1, 1], r[2, 1]) * dc[1]
                     - wp.vec3(r[0, 2], r[1, 2], r[2, 2]) * dc[2])
    clip = cams.clip[img]
    zinv = 1.0 / dc[2]
    return cams.eye[img] + d * (clip[0] * zinv), d, (clip[1] - clip[0]) * zinv


@wp.func
def to_camera(cams: Cams, img: wp.int32, n: wp.vec3) -> wp.vec3:
    """World -> Cycles camera space (x right, y up, +z forward): (x.n, y.n, -z.n)."""
    r = cams.rot[img]
    return wp.vec3(r[0, 0] * n[0] + r[1, 0] * n[1] + r[2, 0] * n[2], r[0, 1] * n[0] + r[1, 1] * n[1] + r[2, 1] * n[2],
                   -(r[0, 2] * n[0] + r[1, 2] * n[1] + r[2, 2] * n[2]))


@wp.func
def max3(v: wp.vec3) -> float:
    return wp.max(wp.max(v[0], v[1]), v[2])


@wp.func
def miss_radiance(lt: Lights, prm: Params, thr: wp.vec3, d: wp.vec3, bounce: wp.int32, mis_skip: wp.int32,
                  mis_pdf: float, mis_n: wp.vec3) -> wp.vec3:
    """Light picked up by a ray that leaves the scene: the sun disc (inside its cone) and the world colour,
    MIS-weighted against light sampling at the previous vertex (shade_background.h)."""
    out = wp.vec3(0.0)
    if sun_hit(lt, d) and (mis_skip == 0 or lt.sun_visible_camera != 0):
        w = float(1.0)
        if mis_skip == 0:
            w = power_heuristic(mis_pdf, lt.sun_pdf * selection_probs(lt, mis_n)[0])
        out += clamp_light(wp.cw_mul(thr, lt.sun_radiance) * w, bounce - 1, prm)
    if lt.has_bg != 0:
        w = float(1.0)
        if mis_skip == 0 and lt.bg_is_light != 0:
            w = power_heuristic(mis_pdf, selection_probs(lt, mis_n)[1] * bg_pdf(lt, d))
        out += clamp_light(wp.cw_mul(thr, lt.bg_color) * w, bounce - 1, prm)
    return out


@wp.func
def russian_roulette(thr: wp.vec3, bounce: wp.int32, prm: Params, rng: Rng, sample: wp.int32, rofs: wp.int32,
                     sobol: wp.array(dtype=wp.uint32)) -> tuple[bool, wp.vec3]:
    """(survives, new throughput): p = min(sqrt(max throughput), 1) after the minimum bounce (path_state.h)."""
    if bounce <= prm.min_bounce:
        return True, thr
    prob = wp.min(wp.sqrt(max3(thr)), 1.0)
    if prob >= 1.0:
        return True, thr
    if prob == 0.0 or rand1(rng, sample, rofs + PRNG_TERMINATE, sobol) >= prob:
        return False, thr
    return True, thr / prob


@wp.func
def direct_light(geo: Geo, color: wp.vec3, lt: Lights, prm: Params, sobol: wp.array(dtype=wp.uint32), rng: Rng,
                 sample: wp.int32, rofs: wp.int32, world: wp.int32, p: wp.vec3, n: wp.vec3,
                 ng: wp.vec3, thr: wp.vec3, bounce: wp.int32) -> wp.vec3:
    """Next-event estimation: one light per vertex, chosen like Cycles' light tree, with a shadow ray."""
    rl = rand3(rng, sample, rofs + PRNG_LIGHT, sobol)
    lk = select_light(lt, n, rl[2])
    if lk == LIGHT_NONE:
        return wp.vec3(0.0)
    sel = selection_probs(lt, n)
    ldir, lpdf, le, use_mis = light_sample(lt, lk, wp.vec2(rl[0], rl[1]))
    lpdf = lpdf * wp.where(lk == LIGHT_SUN, sel[0], sel[1])
    ev, bpdf = bsdf_eval(color, n, ng, ldir)
    if bpdf <= 0.0 or lpdf <= 0.0:
        return wp.vec3(0.0)
    w = float(1.0)
    if use_mis != 0:
        w = power_heuristic(lpdf, bpdf)
    c = wp.cw_mul(thr, wp.cw_mul(ev, le)) * (w / lpdf)
    if max3(c) <= 0.0 or occluded(geo, world, offset_point(p, ng, prm), ldir, T_INF):
        return wp.vec3(0.0)
    return clamp_light(c, bounce, prm)


@wp.func
def first_ray(cams: Cams, prm: Params, ftab: wp.array(dtype=float), sobol: wp.array(dtype=wp.uint32), rng: Rng,
              img: wp.int32, px: wp.int32, pyr: wp.int32, sample: wp.int32) -> tuple[wp.vec3, wp.vec3, float]:
    """Camera ray through the filter-importance-sampled position in pixel (px, pyr); sample 0 is centred."""
    fu = wp.vec2(0.5, 0.5)
    if sample != 0 or prm.first_centered == 0:
        fu = rand2(rng, sample, PRNG_FILTER, sobol)
    return camera_ray(cams, img, float(px) + lookup_table(ftab, fu[0]), float(pyr) + lookup_table(ftab, fu[1]), prm)


# ---------------------------------------------------------------- one camera path
@wp.func
def trace_path(geo: Geo, mats: Mats, lt: Lights, cams: Cams, prm: Params, ftab: wp.array(dtype=float),
               sobol: wp.array(dtype=wp.uint32), rng: Rng, img: wp.int32, world: wp.int32, px: wp.int32,
               pyr: wp.int32, sample: wp.int32) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """(radiance, denoising albedo, denoising normal) of one sample of one pixel."""
    o, d, tmax = first_ray(cams, prm, ftab, sobol, rng, img, px, pyr, sample)
    thr = wp.vec3(1.0)
    radiance = wp.vec3(0.0)
    alb = wp.vec3(0.0)
    nrm = wp.vec3(0.0)
    mis_n = wp.vec3(0.0)
    mis_pdf = float(0.0)
    bounce = int(0)  # every bounce is diffuse: the diffuse and total bounce counts are equal
    last = wp.min(prm.max_bounce, prm.max_diffuse)
    for _it in range(64):
        rofs = PRNG_BOUNCE_NUM * (bounce + 1)
        inst, t, ng, face = closest_hit(geo, world, o, d, tmax)
        if inst < 0:
            c = miss_radiance(lt, prm, thr, d, bounce, wp.where(bounce == 0, 1, 0), mis_pdf, mis_n)
            radiance += c
            if bounce == 0:
                alb += c
            break
        if bounce >= last:
            break  # no emitters on surfaces: nothing to add at the last vertex
        p = o + d * t
        if wp.dot(ng, d) > 0.0:
            ng = -ng  # backfacing: flip to the ray's side; flat shading (n = ng)
        m = geo.inst_mat[inst]
        alive, thr = russian_roulette(thr, bounce, prm, rng, sample, rofs, sobol)
        if not alive:
            break
        color = surface_color(geo, mats, inst, m, face, p)
        if bounce == 0:
            alb += color
            nrm += wp.where(prm.normal_camera != 0, to_camera(cams, img, ng), ng)
        radiance += direct_light(geo, color, lt, prm, sobol, rng, sample, rofs, world, p, ng, ng, thr, bounce)
        wo, wgt, bpdf = bsdf_sample(color, ng, ng, rand3(rng, sample, rofs + PRNG_SURFACE_BSDF, sobol))
        if bpdf <= 0.0:
            break
        thr = wp.cw_mul(thr, wgt)
        mis_pdf = bpdf
        mis_n = ng
        bounce += 1
        o = offset_point(p, ng, prm)
        d = wo
        tmax = T_INF
    return radiance, alb, nrm


@wp.kernel
def render_kernel(geo: Geo, mats: Mats, lt: Lights, cams: Cams, prm: Params, ftab: wp.array(dtype=float),
                  sobol: wp.array(dtype=wp.uint32), out_rgb: wp.array(dtype=wp.vec3),
                  out_alb: wp.array(dtype=wp.vec3), out_nrm: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    npix = prm.width * prm.height
    img = tid // npix
    pix = tid - img * npix
    row = pix // prm.width
    px = pix - row * prm.width
    pyr = prm.height - 1 - row  # Cycles raster y counts from the bottom row
    kseed = prm.seed_hash
    if img != 0:
        kseed = kseed ^ hash_hp_uint(wp.uint32(img))
    rng = Rng()
    rng.kind = prm.sampler
    rng.pixel = pixel_rng(prm.sampler, px, pyr, kseed)
    rng.seq_len = wp.uint32(prm.total_spp)
    rng.mask = prm.sobol_mask
    acc = wp.vec3(0.0)
    acc_a = wp.vec3(0.0)
    acc_n = wp.vec3(0.0)
    for s in range(prm.spp):
        c, a, n = trace_path(geo, mats, lt, cams, prm, ftab, sobol, rng, img, cams.world[img], px, pyr,
                             prm.sample_offset + s)
        acc += c
        acc_a += a
        acc_n += n
    inv = 1.0 / float(prm.spp)
    out_rgb[tid] = acc * inv
    out_alb[tid] = acc_a * inv
    out_nrm[tid] = acc_n * inv
