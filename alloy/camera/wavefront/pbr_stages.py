# SPDX-License-Identifier: Apache-2.0
"""Stages of the full-material wavefront shade kernel: the megakernel's path loop (``pbr.integrator``) split at the two
ray casts. The stages share ``pbr.path``'s helpers with the megakernel, so shading is the same code.

A path's vertex is processed as: load registers -> area-light emission -> escape or terminate -> hit geometry ->
Russian roulette -> closures (``pbr_shade``) -> denoising albedo -> NEE shadow ray (cast by the tracer) -> BSDF
bounce (next ray). The shadow ray's contribution is added by ``state.shadow_add`` if it is unoccluded.

No ``from __future__ import annotations`` here: Warp resolves struct and function argument types at definition time.

Example (inside a Warp kernel)::

    r = load_reg(ps, p)
    r.rad = r.rad + lights_on_ray(lsc, p2, prm, world, r.o, r.d, r.inst, r.t, r.thr, r.bounce, r.mis_n, 0, r.mis_pdf, 1)
"""
import warp as wp

from ..pbr import lights as L
from ..pbr.bsdf import principled as pr
from ..pbr.bsdf.bssrdf import LABEL_SUBSURFACE_SCATTER
from ..pbr.bsdf.glass import LABEL_TRANSMIT
from ..pbr.bsdf.microfacet import LABEL_DIFFUSE, LABEL_NONE
from ..pbr.path import (BumpTab, CamsV2, Medium, ParamsV2, albedo_step, col_at, count_bounce, escaped, first_ray_v2,
                        lights_on_ray, medium_enter_exit, medium_sigma, mis_after_bounce, smooth_normal, transmittance,
                        uv_at)
from ..pbr.shade_data import ShadeData
from ..pbr.subsurface import entry_bounce, random_walk
from ..tracer.geometry import T_INF, Geo
from ..tracer.integrator import Params, clamp_light, offset_point, russian_roulette
from ..tracer.sampling import PRNG_BOUNCE_NUM, PRNG_LIGHT, PRNG_SURFACE_BSDF, Rng, rand2, rand3
from .bevel import BevelBuf, bevel_active, bevel_seed, bevel_setup
from .bevel_flat import BevelFlat, bevel_flat
from .state import (F_ANCESTOR, F_CAMERA, F_DEN, F_MIS_HT, F_MIS_SKIP, F_PATH_DIFFUSE, F_SSS_EXIT, F_TERM_NEXT,
                    PathState, Wave, enqueue, item_of, item_rng, path_of)


@wp.struct
class Reg:
    """A path's registers while its vertex is shaded (loaded once, stored at the stage boundaries)."""
    o: wp.vec3
    d: wp.vec3
    t: float
    inst: wp.int32
    flags: wp.int32
    thr: wp.vec3
    rad: wp.vec3
    alb: wp.vec3
    nrm: wp.vec3
    den_thr: wp.vec3
    mis_n: wp.vec3
    mis_pdf: float
    min_ray_pdf: float
    bounce: wp.int32


@wp.struct
class Hit:
    """Geometry of the hit point: world position, flipped geometric / shading normals, ray-differential footprint."""
    pt: wp.vec3
    ng: wp.vec3
    ns: wp.vec3
    sd_dP: float
    m: wp.int32  # mesh
    g: wp.int32  # global triangle
    backfacing: wp.int32
    uv: wp.vec3  # (u, v, 0) of the mesh's UV layer at the hit
    col: wp.vec3  # the colour attribute at the hit


@wp.func
def load_reg(ps: PathState, p: wp.int32) -> Reg:
    r = Reg()
    r.o = ps.o[p]
    r.d = ps.d[p]
    r.inst = ps.hit_inst[p]
    r.t = ps.hit_t[p]
    r.flags = ps.flags[p]
    r.thr = ps.thr[p]
    r.rad = ps.rad[p]
    r.alb = ps.alb[p]
    r.nrm = ps.nrm[p]
    r.den_thr = ps.den_thr[p]
    r.mis_n = ps.mis_n[p]
    r.mis_pdf = ps.mis_pdf[p]
    r.min_ray_pdf = ps.min_ray_pdf[p]
    r.bounce = ps.bounce[p]
    return r


@wp.func
def medium_thr(ps: PathState, p: wp.int32, bt: BumpTab, thr: wp.vec3) -> wp.vec3:
    """The throughput after the ray's segment through the absorbing volumes it is in (Beer-Lambert to the hit, or
    nothing left if it escapes); unchanged outside volumes. The shade kernel and the Bevel prepass both use it, so
    Russian roulette sees the same throughput in both."""
    if ps.vst_mid[p][0] < 0:
        return thr
    md = Medium()
    md.inst = ps.vst_inst[p]
    md.mid = ps.vst_mid[p]
    t = wp.where(ps.hit_inst[p] >= 0, ps.hit_t[p], 1e30)
    return wp.cw_mul(thr, transmittance(medium_sigma(bt, md), t))


@wp.func
def mis_ht_of(r: Reg) -> wp.int32:
    return wp.where((r.flags & F_MIS_HT) != 0, 1, 0)


@wp.func
def store_guides(ps: PathState, p: wp.int32, r: Reg):
    ps.rad[p] = r.rad
    ps.alb[p] = r.alb
    ps.nrm[p] = r.nrm


@wp.func
def mis_skip_of(r: Reg) -> wp.int32:
    return wp.where((r.flags & F_MIS_SKIP) != 0, 1, 0)


@wp.func
def lamp_emission(r: Reg, c: wp.vec3) -> Reg:
    """Add the emission of lamps the ray crossed to the radiance and (while the guides are unfinished) the albedo,
    as Cycles' film_write_emission_or_background_pass."""
    out = r
    out.rad = r.rad + c
    if (r.flags & F_DEN) != 0:
        out.alb = r.alb + wp.cw_mul(r.den_thr, c)
    return out


@wp.func
def escape_reg(r: Reg, lsc: L.LightScene, p2: ParamsV2, prm: Params, world: wp.int32) -> Reg:
    """A ray that left the scene: world and distant-light radiance, added to the radiance and (while the guides are
    unfinished) the albedo."""
    out = r
    c = escaped(lsc, p2, prm, world, r.o, r.d, r.thr, r.bounce, r.mis_n, mis_ht_of(r), r.mis_pdf, mis_skip_of(r))
    out.rad = r.rad + c
    if (r.flags & F_DEN) != 0:
        out.alb = r.alb + wp.cw_mul(r.den_thr, c)
    return out


@wp.func
def world_face_normal(geo: Geo, inst: wp.int32, face: wp.int32) -> wp.vec3:
    return wp.normalize(wp.transform_vector(geo.inst_xf[inst], wp.mesh_eval_face_normal(geo.inst_mesh[inst], face)))


@wp.func
def hit_geometry(ps: PathState, p: wp.int32, geo: Geo, sd: ShadeData, r: Reg) -> Hit:
    """Hit point, normals flipped to the ray's side, and the footprint of the ray differential."""
    face = ps.hit_prim[p]
    h = Hit()
    h.ng = world_face_normal(geo, r.inst, face)
    h.pt = r.o + r.d * r.t
    h.sd_dP = ps.rdP[p] + r.t * ps.rdD[p]
    ns, m, g = smooth_normal(geo, sd, r.inst, face, ps.hit_u[p], ps.hit_v[p])
    h.ns = ns
    h.m = m
    h.g = g
    h.uv = uv_at(sd, m, face, ps.hit_u[p], ps.hit_v[p])
    h.col = col_at(sd, m, face, ps.hit_u[p], ps.hit_v[p])
    h.backfacing = 0
    if wp.dot(h.ng, r.d) > 0.0:
        h.ng = -h.ng
        h.ns = -h.ns
        h.backfacing = 1
    return h


@wp.func
def nee_sample(p: wp.int32, lsc: L.LightScene, world: wp.int32, rl: wp.vec3, pt: wp.vec3, ns: wp.vec3,
               ht: wp.int32) -> L.LightSample:
    """The light sample of a vertex. A separate function on purpose: inlining ``L.light_sample`` into the caller
    changes the float rounding of the shadow ray's contribution (compiler contraction), and the wavefront must
    reproduce the reference renders bit for bit."""
    return L.light_sample(lsc, world, rl, pt, ns, ht)


@wp.func
def nee_ray(ps: PathState, p: wp.int32, q_sh: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
            lsc: L.LightScene, cl: pr.PrincipledClosures, prm: Params, rl: wp.vec3, world: wp.int32, wi: wp.vec3,
            h: Hit, thr: wp.vec3, bounce: wp.int32):
    """Light sample + BSDF eval; writes the shadow ray and its contribution, queues the path for the shadow cast."""
    ls = nee_sample(p, lsc, world, rl, h.pt, h.ns, cl.glass.active)
    if ls.valid == 0:
        return
    ev = pr.closures_eval(cl, wi, ls.D)
    if wp.max(wp.max(ev.f[0], ev.f[1]), ev.f[2]) > 0.0:  # pdf 0 (no closure samples ls.D): MIS weight 1, as Cycles
        w = float(1.0)
        if ls.use_mis != 0:
            w = L.power_heuristic(ls.pdf, ev.pdf)
        c = wp.cw_mul(thr, wp.cw_mul(ev.f, ls.L)) * (w / ls.pdf)
        side = wp.where(wp.dot(ls.D, h.ng) >= 0.0, h.ng, -h.ng)
        ps.sh_o[p] = offset_point(h.pt, side, prm)
        ps.sh_d[p] = ls.D
        ps.sh_tmax[p] = wp.min(ls.t * 0.9999, T_INF)
        ps.sh_L[p] = clamp_light(c, bounce, prm)
        enqueue(q_sh, counts, 1, p)


@wp.func
def sss_bounce(ps: PathState, p: wp.int32, q_out: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
               cl: pr.PrincipledClosures, prm: Params, geo: Geo, rng: Rng, sample: wp.int32, rofs: wp.int32,
               sobol: wp.array(dtype=wp.uint32), wi: wp.vec3, h: Hit, r: Reg, weight: wp.vec3, den: wp.int32):
    """The BSSRDF was picked: entry bounce and random walk now; the exit becomes the path's next vertex without a
    cast (its hit goes to x_*, restored by ``wf_sss_fixup``). Bounce counters and MIS state stay (Cycles)."""
    ok, wo = entry_bounce(cl.sss, wi, h.ng, rand2(rng, sample, rofs + PRNG_BOUNCE_NUM, sobol))
    if ok == 0:
        return
    w = random_walk(geo, r.inst, h.pt, offset_point(h.pt, -h.ng, prm), wo, h.ns, cl.sss, wp.cw_mul(r.thr, weight), rng,
                    sample, rofs + PRNG_BOUNCE_NUM, sobol)
    if w.ok == 0:
        return
    ps.thr[p] = w.thr
    ps.den_thr[p] = r.den_thr
    ps.rng_extra[p] = ps.rng_extra[p] + 2
    ps.x_t[p] = w.t
    ps.x_u[p] = w.u
    ps.x_v[p] = w.v
    ps.x_prim[p] = w.face
    ps.x_inst[p] = r.inst
    ps.o[p] = w.p + w.d * w.t  # reached "from outside", as Cycles
    ps.d[p] = -w.d
    ps.tmax[p] = 0.0
    ps.rdP[p] = h.sd_dP
    keep = F_MIS_SKIP | F_PATH_DIFFUSE | F_MIS_HT | F_ANCESTOR
    ps.flags[p] = (r.flags & keep) | wp.where(den != 0, F_DEN, 0) | F_SSS_EXIT
    enqueue(q_out, counts, 0, p)


@wp.func
def bounce_ray(ps: PathState, p: wp.int32, q_out: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
               cl: pr.PrincipledClosures, prm: Params, p2: ParamsV2, rb: wp.vec3, wi: wp.vec3, h: Hit, r: Reg,
               den: wp.int32, mid: wp.int32, vol_on: wp.int32, geo: Geo, rng: Rng, sample: wp.int32, rofs: wp.int32,
               sobol: wp.array(dtype=wp.uint32)):
    """Sample the BSDF; on success update the path (throughput, MIS state, counters, flags, absorbing volumes), write
    the next ray and queue the path. ``den``: the denoising guides are still unfinished; ``mid`` / ``vol_on``: the
    hit's material and whether it has an absorption volume (entered or left on a refraction)."""
    s = pr.closures_sample(cl, wi, rb)
    if s.label == LABEL_SUBSURFACE_SCATTER:
        sss_bounce(ps, p, q_out, counts, cl, prm, geo, rng, sample, rofs, sobol, wi, h, r, s.weight, den)
        return
    if s.label == LABEL_NONE or not (s.pdf > 0.0):
        return
    mis_skip, mis_pdf, mis_n, min_ray_pdf = mis_after_bounce(s.label, s.pdf, h.ns, mis_skip_of(r), r.mis_pdf, r.mis_n,
                                                            r.min_ray_pdf)
    bounce, db, gb, tb, term = count_bounce(s.label, r.bounce, ps.dbounce[p], ps.gbounce[p], ps.tbounce[p], prm, p2)
    diffuse = (s.label & LABEL_DIFFUSE) != 0
    nf = wp.where(den != 0, F_DEN, 0) | wp.where(mis_skip != 0, F_MIS_SKIP, 0) | wp.where(term != 0, F_TERM_NEXT, 0)
    nf = nf | wp.where(diffuse, F_PATH_DIFFUSE, 0) | wp.where(cl.glass.active != 0, F_MIS_HT, 0)
    nf = nf | wp.where(diffuse or (r.flags & F_ANCESTOR) != 0, F_ANCESTOR, 0)
    ps.dbounce[p] = db
    ps.gbounce[p] = gb
    ps.tbounce[p] = tb
    if (s.label & LABEL_TRANSMIT) != 0 and vol_on != 0:
        md = Medium()
        md.inst = ps.vst_inst[p]
        md.mid = ps.vst_mid[p]
        md = medium_enter_exit(md, r.inst, mid, h.backfacing != 0)
        ps.vst_inst[p] = md.inst
        ps.vst_mid[p] = md.mid
    ps.o[p] = offset_point(h.pt, wp.where(wp.dot(s.wo, h.ng) >= 0.0, h.ng, -h.ng), prm)
    ps.d[p] = s.wo
    ps.tmax[p] = T_INF
    ps.rdP[p] = h.sd_dP
    ps.thr[p] = wp.cw_mul(r.thr, s.weight)
    ps.den_thr[p] = r.den_thr
    ps.mis_n[p] = mis_n
    ps.mis_pdf[p] = mis_pdf
    ps.min_ray_pdf[p] = min_ray_pdf
    ps.bounce[p] = bounce
    ps.flags[p] = nf
    enqueue(q_out, counts, 0, p)


@wp.kernel(module="unique", enable_backward=False)
def wf_sss_fixup(ps: PathState, q_in: wp.array(dtype=wp.int32), identity: wp.int32):
    """Restore the stored exit hit of every path whose next vertex is a random walk's exit (it was not traced)."""
    p = path_of(q_in, identity, wp.tid())
    if (ps.flags[p] & F_SSS_EXIT) != 0:
        ps.hit_t[p] = ps.x_t[p]
        ps.hit_u[p] = ps.x_u[p]
        ps.hit_v[p] = ps.x_v[p]
        ps.hit_prim[p] = ps.x_prim[p]
        ps.hit_inst[p] = ps.x_inst[p]


@wp.kernel(module="unique", enable_backward=False)
def wf_camera(ps: PathState, wv: Wave, cams: CamsV2, prm: Params, ftab: wp.array(dtype=float),
              sobol: wp.array(dtype=wp.uint32), pix_rng: wp.array(dtype=wp.uint32)):
    """Camera ray and fresh path state of every work item of the wave."""
    p = wp.tid()
    it = item_of(wv, prm, p)
    rng = item_rng(prm, pix_rng, it.pix)
    o, d, tmax, rdD = first_ray_v2(cams, prm, ftab, sobol, rng, it.img, it.px, it.pyr, it.sample)
    ps.o[p] = o
    ps.d[p] = d
    ps.tmax[p] = tmax
    ps.world[p] = it.world
    ps.rdD[p] = rdD
    ps.rdP[p] = 0.0
    ps.thr[p] = wp.vec3(1.0)
    ps.den_thr[p] = wp.vec3(1.0)
    ps.mis_n[p] = wp.vec3(0.0)
    ps.mis_pdf[p] = 0.0
    ps.min_ray_pdf[p] = 3.4e38
    ps.bounce[p] = 0
    ps.dbounce[p] = 0
    ps.gbounce[p] = 0
    ps.tbounce[p] = 0
    ps.rng_extra[p] = 0
    ps.vst_inst[p] = wp.vec4i(-1, -1, -1, -1)
    ps.vst_mid[p] = wp.vec4i(-1, -1, -1, -1)
    ps.flags[p] = F_DEN | F_MIS_SKIP | F_CAMERA
    ps.rad[p] = wp.vec3(0.0)
    ps.alb[p] = wp.vec3(0.0)
    ps.nrm[p] = wp.vec3(0.0)


@wp.kernel(module="unique", enable_backward=False)
def wf_bevel_prepass(ps: PathState, wv: Wave, q_in: wp.array(dtype=wp.int32), identity: wp.int32,
                     q_bv: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), geo: Geo, sd: ShadeData,
                     bt: BumpTab, prm: Params, sobol: wp.array(dtype=wp.uint32), pix_rng: wp.array(dtype=wp.uint32),
                     bb: BevelBuf, bf: BevelFlat):
    """Bevel rays of every vertex the shade kernel will bevel (same inputs, computed the same way: hit, not
    terminated, survives Russian roulette, material with Bevel, ``bevel_active``)."""
    p = path_of(q_in, identity, wp.tid())
    inst = ps.hit_inst[p]
    sss_exit = (ps.flags[p] & F_SSS_EXIT) != 0
    if inst < 0 or ((ps.flags[p] & F_TERM_NEXT) != 0 and not sss_exit):
        return
    face = ps.hit_prim[p]
    m = sd.inst_src[inst]
    mid = sd.tri_mat[sd.mesh_tri_off[m] + face]
    if bt.bevel_on[mid] == 0 or not bevel_active(bt.bevel_radius[mid], bt.bevel_samples[mid], ps.min_ray_pdf[p]):
        return
    it = item_of(wv, prm, p)
    rng = item_rng(prm, pix_rng, it.pix)
    bounce = ps.bounce[p]
    if not sss_exit:  # a walk's exit has no Russian roulette
        alive, thr = russian_roulette(medium_thr(ps, p, bt, ps.thr[p]), bounce, prm, rng, it.sample,
                                      PRNG_BOUNCE_NUM * (bounce + 1 + ps.rng_extra[p]), sobol)
        if not alive:
            return
    d = ps.d[p]
    ng = world_face_normal(geo, inst, face)
    if wp.dot(ng, d) > 0.0:
        ng = -ng
    p_frame = wp.transform_point(geo.inst_inv[inst], ps.o[p] + d * ps.hit_t[p])
    if bevel_flat(bf, sd.bevel_mesh, sd.mesh_tri_off[m] + face, p_frame, bt.bevel_radius[mid]):
        bb.n[p] = -1
        return
    rot_i = wp.quat_to_matrix(wp.transform_get_rotation(geo.inst_xf[inst]))
    bevel_setup(bb, p, m, p_frame, wp.transpose(rot_i) * ng, bt.bevel_radius[mid], bt.bevel_samples[mid],
                bevel_seed(rng.pixel, it.sample, bounce + 64 * ps.rng_extra[p]))
    enqueue(q_bv, counts, 2, p)
