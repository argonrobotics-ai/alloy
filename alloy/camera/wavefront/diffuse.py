# SPDX-License-Identifier: Apache-2.0
"""Diffuse (Lambert) wavefront kernels: ``tracer.integrator.trace_path`` split at the two ray casts, same material hooks,
lights and sampler. Compiled with fast_math like the diffuse megakernel.

Example::

    wf = DiffuseWavefront(renderer, tracer, WavefrontOptions.from_env())    # renderer: tracer.renderer.Renderer
    wf.render(spp, prm)                                                      # fills renderer.rgb / albedo / normal
"""
from __future__ import annotations

import sys
import weakref

import warp as wp

from ..tracer.geometry import T_INF, Geo
from ..tracer.integrator import (Cams, Mats, Params, bsdf_eval, bsdf_sample, camera_ray, clamp_light, max3,
                                 offset_point, russian_roulette, surface_color, to_camera)
from ..tracer.sampling import (PRNG_BOUNCE_NUM, PRNG_FILTER, PRNG_LIGHT, PRNG_SURFACE_BSDF, lookup_table, power_heuristic,
                               rand2, rand3)
from ..tracer.sky import LIGHT_NONE, LIGHT_SUN, Lights, bg_pdf, light_sample, select_light, selection_probs, sun_hit
from .host import Stages, Wavefront
from .knobs import WavefrontOptions
from .state import (F_CAMERA, F_DEN, F_MIS_SKIP, F_TERM_NEXT, PathState, Wave, enqueue, item_of, item_rng, path_of,
                    pixel_rng_kernel)

wp.set_module_options({"fast_math": True}, sys.modules[__name__])


@wp.kernel(enable_backward=False)
def wf_camera(ps: PathState, wv: Wave, cams: Cams, prm: Params, ftab: wp.array(dtype=float),
              sobol: wp.array(dtype=wp.uint32), pix_rng: wp.array(dtype=wp.uint32)):
    p = wp.tid()
    it = item_of(wv, prm, p)
    rng = item_rng(prm, pix_rng, it.pix)
    fu = wp.vec2(0.5, 0.5)
    if it.sample != 0 or prm.first_centered == 0:
        fu = rand2(rng, it.sample, PRNG_FILTER, sobol)
    o, d, tmax = camera_ray(cams, it.img, float(it.px) + lookup_table(ftab, fu[0]),
                            float(it.pyr) + lookup_table(ftab, fu[1]), prm)
    ps.o[p] = o
    ps.d[p] = d
    ps.tmax[p] = tmax
    ps.world[p] = it.world
    ps.thr[p] = wp.vec3(1.0)
    ps.mis_n[p] = wp.vec3(0.0)
    ps.mis_pdf[p] = 0.0
    ps.bounce[p] = 0
    ps.dbounce[p] = 0
    ps.flags[p] = F_DEN | F_MIS_SKIP | F_CAMERA
    ps.rad[p] = wp.vec3(0.0)
    ps.alb[p] = wp.vec3(0.0)
    ps.nrm[p] = wp.vec3(0.0)


@wp.func
def escape(ps: PathState, p: wp.int32, lt: Lights, prm: Params, d: wp.vec3):
    """A ray that left the scene: the sun disc and the world colour, MIS-weighted against the previous vertex."""
    flags = ps.flags[p]
    thr, bounce, mis_n, mis_pdf = ps.thr[p], ps.bounce[p], ps.mis_n[p], ps.mis_pdf[p]
    mis_skip = (flags & F_MIS_SKIP) != 0
    den = (flags & F_DEN) != 0
    radiance = ps.rad[p]
    alb = ps.alb[p]
    if sun_hit(lt, d) and ((not mis_skip) or lt.sun_visible_camera != 0):
        w = float(1.0)
        if not mis_skip:
            w = power_heuristic(mis_pdf, lt.sun_pdf * selection_probs(lt, mis_n)[0])
        c = clamp_light(wp.cw_mul(thr, lt.sun_radiance) * w, bounce - 1, prm)
        radiance += c
        if den:
            alb += c
    if lt.has_bg != 0:
        w = float(1.0)
        if (not mis_skip) and lt.bg_is_light != 0:
            w = power_heuristic(mis_pdf, selection_probs(lt, mis_n)[1] * bg_pdf(lt, d))
        c = clamp_light(wp.cw_mul(thr, lt.bg_color) * w, bounce - 1, prm)
        radiance += c
        if den:
            alb += c
    ps.rad[p] = radiance
    ps.alb[p] = alb


@wp.func
def nee_ray(ps: PathState, p: wp.int32, q_sh: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
            color: wp.vec3, lt: Lights, prm: Params, rl: wp.vec3, pt: wp.vec3, n: wp.vec3, ng: wp.vec3, thr: wp.vec3,
            bounce: wp.int32):
    """Next-event estimation: pick a light, write the shadow ray and its contribution, queue it."""
    lk = select_light(lt, n, rl[2])
    if lk == LIGHT_NONE:
        return
    sel = selection_probs(lt, n)
    ldir, lpdf, le, use_mis = light_sample(lt, lk, wp.vec2(rl[0], rl[1]))
    lpdf = lpdf * wp.where(lk == LIGHT_SUN, sel[0], sel[1])
    ev, bpdf = bsdf_eval(color, n, ng, ldir)
    if bpdf > 0.0 and lpdf > 0.0:
        w = float(1.0)
        if use_mis != 0:
            w = power_heuristic(lpdf, bpdf)
        c = wp.cw_mul(thr, wp.cw_mul(ev, le)) * (w / lpdf)
        if max3(c) > 0.0:
            ps.sh_o[p] = offset_point(pt, ng, prm)
            ps.sh_d[p] = ldir
            ps.sh_tmax[p] = T_INF
            ps.sh_L[p] = clamp_light(c, bounce, prm)
            enqueue(q_sh, counts, 1, p)


@wp.func
def bounce_ray(ps: PathState, p: wp.int32, q_out: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
               color: wp.vec3, prm: Params, rb: wp.vec3, pt: wp.vec3, n: wp.vec3, ng: wp.vec3, thr: wp.vec3,
               bounce: wp.int32):
    """Sample the diffuse lobe; on success write the next ray and the path counters, and queue the path."""
    wo, wgt, bpdf = bsdf_sample(color, n, ng, rb)
    if bpdf <= 0.0:
        return
    nf = int(0)
    nb = bounce + 1
    if nb >= prm.max_bounce:
        nf = nf | F_TERM_NEXT
    db = ps.dbounce[p] + 1
    if db >= prm.max_diffuse:
        nf = nf | F_TERM_NEXT
    ps.dbounce[p] = db
    ps.thr[p] = wp.cw_mul(thr, wgt)
    ps.mis_pdf[p] = bpdf
    ps.mis_n[p] = n
    ps.bounce[p] = nb
    ps.flags[p] = nf
    ps.o[p] = offset_point(pt, ng, prm)
    ps.d[p] = wo
    ps.tmax[p] = T_INF
    enqueue(q_out, counts, 0, p)


@wp.kernel(enable_backward=False)
def wf_shade(ps: PathState, wv: Wave, q_in: wp.array(dtype=wp.int32), identity: wp.int32,
             q_out: wp.array(dtype=wp.int32), q_sh: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
             geo: Geo, mats: Mats, lt: Lights, cams: Cams, prm: Params, sobol: wp.array(dtype=wp.uint32),
             pix_rng: wp.array(dtype=wp.uint32)):
    p = path_of(q_in, identity, wp.tid())
    it = item_of(wv, prm, p)
    rng = item_rng(prm, pix_rng, it.pix)
    d = ps.d[p]
    inst = ps.hit_inst[p]
    flags = ps.flags[p]
    if inst < 0:
        escape(ps, p, lt, prm, d)
        return
    if (flags & F_TERM_NEXT) != 0:
        return
    pt = ps.o[p] + d * ps.hit_t[p]
    ng = wp.normalize(wp.transform_vector(geo.inst_xf[inst], wp.mesh_eval_face_normal(geo.inst_mesh[inst],
                                                                                        ps.hit_prim[p])))
    if wp.dot(ng, d) > 0.0:
        ng = -ng
    m = geo.inst_mat[inst]
    bounce = ps.bounce[p]
    rofs = PRNG_BOUNCE_NUM * (bounce + 1)
    alive, thr = russian_roulette(ps.thr[p], bounce, prm, rng, it.sample, rofs, sobol)
    if not alive:
        return
    color = surface_color(geo, mats, inst, m, ps.hit_prim[p], pt)
    if (flags & F_DEN) != 0:
        ps.alb[p] = ps.alb[p] + color
        ps.nrm[p] = ps.nrm[p] + wp.where(prm.normal_camera != 0, to_camera(cams, it.img, ng), ng)
    nee_ray(ps, p, q_sh, counts, color, lt, prm, rand3(rng, it.sample, rofs + PRNG_LIGHT, sobol), pt, ng, ng, thr,
            bounce)
    bounce_ray(ps, p, q_out, counts, color, prm, rand3(rng, it.sample, rofs + PRNG_SURFACE_BSDF, sobol), pt, ng, ng,
               thr, bounce)


class DiffuseWavefront:
    """Wavefront renderer for a diffuse :class:`~alloy.camera.tracer.renderer.Renderer`."""

    def __init__(self, r, tracer, opts: WavefrontOptions):
        # a weak link back: the renderer owns this object, and a cycle would keep GPU buffers alive until gc runs
        self.r, self.dev = weakref.proxy(r), r.sc.device
        n = r.n_img * r.w * r.h
        self.wf = Wavefront(tracer, n, self.dev, r.cams.world, opts.paths_per_wave(self.dev, False), opts.sort_min)
        self.pix_rng = wp.zeros(n, dtype=wp.uint32, device=self.dev)

    def render(self, spp: int, prm: Params) -> None:
        """Mean over ``spp`` samples into ``r.rgb`` / ``r.albedo`` / ``r.normal``."""
        r, wf, dev = self.r, self.wf, self.dev
        wp.launch(pixel_rng_kernel, dim=self.pix_rng.shape[0], inputs=[prm], outputs=[self.pix_rng], device=dev)

        def camera(wv, n):
            wp.launch(wf_camera, dim=n, inputs=[wf.ps, wv, r.cams, prm, r.ftab, r.sobol, self.pix_rng], device=dev)

        def shade(wv, n, q_in, q_out, q_sh, counts, first):
            wp.launch(wf_shade, dim=n, inputs=[wf.ps, wv, q_in, int(first), q_out, q_sh, counts, r.sc.geo, r.mats,
                                               r.lights, r.cams, prm, r.sobol, self.pix_rng], device=dev,
                      block_dim=r.block_dim)

        wf.run(spp, Stages(camera, shade), r.rgb, r.albedo, r.normal)
