# SPDX-License-Identifier: Apache-2.0
"""Full-material megakernel: the path loop of ``alloy.camera.tracer`` (Cycles 4.5 conventions and sampler) with
Principled materials, generated shader inputs, smooth corner normals, area lights + HDRI world, a thin-lens camera,
glossy bounce counting, filter glossy and the reflective-caustics rule. Built per material set by
``make_kernel(module)`` (the module from ``dispatch.generate``).

Bump uses Cycles' compact ray differentials (camera dD, dP carried along the path) and the height subgraph at the
three bump points, applied to the Bevel output when the material bevels.

No ``from __future__ import annotations`` here: Warp resolves the closures' argument types at definition time.

Example::

    kernel = make_kernel(materials_module)
    wp.launch(kernel, dim=n_images * width * height, inputs=[...], outputs=[rgb, albedo, normal])
"""

import os

import warp as wp

from ..tracer.geometry import Geo
from ..tracer.integrator import Params, russian_roulette
from ..tracer.sampling import PRNG_SURFACE_BSDF, Rng, hash_hp_uint, pixel_rng, rand3
from . import lights as L
from . import nodes as nd
from .bsdf import principled as pr
from .bsdf.bssrdf import LABEL_SUBSURFACE_SCATTER
from .mk_path import Hit, MkPath, Surf, mk_bounce, mk_next_hit, mk_rofs, mk_start, mk_subsurface
from .path import (BumpTab, CamsV2, ParamsV2, albedo_step, bevel_at, col_at, exit_closures, geo_rot, nee_v2,
                   path_flags, smooth_normal, uv_at)
from .shade_data import ShadeData

__all__ = ["BumpTab", "CamsV2", "ParamsV2", "make_kernel"]


def kernel_options() -> dict:
    """CUDA compile options (env, for tuning): ALLOY_CAMERA_MAXREG caps registers per thread."""
    mr = int(os.environ.get("ALLOY_CAMERA_MAXREG", "0"))
    return {"cuda_max_registers": mr} if mr > 0 else {}


def make_kernel(disp):
    """Compile the megakernel against a generated material module (``mat_params``, ``mat_height``, ``Images``)."""
    bump_at = _make_bump(disp.mat_height, disp.Images)
    closures_at = _make_closures(disp.mat_params, bump_at, disp.Images)
    shade_vertex = _make_shade(closures_at, _make_exit_normal(closures_at, disp.Images), disp.Images)
    return _make_render(_make_trace(shade_vertex, disp.Images), disp.Images)


def _make_bump(mat_height, Images):
    """The Bump node evaluation for this material set."""

    @wp.func
    def bump_at(geo: Geo, sd: ShadeData, bt: BumpTab, im: Images, inst: wp.int32, m: wp.int32, mid: wp.int32,
                p: wp.vec3, ng: wp.vec3, n_base: wp.vec3, n_obj: wp.vec3, uv: wp.vec3, col: wp.vec3, rnd: float,
                sd_dP: float) -> wp.vec3:
        """Bump node normal: height at P, P + dPdx, P + dPdy (object space) on top of n_base. The UV and colour
        attribute are those of P at the three points (no UV differentials: an image bump on UVs is flat)."""
        fw = bt.filter_width[mid]
        pc, pxx, pyy = nd.bump_sample_points(p, ng, sd_dP, fw)
        to_obj = sd.mesh_obj_inv[m]
        hc = mat_height(mid, wp.transform_point(to_obj, wp.transform_point(geo.inst_inv[inst], pc)), n_obj, uv, col, rnd, im)
        hx = mat_height(mid, wp.transform_point(to_obj, wp.transform_point(geo.inst_inv[inst], pxx)), n_obj, uv, col, rnd, im)
        hy = mat_height(mid, wp.transform_point(to_obj, wp.transform_point(geo.inst_inv[inst], pyy)), n_obj, uv, col, rnd, im)
        return nd.bump_normal(n_base, ng, sd_dP, hc, hx, hy, bt.strength[mid], bt.distance[mid], bt.invert[mid], fw)

    return bump_at


def _make_closures(mat_params, bump_at, Images):
    """Principled closures at a hit for this material set."""

    @wp.func
    def closures_at(geo: Geo, sd: ShadeData, bt: BumpTab, p2: ParamsV2, lut: wp.array(dtype=float), im: Images,
                    rng: Rng, inst: wp.int32, m: wp.int32, g: wp.int32, p: wp.vec3, ng: wp.vec3, ns: wp.vec3,
                    backfacing: bool, sd_dP: float, sample: wp.int32, bounce: wp.int32, min_ray_pdf: float,
                    wi: wp.vec3, path_diffuse: wp.int32, uv: wp.vec3, col: wp.vec3,
                    ancestor: wp.int32) -> pr.PrincipledClosures:
        """The hit's Principled closures: shader inputs in object space, Bevel, Bump, tangent, filter glossy."""
        p_frame = wp.transform_point(geo.inst_inv[inst], p)
        n_frame = wp.transform_vector(geo.inst_inv[inst], ns)
        p_obj = wp.transform_point(sd.mesh_obj_inv[m], p_frame)
        n_obj = wp.normalize(sd.mesh_n_obj[m] * n_frame)
        mid = sd.tri_mat[g]
        rnd = sd.mesh_rnd[m]
        q = mat_params(mid, p_obj, n_obj, uv, col, rnd, im)
        q.backfacing = wp.where(backfacing, 1, 0)
        n_in = wp.vec3(0.0)
        n_base = ns
        if bt.bevel_on[mid] != 0:
            n_base = bevel_at(geo, sd, bt, rng, inst, m, mid, p_frame, ng, n_frame, backfacing, sample, bounce,
                              min_ray_pdf)
            n_in = n_base
        if bt.on[mid] != 0:
            n_in = bump_at(geo, sd, bt, im, inst, m, mid, p, ng, n_base, n_obj, uv, col, rnd, sd_dP)
            q.coat_normal = n_in
        t_in = wp.vec3(0.0)
        if q.anisotropic > 0.0:
            gen = wp.cw_mul(p_obj - sd.mesh_texloc[m], sd.mesh_texinv[m]) + wp.vec3(0.5)
            t_in = pr.default_tangent(ns, gen, wp.mul(geo_rot(geo, inst), sd.mesh_n_world[m]))
        q.ng = ng
        q.ns = ns
        q.blur_roughness = pr.blur_roughness(p2.blur_glossy, min_ray_pdf)
        return pr.principled_setup(lut, q, n_in, t_in, wi, path_flags(path_diffuse, p2, ancestor))

    return closures_at


def _make_exit_normal(closures_at, Images):
    """The BSSRDF normal at a random walk's exit for this material set."""

    @wp.func
    def exit_normal(geo: Geo, sd: ShadeData, bt: BumpTab, p2: ParamsV2, lut: wp.array(dtype=float), im: Images,
                    rng: Rng, inst: wp.int32, m: wp.int32, g: wp.int32, p: wp.vec3, ng: wp.vec3, ns: wp.vec3,
                    sd_dP: float, sample: wp.int32, bounce: wp.int32, min_ray_pdf: float, wi: wp.vec3, uv: wp.vec3,
                    col: wp.vec3) -> wp.vec3:
        """surface_shader_bssrdf_normal: the BSSRDF's normal at a walk's exit when the material bumps it (shader
        evaluated there), else the smooth normal."""
        mid = sd.tri_mat[g]
        if bt.on[mid] == 0 and bt.bevel_on[mid] == 0:
            return ns
        c = closures_at(geo, sd, bt, p2, lut, im, rng, inst, m, g, p, ng, ns, False, sd_dP, sample, bounce,
                        min_ray_pdf, wi, 0, uv, col, 0)
        return wp.where(c.sss.active != 0, c.sss.n, ns)

    return exit_normal


def _make_shade(closures_at, exit_normal, Images):
    """The shading of one path vertex for this material set (``shade_vertex``)."""

    @wp.func
    def shade_vertex(geo: Geo, sd: ShadeData, bt: BumpTab, p2: ParamsV2, lut: wp.array(dtype=float), im: Images,
                     rng: Rng, cams: CamsV2, img: wp.int32, prm: Params, sobol: wp.array(dtype=wp.uint32),
                     sample: wp.int32, st_in: MkPath, h: Hit):
        """(path, surface, closures, alive) at a hit: normals facing the ray, attributes, Russian roulette and the
        material's closures (denoising albedo on the way), or a random walk's exit as a white Lambertian."""
        st = st_in
        ns, m, g = smooth_normal(geo, sd, h.inst, h.face, h.u, h.v)
        sf = Surf()
        sf.p = st.o + st.d * h.t
        sf.dP = st.rdP + h.t * st.rdD
        sf.m = m
        sf.g = g
        sf.uv = uv_at(sd, m, h.face, h.u, h.v)
        sf.col = col_at(sd, m, h.face, h.u, h.v)
        sf.backfacing = wp.dot(h.ng, st.d) > 0.0
        sf.ng = wp.where(sf.backfacing, -h.ng, h.ng)
        sf.ns = wp.where(sf.backfacing, -ns, ns)
        vb = st.bounce + 64 * st.rng_extra
        if st.sss_exit != 0:  # no Russian roulette at a walk's exit (Cycles)
            n = exit_normal(geo, sd, bt, p2, lut, im, rng, h.inst, m, g, sf.p, sf.ng, sf.ns, sf.dP, sample, vb,
                            st.min_ray_pdf, -st.d, sf.uv, sf.col)
            st.sss_exit = 0
            return st, sf, exit_closures(n, sf.ns, sf.ng), 1
        cl = pr.PrincipledClosures()
        if st.term_next != 0:
            return st, sf, cl, 0
        alive, thr_rr = russian_roulette(st.thr, st.bounce, prm, rng, sample, mk_rofs(st), sobol)
        if not alive:
            return st, sf, cl, 0
        st.thr = thr_rr
        cl = closures_at(geo, sd, bt, p2, lut, im, rng, h.inst, m, g, sf.p, sf.ng, sf.ns, sf.backfacing, sf.dP,
                         sample, vb, st.min_ray_pdf, -st.d, st.path_diffuse, sf.uv, sf.col, st.ancestor)
        if st.den != 0:
            den, den_thr, alb, nrm = albedo_step(lut, cl, -st.d, cams, img, prm, st.den_thr, st.alb, st.nrm)
            st.den = den
            st.den_thr = den_thr
            st.alb = alb
            st.nrm = nrm
        return st, sf, cl, 1

    return shade_vertex


def _make_trace(shade_vertex, Images):
    """One camera path (the megakernel's loop) for this material set: (radiance, albedo, normal)."""

    @wp.func
    def trace_v2(geo: Geo, sd: ShadeData, bt: BumpTab, lsc: L.LightScene, cams: CamsV2, prm: Params, p2: ParamsV2,
                 ftab: wp.array(dtype=float), sobol: wp.array(dtype=wp.uint32), lut: wp.array(dtype=float),
                 im: Images, rng: Rng, img: wp.int32, world: wp.int32, px: wp.int32, pyr: wp.int32, sample: wp.int32):
        st = mk_start(cams, prm, ftab, sobol, rng, img, px, pyr, sample)
        for _it in range(64):
            st, h, alive = mk_next_hit(geo, bt, lsc, p2, prm, world, st)
            if alive == 0:
                break
            st, sf, cl, alive = shade_vertex(geo, sd, bt, p2, lut, im, rng, cams, img, prm, sobol, sample, st, h)
            if alive == 0:
                break
            rofs = mk_rofs(st)
            st.rad = st.rad + nee_v2(geo, lsc, cl, prm, sobol, rng, sample, rofs, world, -st.d, sf.p, sf.ns, sf.ng,
                                     st.thr, st.bounce, cl.glass.active)
            s = pr.closures_sample(cl, -st.d, rand3(rng, sample, rofs + PRNG_SURFACE_BSDF, sobol))
            if s.label == LABEL_SUBSURFACE_SCATTER:
                st, alive = mk_subsurface(geo, prm, rng, sample, sobol, st, h, sf, cl, s.weight)
            else:
                st, alive = mk_bounce(st, sf, h, s, cl.glass.active, sd.tri_mat[sf.g], bt, prm, p2)
            if alive == 0:
                break
        return st.rad, st.alb, st.nrm

    return trace_v2


def _make_render(trace_v2, Images):
    """The megakernel: one thread per pixel of every image, ``prm.spp`` paths each."""

    @wp.kernel(module="unique", enable_backward=False, **kernel_options())
    def render_v2(geo: Geo, sd: ShadeData, bt: BumpTab, lsc: L.LightScene, cams: CamsV2, prm: Params, p2: ParamsV2,
                  ftab: wp.array(dtype=float), sobol: wp.array(dtype=wp.uint32), lut: wp.array(dtype=float),
                  im: Images, out_rgb: wp.array(dtype=wp.vec3), out_alb: wp.array(dtype=wp.vec3),
                  out_nrm: wp.array(dtype=wp.vec3)):
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
            c, a, n = trace_v2(geo, sd, bt, lsc, cams, prm, p2, ftab, sobol, lut, im, rng, img, cams.world[img], px,
                               pyr, prm.sample_offset + s)
            acc += c
            acc_a += a
            acc_n += n
        inv = 1.0 / float(prm.spp)
        out_rgb[tid] = acc * inv
        out_alb[tid] = acc_a * inv
        out_nrm[tid] = acc_n * inv

    return render_v2
