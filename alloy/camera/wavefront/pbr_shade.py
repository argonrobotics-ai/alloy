# SPDX-License-Identifier: Apache-2.0
"""The full-material wavefront shade kernels, built per generated material module (``dispatch.generate``).

``make_shade(SPEC)`` compiles one kernel: ``SPEC >= 0`` = only material SPEC (its paths are one contiguous range of the
material-sorted queue), with the material, its Bevel and Bump flags as compile-time constants so the compiler drops the
other materials' code; ``SPEC = -1`` = any material; ``SPEC = -2`` = escaped / terminated paths only.

No ``from __future__ import annotations`` here: Warp resolves the closures' argument types at definition time.

Example::

    shade = ShadeKernels(disp, block_dim=32, warps=12)
    generic = shade.generic          # any material
    spec3 = shade.specialised(3, bevel_on=0, bump_on=1)
"""
import warp as wp

from ..pbr import lights as L
from ..pbr.bsdf import principled as pr
from ..pbr.integrator import _make_bump, kernel_options
from ..pbr.path import BumpTab, CamsV2, ParamsV2, albedo_step, exit_closures, geo_rot, lights_on_ray, path_flags
from ..pbr.shade_data import ShadeData
from ..tracer.geometry import Geo
from ..tracer.integrator import Params, russian_roulette
from ..tracer.sampling import PRNG_BOUNCE_NUM, PRNG_LIGHT, PRNG_SURFACE_BSDF, Rng, rand3
from .bevel import BevelBuf, bevel_n, bevel_seed
from .pbr_stages import (Hit, bounce_ray, escape_reg, hit_geometry, lamp_emission, load_reg, medium_thr, mis_ht_of,
                         mis_skip_of, nee_ray, store_guides)
from .state import (F_ANCESTOR, F_DEN, F_PATH_DIFFUSE, F_SSS_EXIT, F_TERM_NEXT, PathState, Wave, item_of, item_rng,
                    path_of)


def shade_kernel_options(block_dim: int, warps: int) -> dict:
    """Compile options of the shade kernels: ``ALLOY_CAMERA_MAXREG`` if set, else ``__launch_bounds__`` for ``warps``
    resident warps per SM, i.e. at most 168 registers for 12 (255 otherwise, 8 warps); ``warps <= 0``: no bound."""
    opts = kernel_options()
    if opts.get("launch_bounds") or opts.get("cuda_max_registers") or warps <= 0:
        return opts
    return {**opts, "launch_bounds": (block_dim, max(1, warps * 32 // block_dim))}


@wp.func
def default_tangent_at(sd: ShadeData, geo: Geo, inst: wp.int32, m: wp.int32, ns: wp.vec3, p_obj: wp.vec3) -> wp.vec3:
    """Anisotropy tangent from the Generated coordinates (Cycles' default tangent)."""
    gen = wp.cw_mul(p_obj - sd.mesh_texloc[m], sd.mesh_texinv[m]) + wp.vec3(0.5)
    return pr.default_tangent(ns, gen, wp.mul(geo_rot(geo, inst), sd.mesh_n_world[m]))


@wp.func
def bevel_at(geo: Geo, sd: ShadeData, bt: BumpTab, bb: BevelBuf, rng: Rng, p: wp.int32, inst: wp.int32, m: wp.int32,
             mid: wp.int32, p_frame: wp.vec3, n_frame: wp.vec3, h: Hit, min_ray_pdf: float, sample: wp.int32,
             bounce: wp.int32) -> wp.vec3:
    """World-space Bevel normal of the hit (gathered from the traced hits; the input normal in flat regions)."""
    rot_i = geo_rot(geo, inst)
    bn = bevel_n(bb, p, sd.bevel_mesh, sd.tri_n, sd.mesh_tri_off[m], p_frame, wp.transpose(rot_i) * h.ng, n_frame,
                 h.backfacing, bt.bevel_radius[mid], bt.bevel_samples[mid], min_ray_pdf,
                 bevel_seed(rng.pixel, sample, bounce))
    return wp.normalize(rot_i * bn)


def _make_shade(disp, block_dim: int, warps: int, SPEC: int = -1, SPEC_BEVEL: int = 0, SPEC_BUMP: int = 0):
    Images = disp.Images
    MP = disp.MAT_PARAMS[SPEC] if SPEC >= 0 else disp.mat_params  # material k only: its constants fold into the BSDF
    MH = disp.MAT_HEIGHT[SPEC] if SPEC >= 0 else disp.mat_height
    bump_at = _make_bump(MH, Images)

    @wp.kernel(module="unique", enable_backward=False, **shade_kernel_options(block_dim, warps))
    def wf_shade(ps: PathState, wv: Wave, q_in: wp.array(dtype=wp.int32), identity: wp.int32,
                 q_out: wp.array(dtype=wp.int32), q_sh: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
                 geo: Geo, sd: ShadeData, bt: BumpTab, lsc: L.LightScene, cams: CamsV2, prm: Params, p2: ParamsV2,
                 sobol: wp.array(dtype=wp.uint32), lut: wp.array(dtype=float), im: Images,
                 pix_rng: wp.array(dtype=wp.uint32), bb: BevelBuf, off: wp.int32):
        p = path_of(q_in, identity, off + wp.tid())
        it = item_of(wv, prm, p)
        rng = item_rng(prm, pix_rng, it.pix)
        r = load_reg(ps, p)
        sss_exit = (r.flags & F_SSS_EXIT) != 0  # a random walk's exit: no cast, emission or roulette (Cycles)
        if not sss_exit:
            r.thr = medium_thr(ps, p, bt, r.thr)
            r = lamp_emission(r, lights_on_ray(lsc, p2, prm, it.world, r.o, r.d, r.inst, r.t, r.thr, r.bounce,
                                               r.mis_n, mis_ht_of(r), r.mis_pdf, mis_skip_of(r)))
            if r.inst < 0:
                store_guides(ps, p, escape_reg(r, lsc, p2, prm, it.world))
                return
            if (r.flags & F_TERM_NEXT) != 0:
                store_guides(ps, p, r)
                return
        if wp.static(SPEC == -2):  # escape-only kernel: launched on misses and terminated paths, which returned above
            return
        h = hit_geometry(ps, p, geo, sd, r)
        extra = ps.rng_extra[p]
        rofs = PRNG_BOUNCE_NUM * (r.bounce + 1 + extra)
        if not sss_exit:
            alive, thr = russian_roulette(r.thr, r.bounce, prm, rng, it.sample, rofs, sobol)
            if not alive:
                store_guides(ps, p, r)
                return
            r.thr = thr
        wi = -r.d
        # The Principled closures are built inline: as a separate function the compiler contracts the float
        # arithmetic differently and the render is no longer bit-identical to the reference (bevel_at, bump_at,
        # default_tangent_at and path_flags are fine as functions).
        m = h.m
        p_frame = wp.transform_point(geo.inst_inv[r.inst], h.pt)
        n_frame = wp.transform_vector(geo.inst_inv[r.inst], h.ns)
        p_obj = wp.transform_point(sd.mesh_obj_inv[m], p_frame)
        n_obj = wp.normalize(sd.mesh_n_obj[m] * n_frame)
        mid = sd.tri_mat[h.g]
        bevel_on = bt.bevel_on[mid]
        bump_on = bt.on[mid]
        if wp.static(SPEC >= 0):  # material-specialised kernel: compile-time material, Bevel and Bump
            mid = SPEC
            bevel_on = SPEC_BEVEL
            bump_on = SPEC_BUMP
        rnd = sd.mesh_rnd[m]
        q = MP(mid, p_obj, n_obj, h.uv, h.col, rnd, im)
        q.backfacing = h.backfacing
        n_in = wp.vec3(0.0)
        n_base = h.ns
        if bevel_on != 0:
            n_base = bevel_at(geo, sd, bt, bb, rng, p, r.inst, m, mid, p_frame, n_frame, h, r.min_ray_pdf, it.sample,
                              r.bounce + 64 * extra)
            n_in = n_base
        if bump_on != 0:
            n_in = bump_at(geo, sd, bt, im, r.inst, m, mid, h.pt, h.ng, n_base, n_obj, h.uv, h.col, rnd, h.sd_dP)
            q.coat_normal = n_in
        t_in = wp.vec3(0.0)
        if q.anisotropic > 0.0:
            t_in = default_tangent_at(sd, geo, r.inst, m, h.ns, p_obj)
        q.ng = h.ng
        q.ns = h.ns
        q.blur_roughness = pr.blur_roughness(p2.blur_glossy, r.min_ray_pdf)
        pf = path_flags(wp.where((r.flags & F_PATH_DIFFUSE) != 0, 1, 0), p2,
                        wp.where((r.flags & F_ANCESTOR) != 0, 1, 0))
        cl = pr.principled_setup(lut, q, n_in, t_in, wi, pf)
        den = r.flags & F_DEN
        if sss_exit:  # lit as a white Lambertian with the BSSRDF's (bevelled, bumped) normal there
            cl = exit_closures(wp.where(cl.sss.active != 0, cl.sss.n, h.ns), h.ns, h.ng)
        elif den != 0:
            a_den, a_thr, a_alb, a_nrm = albedo_step(lut, cl, wi, cams, it.img, prm, r.den_thr, r.alb, r.nrm)
            den = a_den
            r.den_thr = a_thr
            r.alb = a_alb
            r.nrm = a_nrm
        nee_ray(ps, p, q_sh, counts, lsc, cl, prm, rand3(rng, it.sample, rofs + PRNG_LIGHT, sobol), it.world, wi, h,
                r.thr, r.bounce)
        store_guides(ps, p, r)
        bounce_ray(ps, p, q_out, counts, cl, prm, p2, rand3(rng, it.sample, rofs + PRNG_SURFACE_BSDF, sobol), wi, h, r,
                   den, mid, bt.vol_on[mid], geo, rng, it.sample, rofs, sobol)

    return wf_shade


class ShadeKernels:
    """The shade kernels of one material module: the generic one, one per material (built on demand) and the
    escape-only one."""

    def __init__(self, disp, block_dim: int, warps: int):
        self.disp, self.block_dim, self.warps = disp, block_dim, warps
        self.generic = _make_shade(disp, block_dim, warps)
        self._spec: dict = {}
        self._escape = None

    def specialised(self, mat: int, bevel_on: int, bump_on: int):
        """The kernel for material ``mat`` with its Bevel / Bump flags folded in."""
        if not self.disp.MAT_PARAMS:  # a module of generic materials only: they read their rows, no per-material code
            return self.generic
        key = (mat, int(bevel_on), int(bump_on))
        if key not in self._spec:
            self._spec[key] = _make_shade(self.disp, self.block_dim, self.warps, mat, int(bevel_on), int(bump_on))
        return self._spec[key]

    @property
    def escape(self):
        """The kernel for escaped and terminated paths only."""
        if self._escape is None:
            self._escape = _make_shade(self.disp, self.block_dim, self.warps, -2)
        return self._escape
