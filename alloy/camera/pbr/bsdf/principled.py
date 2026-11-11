# SPDX-License-Identifier: Apache-2.0
"""Cycles 4.5 Principled BSDF in NVIDIA Warp, for the inputs production scene C3 uses (see README.md).

Port of kernel/svm/closure.h (CLOSURE_BSDF_PRINCIPLED_ID: layering of sheen, coat, metal, specular and diffuse,
albedo scaling, caustics rule), kernel/integrator/surface_shader.h (closure pick by sample weight, one-sample MIS
over closures, filter-glossy blur) and kernel/film/denoising_passes.h (denoising albedo), tag v4.5.0.

All vectors are world space and unit length. Conventions as Cycles:
  wi points from the surface towards the previous vertex (the viewer), wo towards the next one / the light;
  ng, ns (and so n) face wi (Cycles flips them for back-facing hits);
  f = BSDF * cos(theta_o) (the cosine is included), pdf is per solid angle.
Usage: lut = bsdf_util.load_tables(device) once; pass it to every call.
"""
import warp as wp

from .util import (CLOSURE_WEIGHT_CUTOFF, F0_from_ior, average3, bump_shadowing_term, closure_layering_weight,
                       fresnel_f82tint_B, ior_from_F0, is_equal3, load_tables,  # noqa: F401  (re-export)
                       maybe_ensure_valid_specular_reflection, mix3, rotate_around_axis, safe_normalize,
                       safe_normalize_fallback, saturate3, vec3s)
from .glass import glass_estimate_albedo, glass_eval, glass_label, glass_sample
from .lobes import Diffuse, Sheen, diffuse_eval, diffuse_sample, sheen_eval, sheen_sample, sheen_setup
from .microfacet import (FRESNEL_DIELECTRIC, FRESNEL_F82_TINT, FRESNEL_GENERALIZED_SCHLICK, LABEL_DIFFUSE,
                        LABEL_NONE, LobeEval, LobeSample, Microfacet, mf_estimate_albedo, mf_eval, mf_finish_setup,
                        mf_label, mf_sample)
from .bssrdf import LABEL_SUBSURFACE_SCATTER
from .principled_structs import (LOBE_DIFFUSE, LOBE_GLASS, LOBE_SPEC, LOBE_SSS, LOBE_SSS_DIFFUSE,  # noqa: F401
                                 N_LOBES, PATH_DIFFUSE, PATH_DIFFUSE_ANCESTOR, PATH_NO_REFLECTIVE_CAUSTICS,
                                 PATH_NO_REFRACTIVE_CAUSTICS, PrincipledClosures, PrincipledParams)
from .principled_layers import principled_setup  # noqa: F401


@wp.struct
class BsdfEval:
    f: wp.vec3    # sum over closures of weight * BSDF * cos(theta_o)
    pdf: float    # MIS'd pdf: sum(sample_weight * pdf) / sum(sample_weight)


@wp.struct
class BsdfSample:
    wo: wp.vec3
    weight: wp.vec3  # throughput multiplier = f / pdf (zero if the sample failed)
    pdf: float       # MIS'd pdf of wo (for MIS with light sampling and for min_ray_pdf)
    label: int       # LABEL_* of the sampled closure; LABEL_NONE = terminate the path


@wp.struct
class DenoiseAlbedo:
    diffuse: wp.vec3   # albedo of diffuse-like closures (roughness^2 > 0.075^2)
    specular: wp.vec3  # albedo of specular-like closures
    normal: wp.vec3    # sample-weighted closure normal (world space), divided by the weight sum when done == 1
    done: int          # 1: write albedo = feature_throughput * diffuse (and the normal) here and stop;
                       # 0: feature_throughput *= specular and wait for the next vertex


@wp.func
def principled_default() -> PrincipledParams:
    """Blender 4.5 Principled BSDF socket defaults (ng, ns must still be set by the caller)."""
    p = PrincipledParams()
    p.base_color = wp.vec3(0.8, 0.8, 0.8)
    p.roughness = 0.5
    p.ior = 1.5
    p.specular_ior_level = 0.5
    p.specular_tint = vec3s(1.0)
    p.sheen_roughness = 0.5
    p.sheen_tint = vec3s(1.0)
    p.coat_roughness = 0.03
    p.coat_ior = 1.5
    p.coat_tint = vec3s(1.0)
    p.subsurface_radius = wp.vec3(1.0, 0.2, 0.1)
    p.subsurface_scale = 0.05
    return p


@wp.func
def blur_roughness(blur_glossy: float, min_ray_pdf: float) -> float:
    """Filter Glossy (surface_shader_prepare_closures). blur_glossy = scene cycles.blur_glossy (C3: 0.3);
    min_ray_pdf = min of the MIS'd BSDF pdfs of all previous bounces (start at 3.4e38 for camera rays)."""
    if blur_glossy == 0.0:
        return 0.0
    blur_pdf = (1.0 / blur_glossy) * min_ray_pdf
    if blur_pdf < 1.0:
        return wp.sqrt(1.0 - blur_pdf) * 0.5
    return 0.0


@wp.func
def default_tangent(ns: wp.vec3, generated: wp.vec3, normal_to_world: wp.mat33) -> wp.vec3:
    """Tangent Cycles feeds an unlinked Principled Tangent socket (Geometry.Tangent, primitive_tangent):
    radial around the object's Z axis. generated = texture-space coordinate (co - texspace_location) * 0.5 /
    texspace_size + 0.5; normal_to_world = transpose(inverse(object-to-world 3x3))."""
    data = normal_to_world * wp.vec3(-(generated[1] - 0.5), generated[0] - 0.5, 0.0)
    return wp.cross(ns, wp.normalize(wp.cross(data, ns)))


@wp.func
def _mf_of(c: PrincipledClosures, j: int) -> Microfacet:
    if j == 1:
        return c.coat
    if j == 2:
        return c.metal
    return c.spec


@wp.func
def lobe_active(c: PrincipledClosures, j: int) -> int:
    if j == 0:
        return c.sheen.active
    if j == LOBE_DIFFUSE:
        return c.diffuse.active
    if j == LOBE_GLASS:
        return c.glass.active
    if j == LOBE_SSS:
        return c.sss.active
    if j == LOBE_SSS_DIFFUSE:
        return c.sss_diffuse.active
    return _mf_of(c, j).active


@wp.func
def lobe_sample_weight(c: PrincipledClosures, j: int) -> float:
    if j == 0:
        return c.sheen.sample_weight
    if j == LOBE_DIFFUSE:
        return c.diffuse.sample_weight
    if j == LOBE_GLASS:
        return c.glass.sample_weight
    if j == LOBE_SSS:
        return c.sss.sample_weight
    if j == LOBE_SSS_DIFFUSE:
        return c.sss_diffuse.sample_weight
    return _mf_of(c, j).sample_weight


@wp.func
def lobe_weight(c: PrincipledClosures, j: int) -> wp.vec3:
    if j == 0:
        return c.sheen.weight
    if j == LOBE_DIFFUSE:
        return c.diffuse.weight
    if j == LOBE_GLASS:
        return c.glass.weight
    if j == LOBE_SSS:
        return c.sss.weight
    if j == LOBE_SSS_DIFFUSE:
        return c.sss_diffuse.weight
    return _mf_of(c, j).weight


@wp.func
def lobe_normal(c: PrincipledClosures, j: int) -> wp.vec3:
    if j == 0:
        return c.sheen.n
    if j == LOBE_DIFFUSE:
        return c.diffuse.n
    if j == LOBE_GLASS:
        return c.glass.n
    if j == LOBE_SSS:
        return c.sss.n
    if j == LOBE_SSS_DIFFUSE:
        return c.sss_diffuse.n
    return _mf_of(c, j).n


@wp.func
def _bump(c: PrincipledClosures, n: wp.vec3, wo: wp.vec3) -> float:
    """bsdf_eval / bsdf_sample: diffuse-class closures whose N differs from sd->N get the bump shadowing term."""
    if is_equal3(n, c.ns):
        return 1.0
    return bump_shadowing_term(c.bump_correction, c.ns, n, wo)


@wp.func
def lobe_eval(c: PrincipledClosures, j: int, wi: wp.vec3, wo: wp.vec3) -> LobeEval:
    """bsdf_eval of closure j (weight not applied)."""
    if j == 0:
        e = sheen_eval(c.sheen, wo)
        e.f = e.f * _bump(c, c.sheen.n, wo)
        return e
    if j == LOBE_DIFFUSE:
        e = diffuse_eval(c.diffuse, wo)
        e.f = e.f * _bump(c, c.diffuse.n, wo)
        return e
    if j == LOBE_GLASS:
        return glass_eval(c.glass, c.ns, wi, wo)
    if j == LOBE_SSS:  # a BSSRDF has no BSDF value; it only takes part in the closure pick
        e = LobeEval()
        return e
    if j == LOBE_SSS_DIFFUSE:
        e = diffuse_eval(c.sss_diffuse, wo)
        e.f = e.f * _bump(c, c.sss_diffuse.n, wo)
        return e
    return mf_eval(_mf_of(c, j), c.ns, wi, wo)


@wp.func
def lobe_sample(c: PrincipledClosures, j: int, wi: wp.vec3, u: wp.vec3) -> LobeSample:
    """bsdf_sample of closure j (weight not applied)."""
    if j == 0:
        s = sheen_sample(c.sheen, c.ng, wp.vec2(u[0], u[1]))
        s.f = s.f * _bump(c, c.sheen.n, s.wo)
        return s
    if j == LOBE_DIFFUSE:
        s = diffuse_sample(c.diffuse, c.ng, wp.vec2(u[0], u[1]))
        s.f = s.f * _bump(c, c.diffuse.n, s.wo)
        return s
    if j == LOBE_GLASS:
        return glass_sample(c.glass, c.ng, wi, u)
    if j == LOBE_SSS_DIFFUSE:
        s = diffuse_sample(c.sss_diffuse, c.ng, wp.vec2(u[0], u[1]))
        s.f = s.f * _bump(c, c.sss_diffuse.n, s.wo)
        return s
    return mf_sample(_mf_of(c, j), c.ng, wi, u)


@wp.func
def closures_eval(c: PrincipledClosures, wi: wp.vec3, wo: wp.vec3) -> BsdfEval:
    """surface_shader_bsdf_eval: sum of all closures, balance-heuristic pdf over closures."""
    r = BsdfEval()
    r.f = vec3s(0.0)
    sum_pdf = float(0.0)
    sum_sw = float(0.0)
    for j in range(N_LOBES):
        if lobe_active(c, j) == 1:
            sw = lobe_sample_weight(c, j)
            e = lobe_eval(c, j, wi, wo)
            if e.pdf != 0.0:
                r.f = r.f + wp.cw_mul(lobe_weight(c, j), e.f)
                sum_pdf = sum_pdf + e.pdf * sw
            sum_sw = sum_sw + sw
    r.pdf = 0.0
    if sum_sw > 0.0:
        r.pdf = sum_pdf / sum_sw
    return r


@wp.func
def closures_sample(c: PrincipledClosures, wi: wp.vec3, u: wp.vec3) -> BsdfSample:
    """surface_shader_bsdf_bssrdf_pick + surface_shader_bsdf_sample_closure. u in [0,1)^3; u[2] picks the closure
    and is then rescaled and reused (as Cycles)."""
    r = BsdfSample()
    r.weight = vec3s(0.0)
    r.label = LABEL_NONE
    total_sw = float(0.0)
    for j in range(N_LOBES):
        if lobe_active(c, j) == 1:
            total_sw = total_sw + lobe_sample_weight(c, j)
    if total_sw <= 0.0:
        return r
    x = u[2] * total_sw
    partial = float(0.0)
    pick = int(-1)
    uz = u[2]
    for j in range(N_LOBES):
        if lobe_active(c, j) == 1 and pick < 0:
            sw = lobe_sample_weight(c, j)
            if x < partial + sw:
                pick = j
                uz = (x - partial) / sw
            partial = partial + sw
    if pick < 0:
        return r
    if pick == LOBE_SSS:  # subsurface_bounce: the caller walks; weight = surface_shader_bssrdf_sample_weight
        r.weight = c.sss.weight * (total_sw / c.sss.sample_weight)
        r.pdf = 1.0
        r.label = LABEL_SUBSURFACE_SCATTER
        return r
    s = lobe_sample(c, pick, wi, wp.vec3(u[0], u[1], uz))
    r.wo = s.wo
    if s.pdf == 0.0:
        return r
    f = wp.cw_mul(lobe_weight(c, pick), s.f)
    pdf = s.pdf * lobe_sample_weight(c, pick)
    for j in range(N_LOBES):
        if j != pick and lobe_active(c, j) == 1:
            e = lobe_eval(c, j, wi, s.wo)
            if e.pdf != 0.0:
                f = f + wp.cw_mul(lobe_weight(c, j), e.f)
                pdf = pdf + e.pdf * lobe_sample_weight(c, j)
    pdf = pdf / total_sw
    if f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0:
        return r
    r.pdf = pdf
    r.weight = f / pdf
    r.label = s.label
    return r


@wp.func
def closures_albedo(lut: wp.array(dtype=float), c: PrincipledClosures, wi: wp.vec3) -> DenoiseAlbedo:
    """film_write_denoising_features_surface: split closure albedo into diffuse-like and specular-like."""
    r = DenoiseAlbedo()
    r.diffuse = vec3s(0.0)
    r.specular = vec3s(0.0)
    r.normal = vec3s(0.0)
    sum_w = float(0.0)
    sum_nonspec = float(0.0)
    for j in range(N_LOBES):
        if lobe_active(c, j) == 1:
            sw = lobe_sample_weight(c, j)
            r.normal = r.normal + lobe_normal(c, j) * sw
            sum_w = sum_w + sw
            albedo = lobe_weight(c, j)
            rough_sq = float(1.0)
            if j == LOBE_GLASS:
                g = c.glass
                albedo = wp.cw_mul(g.weight, glass_estimate_albedo(lut, g, wp.dot(wi, g.n), 1, 1))
                rough_sq = g.alpha * g.alpha
            elif j >= 1 and j <= LOBE_SPEC:
                m = _mf_of(c, j)
                albedo = wp.cw_mul(m.weight, mf_estimate_albedo(lut, m, wp.dot(wi, m.n)))
                rough_sq = m.alpha_x * m.alpha_y
            if rough_sq > 0.075 * 0.075:
                r.diffuse = r.diffuse + albedo
                sum_nonspec = sum_nonspec + sw
            else:
                r.specular = r.specular + albedo
    r.done = 0
    if sum_w == 0.0 or sum_nonspec * 4.0 > sum_w:
        r.done = 1
    if r.done == 1 and sum_w != 0.0:
        r.normal = r.normal / sum_w
    return r


@wp.func
def closures_label(c: PrincipledClosures, j: int) -> int:
    if j == 0 or j == LOBE_DIFFUSE or j == LOBE_SSS_DIFFUSE:
        return 2 | LABEL_DIFFUSE
    if j == LOBE_SSS:
        return LABEL_SUBSURFACE_SCATTER
    if j == LOBE_GLASS:
        return glass_label(c.glass)
    return mf_label(_mf_of(c, j))


# ---------------------------------------------------------------- one-call API (setup is redone per call)
@wp.func
def principled_eval(lut: wp.array(dtype=float), p: PrincipledParams, n: wp.vec3, t: wp.vec3, wi: wp.vec3,
                    wo: wp.vec3, path_flags: int) -> BsdfEval:
    return closures_eval(principled_setup(lut, p, n, t, wi, path_flags), wi, wo)


@wp.func
def principled_sample(lut: wp.array(dtype=float), p: PrincipledParams, n: wp.vec3, t: wp.vec3, wi: wp.vec3,
                      u: wp.vec3, path_flags: int) -> BsdfSample:
    return closures_sample(principled_setup(lut, p, n, t, wi, path_flags), wi, u)


@wp.func
def principled_albedo(lut: wp.array(dtype=float), p: PrincipledParams, n: wp.vec3, wi: wp.vec3,
                      path_flags: int) -> DenoiseAlbedo:
    return closures_albedo(lut, principled_setup(lut, p, n, vec3s(0.0), wi, path_flags), wi)
