# SPDX-License-Identifier: Apache-2.0
"""Principled BSDF closure setup, layer by layer.

Port of kernel/svm/closure.h (CLOSURE_BSDF_PRINCIPLED_ID), tag v4.5.0: sheen, coat, metal, transmission (glass),
specular, diffuse,
then the filter-glossy blur of surface_shader.h. Split out of principled.py; each layer helper takes the closures
and the running layering weight and returns both updated.
"""
import warp as wp

from .bssrdf import bssrdf_new, bssrdf_setup
from .glass import glass_new, glass_setup
from .lobes import Diffuse, Sheen, sheen_setup
from .microfacet import (FRESNEL_DIELECTRIC, FRESNEL_F82_TINT, FRESNEL_GENERALIZED_SCHLICK, Microfacet,
                         mf_estimate_albedo, mf_finish_setup)
from .principled_structs import (PATH_DIFFUSE, PATH_DIFFUSE_ANCESTOR, PATH_NO_REFLECTIVE_CAUSTICS,  # noqa: F401
                                 PATH_NO_REFRACTIVE_CAUSTICS, PrincipledClosures, PrincipledParams)
from .util import (CLOSURE_WEIGHT_CUTOFF, F0_from_ior, average3, closure_layering_weight, fresnel_f82tint_B,
                   ior_from_F0, is_equal3, maybe_ensure_valid_specular_reflection, mix3, rotate_around_axis,
                   safe_normalize, safe_normalize_fallback, saturate3, vec3s)


@wp.func
def _new_lobe(w_in: wp.vec3) -> Microfacet:
    """bsdf_alloc + bsdf_microfacet_ggx_setup defaults. active = 0 below CLOSURE_WEIGHT_CUTOFF."""
    m = Microfacet()
    m.weight = wp.max(w_in, vec3s(0.0))
    m.sample_weight = wp.abs(average3(m.weight))
    m.active = 0
    if m.sample_weight >= CLOSURE_WEIGHT_CUTOFF:
        m.active = 1
    m.energy_scale = 1.0
    return m


@wp.func
def _layer_albedo(lut: wp.array(dtype=float), m: Microfacet, wi: wp.vec3) -> wp.vec3:
    """bsdf_albedo(sc, reflection=true, transmission=false) of a microfacet closure."""
    return wp.cw_mul(m.weight, mf_estimate_albedo(lut, m, wp.dot(wi, m.n)))


@wp.func
def _blur(m_in: Microfacet, r: float) -> Microfacet:
    m = m_in
    m.alpha_x = wp.max(r, m.alpha_x)
    m.alpha_y = wp.max(r, m.alpha_y)
    return m


@wp.func
def _setup_sheen(lut: wp.array(dtype=float), p: PrincipledParams, c_in: PrincipledClosures, n: wp.vec3,
                 coat_n: wp.vec3, wi: wp.vec3, weight_in: wp.vec3,
                 coat_weight: float):
    """First layer: sheen."""
    c = c_in
    weight = weight_in
    sheen_weight = wp.max(p.sheen_weight, 0.0)
    if sheen_weight > CLOSURE_WEIGHT_CUTOFF:
        s = Sheen()
        s.weight = wp.max(wp.cw_mul(sheen_weight * wp.max(p.sheen_tint, vec3s(0.0)), weight), vec3s(0.0))
        s.sample_weight = wp.abs(average3(s.weight))
        if s.sample_weight >= CLOSURE_WEIGHT_CUTOFF:
            s.active = 1
            s.n = safe_normalize(mix3(n, coat_n, wp.clamp(coat_weight, 0.0, 1.0)))
            s.roughness = wp.clamp(p.sheen_roughness, 0.0, 1.0)
            s = sheen_setup(lut, s, wi)
            if s.active == 1:
                weight = closure_layering_weight(s.weight, weight)
            c.sheen = s
    return c, weight


@wp.func
def _coat_lobe_config(lut: wp.array(dtype=float), m_in: Microfacet, coat_n: wp.vec3, coat_ior: float,
                      coat_roughness: float, wi: wp.vec3) -> Microfacet:
    """Coat lobe parameters (dielectric Fresnel, squared roughness) and bsdf setup."""
    m = m_in
    m.n = coat_n
    m.ior = coat_ior
    cr = wp.clamp(coat_roughness, 0.0, 1.0)
    m.alpha_x = wp.clamp(cr * cr, 0.0, 1.0)
    m.alpha_y = m.alpha_x
    m.fresnel_type = FRESNEL_DIELECTRIC
    return mf_finish_setup(lut, m, wi, 1)


@wp.func
def _coat_tint_weight(p: PrincipledParams, coat_n: wp.vec3, coat_ior: float, wi: wp.vec3, weight: wp.vec3,
                      coat_weight: float) -> wp.vec3:
    """Absorption of the coat tint over the refracted path length (Cycles coat tint)."""
    coat_tint = wp.max(p.coat_tint, vec3s(0.0))
    if is_equal3(coat_tint, vec3s(1.0)):
        return weight
    cos_ni = wp.dot(wi, coat_n)
    cos_nt = wp.sqrt(1.0 - (1.0 / coat_ior) * (1.0 / coat_ior) * (1.0 - cos_ni * cos_ni))
    depth = 1.0 / cos_nt
    tint = wp.vec3(wp.pow(coat_tint[0], depth), wp.pow(coat_tint[1], depth), wp.pow(coat_tint[2], depth))
    return wp.cw_mul(weight, mix3(vec3s(1.0), tint, coat_weight))


@wp.func
def _setup_metal(lut: wp.array(dtype=float), c_in: PrincipledClosures, wi: wp.vec3, valid_n: wp.vec3, t: wp.vec3,
                 weight_in: wp.vec3, base: wp.vec3, spec_tint: wp.vec3, alpha_x: float, alpha_y: float,
                 metallic: float, multi: int, refl_caustics: bool):
    """Metallic component (F82-tint)."""
    c = c_in
    weight = weight_in
    if metallic > CLOSURE_WEIGHT_CUTOFF:
        if refl_caustics:
            m = _new_lobe(metallic * weight)
            if m.active == 1:
                m.n = valid_n
                m.ior = 1.0
                m.t = t
                m.alpha_x = wp.clamp(alpha_x, 0.0, 1.0)
                m.alpha_y = wp.clamp(alpha_y, 0.0, 1.0)
                m.fresnel_type = FRESNEL_F82_TINT
                m.f0 = wp.min(base, vec3s(1.0))
                f82 = wp.min(spec_tint, vec3s(1.0))
                if not is_equal3(f82, vec3s(1.0)):
                    m.b = fresnel_f82tint_B(m.f0, f82)
                m = mf_finish_setup(lut, m, wi, multi)
                c.metal = m
        weight = weight * (1.0 - metallic)
    return c, weight


@wp.func
def _specular_eta(spec_level_in: float, ior: float) -> float:
    """The Specular IOR Level adjustment of the IOR (used by the specular layer and the subsurface entry)."""
    eta = ior
    spec_level = wp.max(spec_level_in, 0.0)
    if spec_level != 0.5:
        eta = ior_from_F0(F0_from_ior(ior) * 2.0 * spec_level)
        if ior < 1.0:
            eta = 1.0 / eta
    return eta


@wp.func
def _setup_transmission(lut: wp.array(dtype=float), p: PrincipledParams, c_in: PrincipledClosures, wi: wp.vec3,
                        valid_n: wp.vec3, weight_in: wp.vec3, base: wp.vec3, spec_tint: wp.vec3, roughness: float,
                        ior: float, multi: int, refl_caustics: bool, refr_caustics: bool):
    """Transmission component: a GGX glass closure (generalized Schlick Fresnel, transmission tint sqrt(base))."""
    c = c_in
    weight = weight_in
    tw = wp.clamp(p.transmission_weight, 0.0, 1.0)
    if tw > CLOSURE_WEIGHT_CUTOFF:
        if refl_caustics or refr_caustics:
            g = glass_new(tw * weight)
            if g.active == 1:
                rel_ior = wp.where(p.backfacing != 0, 1.0 / ior, ior)
                cb = wp.min(base, vec3s(1.0))
                rt = wp.where(refl_caustics, vec3s(1.0), vec3s(0.0))
                tt = wp.where(refr_caustics, wp.vec3(wp.sqrt(cb[0]), wp.sqrt(cb[1]), wp.sqrt(cb[2])), vec3s(0.0))
                c.glass = glass_setup(lut, g, valid_n, roughness * roughness, rel_ior, F0_from_ior(ior) * spec_tint, rt,
                                      tt, wi, multi)
        weight = weight * (1.0 - tw)
    return c, weight


@wp.func
def _setup_specular(lut: wp.array(dtype=float), p: PrincipledParams, c_in: PrincipledClosures, wi: wp.vec3,
                    valid_n: wp.vec3, t: wp.vec3, weight_in: wp.vec3, spec_tint: wp.vec3, alpha_x: float,
                    alpha_y: float, ior: float, multi: int,
                    refl_caustics: bool):
    """Specular IOR level adjustment and dielectric specular layer."""
    c = c_in
    weight = weight_in
    eta = _specular_eta(p.specular_ior_level, ior)
    f0 = F0_from_ior(ior)
    if wp.max(p.specular_ior_level, 0.0) != 0.5:
        f0 = f0 * 2.0 * wp.max(p.specular_ior_level, 0.0)
    if refl_caustics and eta != 1.0:
        m = _new_lobe(weight)
        if m.active == 1:
            m.n = valid_n
            m.ior = eta
            m.t = t
            m.alpha_x = wp.clamp(alpha_x, 0.0, 1.0)
            m.alpha_y = wp.clamp(alpha_y, 0.0, 1.0)
            m.fresnel_type = FRESNEL_GENERALIZED_SCHLICK
            m.f0 = saturate3(f0 * spec_tint)
            m = mf_finish_setup(lut, m, wi, multi)
            weight = closure_layering_weight(_layer_albedo(lut, m, wi), weight)
            c.spec = m
    return c, weight


@wp.func
def _setup_diffuse(c_in: PrincipledClosures, p: PrincipledParams, base: wp.vec3, weight: wp.vec3, n: wp.vec3,
                   valid_n: wp.vec3, roughness: float, eta: float, ancestor: int) -> PrincipledClosures:
    """Diffuse/subsurface component: a random-walk BSSRDF of weight clamped_base * subsurface_weight (falling back
    to diffuse after a diffuse bounce), then Lambert diffuse of weight base * (1 - subsurface_weight) (diffuse
    roughness 0)."""
    c = c_in
    sw = wp.clamp(p.subsurface_weight, 0.0, 1.0)
    cb = wp.min(base, vec3s(1.0))
    b = bssrdf_new(wp.cw_mul(cb, weight) * sw)
    radius = p.subsurface_radius * p.subsurface_scale
    sss, sss_diffuse = bssrdf_setup(b, valid_n, cb, radius, roughness * roughness, eta, p.subsurface_anisotropy,
                                    ancestor)
    c.sss = sss
    c.sss_diffuse = sss_diffuse
    d = Diffuse()
    d.weight = wp.max(wp.cw_mul(base, weight) * (1.0 - sw), vec3s(0.0))
    d.sample_weight = wp.abs(average3(d.weight))
    d.active = 0
    if d.sample_weight >= CLOSURE_WEIGHT_CUTOFF:
        d.active = 1
    d.n = n
    c.diffuse = d
    return c


@wp.func
def _anisotropic_alpha(roughness: float, anisotropic: float, rotation: float, t_in: wp.vec3, n: wp.vec3):
    """GGX alphas from roughness and the anisotropy stretch; the tangent is rotated by anisotropic_rotation.
    Returns (alpha_x, alpha_y, t)."""
    t = t_in
    alpha_x = roughness * roughness
    alpha_y = alpha_x
    if anisotropic > 0.0:
        aspect = wp.sqrt(1.0 - anisotropic * 0.9)
        alpha_x = alpha_x / aspect
        alpha_y = alpha_y * aspect
        if rotation != 0.0:
            t = rotate_around_axis(t, n, rotation * 6.28318530717958647692)
    return alpha_x, alpha_y, t


@wp.func
def principled_setup(lut: wp.array(dtype=float), p: PrincipledParams, n_in: wp.vec3, t_in: wp.vec3, wi: wp.vec3,
                     path_flags: int) -> PrincipledClosures:
    """svm_node_closure_bsdf (Principled) + bsdf setups + filter-glossy blur. n_in = Normal socket (zero =
    unlinked = ns), t_in = Tangent socket (default_tangent when unlinked; only used if anisotropic > 0)."""
    c = PrincipledClosures()
    c.ns = p.ns
    c.ng = p.ng
    c.bump_correction = 1 - p.no_bump_correction
    n = safe_normalize_fallback(n_in, p.ns)
    metallic = wp.clamp(p.metallic, 0.0, 1.0)
    roughness = wp.clamp(p.roughness, 0.0, 1.0)
    spec_tint = wp.max(p.specular_tint, vec3s(0.0))
    anisotropic = wp.clamp(p.anisotropic, 0.0, 1.0)
    coat_weight = wp.max(p.coat_weight, 0.0)
    ior = wp.max(p.ior, 1e-5)
    multi = 1 - p.single_scatter_ggx
    t = t_in
    valid_n = maybe_ensure_valid_specular_reflection(p.ng, wi, n, c.bump_correction)
    coat_n = safe_normalize_fallback(p.coat_normal, p.ns)
    base = wp.max(p.base_color, vec3s(0.0))
    weight = vec3s(1.0)
    alpha_x, alpha_y, t = _anisotropic_alpha(roughness, anisotropic, p.anisotropic_rotation, t, n)
    refl_caustics = (path_flags & PATH_NO_REFLECTIVE_CAUSTICS) == 0 or (path_flags & PATH_DIFFUSE) == 0
    refr_caustics = (path_flags & PATH_NO_REFRACTIVE_CAUSTICS) == 0 or (path_flags & PATH_DIFFUSE) == 0

    c, weight = _setup_sheen(lut, p, c, n, coat_n, wi, weight, coat_weight)
    if coat_weight > CLOSURE_WEIGHT_CUTOFF:
        coat_n = maybe_ensure_valid_specular_reflection(p.ng, wi, coat_n, c.bump_correction)
        coat_ior = wp.max(p.coat_ior, 1.0)
        if refl_caustics:
            m = _new_lobe(coat_weight * weight)
            if m.active == 1:
                m = _coat_lobe_config(lut, m, coat_n, coat_ior, p.coat_roughness, wi)
                weight = closure_layering_weight(_layer_albedo(lut, m, wi), weight)
                c.coat = m
        weight = _coat_tint_weight(p, coat_n, coat_ior, wi, weight, coat_weight)
    c, weight = _setup_metal(lut, c, wi, valid_n, t, weight, base, spec_tint, alpha_x, alpha_y, metallic, multi,
                             refl_caustics)
    c, weight = _setup_transmission(lut, p, c, wi, valid_n, weight, base, spec_tint, roughness, ior, multi,
                                    refl_caustics, refr_caustics)
    c, weight = _setup_specular(lut, p, c, wi, valid_n, t, weight, spec_tint, alpha_x, alpha_y, ior, multi,
                                refl_caustics)
    ancestor = wp.where((path_flags & PATH_DIFFUSE_ANCESTOR) != 0, 1, 0)
    c = _setup_diffuse(c, p, base, weight, n, valid_n, roughness, _specular_eta(p.specular_ior_level, ior), ancestor)

    # Filter glossy: blur after setup (energy scale and sample weights keep the unblurred roughness, as Cycles)
    if p.blur_roughness > 0.0:
        c.coat = _blur(c.coat, p.blur_roughness)
        c.metal = _blur(c.metal, p.blur_roughness)
        c.spec = _blur(c.spec, p.blur_roughness)
        c.glass.alpha = wp.max(p.blur_roughness, c.glass.alpha)
    return c
