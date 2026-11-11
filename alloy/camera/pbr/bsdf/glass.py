# SPDX-License-Identifier: Apache-2.0
"""GGX glass lobe of Cycles 4.5 (kernel/closure/bsdf_microfacet.h, tag v4.5.0): the Principled BSDF's transmission
component (CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID with FRESNEL_GENERALIZED_SCHLICK, exponent < 0).

Reflection and refraction through a rough (or, below BSDF_ROUGHNESS_SQ_THRESH, smooth) dielectric boundary, the
reflect/refract choice made by the Fresnel energy split, multiscatter energy compensation from the glass tables
(``ggx_glass_E`` / ``ggx_glass_inv_E`` for rays leaving the denser medium). ``ior`` is relative: the material's IOR
on a front-facing hit, 1 / IOR on a back-facing one (the ray leaves the object). Conventions as ``microfacet.py``:
wi and the normals face the viewer; f includes the cosine; pdf per solid angle.

Example (inside a Warp function)::

    g = glass_setup(lut, glass_new(weight), n, alpha, ior, f0, refl_tint, trans_tint, wi, 1)
    s = glass_sample(g, ng, wi, u)          # s.label has LABEL_TRANSMIT when it refracted
"""
import warp as wp

from .microfacet import (LABEL_GLOSSY, LABEL_NONE, LABEL_REFLECT, LABEL_SINGULAR, LobeEval, LobeSample, ggx_D,
                         ggx_sample_vndf)
from .util import (BSDF_ROUGHNESS_SQ_THRESH, CLOSURE_WEIGHT_CUTOFF, OFS_GEN_SCHLICK_IOR_S, OFS_GGX_GLASS_E,
                   OFS_GGX_GLASS_EAVG, OFS_GGX_GLASS_INV_E, OFS_GGX_GLASS_INV_EAVG, F0_from_ior, average3,
                   fresnel_dielectric_Fss, ggx_lambda, is_equal3, is_zero3, lut_read_2d, lut_read_3d,
                   make_orthonormals_a, mix3, safe_sqrt, saturate3, vec3s)

LABEL_TRANSMIT = wp.constant(1)


@wp.struct
class Glass:
    active: int
    weight: wp.vec3
    sample_weight: float
    n: wp.vec3
    alpha: float          # isotropic GGX alpha (roughness^2); glass ignores anisotropy, as Cycles
    ior: float            # relative IOR across the boundary (1 / IOR from inside)
    energy_scale: float
    f0: wp.vec3           # F0_from_ior(IOR) * specular tint, saturated
    refl_tint: wp.vec3    # 1, or 0 without reflective caustics
    trans_tint: wp.vec3   # sqrt(base colour), or 0 without refractive caustics


@wp.struct
class GlassFresnel:
    reflectance: wp.vec3
    transmittance: wp.vec3
    cos_t: float          # cosine of the refracted ray with the normal (negative: other side); 0 on TIR


@wp.func
def glass_new(w_in: wp.vec3) -> Glass:
    """bsdf_alloc + bsdf_microfacet_ggx_glass_setup defaults; active = 0 below CLOSURE_WEIGHT_CUTOFF."""
    g = Glass()
    g.weight = wp.max(w_in, vec3s(0.0))
    g.sample_weight = wp.abs(average3(g.weight))
    g.active = wp.where(g.sample_weight >= CLOSURE_WEIGHT_CUTOFF, 1, 0)
    g.energy_scale = 1.0
    return g


@wp.func
def glass_eval_flag(g: Glass) -> bool:
    return g.alpha * g.alpha > BSDF_ROUGHNESS_SQ_THRESH


@wp.func
def fresnel_dielectric_t(cos_theta_i_in: float, eta: float) -> wp.vec2:
    """(F, cos_theta_t) of bsdf_util.h fresnel_dielectric; cos_theta_t = 0 on total internal reflection."""
    eta_cos_theta_t_sq = eta * eta - (1.0 - cos_theta_i_in * cos_theta_i_in)
    if eta_cos_theta_t_sq <= 0.0:
        return wp.vec2(1.0, 0.0)
    cos_theta_i = wp.abs(cos_theta_i_in)
    cos_theta_t = -safe_sqrt(eta_cos_theta_t_sq) / eta
    r_s = (cos_theta_i + eta * cos_theta_t) / (cos_theta_i - eta * cos_theta_t)
    r_p = (cos_theta_t + eta * cos_theta_i) / (eta * cos_theta_i - cos_theta_t)
    return wp.vec2(0.5 * (r_s * r_s + r_p * r_p), cos_theta_t)


@wp.func
def glass_fresnel(g: Glass, cos_theta_i: float) -> GlassFresnel:
    """microfacet_fresnel, generalized Schlick with exponent < 0: the real dielectric curve remapped to f0..1."""
    r = GlassFresnel()
    ft = fresnel_dielectric_t(cos_theta_i, g.ior)
    f0_real = F0_from_ior(g.ior)
    s = wp.clamp((ft[0] - f0_real) / (1.0 - f0_real), 0.0, 1.0)
    f = mix3(g.f0, vec3s(1.0), s)
    r.reflectance = wp.cw_mul(f, g.refl_tint)
    r.transmittance = wp.cw_mul(vec3s(1.0) - f, g.trans_tint)
    r.cos_t = ft[1]
    return r


@wp.func
def glass_estimate_albedo(lut: wp.array(dtype=float), g: Glass, cos_ni: float, reflection: int,
                          transmission: int) -> wp.vec3:
    """bsdf_microfacet_estimate_albedo: the reflection part from the generalized-Schlick table, the transmission
    part from the Fresnel at the incident angle (weight not included)."""
    fr = glass_fresnel(g, cos_ni)
    refl = fr.reflectance * float(reflection)
    trans = fr.transmittance * float(transmission)
    if not is_zero3(refl):
        rough = wp.sqrt(g.alpha)
        z = wp.sqrt(wp.abs((g.ior - 1.0) / (g.ior + 1.0)))
        s = lut_read_3d(lut, rough, cos_ni, z, OFS_GEN_SCHLICK_IOR_S, 16, 16, 16)
        refl = wp.cw_mul(mix3(g.f0, vec3s(1.0), s), g.refl_tint)
    return refl + trans


@wp.func
def glass_setup(lut: wp.array(dtype=float), g_in: Glass, n: wp.vec3, alpha: float, ior: float, f0: wp.vec3,
                refl_tint: wp.vec3, trans_tint: wp.vec3, wi: wp.vec3, preserve_energy: int) -> Glass:
    """bsdf_microfacet_ggx_glass_setup + bsdf_microfacet_setup_fresnel_generalized_schlick (+ preserve energy)."""
    g = g_in
    g.n = n
    g.alpha = wp.clamp(alpha, 0.0, 1.0)
    g.ior = ior
    g.f0 = saturate3(f0)
    g.refl_tint = refl_tint
    g.trans_tint = trans_tint
    cos_ni = wp.dot(wi, n)
    g.sample_weight = g.sample_weight * average3(glass_estimate_albedo(lut, g, cos_ni, 1, 1))
    if preserve_energy == 0:
        return g
    fss = trans_tint
    if is_zero3(trans_tint):
        f0r = F0_from_ior(ior)
        s = wp.clamp((fresnel_dielectric_Fss(ior) - f0r) / (1.0 - f0r), 0.0, 1.0)
        fss = wp.cw_mul(refl_tint, mix3(g.f0, vec3s(1.0), s))
    return _preserve_energy(lut, g, cos_ni, fss)


@wp.func
def _preserve_energy(lut: wp.array(dtype=float), g_in: Glass, mu: float, fss: wp.vec3) -> Glass:
    """microfacet_ggx_preserve_energy for CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID."""
    g = g_in
    rough = wp.sqrt(g.alpha)
    ior = g.ior
    ofs = OFS_GGX_GLASS_E
    avg_ofs = OFS_GGX_GLASS_EAVG
    if ior < 1.0:
        ior = 1.0 / ior
        ofs = OFS_GGX_GLASS_INV_E
        avg_ofs = OFS_GGX_GLASS_INV_EAVG
    z = wp.sqrt(wp.abs((ior - 1.0) / (ior + 1.0)))
    e = lut_read_3d(lut, rough, mu, z, ofs, 16, 16, 16)
    e_avg = lut_read_2d(lut, rough, z, avg_ofs, 16, 16)
    missing_factor = (1.0 - e) / e
    g.energy_scale = 1.0 + missing_factor
    if not is_equal3(fss, vec3s(1.0)):
        fms = wp.cw_div(fss * e_avg, vec3s(1.0) - fss * (1.0 - e_avg))
        darkening = (vec3s(1.0) + fms * missing_factor) / g.energy_scale
        g.weight = wp.cw_mul(g.weight, darkening)
        g.sample_weight = g.sample_weight * average3(darkening)
    return g


@wp.func
def glass_eval(g: Glass, ns: wp.vec3, wi: wp.vec3, wo: wp.vec3) -> LobeEval:
    """bsdf_microfacet_ggx_eval for the glass closure (Cycles passes sd->N as the hemisphere normal)."""
    r = LobeEval()
    r.f = vec3s(0.0)
    r.pdf = 0.0
    n = g.n
    cos_ni = wp.dot(n, wi)
    cos_no = wp.dot(n, wo)
    is_trans = cos_no < 0.0
    if cos_ni <= 0.0 or (not glass_eval_flag(g)) or ((wp.dot(ns, wo) < 0.0) != is_trans):
        return r
    h = wi + wo
    if is_trans:
        h = -(g.ior * wo + wi)
    lh = wp.length(h)
    inv_len_h = wp.where(lh != 0.0, 1.0 / lh, 0.0)
    h = h * inv_len_h
    cos_hi = wp.dot(h, wi)
    fr = glass_fresnel(g, cos_hi)
    if is_zero3(fr.reflectance) and is_zero3(fr.transmittance):
        return r
    alpha2 = g.alpha * g.alpha
    d = ggx_D(alpha2, wp.dot(n, h))
    lambda_i = ggx_lambda(alpha2, cos_ni)
    lambda_o = ggx_lambda(alpha2, cos_no)
    common = d / cos_ni * 0.25
    if is_trans:
        common = d / cos_ni * (g.ior * inv_len_h) * (g.ior * inv_len_h) * wp.abs(cos_hi * wp.dot(h, wo))
    pdf_reflect = average3(fr.reflectance) / average3(fr.reflectance + fr.transmittance)
    lobe_pdf = wp.where(is_trans, 1.0 - pdf_reflect, pdf_reflect)
    r.pdf = common * lobe_pdf / (1.0 + lambda_i)
    f = wp.where(is_trans, fr.transmittance, fr.reflectance)
    r.f = f * (common / (1.0 + lambda_o + lambda_i) * g.energy_scale)
    return r


@wp.func
def glass_sample(g: Glass, ng: wp.vec3, wi: wp.vec3, u: wp.vec3) -> LobeSample:
    """bsdf_microfacet_ggx_sample for the glass closure: VNDF half vector, then reflect or refract by the Fresnel
    energy split (u[2]). Cycles uses the true geometric normal sd->Ng for the side test here."""
    r = LobeSample()
    r.wo = vec3s(0.0)
    r.f = vec3s(0.0)
    r.pdf = 0.0
    r.label = LABEL_NONE
    n = g.n
    cos_ni = wp.dot(n, wi)
    if cos_ni <= 0.0:
        return r
    singular = not glass_eval_flag(g)
    h = n
    local_h = wp.vec3(0.0, 0.0, 1.0)
    if not singular:
        x = make_orthonormals_a(n)
        y = wp.cross(n, x)
        local_i = wp.vec3(wp.dot(x, wi), wp.dot(y, wi), cos_ni)
        local_h = ggx_sample_vndf(local_i, g.alpha, g.alpha, wp.vec2(u[0], u[1]))
        h = local_h[0] * x + local_h[1] * y + local_h[2] * n
    cos_hi = wp.dot(h, wi)
    fr = glass_fresnel(g, cos_hi)
    if is_zero3(fr.reflectance) and is_zero3(fr.transmittance):
        return r
    pdf_reflect = average3(fr.reflectance) / average3(fr.reflectance + fr.transmittance)
    do_refract = u[2] >= pdf_reflect
    inv_eta = wp.where(g.ior != 0.0, 1.0 / g.ior, 0.0)
    cos_ho = fr.cos_t
    wo = 2.0 * cos_hi * h - wi
    if do_refract:
        wo = (inv_eta * wp.dot(h, wi) + cos_ho) * h - inv_eta * wi
    if (wp.dot(ng, wo) < 0.0) != do_refract:
        return r
    r.wo = wo
    return _glass_sample_tail(g, r, fr, do_refract, singular, pdf_reflect, cos_ni, cos_hi, cos_ho, inv_eta,
                              local_h[2], wo)


@wp.func
def _glass_sample_tail(g: Glass, r_in: LobeSample, fr: GlassFresnel, do_refract: bool, singular_in: bool,
                       pdf_reflect: float, cos_ni: float, cos_hi: float, cos_ho: float, inv_eta: float,
                       cos_nh: float, wo: wp.vec3) -> LobeSample:
    """f, pdf and label of a sampled glass direction (bsdf_microfacet_sample after the direction)."""
    r = r_in
    f = fr.reflectance
    pdf = pdf_reflect
    singular = singular_in
    if do_refract:
        f = fr.transmittance
        pdf = 1.0 - pdf_reflect
        singular = singular or wp.abs(g.ior - 1.0) < 1e-4
    if singular:
        pdf = pdf * 1e6
        f = f * 1e6
    else:
        alpha2 = g.alpha * g.alpha
        d = ggx_D(alpha2, cos_nh)
        lambda_o = ggx_lambda(alpha2, wp.dot(g.n, wo))
        lambda_i = ggx_lambda(alpha2, cos_ni)
        common = d / cos_ni * 0.25
        if do_refract:
            q = cos_ho + cos_hi * inv_eta
            common = d / cos_ni * wp.abs(cos_hi * cos_ho) / (q * q)
        pdf = pdf * (common / (1.0 + lambda_i))
        f = f * (common / (1.0 + lambda_i + lambda_o))
    r.f = f * g.energy_scale
    r.pdf = pdf
    r.label = wp.where(do_refract, LABEL_TRANSMIT, LABEL_REFLECT) | wp.where(singular, LABEL_SINGULAR, LABEL_GLOSSY)
    return r


@wp.func
def glass_label(g: Glass) -> int:
    return LABEL_REFLECT | LABEL_TRANSMIT | wp.where(glass_eval_flag(g), LABEL_GLOSSY, LABEL_SINGULAR)
