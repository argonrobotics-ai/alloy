# SPDX-License-Identifier: Apache-2.0
"""GGX microfacet lobe of Cycles 4.5 (kernel/closure/bsdf_microfacet.h, tag v4.5.0), reflection only.

The Principled BSDF of C3 uses three GGX lobes, each with its own Fresnel:
  coat      FRESNEL_DIELECTRIC        (bsdf_microfacet_setup_fresnel_dielectric, always energy-preserving)
  metal     FRESNEL_F82_TINT          (bsdf_microfacet_setup_fresnel_f82_tint, energy-preserving if MULTI_GGX)
  specular  FRESNEL_GENERALIZED_SCHLICK with exponent < 0, f90 = 1, no thin film
            (bsdf_microfacet_setup_fresnel_generalized_schlick, energy-preserving if MULTI_GGX)
Transmission (glass, refraction) is not ported: C3 is opaque.
"""
import warp as wp

from .util import (BSDF_ROUGHNESS_SQ_THRESH, F0_from_ior, M_1_PI, OFS_GEN_SCHLICK_IOR_S, OFS_GEN_SCHLICK_S,
                       OFS_GGX_E, OFS_GGX_EAVG, average3, fresnel_dielectric, fresnel_dielectric_Fss,
                       fresnel_f82_Fss, ggx_lambda, is_equal3, is_zero3, lut_read, lut_read_2d, lut_read_3d,
                       make_orthonormals_a, make_orthonormals_tangent_a, mix3, safe_sqrt, sample_uniform_disk,
                       saturate3, vec3s)

FRESNEL_DIELECTRIC = wp.constant(1)
FRESNEL_GENERALIZED_SCHLICK = wp.constant(4)
FRESNEL_F82_TINT = wp.constant(5)

LABEL_NONE = wp.constant(0)
LABEL_REFLECT = wp.constant(2)
LABEL_DIFFUSE = wp.constant(4)
LABEL_GLOSSY = wp.constant(8)
LABEL_SINGULAR = wp.constant(16)


@wp.struct
class Microfacet:
    active: int           # 0: closure not allocated (weight below CLOSURE_WEIGHT_CUTOFF or caustics rule)
    weight: wp.vec3       # sc->weight (includes the multi-bounce Fresnel darkening)
    sample_weight: float  # sc->sample_weight
    n: wp.vec3
    t: wp.vec3
    alpha_x: float
    alpha_y: float
    ior: float
    energy_scale: float   # 1/E(rough, mu): multiscatter GGX compensation
    fresnel_type: int
    f0: wp.vec3           # F82: base colour; generalized Schlick: f0 * specular tint (saturated)
    b: wp.vec3            # F82 tint B term


@wp.struct
class LobeEval:
    f: wp.vec3     # BSDF * cos(theta_o), Cycles convention (weight NOT applied)
    pdf: float


@wp.struct
class LobeSample:
    wo: wp.vec3
    f: wp.vec3     # BSDF * cos(theta_o) of the sampled lobe (weight NOT applied)
    pdf: float     # lobe pdf (solid angle)
    label: int


@wp.func
def mf_eval_flag(m: Microfacet) -> bool:
    return m.alpha_x * m.alpha_y > BSDF_ROUGHNESS_SQ_THRESH


@wp.func
def mf_fresnel(m: Microfacet, cos_theta_i: float) -> wp.vec3:
    """microfacet_fresnel, reflectance only (transmittance is zero for reflection-only closures)."""
    if m.fresnel_type == FRESNEL_DIELECTRIC:
        return vec3s(fresnel_dielectric(cos_theta_i, m.ior))
    if m.fresnel_type == FRESNEL_F82_TINT:
        mu = wp.clamp(1.0 - cos_theta_i, 0.0, 1.0)
        mu5 = (mu * mu) * (mu * mu) * mu
        f_schlick = mix3(m.f0, vec3s(1.0), mu5)
        return saturate3(f_schlick - m.b * (cos_theta_i * mu5 * mu))
    # Generalized Schlick, exponent < 0: real Fresnel remapped from F0_real..1 to f0..f90 (= 1).
    f_real = fresnel_dielectric(cos_theta_i, m.ior)
    f0_real = F0_from_ior(m.ior)
    s = wp.clamp((f_real - f0_real) / (1.0 - f0_real), 0.0, 1.0)
    return mix3(m.f0, vec3s(1.0), s)


@wp.func
def mf_estimate_albedo(lut: wp.array(dtype=float), m: Microfacet, cos_ni: float) -> wp.vec3:
    """bsdf_microfacet_estimate_albedo(reflection=true, transmission=true), NOT including the closure weight."""
    reflectance = mf_fresnel(m, cos_ni)
    if is_zero3(reflectance):
        return reflectance
    rough = wp.sqrt(wp.sqrt(m.alpha_x * m.alpha_y))
    if m.fresnel_type == FRESNEL_GENERALIZED_SCHLICK:
        z = wp.sqrt(wp.abs((m.ior - 1.0) / (m.ior + 1.0)))
        s = lut_read_3d(lut, rough, cos_ni, z, OFS_GEN_SCHLICK_IOR_S, 16, 16, 16)
        return mix3(m.f0, vec3s(1.0), s)
    if m.fresnel_type == FRESNEL_F82_TINT:
        s = lut_read_3d(lut, rough, cos_ni, 0.5, OFS_GEN_SCHLICK_S, 16, 16, 16)
        return mix3(m.f0, vec3s(1.0), s)
    if m.ior > 1.0:
        z = wp.sqrt(wp.abs((m.ior - 1.0) / (m.ior + 1.0)))
        s = lut_read_3d(lut, rough, cos_ni, z, OFS_GEN_SCHLICK_IOR_S, 16, 16, 16)
        return vec3s(F0_from_ior(m.ior) + s * (1.0 - F0_from_ior(m.ior)))
    return reflectance


@wp.func
def mf_fss(m: Microfacet) -> wp.vec3:
    """The single-scattering Fresnel average each setup_fresnel_* passes to microfacet_ggx_preserve_energy."""
    if m.fresnel_type == FRESNEL_DIELECTRIC:
        return vec3s(fresnel_dielectric_Fss(m.ior))
    if m.fresnel_type == FRESNEL_F82_TINT:
        return fresnel_f82_Fss(m.f0, m.b)
    f0r = F0_from_ior(m.ior)
    s = wp.clamp((fresnel_dielectric_Fss(m.ior) - f0r) / (1.0 - f0r), 0.0, 1.0)
    return mix3(m.f0, vec3s(1.0), s)


@wp.func
def mf_finish_setup(lut: wp.array(dtype=float), m_in: Microfacet, wi: wp.vec3, preserve_energy: int) -> Microfacet:
    """Tail of bsdf_microfacet_setup_fresnel_*: albedo-scaled sample weight, then microfacet_ggx_preserve_energy."""
    m = m_in
    cos_ni = wp.dot(wi, m.n)
    m.sample_weight = m.sample_weight * average3(mf_estimate_albedo(lut, m, cos_ni))
    if preserve_energy == 0:
        return m
    rough = wp.sqrt(wp.sqrt(m.alpha_x * m.alpha_y))
    e = lut_read_2d(lut, rough, cos_ni, OFS_GGX_E, 32, 32)
    e_avg = lut_read(lut, rough, OFS_GGX_EAVG, 32)
    missing_factor = (1.0 - e) / e
    m.energy_scale = 1.0 + missing_factor
    fss = mf_fss(m)
    if not is_equal3(fss, vec3s(1.0)):
        fms = wp.cw_div(fss * e_avg, vec3s(1.0) - fss * (1.0 - e_avg))
        darkening = (vec3s(1.0) + fms * missing_factor) / m.energy_scale
        m.weight = wp.cw_mul(m.weight, darkening)
        m.sample_weight = m.sample_weight * average3(darkening)
    return m


@wp.func
def ggx_D(alpha2: float, cos_nh: float) -> float:
    cos_nh2 = wp.min(cos_nh * cos_nh, 1.0)
    one_minus = 1.0 - cos_nh2
    d = one_minus + alpha2 * cos_nh2
    return alpha2 / (3.14159265358979323846 * d * d)


@wp.func
def ggx_aniso_D(alpha_x: float, alpha_y: float, h_in: wp.vec3) -> float:
    h = wp.vec3(h_in[0] / alpha_x, h_in[1] / alpha_y, h_in[2])
    l2 = wp.dot(h, h)
    return M_1_PI / (alpha_x * alpha_y * l2 * l2)


@wp.func
def ggx_aniso_lambda(alpha_x: float, alpha_y: float, v: wp.vec3) -> float:
    ax = alpha_x * v[0]
    ay = alpha_y * v[1]
    sqr_alpha_tan_n = (ax * ax + ay * ay) / (v[2] * v[2])
    return 0.5 * (wp.sqrt(1.0 + sqr_alpha_tan_n) - 1.0)


@wp.func
def mf_eval(m: Microfacet, ns: wp.vec3, wi: wp.vec3, wo: wp.vec3) -> LobeEval:
    """bsdf_microfacet_ggx_eval. Cycles passes sd->N (smooth normal) as the 'Ng' of the hemisphere test here."""
    r = LobeEval()
    r.f = vec3s(0.0)
    r.pdf = 0.0
    n = m.n
    cos_ni = wp.dot(n, wi)
    cos_no = wp.dot(n, wo)
    cos_ngo = wp.dot(ns, wo)
    if cos_ni <= 0.0 or (not mf_eval_flag(m)) or cos_no < 0.0 or cos_ngo < 0.0:
        return r
    h = wi + wo
    lh = wp.length(h)
    inv_len_h = 0.0
    if lh != 0.0:
        inv_len_h = 1.0 / lh
    h = h * inv_len_h
    cos_hi = wp.dot(h, wi)
    refl = mf_fresnel(m, cos_hi)
    if is_zero3(refl):
        return r
    cos_nh = wp.dot(n, h)
    d = float(0.0)
    lambda_i = float(0.0)
    lambda_o = float(0.0)
    if m.alpha_x == m.alpha_y:
        alpha2 = m.alpha_x * m.alpha_y
        d = ggx_D(alpha2, cos_nh)
        lambda_i = ggx_lambda(alpha2, cos_ni)
        lambda_o = ggx_lambda(alpha2, cos_no)
    else:
        x = make_orthonormals_tangent_a(n, m.t)
        y = wp.cross(n, x)
        d = ggx_aniso_D(m.alpha_x, m.alpha_y, wp.vec3(wp.dot(x, h), wp.dot(y, h), cos_nh))
        lambda_i = ggx_aniso_lambda(m.alpha_x, m.alpha_y, wp.vec3(wp.dot(x, wi), wp.dot(y, wi), cos_ni))
        lambda_o = ggx_aniso_lambda(m.alpha_x, m.alpha_y, wp.vec3(wp.dot(x, wo), wp.dot(y, wo), cos_no))
    common = d / cos_ni * 0.25
    r.pdf = common / (1.0 + lambda_i)
    r.f = refl * (common / (1.0 + lambda_o + lambda_i) * m.energy_scale)
    return r


@wp.func
def ggx_sample_vndf(wi: wp.vec3, alpha_x: float, alpha_y: float, u: wp.vec2) -> wp.vec3:
    """microfacet_ggx_sample_vndf (Heitz 2018), local frame."""
    wi_ = wp.normalize(wp.vec3(alpha_x * wi[0], alpha_y * wi[1], wi[2]))
    lensq = wi_[0] * wi_[0] + wi_[1] * wi_[1]
    t1 = wp.vec3(1.0, 0.0, 0.0)
    t2 = wp.vec3(0.0, 1.0, 0.0)
    if lensq > 1e-7:
        t1 = wp.vec3(-wi_[1], wi_[0], 0.0) * (1.0 / wp.sqrt(lensq))
        t2 = wp.cross(wi_, t1)
    t = sample_uniform_disk(u)
    s = safe_sqrt(1.0 - t[0] * t[0])
    ty = s + (0.5 * (1.0 + wi_[2])) * (t[1] - s)
    hz = safe_sqrt(1.0 - (t[0] * t[0] + ty * ty))
    h_ = t[0] * t1 + ty * t2 + hz * wi_
    return wp.normalize(wp.vec3(alpha_x * h_[0], alpha_y * h_[1], wp.max(0.0, h_[2])))


@wp.func
def _mf_sample_glossy(m: Microfacet, r_in: LobeSample, refl: wp.vec3, wo: wp.vec3, cos_ni: float, cos_hi: float,
                      local_h: wp.vec3, local_i: wp.vec3) -> LobeSample:
    """Glossy tail of mf_sample: pdf and f of the sampled half vector (isotropic or anisotropic GGX)."""
    r = r_in
    n = m.n
    d = float(0.0)
    lambda_i = float(0.0)
    lambda_o = float(0.0)
    if m.alpha_x == m.alpha_y:
        alpha2 = m.alpha_x * m.alpha_y
        d = ggx_D(alpha2, local_h[2])
        lambda_o = ggx_lambda(alpha2, wp.dot(n, wo))
        lambda_i = ggx_lambda(alpha2, cos_ni)
    else:
        local_o = 2.0 * cos_hi * local_h - local_i
        d = ggx_aniso_D(m.alpha_x, m.alpha_y, local_h)
        lambda_o = ggx_aniso_lambda(m.alpha_x, m.alpha_y, local_o)
        lambda_i = ggx_aniso_lambda(m.alpha_x, m.alpha_y, local_i)
    common = d / cos_ni * 0.25
    r.pdf = common / (1.0 + lambda_i)
    r.f = refl * (common / (1.0 + lambda_i + lambda_o) * m.energy_scale)
    r.label = LABEL_REFLECT | LABEL_GLOSSY
    return r


@wp.func
def mf_sample(m: Microfacet, ng: wp.vec3, wi: wp.vec3, u: wp.vec3) -> LobeSample:
    """bsdf_microfacet_ggx_sample, reflection only. Cycles uses the true geometric normal sd->Ng here."""
    r = LobeSample()
    r.wo = vec3s(0.0)
    r.f = vec3s(0.0)
    r.pdf = 0.0
    r.label = LABEL_NONE
    n = m.n
    cos_ni = wp.dot(n, wi)
    if cos_ni <= 0.0:
        return r
    singular = not mf_eval_flag(m)
    h = n
    local_h = wp.vec3(0.0, 0.0, 1.0)
    local_i = wp.vec3(0.0, 0.0, cos_ni)
    if not singular:
        x = wp.vec3(0.0, 0.0, 0.0)
        if m.alpha_x == m.alpha_y:
            x = make_orthonormals_a(n)
        else:
            x = make_orthonormals_tangent_a(n, m.t)
        y = wp.cross(n, x)
        local_i = wp.vec3(wp.dot(x, wi), wp.dot(y, wi), cos_ni)
        local_h = ggx_sample_vndf(local_i, m.alpha_x, m.alpha_y, wp.vec2(u[0], u[1]))
        h = local_h[0] * x + local_h[1] * y + local_h[2] * n
    cos_hi = wp.dot(h, wi)
    refl = mf_fresnel(m, cos_hi)
    if is_zero3(refl):
        return r
    wo = 2.0 * cos_hi * h - wi
    if wp.dot(ng, wo) < 0.0:
        return r
    r.wo = wo
    if singular:
        r.pdf = 1e6
        r.f = refl * (1e6 * m.energy_scale)
        r.label = LABEL_REFLECT | LABEL_SINGULAR
        return r
    return _mf_sample_glossy(m, r, refl, wo, cos_ni, cos_hi, local_h, local_i)


@wp.func
def mf_label(m: Microfacet) -> int:
    if mf_eval_flag(m):
        return LABEL_REFLECT | LABEL_GLOSSY
    return LABEL_REFLECT | LABEL_SINGULAR
