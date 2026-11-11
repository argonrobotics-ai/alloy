# SPDX-License-Identifier: Apache-2.0
"""Sheen (microfiber LTC, kernel/closure/bsdf_sheen.h) and Lambert diffuse (bsdf_diffuse.h) of Cycles 4.5."""
import warp as wp

from .util import (M_1_PI, OFS_SHEEN_LTC, lut_read_2d, make_orthonormals_a, safe_normalize, safe_sqrt,
                       sample_uniform_disk, vec3s)
from .microfacet import LABEL_DIFFUSE, LABEL_REFLECT, LobeEval, LobeSample


@wp.struct
class Sheen:
    active: int
    weight: wp.vec3
    sample_weight: float
    n: wp.vec3
    t: wp.vec3
    b: wp.vec3
    roughness: float
    transform_a: float
    transform_b: float


@wp.struct
class Diffuse:
    active: int
    weight: wp.vec3
    sample_weight: float
    n: wp.vec3


@wp.func
def sheen_setup(lut: wp.array(dtype=float), s_in: Sheen, wi: wp.vec3) -> Sheen:
    """bsdf_sheen_setup: LTC lookup; scales weight and sample weight by the tabulated albedo.
    Sets active = 0 if the LTC is invalid (Cycles turns the closure into CLOSURE_NONE_ID)."""
    s = s_in
    s.roughness = wp.clamp(s.roughness, 1e-3, 1.0)
    # make_orthonormals_safe_tangent(N, wi, &T, &B)
    b = safe_normalize(wp.cross(s.n, wi))
    t = wp.cross(b, s.n)
    if wp.dot(b, b) < 0.99:
        t = make_orthonormals_a(s.n)
        b = wp.cross(s.n, t)
    s.t = t
    s.b = b
    cos_ni = wp.dot(s.n, wi)
    s.transform_a = lut_read_2d(lut, cos_ni, s.roughness, OFS_SHEEN_LTC, 32, 32)
    s.transform_b = lut_read_2d(lut, cos_ni, s.roughness, OFS_SHEEN_LTC + 32 * 32, 32, 32)
    albedo = lut_read_2d(lut, cos_ni, s.roughness, OFS_SHEEN_LTC + 2 * 32 * 32, 32, 32)
    if wp.abs(s.transform_a) < 1e-5 or albedo < 1e-5:
        s.active = 0
        s.sample_weight = 0.0
        return s
    s.weight = s.weight * albedo
    s.sample_weight = s.sample_weight * albedo
    return s


@wp.func
def sheen_ltc_value(s: Sheen, local_o: wp.vec3) -> float:
    a = s.transform_a
    b = s.transform_b
    p = a * local_o[0] + b * local_o[2]
    q = a * local_o[1]
    len_sqr = p * p + q * q + local_o[2] * local_o[2]
    r = a / len_sqr
    return M_1_PI * local_o[2] * r * r


@wp.func
def sheen_eval(s: Sheen, wo: wp.vec3) -> LobeEval:
    r = LobeEval()
    r.f = vec3s(0.0)
    r.pdf = 0.0
    if wp.dot(s.n, wo) <= 0.0:
        return r
    local_o = wp.vec3(wp.dot(wo, s.t), wp.dot(wo, s.b), wp.dot(wo, s.n))
    if local_o[2] <= 0.0:
        return r
    val = sheen_ltc_value(s, local_o)
    r.pdf = val
    r.f = vec3s(val)
    return r


@wp.func
def sheen_sample(s: Sheen, ng: wp.vec3, u: wp.vec2) -> LobeSample:
    r = LobeSample()
    disk = sample_uniform_disk(u)
    disk_z = safe_sqrt(1.0 - wp.dot(disk, disk))
    local_o = wp.normalize(wp.vec3(disk[0] - disk_z * s.transform_b, disk[1], disk_z * s.transform_a))
    r.wo = local_o[0] * s.t + local_o[1] * s.b + local_o[2] * s.n
    r.label = LABEL_REFLECT | LABEL_DIFFUSE
    r.f = vec3s(0.0)
    r.pdf = 0.0
    if wp.dot(ng, r.wo) <= 0.0:
        return r
    val = sheen_ltc_value(s, local_o)
    r.pdf = val
    r.f = vec3s(val)
    return r


@wp.func
def diffuse_eval(d: Diffuse, wo: wp.vec3) -> LobeEval:
    r = LobeEval()
    c = wp.max(wp.dot(d.n, wo), 0.0) * M_1_PI
    r.pdf = c
    r.f = vec3s(c)
    return r


@wp.func
def diffuse_sample(d: Diffuse, ng: wp.vec3, u: wp.vec2) -> LobeSample:
    """bsdf_diffuse_sample: sample_cos_hemisphere(N) with make_orthonormals(N); rejected below Ng."""
    r = LobeSample()
    disk = sample_uniform_disk(u)
    costheta = safe_sqrt(1.0 - wp.dot(disk, disk))
    t = make_orthonormals_a(d.n)
    b = wp.cross(d.n, t)
    r.wo = disk[0] * t + disk[1] * b + costheta * d.n
    r.label = LABEL_REFLECT | LABEL_DIFFUSE
    r.pdf = costheta * M_1_PI
    r.f = vec3s(r.pdf)
    if wp.dot(ng, r.wo) <= 0.0:
        r.pdf = 0.0
        r.f = vec3s(0.0)
    return r

