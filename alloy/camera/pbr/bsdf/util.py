# SPDX-License-Identifier: Apache-2.0
"""Shared Warp helpers for the Cycles 4.5 Principled BSDF port (tag v4.5.0, commit 8cb6b388).

Ports, line for line where it matters:
  util/math_float3.h (make_orthonormals, rotate_around_axis, safe_normalize*), kernel/sample/mapping.h
  (sample_uniform_disk, make_orthonormals_tangent / _safe_tangent), kernel/util/lookup_table.h (lookup_table_read*),
  kernel/closure/bsdf_util.h (fresnel_dielectric, F0/ior, Fss fits, F82 B, closure_layering_weight,
  ensure_valid_specular_reflection), kernel/closure/bsdf.h (bump_shadowing_term).
"""
from pathlib import Path

import numpy as np
import warp as wp

M_PI = wp.constant(3.14159265358979323846)
M_1_PI = wp.constant(0.31830988618379067154)
M_PI_2 = wp.constant(1.57079632679489661923)
M_PI_4 = wp.constant(0.78539816339744830962)
CLOSURE_WEIGHT_CUTOFF = wp.constant(1e-5)
BSDF_ROUGHNESS_SQ_THRESH = wp.constant(2e-10)

# Offsets into the single flat LUT array (same packing idea as Cycles' kernel_data.tables).
# The tables are scene/shader.tables of Cycles 4.5 (Apache-2.0; source_sha256 in the npz is that file's hash).
_TABLE_ORDER = ("ggx_E", "ggx_Eavg", "sheen_ltc", "ggx_gen_schlick_ior_s", "ggx_gen_schlick_s", "ggx_glass_E",
                "ggx_glass_Eavg", "ggx_glass_inv_E", "ggx_glass_inv_Eavg")
OFS_GGX_E = wp.constant(0)
OFS_GGX_EAVG = wp.constant(1024)
OFS_SHEEN_LTC = wp.constant(1056)
OFS_GEN_SCHLICK_IOR_S = wp.constant(4128)
OFS_GEN_SCHLICK_S = wp.constant(8224)
OFS_GGX_GLASS_E = wp.constant(12320)
OFS_GGX_GLASS_EAVG = wp.constant(16416)
OFS_GGX_GLASS_INV_E = wp.constant(16672)
OFS_GGX_GLASS_INV_EAVG = wp.constant(20768)
LUT_SIZE = 21024


def load_tables(device=None) -> wp.array:
    """The Cycles BSDF lookup tables as one flat float32 Warp array (pass it as `lut` to the library)."""
    data = np.load(Path(__file__).with_name("cycles_tables.npz"))
    flat = np.concatenate([data[k] for k in _TABLE_ORDER]).astype(np.float32)
    assert flat.size == LUT_SIZE
    return wp.array(flat, dtype=float, device=device)


# ---------------------------------------------------------------- lookup tables (kernel/util/lookup_table.h)
@wp.func
def lut_read(lut: wp.array(dtype=float), x_in: float, offset: int, size: int) -> float:
    x = wp.clamp(x_in, 0.0, 1.0) * float(size - 1)
    index = wp.min(int(x), size - 1)
    nindex = wp.min(index + 1, size - 1)
    t = x - float(index)
    data0 = lut[index + offset]
    if t == 0.0:
        return data0
    data1 = lut[nindex + offset]
    return (1.0 - t) * data0 + t * data1


@wp.func
def lut_read_2d(lut: wp.array(dtype=float), x: float, y_in: float, offset: int, xsize: int, ysize: int) -> float:
    y = wp.clamp(y_in, 0.0, 1.0) * float(ysize - 1)
    index = wp.min(int(y), ysize - 1)
    nindex = wp.min(index + 1, ysize - 1)
    t = y - float(index)
    data0 = lut_read(lut, x, offset + xsize * index, xsize)
    if t == 0.0:
        return data0
    data1 = lut_read(lut, x, offset + xsize * nindex, xsize)
    return (1.0 - t) * data0 + t * data1


@wp.func
def lut_read_3d(lut: wp.array(dtype=float), x: float, y: float, z_in: float, offset: int, xsize: int, ysize: int,
                zsize: int) -> float:
    z = wp.clamp(z_in, 0.0, 1.0) * float(zsize - 1)
    index = wp.min(int(z), zsize - 1)
    nindex = wp.min(index + 1, zsize - 1)
    t = z - float(index)
    data0 = lut_read_2d(lut, x, y, offset + xsize * ysize * index, xsize, ysize)
    if t == 0.0:
        return data0
    data1 = lut_read_2d(lut, x, y, offset + xsize * ysize * nindex, xsize, ysize)
    return (1.0 - t) * data0 + t * data1


# ---------------------------------------------------------------- small vector helpers
@wp.func
def vec3s(s: float) -> wp.vec3:
    return wp.vec3(s, s, s)


@wp.func
def saturate3(v: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.clamp(v[0], 0.0, 1.0), wp.clamp(v[1], 0.0, 1.0), wp.clamp(v[2], 0.0, 1.0))


@wp.func
def average3(v: wp.vec3) -> float:
    return (v[0] + v[1] + v[2]) * (1.0 / 3.0)


@wp.func
def reduce_max3(v: wp.vec3) -> float:
    return wp.max(v[0], wp.max(v[1], v[2]))


@wp.func
def is_zero3(v: wp.vec3) -> bool:
    return v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0


@wp.func
def is_equal3(a: wp.vec3, b: wp.vec3) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def mix3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:
    return a + t * (b - a)


@wp.func
def safe_sqrt(x: float) -> float:
    return wp.sqrt(wp.max(x, 0.0))


@wp.func
def safe_normalize(a: wp.vec3) -> wp.vec3:
    t = wp.length(a)
    if t != 0.0:
        return a * (1.0 / t)
    return a


@wp.func
def safe_normalize_fallback(a: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    t = wp.length(a)
    if t != 0.0:
        return a * (1.0 / t)
    return fallback


@wp.func
def safe_divide_color(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    r = wp.vec3(0.0, 0.0, 0.0)
    for i in range(3):
        if b[i] != 0.0:
            r[i] = a[i] / b[i]
    return r


# ---------------------------------------------------------------- bases and mappings
@wp.func
def make_orthonormals_a(n: wp.vec3) -> wp.vec3:
    """First vector of Cycles' make_orthonormals(N, &a, &b); b = cross(N, a)."""
    a = wp.vec3(n[2] - n[1], n[0] + n[2], -n[1] - n[0])
    if n[0] != n[1] or n[0] != n[2]:
        a = wp.vec3(n[2] - n[1], n[0] - n[2], n[1] - n[0])
    return wp.normalize(a)


@wp.func
def make_orthonormals_tangent_a(n: wp.vec3, t: wp.vec3) -> wp.vec3:
    """make_orthonormals_tangent: b = normalize(cross(N, T)), a = cross(b, N). Returns a; b = cross(N, a)."""
    b = wp.normalize(wp.cross(n, t))
    return wp.cross(b, n)


@wp.func
def rotate_around_axis(p: wp.vec3, axis: wp.vec3, angle: float) -> wp.vec3:
    c = wp.cos(angle)
    s = wp.sin(angle)
    x = ((c + (1.0 - c) * axis[0] * axis[0]) * p[0]) + (((1.0 - c) * axis[0] * axis[1] - axis[2] * s) * p[1]) + \
        (((1.0 - c) * axis[0] * axis[2] + axis[1] * s) * p[2])
    y = (((1.0 - c) * axis[0] * axis[1] + axis[2] * s) * p[0]) + ((c + (1.0 - c) * axis[1] * axis[1]) * p[1]) + \
        (((1.0 - c) * axis[1] * axis[2] - axis[0] * s) * p[2])
    z = (((1.0 - c) * axis[0] * axis[2] - axis[1] * s) * p[0]) + (((1.0 - c) * axis[1] * axis[2] + axis[0] * s) *
                                                                  p[1]) + ((c + (1.0 - c) * axis[2] * axis[2]) * p[2])
    return wp.vec3(x, y, z)


@wp.func
def sample_uniform_disk(u: wp.vec2) -> wp.vec2:
    """Shirley-Chiu concentric mapping, as kernel/sample/mapping.h."""
    a = 2.0 * u[0] - 1.0
    b = 2.0 * u[1] - 1.0
    if a == 0.0 and b == 0.0:
        return wp.vec2(0.0, 0.0)
    r = b
    phi = M_PI_2 - M_PI_4 * (a / b)
    if a * a > b * b:
        r = a
        phi = M_PI_4 * (b / a)
    return wp.vec2(r * wp.cos(phi), r * wp.sin(phi))


# ---------------------------------------------------------------- Fresnel (kernel/closure/bsdf_util.h)
@wp.func
def fresnel_dielectric(cos_theta_i_in: float, eta: float) -> float:
    eta_cos_theta_t_sq = eta * eta - (1.0 - cos_theta_i_in * cos_theta_i_in)
    if eta_cos_theta_t_sq <= 0.0:
        return 1.0
    cos_theta_i = wp.abs(cos_theta_i_in)
    cos_theta_t = -safe_sqrt(eta_cos_theta_t_sq) / eta
    r_s = (cos_theta_i + eta * cos_theta_t) / (cos_theta_i - eta * cos_theta_t)
    r_p = (cos_theta_t + eta * cos_theta_i) / (eta * cos_theta_i - cos_theta_t)
    return 0.5 * (r_s * r_s + r_p * r_p)


@wp.func
def fresnel_dielectric_Fss(eta: float) -> float:
    if eta < 1.0:
        return 0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607))
    return (eta - 1.0) / (4.08567 + 1.00071 * eta)


@wp.func
def F0_from_ior(ior: float) -> float:
    r = (ior - 1.0) / (ior + 1.0)
    return r * r


@wp.func
def ior_from_F0(f0: float) -> float:
    sqrt_f0 = wp.sqrt(wp.clamp(f0, 0.0, 0.99))
    return (1.0 + sqrt_f0) / (1.0 - sqrt_f0)


@wp.func
def fresnel_f82_Fss(f0: wp.vec3, b: wp.vec3) -> wp.vec3:
    return mix3(f0, vec3s(1.0), 1.0 / 21.0) - b * (1.0 / 126.0)


@wp.func
def fresnel_f82tint_B(f0: wp.vec3, tint: wp.vec3) -> wp.vec3:
    f = 6.0 / 7.0
    f5 = (f * f) * (f * f) * f
    f_schlick = mix3(f0, vec3s(1.0), f5)
    return wp.cw_mul(f_schlick * (7.0 / (f5 * f)), vec3s(1.0) - tint)


@wp.func
def closure_layering_weight(layer_albedo: wp.vec3, weight: wp.vec3) -> wp.vec3:
    return weight * wp.clamp(1.0 - reduce_max3(safe_divide_color(layer_albedo, weight)), 0.0, 1.0)


# ---------------------------------------------------------------- shading-normal corrections
@wp.func
def ensure_valid_specular_reflection(ng: wp.vec3, wi: wp.vec3, n: wp.vec3) -> wp.vec3:
    r = 2.0 * wp.dot(n, wi) * n - wi
    iz = wp.dot(wi, ng)
    threshold = wp.min(0.9 * iz, 0.01)
    if wp.dot(ng, r) >= threshold:
        return n
    x = safe_normalize_fallback(n - wp.dot(n, ng) * ng, n)
    ix = wp.dot(wi, x)
    a = ix * ix + iz * iz
    b = 2.0 * (a + iz * threshold)
    c = (threshold + iz) * (threshold + iz)
    nz2 = 0.25 * (b - safe_sqrt(b * b - 4.0 * a * c)) / a
    if ix < 0.0:
        nz2 = 0.25 * (b + safe_sqrt(b * b - 4.0 * a * c)) / a
    return safe_sqrt(1.0 - nz2) * x + safe_sqrt(nz2) * ng


@wp.func
def maybe_ensure_valid_specular_reflection(ng: wp.vec3, wi: wp.vec3, n: wp.vec3, bump_correction: int) -> wp.vec3:
    """Triangles only (no curves in C3). Active when the material's use_bump_map_correction is on (default)."""
    if bump_correction == 0 or is_equal3(ng, n):
        return n
    return ensure_valid_specular_reflection(ng, wi, n)


@wp.func
def ggx_lambda(alpha2: float, cos_n: float) -> float:
    """bsdf_lambda<GGX>: Smith Lambda, isotropic."""
    return 0.5 * (wp.sqrt(1.0 + alpha2 * wp.max(1.0 / (cos_n * cos_n) - 1.0, 0.0)) - 1.0)


@wp.func
def bump_shadowing_term(bump_correction: int, ng: wp.vec3, n: wp.vec3, i: wp.vec3) -> float:
    """kernel/closure/bsdf.h bump_shadowing_term; Cycles calls it with ng = sd->N (smooth normal), n = closure N."""
    cos_ngi = wp.dot(ng, i)
    cos_ngn = wp.dot(ng, n)
    cos_ni = wp.dot(n, i)
    if cos_ngi * cos_ngn * cos_ni < 0.0:
        return 0.0
    if bump_correction == 0:
        return 1.0
    cos_i = wp.abs(cos_ngi)
    cos_d = wp.abs(cos_ngn)
    if cos_d >= 1.0 or cos_i >= 1.0:
        return 1.0
    if cos_i < 1e-6:
        return 0.0
    tan2_d = 1.0 / (cos_d * cos_d) - 1.0
    bump_alpha2 = wp.clamp(0.125 * tan2_d, 0.0, 1.0)
    return 1.0 / (1.0 + ggx_lambda(bump_alpha2, cos_i))
