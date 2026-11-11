# SPDX-License-Identifier: Apache-2.0
"""Shared constants and math helpers, ported from Cycles 4.5.0 (util/math_*.h, kernel/sample/mapping.h)."""
import warp as wp

M_PI = wp.constant(3.14159265358979323846)
M_2PI = wp.constant(6.28318530717958647692)
M_1_PI = wp.constant(0.31830988618379067154)
M_1_2PI = wp.constant(0.15915494309189533577)
M_PI_2 = wp.constant(1.57079632679489661923)
M_PI_4 = wp.constant(0.78539816339744830962)
FLT_MAX = wp.constant(3.402823466e38)

# light kinds (emitters and LightSample.kind)
KIND_AREA = wp.constant(0)
KIND_BACKGROUND = wp.constant(1)
KIND_DISTANT = wp.constant(2)


@wp.func
def safe_sqrt(x: float):
    return wp.sqrt(wp.max(x, 0.0))


@wp.func
def sin_from_cos(c: float):
    return safe_sqrt(1.0 - c * c)


@wp.func
def safe_divide(a: float, b: float):
    if b != 0.0:
        return a / b
    return 0.0


@wp.func
def safe_asin(x: float):
    return wp.asin(wp.clamp(x, -1.0, 1.0))


@wp.func
def safe_acos(x: float):
    return wp.acos(wp.clamp(x, -1.0, 1.0))


@wp.func
def sqr(x: float):
    return x * x


@wp.func
def safe_normalize_len(v: wp.vec3):
    """Returns (unit vector, length); zero vector when the length is 0 (Cycles safe_normalize_len)."""
    l = wp.length(v)
    if l == 0.0:
        return wp.vec3(0.0, 0.0, 0.0), 0.0
    return v / l, l


@wp.func
def power_heuristic(a: float, b: float):
    """kernel/sample/mis.h power_heuristic (beta 2). Returns 0 for 0/0 (a == b == 0 would be NaN in Cycles)."""
    d = a * a + b * b
    if d == 0.0:
        return 0.0
    return (a * a) / d


@wp.func
def make_orthonormals(n: wp.vec3):
    """util/math_float3.h make_orthonormals: returns (a, b)."""
    a = wp.vec3(0.0)
    if n[0] != n[1] or n[0] != n[2]:
        a = wp.vec3(n[2] - n[1], n[0] - n[2], n[1] - n[0])
    else:
        a = wp.vec3(n[2] - n[1], n[0] + n[2], -n[1] - n[0])
    a = wp.normalize(a)
    b = wp.normalize(wp.cross(n, a))
    return a, b


@wp.func
def sample_uniform_disk(r: wp.vec2):
    """kernel/sample/mapping.h sample_uniform_disk: Shirley-Chiu concentric mapping, unit disk."""
    a = 2.0 * r[0] - 1.0
    b = 2.0 * r[1] - 1.0
    if a == 0.0 and b == 0.0:
        return wp.vec2(0.0, 0.0)
    rad = float(0.0)
    phi = float(0.0)
    if a * a > b * b:
        rad = a
        phi = M_PI_4 * (b / a)
    else:
        rad = b
        phi = M_PI_2 - M_PI_4 * (a / b)
    return wp.vec2(rad * wp.cos(phi), rad * wp.sin(phi))


@wp.func
def sample_uniform_cone(n: wp.vec3, one_minus_cos_angle: float, rand: wp.vec2):
    """mapping.h sample_uniform_cone. Returns (direction, pdf)."""
    if one_minus_cos_angle > 0.0:
        xy = sample_uniform_disk(rand)
        r2 = xy[0] * xy[0] + xy[1] * xy[1]
        cos_theta = 1.0 - r2 * one_minus_cos_angle
        xy = xy * safe_sqrt(one_minus_cos_angle * (2.0 - one_minus_cos_angle * r2))
        t, b = make_orthonormals(n)
        return xy[0] * t + xy[1] * b + cos_theta * n, M_1_2PI / one_minus_cos_angle
    return n, 1.0


@wp.struct
class LightSample:
    """Result of sampling or hitting a light. L is radiance (strength * eval_fac * emission), pdf is solid angle
    density INCLUDING the light-selection probability (Cycles `ls->pdf *= ls->pdf_selection`)."""
    valid: int
    kind: int        # KIND_*
    id: int          # index inside its kind (area light index; 0 for background; sun index)
    use_mis: int
    L: wp.vec3
    D: wp.vec3       # unit direction from the shading point towards the light
    t: float         # distance to the light point (FLT_MAX for distant / background)
    pdf: float
    pdf_sel: float
    P: wp.vec3
    Ng: wp.vec3
