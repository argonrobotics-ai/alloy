# SPDX-License-Identifier: Apache-2.0
"""Rectangle area lights. Port of intern/cycles/kernel/light/area.h and the area branch of
scene/light.cpp (device_update_lights) at tag v4.5.0.

Radiance of a normalized area light: L = strength_rgb / (pi * area)  (strength = colour * power in W,
`eval_fac = M_1_PI_F * invarea` in area.h:area_light_eval). Spread 180 deg (Blender default, the C3 value)
sets tan_half_spread = FLT_MAX, which makes the spread attenuation 1 and the spread clamp a no-op; the general
spread < 180 path is ported for the rectangle case only (ellipse/disk shapes are not supported)."""
import numpy as np
import warp as wp

from .common import (FLT_MAX, M_1_PI, M_PI, M_PI_2, LightSample, KIND_AREA, safe_asin, safe_divide,
                           safe_normalize_len, safe_sqrt, sqr)


@wp.struct
class AreaLight:
    co: wp.vec3
    axis_u: wp.vec3        # unit
    axis_v: wp.vec3        # unit
    dir: wp.vec3           # unit, emitting side (-Z column of matrix_world)
    len_u: float           # full side length (m, with object scale)
    len_v: float
    invarea: float         # 1/area when normalize, else 1
    strength: wp.vec3      # colour * power [W] * 2^exposure
    tan_half_spread: float  # FLT_MAX for spread == pi
    normalize_spread: float
    use_mis: int
    normalized: int


@wp.func
def rotate_around_axis(p: wp.vec3, axis: wp.vec3, angle: float):
    costheta = wp.cos(angle)
    sintheta = wp.sin(angle)
    return (p * costheta + wp.cross(axis, p) * sintheta + axis * wp.dot(axis, p) * (1.0 - costheta))


@wp.func
def _rect_sample_point(P: wp.vec3, x: wp.vec3, y: wp.vec3, z: wp.vec3, z0: float, x0: float, x1: float, y0: float,
                       y1: float, nz0: float, nz2: float, S: float, g2: float, g3: float, z02: float,
                       rand: wp.vec2) -> wp.vec3:
    """Spherical rectangle sampling of the point on the light (area_light_rect_sample, sample_coord != 0)."""
    b0 = nz0
    b1 = nz2
    b0sq = b0 * b0
    au = rand[0] * S + g2 + g3
    fu = safe_divide(wp.cos(au) * b0 + b1, wp.sin(au))
    cu = 1.0 / wp.sqrt(fu * fu + b0sq)
    if fu < 0.0:
        cu = -cu
    cu = wp.clamp(cu, -1.0, 1.0)
    xu = -(cu * z0) / wp.max(wp.sqrt(1.0 - cu * cu), 1e-7)
    xu = wp.clamp(xu, x0, x1)
    d2 = xu * xu + z02
    h0 = y0 / wp.sqrt(d2 + y0 * y0)
    h1 = y1 / wp.sqrt(d2 + y1 * y1)
    hv = h0 + rand[1] * (h1 - h0)
    hv2 = hv * hv
    yv = y1
    if hv2 < 1.0 - 1e-6:
        yv = hv * wp.sqrt(d2 / (1.0 - hv2))
    return P + xu * x + yv * y + z0 * z


@wp.func
def area_light_rect_sample(P: wp.vec3, light_p_in: wp.vec3, axis_u: wp.vec3, len_u: float, axis_v: wp.vec3,
                           len_v: float, rand: wp.vec2, sample_coord: int):
    """area.h area_light_rect_sample (Ureña spherical rectangle sampling). Returns (pdf, light_p)."""
    x = axis_u
    y = axis_v
    z = wp.cross(x, y)
    d = light_p_in - P
    z0 = wp.dot(d, z)
    if z0 > 0.0:
        z = -z
        z0 = -z0
    xc = wp.dot(d, x)
    yc = wp.dot(d, y)
    x0 = xc - 0.5 * len_u
    x1 = xc + 0.5 * len_u
    y0 = yc - 0.5 * len_v
    y1 = yc + 0.5 * len_v
    nz0 = -y0
    nz1 = x1
    nz2 = y1
    nz3 = -x0
    z02 = z0 * z0
    nz0 = nz0 / wp.sqrt(nz0 * nz0 + z02)
    nz1 = nz1 / wp.sqrt(nz1 * nz1 + z02)
    nz2 = nz2 / wp.sqrt(nz2 * nz2 + z02)
    nz3 = nz3 / wp.sqrt(nz3 * nz3 + z02)
    g0 = safe_asin(-nz0 * nz1)
    g1 = safe_asin(-nz1 * nz2)
    g2 = safe_asin(-nz2 * nz3)
    g3 = safe_asin(-nz3 * nz0)
    S = -(g0 + g1 + g2 + g3)

    light_p = light_p_in
    if sample_coord != 0:
        light_p = _rect_sample_point(P, x, y, z, z0, x0, x1, y0, y1, nz0, nz2, S, g2, g3, z02, rand)

    pdf = float(0.0)
    mn = wp.min(wp.min(nz0 * nz0, nz1 * nz1), wp.min(nz2 * nz2, nz3 * nz3))
    if S < 1e-5 or mn > 0.99999:
        # tiny solid angle: planar fallback, area.h:101-111
        t = wp.length(d)
        pdf = safe_divide(-t * t * t, z0 * len_u * len_v)
    else:
        pdf = 1.0 / S
    return pdf, light_p


@wp.func
def area_light_spread_attenuation(D: wp.vec3, ng: wp.vec3, tan_half_spread: float, normalize_spread: float):
    """area.h area_light_spread_attenuation. For spread == pi (tan_half_spread == FLT_MAX) this is 1."""
    if tan_half_spread >= 1.0e30:
        return 1.0
    c = wp.dot(-D, ng)
    tan_a = wp.length(wp.cross(-D, ng)) / c
    if tan_half_spread == 0.0:
        if tan_a > 1e-5:
            return 0.0
        return M_PI
    return wp.max((tan_half_spread - tan_a) * normalize_spread, 0.0)


@wp.func
def area_light_spread_clamp_rect(P: wp.vec3, ng: wp.vec3, light_p: wp.vec3, axis_u: wp.vec3, len_u: float,
                                 axis_v: wp.vec3, len_v: float, tan_half_spread: float):
    """Rectangle branch of area_light_spread_clamp_light (sample_rectangle == true). Returns
    (ok, light_p, len_u, len_v): the smaller rectangle covering the region that can reach P."""
    t = wp.dot(ng, P - light_p)
    closest = P - t * ng
    r_spread = t * tan_half_spread
    su = wp.dot(axis_u, closest - light_p)
    sv = wp.dot(axis_v, closest - light_p)
    min_u = wp.max(su - r_spread, -len_u * 0.5)
    max_u = wp.min(su + r_spread, len_u * 0.5)
    min_v = wp.max(sv - r_spread, -len_v * 0.5)
    max_v = wp.min(sv + r_spread, len_v * 0.5)
    if min_u >= max_u or min_v >= max_v:
        return 0, light_p, len_u, len_v
    nlp = light_p + axis_u * (0.5 * (min_u + max_u)) + axis_v * (0.5 * (min_v + max_v))
    return 1, nlp, max_u - min_u, max_v - min_v


@wp.func
def area_light_eval(l: AreaLight, ray_P: wp.vec3, rand: wp.vec2, sample_coord: int, D_in: wp.vec3):
    """area.h area_light_eval (rectangle, not in a volume). Returns (ok, pdf [solid angle, no selection],
    eval_fac, light_P, D, t). With sample_coord == 0 the direction D_in is used (hit by a BSDF ray)."""
    ng = l.dir
    light_p = l.co
    len_u = l.len_u
    len_v = l.len_v
    if l.tan_half_spread < 1.0e30:
        ok, lp2, lu2, lv2 = area_light_spread_clamp_rect(ray_P, ng, light_p, l.axis_u, len_u, l.axis_v, len_v,
                                                         l.tan_half_spread)
        if ok == 0:
            return 0, 0.0, 0.0, light_p, D_in, 0.0
        light_p = lp2
        len_u = lu2
        len_v = lv2
    pdf, lp = area_light_rect_sample(ray_P, light_p, l.axis_u, len_u, l.axis_v, len_v, rand, sample_coord)
    D = D_in
    t = float(0.0)
    if sample_coord != 0:
        D, t = safe_normalize_len(lp - ray_P)
    eval_fac = M_1_PI * l.invarea * area_light_spread_attenuation(D, ng, l.tan_half_spread, l.normalize_spread)
    ok2 = 0
    if eval_fac > 0.0:
        ok2 = 1
    return ok2, pdf, eval_fac, lp, D, t


@wp.func
def area_light_sample(l: AreaLight, P: wp.vec3, rand: wp.vec2):
    """NEE sample of one rectangle light from shading point P (area.h area_light_sample). pdf is solid angle,
    without selection probability. ls.L = strength * eval_fac."""
    ls = LightSample()
    ls.valid = 0
    ls.kind = KIND_AREA
    ls.Ng = l.dir
    if wp.dot(l.co - P, l.dir) > 0.0:
        return ls
    ok, pdf, ef, lp, D, t = area_light_eval(l, P, rand, 1, wp.vec3(0.0))
    if ok == 0:
        return ls
    if l.tan_half_spread < 1.0e30:
        # outside-of-light check (area.h:365-380), margin for the clamped sampler
        inpl = lp - l.co
        eu = (0.5 + 1e-7) * l.len_u + 1e-6
        ev = (0.5 + 1e-7) * l.len_v + 1e-6
        if wp.abs(wp.dot(inpl, l.axis_u)) > eu or wp.abs(wp.dot(inpl, l.axis_v)) > ev:
            return ls
    ls.P = lp
    ls.D = D
    ls.t = t
    ls.pdf = pdf
    ls.L = l.strength * ef
    ls.use_mis = l.use_mis
    if pdf > 0.0:
        ls.valid = 1
    return ls


@wp.func
def area_light_intersect(l: AreaLight, P: wp.vec3, D: wp.vec3, tmin: float, tmax: float):
    """One-sided ray / rectangle test (area.h area_light_intersect, ray_quad_intersect). Returns (hit, t)."""
    if wp.dot(D, l.dir) >= 0.0:
        return 0, 0.0
    denom = wp.dot(D, l.dir)
    t = wp.dot(l.co - P, l.dir) / denom
    if t <= tmin or t >= tmax:
        return 0, 0.0
    hp = P + D * t - l.co
    if wp.abs(wp.dot(hp, l.axis_u)) > 0.5 * l.len_u or wp.abs(wp.dot(hp, l.axis_v)) > 0.5 * l.len_v:
        return 0, 0.0
    return 1, t


@wp.func
def area_light_hit(l: AreaLight, ray_P: wp.vec3, ray_D: wp.vec3, t: float):
    """Emission seen by a BSDF ray that hit light `l` at distance t (area_light_sample_from_intersection).
    ls.pdf = solid angle pdf of NEE sampling this direction from ray_P (no selection prob, multiply by
    light_tree_pdf for MIS)."""
    ls = LightSample()
    ls.valid = 0
    ls.kind = KIND_AREA
    ls.Ng = l.dir
    ok, pdf, ef, lp, D, tt = area_light_eval(l, ray_P, wp.vec2(0.0, 0.0), 0, ray_D)
    if ok == 0:
        return ls
    ls.D = ray_D
    ls.t = t
    ls.P = ray_P + ray_D * t
    ls.pdf = pdf
    ls.L = l.strength * ef
    ls.use_mis = l.use_mis
    ls.valid = 1
    return ls


# ---------------------------------------------------------------------------------------------- host side
def make_area_light(matrix_world, size_x, size_y, color, power_w, spread=np.pi, normalize=True, exposure=0.0,
                    use_mis=True):
    """Build an AreaLight from a Blender light (rectangle). matrix_world is the 4x4 object matrix (the light
    looks along its local -Z); size_x/size_y are Blender `size`/`size_y` in metres (object scale multiplies).
    Mirrors scene/light.cpp:1327-1367 and blender/light.cpp:87 (strength = colour * energy * 2^exposure)."""
    m = np.asarray(matrix_world, dtype=np.float64).reshape(4, 4)
    ext_u = m[:3, 0] * size_x
    ext_v = m[:3, 1] * size_y
    lu = np.linalg.norm(ext_u)
    lv = np.linalg.norm(ext_v)
    area = lu * lv
    half = 0.5 * max(spread, 0.0)
    tan_half = 3.4028235e38 if abs(spread - np.pi) < 1e-6 else float(np.tan(half))
    if half > 0.0:
        nspread = 1.0 / (tan_half - half) if half > 0.05 else 3.0 / half ** 3
        nspread = float(np.float32(nspread))
    else:
        nspread = 3.4028235e38
    d = -m[:3, 2]
    a = AreaLight()
    a.co = wp.vec3(*m[:3, 3])
    a.axis_u = wp.vec3(*(ext_u / lu))
    a.axis_v = wp.vec3(*(ext_v / lv))
    a.dir = wp.vec3(*(d / np.linalg.norm(d)))
    a.len_u = float(lu)
    a.len_v = float(lv)
    a.invarea = float(1.0 / area) if normalize else 1.0
    a.strength = wp.vec3(*(np.asarray(color, np.float64) * power_w * 2.0 ** exposure))
    a.tan_half_spread = tan_half
    a.normalize_spread = nspread
    a.normalized = int(bool(normalize))
    a.use_mis = int(bool(use_mis and area != 0.0 and spread > 0.0))
    return a
