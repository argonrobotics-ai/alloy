# SPDX-License-Identifier: Apache-2.0
"""Device-side lights for the diffuse renderer: one sun (Cycles distant light with angle) + a constant world colour.

Light selection follows Cycles 4.5's light tree for a scene with only distant lights + background: the root's left
(local) child has zero importance, the right child is the distant leaf, and the leaf picks an emitter with two
reservoirs (max / min importance, kernel/light/tree.h light_tree_cluster_select_emitter):
    p(e) = 0.5 * (max_e / sum max + min_e / sum min),  or 1/#valid for the min part when sum min = 0.
Sun importance (distant_light_tree_parameters + light_tree_importance): max = E cos(theta_i - theta_u)+,
min = E cos(theta_i + theta_u)+ cos^3(theta_u); background: max = E, min = 0 (theta_o = pi).
E_sun = average(strength); E_bg = average_radiance * pi * average(emission_estimate) (scene/light_tree.cpp:231).
Directions: sun = uniform cone (kernel/light/distant.h), background = uniform sphere (the 1024x512 importance map
of a constant world is uniform in solid angle up to row discretisation).
"""
from __future__ import annotations

import warp as wp

from .sampling import PI, sample_uniform_cone, sample_uniform_sphere

LIGHT_NONE = wp.constant(-1)
LIGHT_SUN = wp.constant(0)
LIGHT_BG = wp.constant(1)


@wp.struct
class Lights:
    has_sun: wp.int32
    sun_to: wp.vec3  # unit direction from the scene toward the sun (= -klight->co)
    sun_radiance: wp.vec3  # strength * eval_fac, eval_fac = 1 / (pi sin^2(half angle))
    sun_one_minus_cos: float
    sun_cos_half: float
    sun_sin_half: float
    sun_pdf: float  # 1 / (2 pi (1 - cos half)), or 1 for a delta sun
    sun_energy: float
    sun_use_mis: wp.int32  # SHADER_USE_MIS: use_mis and angle > 0
    sun_visible_camera: wp.int32
    has_bg: wp.int32  # background shader non-black
    bg_color: wp.vec3  # radiance (colour * strength)
    bg_is_light: wp.int32  # world sampled as a light (MIS)
    bg_energy: float
    bg_use_map: wp.int32  # 1: Cycles' equirect importance map (constant world: uniform rows), 0: uniform sphere
    bg_marg: wp.array(dtype=wp.vec2)  # (res_y + 1) marginal CDF (x = row value, y = cdf; last x = total)
    bg_res_x: wp.int32
    bg_res_y: wp.int32
    bg_first: wp.int32  # reservoir order in the distant leaf: 1 = background before the sun


@wp.func
def sun_importance(lt: Lights, n: wp.vec3) -> wp.vec2:
    cos_i = wp.dot(lt.sun_to, n)
    sin_i = wp.sqrt(wp.max(1.0 - cos_i * cos_i, 0.0))
    cos_u = lt.sun_cos_half
    sin_u = lt.sun_sin_half
    cmin = float(1.0)
    if cos_i < cos_u:
        cmin = cos_i * cos_u + sin_i * sin_u
    if cmin < 0.0:
        return wp.vec2(0.0)
    cmax = wp.max(cos_i * cos_u - sin_i * sin_u, 0.0)
    return wp.vec2(lt.sun_energy * cmin, lt.sun_energy * cmax * cos_u * cos_u * cos_u)


@wp.func
def selection_probs(lt: Lights, n: wp.vec3) -> wp.vec2:
    """(p_sun, p_bg) at a shading point with normal n (opaque surface)."""
    s = wp.vec2(0.0)
    if lt.has_sun != 0:
        s = sun_importance(lt, n)
    bmax = float(0.0)
    if lt.bg_is_light != 0:
        bmax = lt.bg_energy
    tot_max = s[0] + bmax
    if tot_max <= 0.0:
        return wp.vec2(0.0)
    q_sun = float(0.0)
    q_bg = float(0.0)
    if s[1] > 0.0:
        q_sun = 1.0  # background min importance is 0
    else:
        cnt = float(0.0)
        if s[0] > 0.0:
            cnt += 1.0
        if bmax > 0.0:
            cnt += 1.0
        if s[0] > 0.0:
            q_sun = 1.0 / cnt
        if bmax > 0.0:
            q_bg = 1.0 / cnt
    return wp.vec2(0.5 * (s[0] / tot_max + q_sun), 0.5 * (bmax / tot_max + q_bg))


@wp.func
def select_light(lt: Lights, n: wp.vec3, r: float) -> wp.int32:
    """Pick sun or background like the two-reservoir leaf (order: sun, background)."""
    s = wp.vec2(0.0)
    if lt.has_sun != 0:
        s = sun_importance(lt, n)
    bmax = float(0.0)
    if lt.bg_is_light != 0:
        bmax = lt.bg_energy
    if s[0] + bmax <= 0.0:
        return LIGHT_NONE
    sample_max = r > 0.5
    rr = 2.0 * r
    ws = s[1]
    wb = float(0.0)
    if sample_max:
        rr = rr - 1.0
        ws = s[0]
        wb = bmax
    elif s[1] <= 0.0:  # no min importance: uniform over emitters with max importance
        ws = wp.where(s[0] > 0.0, 1.0, 0.0)
        wb = wp.where(bmax > 0.0, 1.0, 0.0)
    if ws > 0.0 and wb > 0.0:
        if lt.bg_first != 0:
            if rr <= ws / (ws + wb):
                return LIGHT_SUN
            return LIGHT_BG
        if rr <= wb / (ws + wb):
            return LIGHT_BG
        return LIGHT_SUN
    if wb > 0.0:
        return LIGHT_BG
    return LIGHT_SUN


@wp.func
def light_sample(lt: Lights, kind: wp.int32, r: wp.vec2) -> tuple[wp.vec3, float, wp.vec3, wp.int32]:
    """Direction toward the light, direction pdf (solid angle), radiance, MIS flag."""
    if kind == LIGHT_SUN:
        if lt.sun_one_minus_cos > 0.0:
            ng = sample_uniform_cone(-lt.sun_to, lt.sun_one_minus_cos, r)
            return -ng, lt.sun_pdf, lt.sun_radiance, lt.sun_use_mis
        return lt.sun_to, 1.0, lt.sun_radiance, 0
    if lt.bg_use_map != 0:
        d, pdf = bg_map_sample(lt, r)
        return d, pdf, lt.bg_color, 1
    return sample_uniform_sphere(r), 1.0 / (4.0 * PI), lt.bg_color, 1


@wp.func
def bg_map_sample(lt: Lights, r: wp.vec2) -> tuple[wp.vec3, float]:
    """background_map_sample (kernel/light/background.h) for a constant world: lower_bound on the marginal CDF,
    uniform conditional rows (u = r.x), pdf = row_value / (2 pi^2 sin(theta) total)."""
    first = int(0)
    count = lt.bg_res_y
    while count > 0:
        step = count >> 1
        middle = first + step
        if lt.bg_marg[middle][1] < r[1]:
            first = middle + 1
            count -= step + 1
        else:
            count = step
    iv = wp.max(0, first - 1)
    cv = lt.bg_marg[iv]
    cn = lt.bg_marg[iv + 1]
    dv = float(0.0)
    if cn[1] != cv[1]:
        dv = (r[1] - cv[1]) / (cn[1] - cv[1])
    v = (float(iv) + dv) / float(lt.bg_res_y)
    u = r[0]
    st = wp.sin(PI * v)
    pdf = float(0.0)
    den = 2.0 * PI * PI * st * lt.bg_marg[lt.bg_res_y][0]
    if st != 0.0 and den != 0.0:
        pdf = cv[0] / den
    theta = PI - PI * v
    phi = PI - 2.0 * PI * u
    return wp.vec3(wp.sin(theta) * wp.cos(phi), wp.sin(theta) * wp.sin(phi), wp.cos(theta)), pdf


@wp.func
def bg_pdf(lt: Lights, d: wp.vec3) -> float:
    """background_light_pdf direction part (map or uniform)."""
    if lt.bg_use_map == 0:
        return 1.0 / (4.0 * PI)
    v = 1.0 - wp.acos(wp.clamp(d[2], -1.0, 1.0)) / PI
    st = wp.sin(v * PI)
    if st == 0.0:
        return 0.0
    iv = wp.clamp(int(v * float(lt.bg_res_y)), 0, lt.bg_res_y - 1)
    den = 2.0 * PI * PI * st * lt.bg_marg[lt.bg_res_y][0]
    if den == 0.0:
        return 0.0
    return lt.bg_marg[iv][0] / den


@wp.func
def sun_hit(lt: Lights, d: wp.vec3) -> bool:
    """distant_light_sample_from_intersection: a ray inside the sun cone (needs angle > 0 and MIS)."""
    if lt.has_sun == 0 or lt.sun_use_mis == 0:
        return False
    return wp.dot(d, lt.sun_to) >= lt.sun_cos_half
