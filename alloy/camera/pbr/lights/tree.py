# SPDX-License-Identifier: Apache-2.0
"""Light selection: port of Cycles' light tree (kernel/light/tree.h, scene/light_tree.cpp) for the emitter types
the pack uses: rectangle area lights (local lights), the HDRI background and suns (distant lights).

Structure (scene/light_tree.cpp:427-457): root inner node, LEFT = tree of all local lights, RIGHT = one `distant`
leaf holding every distant emitter (background, suns). Cycles builds the local subtree with a SAH split; this
port splits at the median of the longest centroid axis down to single-emitter leaves (identical selection
probabilities for 2 lights; for >2 the topology can differ, the converged image does not). Importance functions,
min/max importance averaging, reservoir sampling and the pdf walk are ported 1:1 (non-volume branch only,
has_transmission ``ht``: 1 at a vertex whose closures transmit (two-sided importance; PATH_RAY_MIS_HAD_TRANSMISSION
for the MIS pdf of the next hit)."""
import numpy as np
import warp as wp

from .area import AreaLight, area_light_sample, area_light_hit
from .common import (FLT_MAX, KIND_AREA, KIND_BACKGROUND, KIND_DISTANT, M_PI, LightSample, safe_acos,
                           safe_normalize_len, safe_sqrt, sin_from_cos, sqr)
from .world import (DistantLight, WorldMap, background_eval, background_light_pdf, background_light_sample,
                          distant_light_hit, distant_light_sample)

NODE_INNER = wp.constant(0)
NODE_LEAF = wp.constant(1)
NODE_DISTANT = wp.constant(2)


@wp.struct
class TreeEmitter:
    kind: int
    idx: int
    energy: float
    theta_o: float
    theta_e: float
    bit_trail: int


@wp.struct
class TreeNode:
    bbox_min: wp.vec3
    bbox_max: wp.vec3
    axis: wp.vec3
    theta_o: float
    theta_e: float
    energy: float
    type: int
    left: int
    right: int
    first_emitter: int
    num_emitters: int


@wp.struct
class LightScene:
    """Everything the light functions need for ALL worlds of a batch (arrays are indexed [world, i])."""
    areas: wp.array2d(dtype=AreaLight)
    suns: wp.array2d(dtype=DistantLight)
    world: WorldMap
    has_world: int
    world_light: int  # 1: the world is also a light (NEE); 0: sampling method NONE, escaped rays only
    n_areas: int
    n_suns: int
    nodes: wp.array2d(dtype=TreeNode)
    emitters: wp.array2d(dtype=TreeEmitter)
    n_emitters: int


# ------------------------------------------------------------------------------------------ importance
@wp.func
def cos_bound_subtended_angle(bmin: wp.vec3, bmax: wp.vec3, centroid: wp.vec3, P: wp.vec3):
    d2 = wp.length_sq(P - centroid)
    r2 = wp.length_sq(bmax - centroid)
    if d2 <= r2:
        return -1.0
    return safe_sqrt(1.0 - r2 / d2)


@wp.func
def light_tree_importance(N: wp.vec3, ht: int, p2c: wp.vec3, cos_theta_u: float, axis: wp.vec3, theta_o: float,
                          theta_e: float, max_distance: float, min_distance: float, energy: float):
    """tree.h light_tree_importance (non-volume; ht = has_transmission). Returns (max_importance, min_importance)."""
    sin_theta_u = sin_from_cos(cos_theta_u)
    cos_theta_i = wp.dot(p2c, N)
    if ht != 0:
        cos_theta_i = wp.abs(cos_theta_i)
    sin_theta_i = sin_from_cos(cos_theta_i)
    cos_min_inc = float(1.0)
    if cos_theta_i < cos_theta_u:
        cos_min_inc = cos_theta_i * cos_theta_u + sin_theta_i * sin_theta_u
    if ht == 0 and cos_min_inc < 0.0:
        return 0.0, 0.0
    cos_max_inc = wp.max(cos_theta_i * cos_theta_u - sin_theta_i * sin_theta_u, 0.0)

    cos_theta = float(0.0)
    sin_theta = float(0.0)
    nv = -p2c
    if axis[0] == nv[0] and axis[1] == nv[1] and axis[2] == nv[2]:
        cos_theta = 1.0
        sin_theta = 0.0
    else:
        cos_theta = wp.dot(axis, nv)
        sin_theta = sin_from_cos(cos_theta)
    cos_t_minus_u = cos_theta * cos_theta_u + sin_theta * sin_theta_u
    sin_theta_o = wp.sin(theta_o)
    cos_theta_o = wp.cos(theta_o)

    cos_min_out = float(0.0)
    if cos_theta >= cos_theta_u or cos_t_minus_u >= cos_theta_o:
        cos_min_out = 1.0
    elif theta_o + theta_e > M_PI or cos_t_minus_u > wp.cos(theta_o + theta_e):
        sin_t_minus_u = sin_from_cos(cos_t_minus_u)
        cos_min_out = cos_t_minus_u * cos_theta_o + sin_t_minus_u * sin_theta_o
    else:
        return 0.0, 0.0

    max_imp = wp.abs(cos_min_inc * energy * cos_min_out * (1.0 / (min_distance * min_distance)))

    min_imp = float(0.0)
    cos_t_plus_u = cos_theta * cos_theta_u - sin_theta * sin_theta_u
    if theta_e - theta_o < 0.0 or cos_theta < 0.0 or cos_theta_u < 0.0 or cos_t_plus_u < wp.cos(theta_e - theta_o):
        min_imp = 0.0
    else:
        sin_t_plus_u = sin_from_cos(cos_t_plus_u)
        cos_max_out = cos_t_plus_u * cos_theta_o - sin_t_plus_u * sin_theta_o
        min_imp = wp.abs(cos_max_inc * energy * cos_max_out / (max_distance * max_distance))
    return max_imp, min_imp


@wp.func
def node_importance(sc: LightScene, w: int, ni: int, P: wp.vec3, N: wp.vec3, ht: int):
    n = sc.nodes[w, ni]
    p2c = wp.vec3(0.0)
    cos_theta_u = float(0.0)
    distance = float(0.0)
    if n.type == NODE_DISTANT:
        p2c = -n.axis
        cos_theta_u = wp.cos(n.theta_o + n.theta_e)
        distance = 1.0
    else:
        centroid = 0.5 * (n.bbox_min + n.bbox_max)
        ext = n.bbox_max - centroid
        vis = wp.dot(N, centroid - P) + wp.dot(wp.vec3(wp.abs(N[0]), wp.abs(N[1]), wp.abs(N[2])),
                                               wp.vec3(wp.abs(ext[0]), wp.abs(ext[1]), wp.abs(ext[2])))
        if vis <= 0.0 and ht == 0:
            return 0.0, 0.0
        p2c, distance = safe_normalize_len(centroid - P)
        cos_theta_u = cos_bound_subtended_angle(n.bbox_min, n.bbox_max, centroid, P)
        distance = wp.max(0.5 * wp.length(centroid - n.bbox_max), distance)
    return light_tree_importance(N, ht, p2c, cos_theta_u, n.axis, n.theta_o, n.theta_e, distance, distance, n.energy)


@wp.func
def emitter_importance(sc: LightScene, w: int, ei: int, P: wp.vec3, N: wp.vec3, ht: int):
    em = sc.emitters[w, ei]
    centroid = wp.vec3(0.0)
    axis = wp.vec3(0.0)
    p2c = wp.vec3(0.0)
    cos_theta_u = float(0.0)
    dist_max = float(1.0)
    dist_min = float(1.0)
    visible = int(1)
    if em.kind == KIND_AREA:
        l = sc.areas[w, em.idx]
        centroid = l.co
        axis = l.dir
        p2c, dmin = safe_normalize_len(centroid - P)
        dist_min = dmin
        dist_max = dmin
        cos_theta_u = FLT_MAX
        eu = l.axis_u * l.len_u
        ev = l.axis_v * l.len_v
        for i in range(4):
            fu = float(i & 1) - 0.5
            fv = 0.5 * float((i & 2) - 1)
            corner = fu * eu + fv * ev + centroid
            p2k, dk = safe_normalize_len(corner - P)
            cos_theta_u = wp.min(cos_theta_u, wp.dot(p2c, p2k))
            dist_max = wp.max(dist_max, dk)
        front = wp.dot(axis, p2c) < 0.0
        above = wp.dot(N, centroid - P) + wp.abs(wp.dot(N, eu)) + wp.abs(wp.dot(N, ev)) > 0.0
        if not (front and above):
            visible = 0
    elif em.kind == KIND_BACKGROUND:
        centroid = wp.vec3(0.0, 0.0, 1.0)
        axis = wp.vec3(0.0, 0.0, -1.0)
        cos_theta_u = -1.0
        p2c = -centroid
    else:
        s = sc.suns[w, em.idx]
        centroid = s.co
        axis = centroid
        cos_theta_u = wp.cos(em.theta_e)
        dist_max = 1.0 / cos_theta_u
        dist_min = 1.0
        p2c = -centroid
    if visible == 0 and ht == 0:
        return 0.0, 0.0
    return light_tree_importance(N, ht, p2c, cos_theta_u, axis, em.theta_o, em.theta_e, dist_max, dist_min, em.energy)


@wp.func
def child_importance(sc: LightScene, w: int, ni: int, P: wp.vec3, N: wp.vec3, ht: int):
    n = sc.nodes[w, ni]
    if n.num_emitters == 1:
        return emitter_importance(sc, w, n.first_emitter, P, N, ht)
    if n.num_emitters != 0:
        return node_importance(sc, w, ni, P, N, ht)
    return 0.0, 0.0


@wp.func
def get_left_probability(sc: LightScene, w: int, li: int, ri: int, P: wp.vec3, N: wp.vec3, ht: int):
    lmax, lmin = child_importance(sc, w, li, P, N, ht)
    rmax, rmin = child_importance(sc, w, ri, P, N, ht)
    tmax = lmax + rmax
    if tmax == 0.0:
        return 0, 0.0
    tmin = lmin + rmin
    pmax = lmax / tmax
    pmin = float(0.0)
    if tmin > 0.0:
        pmin = lmin / tmin
    else:
        a = float(0.0)
        if lmax > 0.0:
            a = 1.0
        b = float(0.0)
        if rmax == 0.0:
            b = 1.0
        pmin = 0.5 * (a + b)
    return 1, 0.5 * (pmax + pmin)


# ------------------------------------------------------------------------------------------ sampling
@wp.func
def sample_reservoir(cur: int, wgt: float, selected: int, sel_w: float, tot: float, rand: float):
    """tree.h sample_reservoir. Returns updated (selected, selected_weight, total_weight, rand)."""
    if not (wgt > 0.0):
        return selected, sel_w, tot, rand
    tot_n = tot + wgt
    if selected == -1:
        return cur, wgt, tot_n, rand
    thresh = wgt / tot_n
    sel_n = selected
    sw = sel_w
    r = float(0.0)
    if rand <= thresh:
        sel_n = cur
        sw = wgt
        r = rand / thresh
    else:
        r = (rand - thresh) / (1.0 - thresh)
    return sel_n, sw, tot_n, wp.clamp(r, 0.0, 1.0)


@wp.func
def _select_uniform_among_important(sc: LightScene, w: int, first_emitter: int, num_emitters: int, has_imp: int,
                                     rand_in: float, P: wp.vec3, N: wp.vec3, ht: int):
    """Min-importance sum is zero: pick uniformly among emitters with non-zero max importance (tree.h
    light_tree_cluster_select_emitter). Returns (selected, importance of selected, sel1, tot1)."""
    rand = rand_in
    selected = int(-1)
    sel1 = float(0.0)
    tot1 = float(0.0)
    hi = int(has_imp)
    for i in range(num_emitters):
        cur = first_emitter + i
        selected, sel1, tot1, rand = sample_reservoir(cur, float(hi & 1), selected, sel1, tot1, rand)
        hi = hi >> 1
    sel0, dmin = emitter_importance(sc, w, selected, P, N, ht)
    return selected, sel0, sel1, tot1


@wp.func
def _importance_pair(sample_max: int, sel_a: float, tot_a: float, sel_b: float, tot_b: float):
    """Order the reservoir sums as importance[0] = max, importance[1] = min. Returns (sel0, tot0, sel1, tot1)."""
    sel0 = sel_b
    tot0 = tot_b
    sel1 = sel_a
    tot1 = tot_a
    if sample_max == 1:
        sel0 = sel_a
        tot0 = tot_a
        sel1 = sel_b
        tot1 = tot_b
    return sel0, tot0, sel1, tot1


@wp.func
def cluster_select_emitter(sc: LightScene, w: int, ni: int, rand_in: float, P: wp.vec3, N: wp.vec3, ht: int):
    """tree.h light_tree_cluster_select_emitter. Returns (emitter index or -1, pdf_factor).
    A = measure used for the reservoir (max importance if sample_max else min), B = the other one."""
    n = sc.nodes[w, ni]
    rand = rand_in
    sample_max = int(0)
    if rand > 0.5:
        sample_max = 1
    if n.num_emitters > 1:
        rand = rand * 2.0 - float(sample_max)
    selected = int(-1)
    sel_a = float(0.0)
    tot_a = float(0.0)
    sel_b = float(0.0)
    tot_b = float(0.0)
    has_imp = int(0)
    num_has = int(0)
    for i in range(n.num_emitters):
        cur = n.first_emitter + i
        imax, imin = emitter_importance(sc, w, cur, P, N, ht)
        wa = imin
        wb = imax
        if sample_max == 1:
            wa = imax
            wb = imin
        selected, sel_a, tot_a, rand = sample_reservoir(cur, wa, selected, sel_a, tot_a, rand)
        if selected == cur:
            sel_b = wb
        tot_b += wb
        if imax > 0.0:
            has_imp |= (1 << i)
            num_has += 1
    if has_imp == 0:
        return -1, 0.0
    # importance[0] = max, importance[1] = min
    sel0, tot0, sel1, tot1 = _importance_pair(sample_max, sel_a, tot_a, sel_b, tot_b)
    if tot1 == 0.0:
        if sample_max == 1:
            sel1 = 1.0
            tot1 = float(num_has)
        else:
            selected, sel0, sel1, tot1 = _select_uniform_among_important(sc, w, n.first_emitter, n.num_emitters,
                                                                         has_imp, rand, P, N, ht)
    return selected, 0.5 * (sel0 / tot0 + sel1 / tot1)


@wp.func
def light_tree_sample(sc: LightScene, w: int, rand: float, P: wp.vec3, N: wp.vec3, ht: int):
    """tree.h light_tree_sample. Returns (ok, emitter index, selection pdf)."""
    pdf_leaf = float(1.0)
    pdf_sel = float(1.0)
    emitter = int(-1)
    ni = int(0)
    r = rand
    done = int(0)
    ok = int(1)
    for it in range(32):
        if done == 0:
            n = sc.nodes[w, ni]
            if n.type != NODE_INNER:
                e, f = cluster_select_emitter(sc, w, ni, r, P, N, ht)
                if e < 0:
                    ok = 0
                else:
                    emitter = e
                    pdf_sel = f
                done = 1
            else:
                okp, lp = get_left_probability(sc, w, n.left, n.right, P, N, ht)
                if okp == 0:
                    ok = 0
                    done = 1
                else:
                    sel, dsc, tot, r2 = sample_reservoir(n.right, 1.0 - lp, n.left, 0.0, lp, r)
                    r = r2
                    ni = sel
                    if sel == n.left:
                        pdf_leaf *= lp
                    else:
                        pdf_leaf *= 1.0 - lp
    return ok, emitter, pdf_sel * pdf_leaf


@wp.func
def light_tree_pdf(sc: LightScene, w: int, P: wp.vec3, N: wp.vec3, ht: int, emitter: int):
    """tree.h light_tree_pdf: probability that light_tree_sample picks `emitter` from (P, N)."""
    pdf = float(1.0)
    bit_trail = sc.emitters[w, emitter].bit_trail
    ni = int(0)
    result = float(-1.0)
    for it in range(32):
        if result < 0.0:
            n = sc.nodes[w, ni]
            if n.type != NODE_INNER:
                tmax = float(0.0)
                tmin = float(0.0)
                smax = float(0.0)
                smin = float(0.0)
                nh = int(0)
                for i in range(n.num_emitters):
                    e = n.first_emitter + i
                    imax, imin = emitter_importance(sc, w, e, P, N, ht)
                    if imax > 0.0:
                        nh += 1
                    if e == emitter:
                        smax = imax
                        smin = imin
                    tmax += imax
                    tmin += imin
                if smax > 0.0:
                    alt = 1.0 / float(nh)
                    if tmin > 0.0:
                        alt = smin / tmin
                    result = pdf * 0.5 * (smax / tmax + alt)
                else:
                    result = 0.0
            else:
                okp, lp = get_left_probability(sc, w, n.left, n.right, P, N, ht)
                if okp == 0:
                    result = 0.0
                else:
                    go_left = (bit_trail & 1) == 0
                    bit_trail = bit_trail >> 1
                    if go_left:
                        ni = n.left
                        pdf *= lp
                    else:
                        ni = n.right
                        pdf *= 1.0 - lp
                    if pdf == 0.0:
                        result = 0.0
    return wp.max(result, 0.0)


@wp.func
def light_sample(sc: LightScene, w: int, rand: wp.vec3, P: wp.vec3, N: wp.vec3, ht: int):
    """Next-event sample (kernel/light/light.h light_sample + light_tree_sample). rand.xy position on the light,
    rand.z selects the light. ls.pdf (solid angle) includes the selection pdf; ls.L is the unshadowed radiance."""
    ls = LightSample()
    ls.valid = 0
    if sc.n_emitters == 0:
        return ls
    ok, e, psel = light_tree_sample(sc, w, rand[2], P, N, ht)
    if ok == 0:
        return ls
    em = sc.emitters[w, e]
    r2 = wp.vec2(rand[0], rand[1])
    if em.kind == KIND_AREA:
        ls = area_light_sample(sc.areas[w, em.idx], P, r2)
    elif em.kind == KIND_BACKGROUND:
        ls = background_light_sample(sc.world, r2)
    else:
        ls = distant_light_sample(sc.suns[w, em.idx], r2)
    ls.id = em.idx
    ls.pdf_sel = psel
    ls.pdf = ls.pdf * psel
    if ls.valid != 0 and not (ls.pdf > 0.0):
        ls.valid = 0
    return ls


@wp.func
def emitter_index(sc: LightScene, kind: int, idx: int):
    """Emitter slot of (kind, idx): local lights first, then background, then suns (host builder order)."""
    if kind == KIND_AREA:
        return idx
    if kind == KIND_BACKGROUND:
        return sc.n_areas
    return sc.n_areas + sc.world_light + idx


@wp.func
def light_hit_mis_weight(sc: LightScene, w: int, ls: LightSample, ray_P: wp.vec3, mis_N: wp.vec3, mis_ht: int,
                         bsdf_pdf: float):
    """Forward MIS weight (sample.h light_sample_mis_weight_forward_lamp/_background): `ls` comes from
    area_light_hit / distant_light_hit (pdf without selection) or background_hit; ray_P / mis_N are the origin and
    shading normal of the PREVIOUS vertex (path mis_origin_n). Returns power_heuristic(bsdf_pdf, ls.pdf * psel)."""
    if ls.use_mis == 0:
        return 1.0
    psel = light_tree_pdf(sc, w, ray_P, mis_N, mis_ht, emitter_index(sc, ls.kind, ls.id))
    pdf = ls.pdf * psel
    if bsdf_pdf + pdf == 0.0:
        return 0.0
    return (bsdf_pdf * bsdf_pdf) / (bsdf_pdf * bsdf_pdf + pdf * pdf)


@wp.func
def background_hit(sc: LightScene, w: int, ray_D: wp.vec3):
    """Escaped ray: world radiance and the NEE pdf of that direction (no selection prob)."""
    from_world = LightSample()
    from_world.kind = KIND_BACKGROUND
    from_world.id = 0
    from_world.D = ray_D
    from_world.t = FLT_MAX
    from_world.valid = 1
    from_world.use_mis = sc.world.use_mis
    from_world.pdf = background_light_pdf(sc.world, ray_D)
    from_world.L = background_eval(sc.world, ray_D)
    return from_world
