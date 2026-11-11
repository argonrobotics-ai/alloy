# SPDX-License-Identifier: Apache-2.0
"""Cycles 4.5 lights as Warp functions: rectangle area lights, the HDRI world with its importance map, distant
lights and Cycles' light tree (selection, NEE pdfs, MIS weights). All @wp.func are float32.

Example::

    from alloy.camera.pbr import lights as L
    area = L.make_area_light(matrix_world, 0.5, 0.5, (1, 1, 1), 100.0)
    scene = L.build_light_scene([[area]] * num_worlds, world=None, device="cuda:0")
"""
from .common import (make_orthonormals, FLT_MAX, KIND_AREA, KIND_BACKGROUND, KIND_DISTANT, LightSample, power_heuristic,  # noqa: F401
                           sample_uniform_cone, sample_uniform_disk)
from .area import (AreaLight, area_light_hit, area_light_intersect, area_light_sample,  # noqa: F401
                         make_area_light)
from .world import (DistantLight, WorldMap, background_eval, background_light_pdf,  # noqa: F401
                          background_light_sample, direction_to_equirectangular, distant_light_hit,
                          distant_light_sample, equirectangular_to_direction, load_hdr, make_distant_light,
                          make_world)
from .tree import (LightScene, background_hit, light_hit_mis_weight, light_sample,  # noqa: F401
                         light_tree_pdf, light_tree_sample)
from .build import build_light_scene  # noqa: F401
import warp as wp


@wp.func
def scene_lights_intersect(sc: LightScene, w: int, P: wp.vec3, D: wp.vec3, tmin: float, tmax: float):
    """Nearest area-light hit along a ray (call for BSDF/indirect rays that are NOT camera rays: the C3 lights are
    invisible to the camera). Returns (hit, t, area light index)."""
    best_t = tmax
    best = int(-1)
    for i in range(sc.n_areas):
        hit, t = area_light_intersect(sc.areas[w, i], P, D, tmin, best_t)
        if hit != 0:
            best_t = t
            best = i
    if best < 0:
        return 0, 0.0, -1
    return 1, best_t, best


@wp.func
def light_nee_mis_weight(ls: LightSample, bsdf_pdf: float):
    """NEE weight (shade_surface.h:384): a light with MIS off is evaluated with bsdf pdf 0 -> weight 1, because BSDF
    rays never hit it; otherwise power_heuristic(ls.pdf, bsdf_pdf)."""
    if ls.use_mis == 0:
        return 1.0
    return power_heuristic(ls.pdf, bsdf_pdf)


@wp.func
def area_light_hit_mis(sc: LightScene, w: int, idx: int, ray_P: wp.vec3, ray_D: wp.vec3, t: float, mis_N: wp.vec3,
                       mis_ht: int, bsdf_pdf: float):
    """Radiance and MIS weight for a BSDF ray that hit area light `idx` (previous vertex: ray_P / mis_N)."""
    ls = area_light_hit(sc.areas[w, idx], ray_P, ray_D, t)
    ls.id = idx
    if ls.valid == 0 or ls.use_mis == 0:
        return wp.vec3(0.0, 0.0, 0.0), 0.0
    return ls.L, light_hit_mis_weight(sc, w, ls, ray_P, mis_N, mis_ht, bsdf_pdf)


@wp.func
def area_lights_emission_along_ray(sc: LightScene, w: int, ray_P: wp.vec3, ray_D: wp.vec3, t_geom: float,
                                   mis_N: wp.vec3, mis_ht: int, bsdf_pdf: float, camera_ray: int):
    """Sum of MIS-weighted emission of every area light hit before `t_geom` (distance of the nearest scene surface,
    1e30 if none). Cycles lights are TRANSPARENT: integrator_shade_light (shade_light.h) adds the emission and the
    ray continues, so the geometry or world behind a light is still seen (and shadow rays ignore lights).
    Camera rays: pass camera_ray = 1 (C3 lights are invisible to the camera, so this is skipped by the caller)."""
    out = wp.vec3(0.0, 0.0, 0.0)
    for i in range(sc.n_areas):
        hit, t = area_light_intersect(sc.areas[w, i], ray_P, ray_D, 0.0, t_geom)
        # lights with MIS off are not intersected by non-camera rays (light.h lights_intersect_impl: !SHADER_USE_MIS)
        if camera_ray == 0 and sc.areas[w, i].use_mis == 0:
            hit = 0
        if hit != 0:
            ls = area_light_hit(sc.areas[w, i], ray_P, ray_D, t)
            ls.id = i
            if ls.valid != 0:
                wgt = float(1.0)
                if camera_ray == 0:
                    wgt = light_hit_mis_weight(sc, w, ls, ray_P, mis_N, mis_ht, bsdf_pdf)
                out += ls.L * wgt
    return out


@wp.func
def escaped_ray_radiance(sc: LightScene, w: int, ray_P: wp.vec3, ray_D: wp.vec3, mis_N: wp.vec3, mis_ht: int,
                         bsdf_pdf: float, camera_ray: int):
    """Radiance (world + suns) of a ray that left the scene, already multiplied by its MIS weight (camera rays and
    specular chains pass camera_ray = 1 / bsdf_pdf = 0: weight 1)."""
    out = wp.vec3(0.0, 0.0, 0.0)
    if sc.has_world != 0:
        ls = background_hit(sc, w, ray_D)
        wgt = float(1.0)
        if camera_ray == 0:
            wgt = light_hit_mis_weight(sc, w, ls, ray_P, mis_N, mis_ht, bsdf_pdf)
        out += ls.L * wgt
    for i in range(sc.n_suns):
        ls = distant_light_hit(sc.suns[w, i], ray_D)
        if ls.valid != 0 and (camera_ray != 0 or ls.use_mis != 0):
            ls.id = i
            wgt2 = float(1.0)
            if camera_ray == 0:
                wgt2 = light_hit_mis_weight(sc, w, ls, ray_P, mis_N, mis_ht, bsdf_pdf)
            out += ls.L * wgt2
    return out
