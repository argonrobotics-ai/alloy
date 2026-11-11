# SPDX-License-Identifier: Apache-2.0
"""Building blocks of the full-material path loop that do not depend on the scene's materials: device structs,
the thin-lens camera ray, smooth shading normals, the Bevel normal, next-event estimation, denoising albedo and
bounce counting (Cycles 4.5 kernel/integrator conventions).

Example (inside a Warp function)::

    o, d, tmax, rdD = first_ray_v2(cams, prm, ftab, sobol, rng, img, px, pyr, sample)
"""
from __future__ import annotations

import warp as wp

from ..tracer.geometry import T_INF, Geo, occluded
from ..tracer.integrator import Params, clamp_light, offset_point
from ..tracer.sampling import PRNG_FILTER, PRNG_LIGHT, Rng, hash_hp_uint, lookup_table, rand2, rand3, sample_uniform_disk
from . import lights as L
from . import nodes as nd
from .bsdf import principled as pr
from .bsdf.glass import LABEL_TRANSMIT
from .bsdf.microfacet import LABEL_DIFFUSE, LABEL_GLOSSY, LABEL_SINGULAR
from .shade_data import ShadeData

PRNG_LENS_TIME = wp.constant(1)


@wp.struct
class CamsV2:
    world: wp.array(dtype=wp.int32)  # per image: the world it looks into
    eye: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.mat33)  # Blender camera axes (looks along -z)
    extent: wp.array(dtype=wp.vec2)
    clip: wp.array(dtype=wp.vec2)
    aperture: wp.array(dtype=float)  # lens radius [m]; 0 = pinhole
    focus: wp.array(dtype=float)  # focus distance along the view axis


@wp.struct
class BumpTab:
    on: wp.array(dtype=wp.int32)  # per material: Bump node present
    strength: wp.array(dtype=float)
    distance: wp.array(dtype=float)
    filter_width: wp.array(dtype=float)
    invert: wp.array(dtype=wp.int32)
    bevel_on: wp.array(dtype=wp.int32)  # per material: Bevel node feeds Normal (and the Bump's normal input)
    bevel_radius: wp.array(dtype=float)
    bevel_samples: wp.array(dtype=wp.int32)
    vol_on: wp.array(dtype=wp.int32)  # per material: an absorption volume on the Volume output (volume.py)
    sigma_a: wp.array(dtype=wp.vec3)  # its absorption coefficient per metre


@wp.struct
class ParamsV2:
    max_glossy: wp.int32  # kernel value (setting + 1)
    blur_glossy: float
    caustics_reflective: wp.int32
    lights_visible_camera: wp.int32
    max_transmission: wp.int32  # kernel value (setting + 1)
    caustics_refractive: wp.int32
    cam_bg_on: wp.int32  # the world shows camera rays a plain colour (Light Path Is Camera Ray mix)
    cam_bg: wp.vec3


@wp.struct
class Medium:
    """Cycles' volume stack for absorption-only volumes (volume_stack.h): the (instance, material) of every object
    the ray is inside, -1 = empty; at most 3 nested volumes (Cycles ignores entries past its stack size)."""
    inst: wp.vec4i
    mid: wp.vec4i


@wp.func
def medium_empty() -> Medium:
    md = Medium()
    md.inst = wp.vec4i(-1, -1, -1, -1)
    md.mid = wp.vec4i(-1, -1, -1, -1)
    return md


@wp.func
def medium_sigma(bt: BumpTab, md: Medium) -> wp.vec3:
    """Total absorption coefficient (per metre) of the volumes the ray is in."""
    s = wp.vec3(0.0)
    for k in range(4):
        if md.mid[k] >= 0:
            s = s + bt.sigma_a[md.mid[k]]
    return s


@wp.func
def medium_enter_exit(md_in: Medium, inst: wp.int32, mid: wp.int32, backfacing: bool) -> Medium:
    """volume_stack_enter_exit after a transmission bounce through a surface whose material has a volume: a
    back-facing hit leaves the object (remove it, shift the rest down), a front-facing one enters it (append unless
    already there or the stack is full)."""
    md = md_in
    if backfacing:
        found = int(0)
        for k in range(3):
            if md.inst[k] == inst and md.mid[k] == mid:
                found = 1
            if found != 0:
                md.inst[k] = md.inst[k + 1]
                md.mid[k] = md.mid[k + 1]
        if found != 0 or (md.inst[3] == inst and md.mid[3] == mid):
            md.inst[3] = -1
            md.mid[3] = -1
        return md
    for k in range(3):
        if md.inst[k] == inst and md.mid[k] == mid:
            return md
        if md.inst[k] < 0:
            md.inst[k] = inst
            md.mid[k] = mid
            return md
    return md


@wp.func
def transmittance(sigma: wp.vec3, t: float) -> wp.vec3:
    """exp(-sigma t) per channel; a ray that never hits anything inside an absorbing volume keeps nothing."""
    out = wp.vec3(1.0)
    for c in range(3):
        if sigma[c] > 0.0:
            out[c] = wp.exp(-sigma[c] * wp.min(t, 1e30))
    return out


@wp.func
def cam_world_dir(r: wp.mat33, v: wp.vec3) -> wp.vec3:
    """Cycles camera space (+z forward) -> world, with Blender's camera matrix r (looks along -z)."""
    return wp.vec3(r[0, 0], r[1, 0], r[2, 0]) * v[0] + wp.vec3(r[0, 1], r[1, 1], r[2, 1]) * v[1] - \
        wp.vec3(r[0, 2], r[1, 2], r[2, 2]) * v[2]


@wp.func
def camera_ray_v2(cams: CamsV2, img: wp.int32, rx: float, ry: float, lens_u: wp.vec2,
                  prm: Params) -> tuple[wp.vec3, wp.vec3, float, float]:
    """camera_sample_perspective with an optional thin lens; also the compact ray differential dD."""
    ext = cams.extent[img]
    pc = wp.vec3((2.0 * rx / float(prm.width) - 1.0) * ext[0], (2.0 * ry / float(prm.height) - 1.0) * ext[1], 1.0)
    p_cam = wp.vec3(0.0)
    d_cam = pc
    ap = cams.aperture[img]
    if ap > 0.0:
        lens = sample_uniform_disk(lens_u) * ap
        pfocus = pc * cams.focus[img]
        p_cam = wp.vec3(lens[0], lens[1], 0.0)
        d_cam = pfocus - p_cam
    r = cams.rot[img]
    d = wp.normalize(cam_world_dir(r, d_cam))
    o = cams.eye[img] + cam_world_dir(r, p_cam)
    zinv = 1.0 / wp.normalize(pc)[2]
    clip = cams.clip[img]
    # compact dD (camera.h:110-118): 0.5 (|dDdx| + |dDdy|), one pixel step
    dcn = wp.normalize(pc)
    dx = wp.normalize(pc + wp.vec3(2.0 * ext[0] / float(prm.width), 0.0, 0.0)) - dcn
    dy = wp.normalize(pc + wp.vec3(0.0, 2.0 * ext[1] / float(prm.height), 0.0)) - dcn
    return o + d * (clip[0] * zinv), d, (clip[1] - clip[0]) * zinv, 0.5 * (wp.length(dx) + wp.length(dy))


@wp.func
def first_ray_v2(cams: CamsV2, prm: Params, ftab: wp.array(dtype=float), sobol: wp.array(dtype=wp.uint32), rng: Rng,
                 img: wp.int32, px: wp.int32, pyr: wp.int32, sample: wp.int32) -> tuple[wp.vec3, wp.vec3, float, float]:
    """(origin, direction, tmax, dD) of the camera ray: filter-sampled pixel position, lens sample when DOF is on."""
    fu = wp.vec2(0.5, 0.5)
    if sample != 0 or prm.first_centered == 0:
        fu = rand2(rng, sample, PRNG_FILTER, sobol)
    lu = wp.vec2(0.0)
    if cams.aperture[img] > 0.0:
        r3 = rand3(rng, sample, PRNG_LENS_TIME, sobol)
        lu = wp.vec2(r3[1], r3[2])
    return camera_ray_v2(cams, img, float(px) + lookup_table(ftab, fu[0]), float(pyr) + lookup_table(ftab, fu[1]),
                         lu, prm)


@wp.func
def geo_rot(geo: Geo, inst: wp.int32) -> wp.mat33:
    """Rotation part of an instance transform (frame world -> world)."""
    return wp.quat_to_matrix(wp.transform_get_rotation(geo.inst_xf[inst]))


@wp.func
def to_cam(cams: CamsV2, img: wp.int32, n: wp.vec3) -> wp.vec3:
    """World -> Cycles camera space (x right, y up, +z forward)."""
    r = cams.rot[img]
    return wp.vec3(r[0, 0] * n[0] + r[1, 0] * n[1] + r[2, 0] * n[2],
                   r[0, 1] * n[0] + r[1, 1] * n[1] + r[2, 1] * n[2],
                   -(r[0, 2] * n[0] + r[1, 2] * n[1] + r[2, 2] * n[2]))


@wp.func
def smooth_normal(geo: Geo, sd: ShadeData, inst: wp.int32, face: wp.int32, bu: float,
                  bv: float) -> tuple[wp.vec3, wp.int32, wp.int32]:
    """(world shading normal from the corner normals, mesh index, global triangle index) of a hit."""
    m = sd.inst_src[inst]
    g = sd.mesh_tri_off[m] + face
    ns = wp.normalize(sd.tri_n[g * 3] * bu + sd.tri_n[g * 3 + 1] * bv + sd.tri_n[g * 3 + 2] * (1.0 - bu - bv))
    return wp.normalize(wp.transform_vector(geo.inst_xf[inst], ns)), m, g


@wp.func
def uv_at(sd: ShadeData, m: wp.int32, face: wp.int32, bu: float, bv: float) -> wp.vec3:
    """(u, v, 0) of a hit: the mesh's corner UVs with the barycentric weights of ``smooth_normal`` (zeros without UVs)."""
    o = sd.uv_off[m]
    if o < 0:
        return wp.vec3(0.0)
    k = (o + face) * 3
    c = sd.uv[k] * bu + sd.uv[k + 1] * bv + sd.uv[k + 2] * (1.0 - bu - bv)
    return wp.vec3(c[0], c[1], 0.0)


@wp.func
def col_at(sd: ShadeData, m: wp.int32, face: wp.int32, bu: float, bv: float) -> wp.vec3:
    """The colour attribute at a hit, interpolated like ``uv_at`` (zeros without the attribute)."""
    o = sd.col_off[m]
    if o < 0:
        return wp.vec3(0.0)
    k = (o + face) * 3
    return sd.col[k] * bu + sd.col[k + 1] * bv + sd.col[k + 2] * (1.0 - bu - bv)


@wp.func
def bevel_at(geo: Geo, sd: ShadeData, bt: BumpTab, rng: Rng, inst: wp.int32, m: wp.int32, mid: wp.int32,
             p_frame: wp.vec3, ng: wp.vec3, n_frame: wp.vec3, backfacing: bool, sample: wp.int32, bounce: wp.int32,
             min_ray_pdf: float) -> wp.vec3:
    """World-space Bevel node normal (nodes.bevel_normal on the frame-space mesh, restricted to the hit object)."""
    rot_i = geo_rot(geo, inst)
    bseed = hash_hp_uint(rng.pixel ^ hash_hp_uint(wp.uint32(sample * 64 + bounce) ^ wp.uint32(0x7A3C11E5)))
    bn = nd.bevel_normal(sd.bevel_mesh, sd.tri_n, sd.tri_obj, m, p_frame, wp.transpose(rot_i) * ng, n_frame,
                         wp.where(backfacing, 1, 0), bt.bevel_radius[mid], bt.bevel_samples[mid], min_ray_pdf, bseed)
    return wp.normalize(rot_i * bn)


@wp.func
def albedo_step(lut: wp.array(dtype=float), cl: pr.PrincipledClosures, wi: wp.vec3, cams: CamsV2, img: wp.int32,
                prm: Params, den_thr: wp.vec3, alb: wp.vec3,
                nrm: wp.vec3) -> tuple[wp.int32, wp.vec3, wp.vec3, wp.vec3]:
    """Cycles' denoising passes: stop at the first diffuse-like vertex, else carry the specular tint onward.
    Returns (still looking, den_thr, alb, nrm)."""
    a = pr.closures_albedo(lut, cl, wi)
    if a.done == 0:
        return 1, wp.cw_mul(den_thr, a.specular), alb, nrm
    n_out = nrm + a.normal
    if prm.normal_camera != 0:
        n_out = nrm + to_cam(cams, img, a.normal)
    return 0, den_thr, alb + wp.cw_mul(den_thr, a.diffuse), n_out


@wp.func
def nee_v2(geo: Geo, lsc: L.LightScene, cl: pr.PrincipledClosures, prm: Params, sobol: wp.array(dtype=wp.uint32),
           rng: Rng, sample: wp.int32, rofs: wp.int32, world: wp.int32, wi: wp.vec3, p: wp.vec3, ns: wp.vec3,
           ng: wp.vec3, thr: wp.vec3, bounce: wp.int32, ht: wp.int32) -> wp.vec3:
    """Next-event estimation through Cycles' light tree, with power-heuristic MIS and a shadow ray. ``ht``: the
    closures transmit (lights behind the surface count; the shadow ray leaves from that side). Shadow rays stop at
    any surface (Cycles: only transparent BSDFs pass shadow rays), so absorbing volumes never attenuate them."""
    ls = L.light_sample(lsc, world, rand3(rng, sample, rofs + PRNG_LIGHT, sobol), p, ns, ht)
    if ls.valid == 0:
        return wp.vec3(0.0)
    ev = pr.closures_eval(cl, wi, ls.D)
    if wp.max(wp.max(ev.f[0], ev.f[1]), ev.f[2]) > 0.0:  # pdf 0 (no closure samples ls.D): MIS weight 1, as Cycles
        w = float(1.0)
        if ls.use_mis != 0:
            w = L.power_heuristic(ls.pdf, ev.pdf)
        c = wp.cw_mul(thr, wp.cw_mul(ev.f, ls.L)) * (w / ls.pdf)
        side = wp.where(wp.dot(ls.D, ng) >= 0.0, ng, -ng)
        if not occluded(geo, world, offset_point(p, side, prm), ls.D, wp.min(ls.t * 0.9999, T_INF)):
            return clamp_light(c, bounce, prm)
    return wp.vec3(0.0)


@wp.func
def exit_closures(n: wp.vec3, ns: wp.vec3, ng: wp.vec3) -> pr.PrincipledClosures:
    """subsurface_shader_data_setup: the closures at a random walk's exit point, one white Lambertian of normal n
    (the BSSRDF's bump normal there, else the smooth normal)."""
    c = pr.PrincipledClosures()
    c.ns = ns
    c.ng = ng
    c.bump_correction = 1
    c.diffuse.active = 1
    c.diffuse.weight = wp.vec3(1.0)
    c.diffuse.sample_weight = 1.0
    c.diffuse.n = n
    return c


@wp.func
def count_bounce(label: wp.int32, bounce: wp.int32, dbounce: wp.int32, gbounce: wp.int32, tbounce: wp.int32,
                 prm: Params, p2: ParamsV2) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32, wp.int32]:
    """path_state_next: (bounce, diffuse, glossy, transmission bounces, terminate at the next vertex). Reflections
    count as diffuse or glossy (singular too), refractions as transmission."""
    b = bounce + 1
    db = dbounce
    gb = gbounce
    tb = tbounce
    term = wp.where(b >= prm.max_bounce, 1, 0)
    if (label & LABEL_TRANSMIT) != 0:
        tb = tb + 1
        if tb >= p2.max_transmission:
            term = 1
    elif (label & LABEL_DIFFUSE) != 0:
        db = db + 1
        if db >= prm.max_diffuse:
            term = 1
    else:
        gb = gb + 1
        if gb >= p2.max_glossy:
            term = 1
    return b, db, gb, tb, term


@wp.func
def path_flags(path_diffuse: wp.int32, p2: ParamsV2, ancestor: wp.int32) -> wp.int32:
    """Principled path flags: the last bounce was diffuse (caustics rules), some bounce was (subsurface falls back to
    diffuse), and the scene's caustics switches."""
    pf = int(0)
    if path_diffuse != 0:
        pf = pf | pr.PATH_DIFFUSE
    if ancestor != 0:
        pf = pf | pr.PATH_DIFFUSE_ANCESTOR
    if p2.caustics_reflective == 0:
        pf = pf | pr.PATH_NO_REFLECTIVE_CAUSTICS
    if p2.caustics_refractive == 0:
        pf = pf | pr.PATH_NO_REFRACTIVE_CAUSTICS
    return pf


@wp.func
def lights_on_ray(lsc: L.LightScene, p2: ParamsV2, prm: Params, world: wp.int32, o: wp.vec3, d: wp.vec3,
                  inst: wp.int32, t: float, thr: wp.vec3, bounce: wp.int32, mis_n: wp.vec3, mis_ht: wp.int32,
                  mis_pdf: float, mis_skip: wp.int32) -> wp.vec3:
    """Emission of every area light the ray crosses before the surface (lights are transparent, shade_light.h);
    camera rays see lights only when they are visible to the camera."""
    if lsc.n_areas > 0 and (bounce > 0 or p2.lights_visible_camera != 0):
        le = L.area_lights_emission_along_ray(lsc, world, o, d, wp.where(inst >= 0, t, T_INF), mis_n, mis_ht,
                                              mis_pdf, mis_skip)
        return clamp_light(wp.cw_mul(thr, le), bounce - 1, prm)
    return wp.vec3(0.0)


@wp.func
def escaped(lsc: L.LightScene, p2: ParamsV2, prm: Params, world: wp.int32, o: wp.vec3, d: wp.vec3, thr: wp.vec3,
            bounce: wp.int32, mis_n: wp.vec3, mis_ht: wp.int32, mis_pdf: float, mis_skip: wp.int32) -> wp.vec3:
    """World (HDRI) and distant-light radiance for a ray that leaves the scene, MIS-weighted and clamped; a camera
    ray sees the world's camera colour when it has one."""
    if bounce == 0 and p2.cam_bg_on != 0:
        return wp.cw_mul(thr, p2.cam_bg)
    lb = L.escaped_ray_radiance(lsc, world, o, d, mis_n, mis_ht, mis_pdf, mis_skip)
    return clamp_light(wp.cw_mul(thr, lb), bounce - 1, prm)


@wp.func
def mis_after_bounce(label: wp.int32, pdf: float, ns: wp.vec3, mis_skip: wp.int32, mis_pdf: float, mis_n: wp.vec3,
                     min_ray_pdf: float) -> tuple[wp.int32, float, wp.vec3, float]:
    """MIS state for the next vertex: singular (mirror) bounces skip MIS; others record the BSDF pdf and normal."""
    if (label & LABEL_SINGULAR) != 0:
        return 1, mis_pdf, mis_n, min_ray_pdf
    return 0, pdf, ns, wp.min(min_ray_pdf, pdf)
