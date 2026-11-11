# SPDX-License-Identifier: Apache-2.0
"""The megakernel's path state and the steps of its loop that do not depend on the material set: the path's start,
the next hit (or the background), a subsurface bounce and a BSDF bounce. ``integrator`` composes them with the
material-specific shading (``closures_at``).

Example (inside the megakernel's path function)::

    st = mk_start(cams, prm, ftab, sobol, rng, img, px, pyr, sample)
    st, h, alive = mk_next_hit(geo, bt, lsc, p2, prm, world, st)
"""
import warp as wp

from ..tracer.geometry import T_INF, Geo, closest_hit_uv
from ..tracer.integrator import Params, offset_point
from ..tracer.sampling import PRNG_BOUNCE_NUM, Rng, rand2
from . import lights as L
from .bsdf import principled as pr
from .bsdf.glass import LABEL_TRANSMIT
from .bsdf.microfacet import LABEL_DIFFUSE, LABEL_NONE
from .path import (BumpTab, CamsV2, Medium, ParamsV2, count_bounce, escaped, first_ray_v2, lights_on_ray,
                   medium_empty, medium_enter_exit, medium_sigma, mis_after_bounce, transmittance)
from .subsurface import entry_bounce, random_walk


@wp.struct
class Hit:
    """A surface hit: instance (-1 = none), distance, geometric normal, face and barycentrics (closest_hit_uv)."""
    inst: wp.int32
    t: float
    ng: wp.vec3
    face: wp.int32
    u: float
    v: float


@wp.struct
class Surf:
    """The shading point of a hit: position, normals (facing the ray), mesh, triangle, attributes."""
    p: wp.vec3
    ng: wp.vec3
    ns: wp.vec3
    m: wp.int32
    g: wp.int32
    uv: wp.vec3
    col: wp.vec3
    backfacing: bool
    dP: float


@wp.struct
class MkPath:
    """One camera path of the megakernel: the ray, throughput and accumulators, MIS and bounce state, the volume
    stack, and a pending random-walk exit (``sss_exit`` = 1: the next vertex is ``x``)."""
    o: wp.vec3
    d: wp.vec3
    tmax: float
    rdP: float
    rdD: float
    thr: wp.vec3
    rad: wp.vec3
    den: wp.int32
    den_thr: wp.vec3
    alb: wp.vec3
    nrm: wp.vec3
    mis_pdf: float
    mis_n: wp.vec3
    mis_skip: wp.int32
    mis_ht: wp.int32
    min_ray_pdf: float
    path_diffuse: wp.int32
    ancestor: wp.int32
    term_next: wp.int32
    bounce: wp.int32
    dbounce: wp.int32
    gbounce: wp.int32
    tbounce: wp.int32
    rng_extra: wp.int32
    sss_exit: wp.int32
    x: Hit
    md: Medium


@wp.func
def _hit(inst: wp.int32, t: float, ng: wp.vec3, face: wp.int32, u: float, v: float) -> Hit:
    h = Hit()
    h.inst = inst
    h.t = t
    h.ng = ng
    h.face = face
    h.u = u
    h.v = v
    return h


@wp.func
def mk_start(cams: CamsV2, prm: Params, ftab: wp.array(dtype=float), sobol: wp.array(dtype=wp.uint32), rng: Rng,
             img: wp.int32, px: wp.int32, pyr: wp.int32, sample: wp.int32) -> MkPath:
    """A path at its camera ray (unit throughput, nothing accumulated, MIS skipped for the first hit)."""
    st = MkPath()
    o, d, tmax, rdD = first_ray_v2(cams, prm, ftab, sobol, rng, img, px, pyr, sample)
    st.o = o
    st.d = d
    st.tmax = tmax
    st.rdD = rdD
    st.min_ray_pdf = 3.4e38
    st.thr = wp.vec3(1.0)
    st.den_thr = wp.vec3(1.0)
    st.den = 1
    st.mis_skip = 1
    st.x = _hit(-1, 0.0, wp.vec3(0.0), 0, 0.0, 0.0)
    st.md = medium_empty()
    return st


@wp.func
def mk_rofs(st: MkPath) -> wp.int32:
    """The vertex's random-number dimension block (two extra blocks per random walk)."""
    return PRNG_BOUNCE_NUM * (st.bounce + 1 + st.rng_extra)


@wp.func
def mk_next_hit(geo: Geo, bt: BumpTab, lsc: L.LightScene, p2: ParamsV2, prm: Params, world: wp.int32, st_in: MkPath):
    """(path, hit, alive): the closest hit of the path's ray (or the pending walk exit), with Beer-Lambert through
    the current volume, lights the ray passes and, on a miss, the background (alive = 0). While the denoising guides
    are unfinished, that emission is added to the albedo guide as well (film_write_emission_or_background_pass)."""
    st = st_in
    if st.sss_exit != 0:
        return st, st.x, 1
    inst, t, ng, face, bu, bv = closest_hit_uv(geo, world, st.o, st.d, st.tmax)
    h = _hit(inst, t, ng, face, bu, bv)
    if st.md.mid[0] >= 0:  # inside an absorbing volume: Beer-Lambert to the next surface (or nothing)
        st.thr = wp.cw_mul(st.thr, transmittance(medium_sigma(bt, st.md), wp.where(inst >= 0, t, 1e30)))
    lc = lights_on_ray(lsc, p2, prm, world, st.o, st.d, inst, t, st.thr, st.bounce, st.mis_n, st.mis_ht, st.mis_pdf,
                       st.mis_skip)
    st.rad = st.rad + lc
    if st.den != 0:  # Cycles writes lamp emission and the background into the denoising albedo too
        st.alb = st.alb + wp.cw_mul(st.den_thr, lc)
    if inst >= 0:
        return st, h, 1
    c = escaped(lsc, p2, prm, world, st.o, st.d, st.thr, st.bounce, st.mis_n, st.mis_ht, st.mis_pdf, st.mis_skip)
    st.rad = st.rad + c
    if st.den != 0:
        st.alb = st.alb + wp.cw_mul(st.den_thr, c)
    return st, h, 0


@wp.func
def mk_subsurface(geo: Geo, prm: Params, rng: Rng, sample: wp.int32, sobol: wp.array(dtype=wp.uint32),
                  st_in: MkPath, h: Hit, sf: Surf, cl: pr.PrincipledClosures, weight: wp.vec3):
    """(path, alive) after a BSSRDF pick: the GGX entry bounce, the random walk, and the exit as the next vertex
    (reached "from outside" along -d, as Cycles)."""
    st = st_in
    rofs = mk_rofs(st)
    ok, wo = entry_bounce(cl.sss, -st.d, sf.ng, rand2(rng, sample, rofs + PRNG_BOUNCE_NUM, sobol))
    if ok == 0:
        return st, 0
    w = random_walk(geo, h.inst, sf.p, offset_point(sf.p, -sf.ng, prm), wo, sf.ns, cl.sss, wp.cw_mul(st.thr, weight),
                    rng, sample, rofs + PRNG_BOUNCE_NUM, sobol)
    if w.ok == 0:
        return st, 0
    st.thr = w.thr
    st.rng_extra = st.rng_extra + 2
    st.sss_exit = 1
    st.x = _hit(h.inst, w.t, w.ng, w.face, w.u, w.v)
    st.o = w.p + w.d * w.t
    st.d = -w.d
    st.rdP = sf.dP
    return st, 1


@wp.func
def mk_bounce(st_in: MkPath, sf: Surf, h: Hit, s: pr.BsdfSample, glass_active: wp.int32, mid: wp.int32,
              bt: BumpTab, prm: Params, p2: ParamsV2):
    """(path, alive) after a BSDF sample: throughput, MIS and bounce counters, the volume stack, the next ray."""
    st = st_in
    if s.label == LABEL_NONE or not (s.pdf > 0.0):
        return st, 0
    st.thr = wp.cw_mul(st.thr, s.weight)
    skip, mpdf, mn, mrp = mis_after_bounce(s.label, s.pdf, sf.ns, st.mis_skip, st.mis_pdf, st.mis_n, st.min_ray_pdf)
    st.mis_skip = skip
    st.mis_pdf = mpdf
    st.mis_n = mn
    st.min_ray_pdf = mrp
    st.mis_ht = glass_active
    st.path_diffuse = wp.where((s.label & LABEL_DIFFUSE) != 0, 1, 0)
    st.ancestor = st.ancestor | st.path_diffuse
    b, db, gb, tb, term = count_bounce(s.label, st.bounce, st.dbounce, st.gbounce, st.tbounce, prm, p2)
    st.bounce = b
    st.dbounce = db
    st.gbounce = gb
    st.tbounce = tb
    st.term_next = term
    if (s.label & LABEL_TRANSMIT) != 0 and bt.vol_on[mid] != 0:
        st.md = medium_enter_exit(st.md, h.inst, mid, sf.backfacing)
    st.o = offset_point(sf.p, wp.where(wp.dot(s.wo, sf.ng) >= 0.0, sf.ng, -sf.ng), prm)
    st.d = wp.vec3(s.wo)
    st.tmax = T_INF
    st.rdP = sf.dP
    return st, 1
