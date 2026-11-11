# SPDX-License-Identifier: Apache-2.0
"""Non-pressure forces, advection and boundary projection for SolverDFSPH.

Non-pressure forces (SPlisHSPlasH TimeStepDFSPH.cpp:159-190 order): gravity, external particle forces and XSPH
velocity smoothing ``v_i += c sum_j (m_j / rho_j)(v_j - v_i) W_ij`` (SPlisHSPlasH XSPH.cpp). They are gathered into
a scratch array first so no thread reads a velocity another thread has already updated.

Projection (solvers §4.2 step 7): after advection each particle is re-tested against the shapes in its slots at the
new body poses; a particle with phi < 0 is moved to the surface and its inward relative normal velocity removed.
The pressure boundary normally keeps phi > 0; projections are counted per world as a leak precursor (LIQ-8).

Sweep (CFL guard, LIQ-8): slots hold the shapes within h at the start of the step, so a particle that moves more
than ``SWEEP`` h in one step (beyond any CFL limit) could cross a wall that was not in its slots, or come out on the
far side of one that was (phi > 0 at the end, so the slot test never fires): measured on the pour scene, a particle
thrown at 51 m/s (16 mm per step) crossed an 8 mm wall. Such a particle's whole path is tested against every shape
of its world (the path in each shape's frame from its start pose to its end pose): boxes exactly by a slab test at
any path length, other shapes sampled at <= r up to ``SWEEP_SAMPLES`` per shape (a fixed sample count alone lets a
0.8 m step, 2,666 m/s, skip an 8 mm wall); at the first entry it is stopped at the last outside point, pushed
``eps`` clear and loses its inward relative normal velocity, as a projection. Speeds are not clamped (health monitors still see them); sweeps are counted per world. Below the
threshold a wall at least h thick cannot be crossed without ending inside a slot shape, which the slot pass handles.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton import GeoType

from .bodies import WORK_SCALE, _fx
from .boundary import ShapeArrays, point_velocity, shape_sdf, shape_world_transform
from .particle_walls import pw_project
from .soft_inertia import soft_project
from .kernels import kernel_w
from .pressure import Particles, is_fluid, nbr_base, nbr_j

SWEEP = 0.5  # [h] a step longer than this is swept against every shape of the world
SWEEP_SAMPLES = 64  # most samples per shape path (spacing <= r up to 64 r per step)
_BISECT = 8
PROJECT_PASSES = 4  # most projection passes: three tiled boxes at a corner need three, plus one that finds none
_GEO_PLANE = wp.constant(int(GeoType.PLANE))
GEO_BOX = wp.constant(int(GeoType.BOX))


@wp.func
def path_samples(length: float, spacing: float) -> int:
    return wp.clamp(int(wp.ceil(length / spacing)) + 1, 8, SWEEP_SAMPLES)


@wp.func
def path_sdf(shapes: ShapeArrays, s: int, l0: wp.vec3, l1: wp.vec3, t: float) -> float:
    _lo, phi, _g = shape_sdf(shapes, s, l0 + t * (l1 - l0))
    return phi


@wp.func
def box_span(half: wp.vec3, l0: wp.vec3, l1: wp.vec3) -> wp.vec2:
    """Parameter interval (t_in, t_out) where the line l0 + t (l1 - l0) is inside the box of half extents ``half``
    (box frame); t_in > t_out when it misses. Slab test: exact at any path length."""
    lo = float(-1.0e30)
    hi = float(1.0e30)
    d = l1 - l0
    for a in range(3):
        if wp.abs(d[a]) > 0.0:
            ta = (-half[a] - l0[a]) / d[a]
            tb = (half[a] - l0[a]) / d[a]
            lo = wp.max(lo, wp.min(ta, tb))
            hi = wp.min(hi, wp.max(ta, tb))
        elif wp.abs(l0[a]) > half[a]:  # parallel to this slab and outside it
            lo = 2.0e30
    return wp.vec2(lo, hi)


@wp.func
def path_entry(shapes: ShapeArrays, s: int, l0: wp.vec3, l1: wp.vec3, samples: int) -> float:
    """Last outside path parameter before the first entry into shape ``s`` (segment l0 -> l1 in its frame), or 2 when
    the path stays outside or starts inside (the slot pass owns a particle that started inside). Boxes are exact
    (``box_span``) at any path length; other shapes are sampled ``samples`` times and bisected."""
    if path_sdf(shapes, s, l0, l1, 0.0) < 0.0:
        return 2.0
    if shapes.geo[s] == GEO_BOX:
        span = box_span(shapes.scale[s], l0, l1)
        if span[0] <= span[1] and span[0] <= 1.0 and span[1] >= 0.0:
            return wp.max(span[0], 0.0)
        return 2.0
    t_out = float(2.0)
    t_lo = float(0.0)
    t_hi = float(2.0)
    for k in range(1, samples + 1):
        t = float(k) / float(samples)
        if t_hi > 1.0:
            if path_sdf(shapes, s, l0, l1, t) < 0.0:
                t_hi = t
            else:
                t_lo = t
    if t_hi <= 1.0:
        for _b in range(_BISECT):
            tm = 0.5 * (t_lo + t_hi)
            if path_sdf(shapes, s, l0, l1, tm) < 0.0:
                t_hi = tm
            else:
                t_lo = tm
        t_out = t_lo
    return t_out


@wp.func
def near_path(shapes: ShapeArrays, s: int, X: wp.transform, x0: wp.vec3, reach: float) -> bool:
    """Bounding-sphere test: can a path of length ``reach`` from x0 (plus margin) touch shape ``s``?"""
    if shapes.geo[s] == _GEO_PLANE:
        return True
    return wp.length(x0 - wp.transform_get_translation(X)) <= shapes.radius[s] + reach


@wp.func
def sweep_entry(shapes: ShapeArrays, w: int, x0: wp.vec3, x1: wp.vec3, body_q0: wp.array[wp.transform],
                body_q1: wp.array[wp.transform], spacing: float):
    """First wall entry along the step path x0 -> x1 over every shape of world ``w``: (shape or -1, local point at
    the last outside parameter, in that shape's frame)."""
    reach = wp.length(x1 - x0) + spacing
    samples = path_samples(wp.length(x1 - x0), spacing)
    best = float(2.0)
    hit = int(-1)
    l_hit = wp.vec3(0.0, 0.0, 0.0)
    for c in range(shapes.start[w], shapes.start[w + 1]):
        s = shapes.ids[c]
        X0 = shape_world_transform(s, shapes.body, shapes.xform, body_q0)
        if near_path(shapes, s, X0, x0, reach):
            l0 = wp.transform_point(wp.transform_inverse(X0), x0)
            l1 = wp.transform_point(wp.transform_inverse(shape_world_transform(s, shapes.body, shapes.xform, body_q1)),
                                    x1)
            t = path_entry(shapes, s, l0, l1, samples)
            if t < best:
                best = t
                hit = s
                l_hit = l0 + t * (l1 - l0)
    return hit, l_hit


@wp.kernel(enable_backward=False)
def nonpressure_kernel(
    p: Particles, h: float, kind: int, dt: float, gravity: wp.array[wp.vec3], particle_f: wp.array[wp.vec3],
    has_f: int, ratio: wp.array[wp.float32], xsph_c: float, vtmp: wp.array[wp.vec3],
):
    i = wp.tid()
    vi = p.qd[i]
    if not is_fluid(p, i):
        vtmp[i] = vi
        return
    a = gravity[p.world[i]]
    if has_f != 0:
        a += particle_f[i] / p.mass[i]
    v = vi + dt * a
    if xsph_c > 0.0:
        xi = p.q[i]
        corr = wp.vec3(0.0, 0.0, 0.0)
        rho_i = ratio[i] * p.rho0[i]
        b_ = nbr_base(p, i)
        for k in range(p.count[i]):
            j = nbr_j(p, b_, i, k)
            # Monaghan's symmetric weight m_j / rho_bar_ij: the pair terms cancel, so XSPH conserves momentum (m_j / rho_j
            # did not where rho_i != rho_j: 0.34 % of P0 after an impact, SCR-OSS3-2 / wp9-results)
            rho_bar = 0.5 * (rho_i + ratio[j] * p.rho0[j])
            if rho_bar > 0.0:
                w = kernel_w(wp.length(xi - p.q[j]), h, kind)
                corr += (p.mass[j] / rho_bar * w) * (p.qd[j] - vi)
        v += xsph_c * corr
    vtmp[i] = v


@wp.kernel(enable_backward=False)
def advect_kernel(p: Particles, dt: float):
    i = wp.tid()
    if is_fluid(p, i):
        p.q[i] = p.q[i] + dt * p.qd[i]


@wp.func
def _wall_work(work: wp.array[wp.int64], s: int, body: int, w: float):
    """A moving wall's work on the particle it pushes out: impulse m (-v_n) n times the wall velocity (bodies.py)."""
    if body >= 0:
        wp.atomic_add(work, s, _fx(w, WORK_SCALE))


@wp.func
def _wall_friction(v: wp.vec3, vb: wp.vec3, n: wp.vec3, vn: float, shapes: ShapeArrays, s: int) -> wp.vec3:
    """Coulomb friction of a projection (Config.wall_friction): after the inward speed -vn is removed, the tangential
    speed relative to the wall drops by up to mu (-vn), never reversing."""
    if shapes.friction == 0:
        return v
    mu = shapes.mu[s]
    if mu <= 0.0:
        return v
    u = v - vb
    vt = u - wp.dot(u, n) * n
    lt = wp.length(vt)
    if lt <= 0.0:
        return v
    return v - wp.min(1.0, mu * (-vn) / lt) * vt


@wp.kernel(enable_backward=False)
def project_kernel(
    p: Particles, slot_shape: wp.array2d[wp.int32], slot_phi: wp.array2d[wp.float32],
    slot_xbl: wp.array2d[wp.vec3], slot_nbl: wp.array2d[wp.vec3], slot_face: wp.array2d[wp.int32],
    slot_vb: wp.array2d[wp.vec3], shapes: ShapeArrays, body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
    eps: float, projected_world: wp.array[wp.int32], body_q0: wp.array[wp.transform], dt: float,
    sweep_len: float, spacing: float, swept_world: wp.array[wp.int32], work: wp.array[wp.int64],
):
    """Move particles that ended inside a wall back to its surface and remove the inward relative velocity.

    The push goes onto the start-of-step tangent plane of each slot (closest point and normal recorded in the shape
    frame by the boundary pass, moved with the body), not along the SDF gradient at the new position: inside a wall
    built from several boxes the nearest face of one box can be an internal seam, and pushing towards it walks the
    particle through the wall (measured: a corner particle of a resting 16 x 16 x 60 column left through the
    floor / wall seams after 4 s). The SDF gradient is the fallback when the particle started inside or the plane
    does not hold it (concave walls). A point within ``eps`` of a surface counts as inside and is pushed ``eps``
    past it: a particle exactly on the face two wall boxes share has phi = 0 for both and would otherwise slide out
    along the seam.

    The passes repeat until one moves nothing (at most ``PROJECT_PASSES``). A point past the end of a wall box reads
    outside it, so at a corner where the floor and two tiled wall boxes meet (``build.add_box_tank``) each push can
    carry the particle into the extent of a box it was outside of: out of the floor into the x wall's extent, back out
    of the x wall into the y wall's. Two passes then left it inside the y wall; there the fallback gradient points at
    an internal seam and the particle slid out along the seams (one particle driven into the (+x, -y) corner at 1 m/s:
    1 mm deep in 10 steps, test_leaks). Two boxes (an edge) settle in two passes."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    x = p.q[i]
    v = p.qd[i]
    moved = int(0)
    if wp.length(dt * v) > sweep_len:  # beyond CFL: sweep the whole path (module doc)
        s, l = sweep_entry(shapes, p.world[i], x - dt * v, x, body_q0, body_q, spacing)
        if s >= 0:
            X = shape_world_transform(s, shapes.body, shapes.xform, body_q)
            _lo, phi, gl = shape_sdf(shapes, s, l)
            n = wp.normalize(wp.transform_vector(X, gl))
            x = wp.transform_point(X, l) + wp.max(eps - phi, 0.0) * n
            vb = point_velocity(shapes.body[s], x, body_q, body_qd, body_com)
            vn = wp.dot(v - vb, n)
            if vn < 0.0:
                v = v - vn * n
                _wall_work(work, s, shapes.body[s], -p.mass[i] * vn * wp.dot(n, vb))
                v = _wall_friction(v, vb, n, vn, shapes, s)
            moved = 1
            wp.atomic_add(swept_world, p.world[i], 1)
    busy = int(1)
    npass = int(0)
    while busy != 0 and npass < PROJECT_PASSES:  # until a pass moves nothing (docstring)
        busy = 0
        npass += 1
        for b in range(4):
            s = slot_shape[i, b]
            if s == shapes.pw.base and shapes.pw.count > 0:  # a particle wall's plane at its end-of-step place
                vb = slot_vb[i, b]
                x, hit = pw_project(slot_xbl[i, b], slot_nbl[i, b], vb, x, dt, eps)
                if hit != 0:
                    vn = wp.dot(v - vb, slot_nbl[i, b])
                    if vn < 0.0:
                        v = v - vn * slot_nbl[i, b]
                    moved = 1
                    busy = 1
            elif s >= shapes.soft.base:  # a soft face at its end-of-step place; cloth: kept on its start side
                x, n, vb, hit = soft_project(shapes.soft, slot_face[i, b], x, dt, slot_nbl[i, b], eps)
                if hit != 0:
                    vn = wp.dot(v - vb, n)
                    if vn < 0.0:
                        v = v - vn * n
                    moved = 1
                    busy = 1
            elif s >= 0:
                X = shape_world_transform(s, shapes.body, shapes.xform, body_q)
                xl = wp.transform_point(wp.transform_inverse(X), x)
                _lo, phi, gl = shape_sdf(shapes, s, xl)
                if phi < eps:
                    n = wp.normalize(wp.transform_vector(X, gl))
                    push = eps - phi
                    if slot_phi[i, b] >= 0.0:
                        n_s = wp.transform_vector(X, slot_nbl[i, b])
                        push_s = eps - wp.dot(n_s, x - wp.transform_point(X, slot_xbl[i, b]))
                        if push_s > 0.0:  # (a concave wall can leave a point inside but beyond the plane)
                            n = n_s
                            push = push_s
                    if push > 0.0:
                        x = x + push * n
                        vb = point_velocity(shapes.body[s], x, body_q, body_qd, body_com)
                        vn = wp.dot(v - vb, n)
                        if vn < 0.0:
                            v = v - vn * n
                            _wall_work(work, s, shapes.body[s], -p.mass[i] * vn * wp.dot(n, vb))
                            v = _wall_friction(v, vb, n, vn, shapes, s)
                        moved = 1
                        busy = 1
    if moved != 0:
        p.q[i] = x
        p.qd[i] = v
        wp.atomic_add(projected_world, p.world[i], 1)


@wp.kernel(enable_backward=False)
def speed_kernel(p: Particles, max_speed: wp.array[wp.float32], nonfinite: wp.array[wp.int32]):
    """Per-world max |v| of active fluid particles (atomic_max is deterministic) and non-finite count."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    v = p.qd[i]
    x = p.q[i]
    s = wp.length(v)
    w = p.world[i]
    if wp.isfinite(s) and wp.isfinite(x[0] + x[1] + x[2]):
        wp.atomic_max(max_speed, w, s)
    else:
        wp.atomic_add(nonfinite, w, 1)
        wp.atomic_max(max_speed, w, 1.0e30)
