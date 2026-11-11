# SPDX-License-Identifier: Apache-2.0
"""Boundary handling from shape signed distance fields (SPEC §3.8, solvers §4.2 step 2).

For every active particle the shapes of its world (plus global shapes) that collide with particles are culled by
bounding radius, evaluated with ``shape_sdf`` (Newton's ``eval_shape_sdf``, newton@ac7f7830:
newton/_src/geometry/soft_contacts_sdf.py:247-300; a mesh without a provisioned volume SDF is queried through its
BVH instead, as Newton's own particle contacts do, SCR-WP9-9) and the nearest ``NB`` within the support radius are
kept as *boundary slots*. Each slot stores the half-space boundary volume Psi(phi) with the order-1 curvature correction,
its gradient dPsi/dd * n (pointing into the wall), the surface point x_b = x - phi n and that point's velocity.
Mesh walls take their curvature from BVH probes instead (boundary_curvature.py, SCR-WP9-10).
The pressure solves read the slots; boundary reactions accumulate per slot (no float atomics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import warp as wp
from alloy._vendor.newton import GeoType, ParticleFlags, ShapeFlags
from alloy._vendor.newton._src.geometry.sdf_texture import TextureSDFData
from alloy._vendor.newton._src.geometry.soft_contacts_sdf import eval_shape_sdf

from .boundary_curvature import curved_volume, face_closest, mesh_mean_curvature
from .kernels import clamp_principal_curvature, halfspace_volume
from .particle_walls import ParticleWallArrays, pw_candidate, pw_plane
from .soft_walls import SoftArrays, soft_candidates, soft_closest

log = logging.getLogger("alloy.sph")

NB = 4  # boundary slots per particle
_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))
_GEO_PLANE = wp.constant(int(GeoType.PLANE))
_GEO_SPHERE = wp.constant(int(GeoType.SPHERE))
_GEO_CAPSULE = wp.constant(int(GeoType.CAPSULE))
_GEO_CYLINDER = wp.constant(int(GeoType.CYLINDER))
_GEO_MESH = wp.constant(int(GeoType.MESH))
_SDF_GEOS = {
    int(GeoType.PLANE), int(GeoType.SPHERE), int(GeoType.CAPSULE), int(GeoType.ELLIPSOID), int(GeoType.CYLINDER),
    int(GeoType.BOX), int(GeoType.MESH), int(GeoType.CONE), int(GeoType.CONVEX_MESH),
}


@dataclass
class BoundarySlots:
    """Per-particle boundary slots, shape (P, NB)."""

    shape: wp.array2d  # int32, -1 = empty
    psi: wp.array2d  # float32, boundary volume fraction
    gpsi: wp.array2d  # vec3, grad Psi wrt x_i [1/m]
    xb: wp.array2d  # vec3, surface point (world)
    vb: wp.array2d  # vec3, surface point velocity (world)
    phi: wp.array2d  # float32, signed distance at the start of the step
    xbl: wp.array2d  # vec3, start-of-step closest surface point in the shape frame (projection plane)
    nbl: wp.array2d  # vec3, start-of-step outward normal in the shape frame
    face: wp.array2d  # int32, soft face of a virtual-shape slot (soft_walls.py), -1 otherwise
    bary: wp.array2d  # vec3, barycentric weights of x_b on that face

    @staticmethod
    def allocate(n: int, device: wp.context.Device) -> BoundarySlots:
        n = max(n, 1)
        shape = wp.full((n, NB), -1, dtype=wp.int32, device=device)
        return BoundarySlots(
            shape=shape,
            psi=wp.zeros((n, NB), dtype=wp.float32, device=device),
            gpsi=wp.zeros((n, NB), dtype=wp.vec3, device=device),
            xb=wp.zeros((n, NB), dtype=wp.vec3, device=device),
            vb=wp.zeros((n, NB), dtype=wp.vec3, device=device),
            phi=wp.zeros((n, NB), dtype=wp.float32, device=device),
            xbl=wp.zeros((n, NB), dtype=wp.vec3, device=device),
            nbl=wp.zeros((n, NB), dtype=wp.vec3, device=device),
            face=wp.full((n, NB), -1, dtype=wp.int32, device=device),
            bary=wp.zeros((n, NB), dtype=wp.vec3, device=device),
        )


def world_shape_table(
    shape_world: np.ndarray, shape_flags: np.ndarray, shape_type: np.ndarray, world_count: int,
    excluded: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CSR (start[W+1], shape ids) of particle-colliding SDF shapes per world, global shapes appended to each."""
    usable = ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0) & np.isin(shape_type, list(_SDF_GEOS))
    if excluded is not None:
        usable &= ~excluded
    glob = np.nonzero(usable & (shape_world < 0))[0]
    start = [0]
    ids: list[int] = []
    for w in range(world_count):
        own = np.nonzero(usable & (shape_world == w))[0]
        ids.extend(int(s) for s in own)
        ids.extend(int(s) for s in glob)
        start.append(len(ids))
    return np.asarray(start, dtype=np.int32), np.asarray(ids, dtype=np.int32)


@wp.func
def shape_world_transform(s: int, shape_body: wp.array[wp.int32], shape_transform: wp.array[wp.transform],
                          body_q: wp.array[wp.transform]) -> wp.transform:
    b = shape_body[s]
    if b >= 0:
        return wp.transform_multiply(body_q[b], shape_transform[s])
    return shape_transform[s]


@wp.func
def point_velocity(b: int, x: wp.vec3, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                   body_com: wp.array[wp.vec3]) -> wp.vec3:
    """Velocity of world point x on body b (zero for static shapes); body_qd = (v_com, omega) in world."""
    if b < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    com = wp.transform_point(body_q[b], body_com[b])
    qd = body_qd[b]
    return wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), x - com)


@wp.func
def mean_curvature(geo: int, scale: wp.vec3, x_local: wp.vec3, grad_local: wp.vec3, h: float) -> float:
    """Order-1 mean curvature H = -(k1 + k2) of the closest surface point (analytic primitives; 0 otherwise)."""
    if geo == _GEO_SPHERE:
        return -2.0 * clamp_principal_curvature(1.0 / scale[0], h)
    if geo == _GEO_CAPSULE:
        k = clamp_principal_curvature(1.0 / scale[0], h)
        if wp.abs(x_local[2]) <= scale[1]:
            return -k
        return -2.0 * k
    if geo == _GEO_CYLINDER:
        if wp.abs(grad_local[2]) < 0.5:
            return -clamp_principal_curvature(1.0 / scale[0], h)
    return 0.0


@wp.struct
class ShapeArrays:
    start: wp.array[wp.int32]
    ids: wp.array[wp.int32]
    geo: wp.array[wp.int32]
    scale: wp.array[wp.vec3]
    xform: wp.array[wp.transform]
    body: wp.array[wp.int32]
    radius: wp.array[wp.float32]
    sdf_index: wp.array[wp.int32]
    tex: wp.array[TextureSDFData]
    mesh: wp.array[wp.uint64]  # Model.shape_source_ptr: wp.Mesh ids, for the BVH path
    bvh: wp.array[wp.int32]  # per model shape, 1: read the mesh BVH (no SDF, or its band misses the reach; sdf_band.py)
    bvh_dist: float  # [m] BVH closest-point search radius (2 h); farther points read +bvh_dist (outside, no slot)
    curvature: int
    offset: float  # wall offset delta [m] (kernels.boundary_fit)
    amplitude: float  # wall volume amplitude A (kernels.boundary_fit)
    taper0: float  # taper start = r (first lattice layer) [m]
    taper1: float  # taper end = 3r (second layer, complete neighbourhood) [m]
    reach: float  # slots kept for phi < reach: taper1, or h when the wall viscosity needs the whole support
    soft: SoftArrays  # soft faces: virtual shapes soft.base + c (soft_walls.py)
    pw: ParticleWallArrays  # another entry's particles: virtual shape pw.base (particle_walls.py)
    mu: wp.array[wp.float32]  # per model shape: Coulomb friction of the wall projection (Config.wall_friction)
    friction: int  # 1: Config.wall_friction


@wp.func
def _mesh_bvh_sdf(mesh: wp.uint64, scale: wp.vec3, xl: wp.vec3, max_dist: float):
    """(phi_lower, phi, grad) of a mesh shape from its BVH: the closest surface point within max_dist [m] (2 h), signed
    by its face normal; no face within it: +max_dist (outside). The callers need exact distances only near the surface:
    slots below the reach (<= h), the sweep's first inside sample (within one sample gap of the surface: <= r up to
    63 r per step) and the projection (below eps). An unbounded search cost 13-24x a baked SDF on JOB2's pour
    (SCR-JOB2-2). Trade-off: a particle deeper than max_dist inside the mesh reads outside and is not pushed out."""
    s_min = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    q = wp.mesh_query_point_sign_normal(mesh, wp.cw_div(xl, scale), max_dist / s_min)
    if not q.result:
        return max_dist, max_dist, wp.vec3(0.0, 0.0, 1.0)
    d = xl - face_closest(mesh, q.face, q.u, q.v, xl, scale, 0.05 * max_dist)  # thin faces (boundary_curvature)
    dist = wp.length(d)
    n = wp.normalize(wp.cw_div(wp.mesh_eval_face_normal(mesh, q.face), scale))  # on the surface: the face normal
    if dist > 1.0e-9:
        n = d * (q.sign / dist)
    phi = q.sign * dist
    return phi, phi, n


@wp.func
def shape_sdf(shapes: ShapeArrays, s: int, xl: wp.vec3):
    """(phi_lower, phi, grad) of shape s at shape-local xl: Newton's eval_shape_sdf, or the BVH query for a mesh
    without a volume SDF (sdf_index -1: eval_shape_sdf would read texture_sdf_table[-1]) or whose SDF's narrow band
    does not reach the liquid's reach (sdf_band.py)."""
    if shapes.bvh[s] != 0:
        return _mesh_bvh_sdf(shapes.mesh[s], shapes.scale[s], xl, shapes.bvh_dist)
    return eval_shape_sdf(shapes.geo[s], shapes.scale[s], xl, shapes.sdf_index[s], shapes.tex)


@wp.func
def _slot_insert(slots_shape: wp.array2d[wp.int32], slots_phi: wp.array2d[wp.float32],
                 slots_face: wp.array2d[wp.int32], i: int, n: int, s: int, phi: float, face: int) -> int:
    """Insert (s, phi, face) into the phi-ascending slot row; returns the slot index used, or -1 if dropped."""
    pos = n
    k = n - 1
    while k >= 0:
        if slots_phi[i, k] <= phi:
            break
        pos = k
        k -= 1
    if pos >= 4:
        return -1
    last = n
    if n >= 4:
        last = 3
    k = last
    while k > pos:
        slots_shape[i, k] = slots_shape[i, k - 1]
        slots_phi[i, k] = slots_phi[i, k - 1]
        slots_face[i, k] = slots_face[i, k - 1]
        k -= 1
    slots_shape[i, pos] = s
    slots_phi[i, pos] = phi
    slots_face[i, pos] = face
    return pos


@wp.kernel(enable_backward=False)
def _find_slots_kernel(
    q: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    phase: wp.array[wp.int8],
    world: wp.array[wp.int32],
    shapes: ShapeArrays,
    body_q: wp.array[wp.transform],
    h: float,
    slot_shape: wp.array2d[wp.int32],
    slot_phi: wp.array2d[wp.float32],
    slot_face: wp.array2d[wp.int32],
):
    i = wp.tid()
    for k in range(4):
        slot_shape[i, k] = -1
        slot_face[i, k] = -1
    if (flags[i] & _ACTIVE) == 0 or int(phase[i]) < 1:
        return
    w = world[i]
    x = q[i]
    n = int(0)
    for c in range(shapes.start[w], shapes.start[w + 1]):
        s = shapes.ids[c]
        X = shape_world_transform(s, shapes.body, shapes.xform, body_q)
        geo = shapes.geo[s]
        if geo != _GEO_PLANE:
            if wp.length(x - wp.transform_get_translation(X)) > shapes.radius[s] + h:
                continue
        xl = wp.transform_point(wp.transform_inverse(X), x)
        _lo, phi, gl = shape_sdf(shapes, s, xl)
        if phi < shapes.reach and phi + shapes.offset < h:
            if _slot_insert(slot_shape, slot_phi, slot_face, i, n, s, phi, -1) >= 0 and n < 4:
                n += 1
    cb, fb, pb = soft_candidates(shapes.soft, x, w)
    for j in range(4):
        if cb[j] >= 0 and pb[j] < shapes.reach and pb[j] + shapes.offset < h:
            if _slot_insert(slot_shape, slot_phi, slot_face, i, n, shapes.soft.base + cb[j], pb[j], fb[j]) >= 0:
                if n < 4:
                    n += 1
    if shapes.pw.count > 0:
        phi = pw_candidate(shapes.pw, x, w)
        if phi < shapes.reach and phi + shapes.offset < h:
            if _slot_insert(slot_shape, slot_phi, slot_face, i, n, shapes.pw.base, phi, -1) >= 0 and n < 4:
                n += 1


@wp.func
def _taper(v: wp.vec2, phi: float, shapes: ShapeArrays) -> wp.vec2:
    """C1 taper to zero at the second lattice layer, whose lattice neighbourhood is complete (zero deficit):
    s = 1 - 3t^2 + 2t^3 keeps value and slope at the first layer and removes the spurious tail force there."""
    if phi <= shapes.taper0:
        return v
    L = shapes.taper1 - shapes.taper0
    t = wp.min((phi - shapes.taper0) / L, 1.0)
    sm = 1.0 - t * t * (3.0 - 2.0 * t)
    dsm = 6.0 * t * (t - 1.0) / L
    return wp.vec2(v[0] * sm, v[1] * sm + v[0] * dsm)


@wp.func
def _fill_soft(i: int, k: int, x: wp.vec3, shapes: ShapeArrays, h: float, kind: int, face: int,
               slot_psi: wp.array2d[wp.float32], slot_gpsi: wp.array2d[wp.vec3], slot_xb: wp.array2d[wp.vec3],
               slot_vb: wp.array2d[wp.vec3], slot_phi: wp.array2d[wp.float32], slot_xbl: wp.array2d[wp.vec3],
               slot_nbl: wp.array2d[wp.vec3], slot_bary: wp.array2d[wp.vec3]):
    """A soft face's slot (soft_walls.py): the flat half-space volume of its closest point, world frame."""
    phi, nrm, xb, vb, bw, _d = soft_closest(shapes.soft, face, x, 0.0)
    v = _taper(halfspace_volume(phi + shapes.offset, h, 0.0, kind) * shapes.amplitude, phi, shapes)
    slot_phi[i, k] = phi
    slot_psi[i, k] = v[0]
    slot_gpsi[i, k] = v[1] * nrm
    slot_xb[i, k] = xb
    slot_vb[i, k] = vb
    slot_xbl[i, k] = xb
    slot_nbl[i, k] = nrm
    slot_bary[i, k] = bw


@wp.func
def _fill_particle_wall(i: int, k: int, x: wp.vec3, shapes: ShapeArrays, h: float, kind: int, world: int,
                        slot_psi: wp.array2d[wp.float32], slot_gpsi: wp.array2d[wp.vec3], slot_xb: wp.array2d[wp.vec3],
                        slot_vb: wp.array2d[wp.vec3], slot_phi: wp.array2d[wp.float32],
                        slot_xbl: wp.array2d[wp.vec3], slot_nbl: wp.array2d[wp.vec3], slot_bary: wp.array2d[wp.vec3]):
    """The particle wall's slot (particle_walls.py): its local plane, filled as an SDF wall at phi."""
    phi, nrm, xb, vb = pw_plane(shapes.pw, x, world)
    v = _taper(halfspace_volume(phi + shapes.offset, h, 0.0, kind) * shapes.amplitude, phi, shapes)
    slot_phi[i, k] = phi
    slot_psi[i, k] = v[0]
    slot_gpsi[i, k] = v[1] * nrm
    slot_xb[i, k] = xb
    slot_vb[i, k] = vb
    slot_xbl[i, k] = xb
    slot_nbl[i, k] = nrm
    slot_bary[i, k] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _fill_one(i: int, k: int, x: wp.vec3, world: int, shapes: ShapeArrays, body_q: wp.array[wp.transform],
              body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], h: float, kind: int,
              slot_shape: wp.array2d[wp.int32], slot_psi: wp.array2d[wp.float32], slot_gpsi: wp.array2d[wp.vec3],
              slot_xb: wp.array2d[wp.vec3], slot_vb: wp.array2d[wp.vec3], slot_phi: wp.array2d[wp.float32],
              slot_xbl: wp.array2d[wp.vec3], slot_nbl: wp.array2d[wp.vec3], slot_face: wp.array2d[wp.int32],
              slot_bary: wp.array2d[wp.vec3]):
    """Raw (uncombined) Psi_eff = A Psi(phi + delta) and its gradient for slot k (kernels.boundary_fit)."""
    s = slot_shape[i, k]
    if s < 0:
        slot_psi[i, k] = 0.0
        slot_gpsi[i, k] = wp.vec3(0.0, 0.0, 0.0)
        return
    if s == shapes.pw.base and shapes.pw.count > 0:
        _fill_particle_wall(i, k, x, shapes, h, kind, world, slot_psi, slot_gpsi, slot_xb, slot_vb, slot_phi,
                            slot_xbl, slot_nbl, slot_bary)
        return
    if s >= shapes.soft.base:
        _fill_soft(i, k, x, shapes, h, kind, slot_face[i, k], slot_psi, slot_gpsi, slot_xb, slot_vb, slot_phi,
                   slot_xbl, slot_nbl, slot_bary)
        return
    X = shape_world_transform(s, shapes.body, shapes.xform, body_q)
    geo = shapes.geo[s]
    xl = wp.transform_point(wp.transform_inverse(X), x)
    _lo, phi, gl = shape_sdf(shapes, s, xl)
    nrm = wp.normalize(wp.transform_vector(X, gl))
    nl = wp.normalize(gl)
    slot_nbl[i, k] = nl
    slot_xbl[i, k] = xl - phi * nl
    H = float(0.0)
    if shapes.curvature > 0:
        H = mean_curvature(geo, shapes.scale[s], xl, gl, h)
    v = halfspace_volume(phi + shapes.offset, h, H, kind) * shapes.amplitude
    if shapes.curvature > 0 and geo == _GEO_MESH and phi < shapes.taper1:  # SCR-WP9-10 (boundary_curvature.py)
        Hm = mesh_mean_curvature(shapes.mesh[s], shapes.scale[s], xl - phi * nl, nl, h, 1 - shapes.bvh[s])
        if Hm != 0.0:
            v = curved_volume(phi, shapes.offset, shapes.amplitude, h, Hm, kind)
    v = _taper(v, phi, shapes)
    xb = x - phi * nrm
    slot_phi[i, k] = phi
    slot_psi[i, k] = v[0]
    slot_gpsi[i, k] = v[1] * nrm
    slot_xb[i, k] = xb
    slot_vb[i, k] = point_velocity(shapes.body[s], xb, body_q, body_qd, body_com)


@wp.kernel(enable_backward=False)
def _fill_slots_kernel(
    q: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    shapes: ShapeArrays,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    h: float,
    kind: int,
    slot_shape: wp.array2d[wp.int32],
    slot_psi: wp.array2d[wp.float32],
    slot_gpsi: wp.array2d[wp.vec3],
    slot_xb: wp.array2d[wp.vec3],
    slot_vb: wp.array2d[wp.vec3],
    slot_phi: wp.array2d[wp.float32],
    slot_xbl: wp.array2d[wp.vec3],
    slot_nbl: wp.array2d[wp.vec3],
    slot_face: wp.array2d[wp.int32],
    slot_bary: wp.array2d[wp.vec3],
):
    """Fill the slots, then combine overlapping walls by inclusion-exclusion:
    Psi_U = sum Psi_b - sum_{b<c} w_bc Psi_b Psi_c, w_bc = clamp(1 + n_b.n_c, 0, 1). The quadrant term Psi_b Psi_c
    is exact for a separable kernel at orthogonal walls (edges, corners) and w -> 0 for facing walls (disjoint
    half-spaces). Each slot keeps Psi_b (1 - o_b/2) and grad Psi_b (1 - o_b), o_b = min(sum_{c!=b} w_bc Psi_c, 1),
    so the slot sums give Psi_U and its exact gradient wherever o_b < 1 (everywhere a resting lattice reaches)."""
    i = wp.tid()
    x = q[i]
    for k in range(4):
        _fill_one(i, k, x, world[i], shapes, body_q, body_qd, body_com, h, kind, slot_shape, slot_psi, slot_gpsi,
                  slot_xb, slot_vb, slot_phi, slot_xbl, slot_nbl, slot_face, slot_bary)
    ps = wp.vec4(slot_psi[i, 0], slot_psi[i, 1], slot_psi[i, 2], slot_psi[i, 3])
    for b in range(4):
        if slot_shape[i, b] >= 0:
            gb = slot_gpsi[i, b]
            ob = float(0.0)
            for c in range(4):
                if c != b and slot_shape[i, c] >= 0:
                    gc = slot_gpsi[i, c]
                    lb = wp.length(gb)
                    lc = wp.length(gc)
                    wbc = float(1.0)
                    if lb > 0.0 and lc > 0.0:
                        wbc = wp.clamp(1.0 + wp.dot(gb, gc) / (lb * lc), 0.0, 1.0)
                    ob += wbc * ps[c]
            # o_b <= 1: the fitted volumes exceed 1 close to a wall, where the quadrant term would turn Psi_U
            # negative and the wall attractive (a corner particle then walks into the corner); the rest lattice has
            # o_b ~ 0.15-0.3 and is unchanged.
            ob = wp.min(ob, 1.0)
            slot_psi[i, b] = ps[b] * (1.0 - 0.5 * ob)
            slot_gpsi[i, b] = gb * (1.0 - ob)


def compute_slots(
    q: wp.array, flags: wp.array, phase: wp.array, world: wp.array, shapes: ShapeArrays, body_q: wp.array,
    body_qd: wp.array, body_com: wp.array, h: float, kind: int, slots: BoundarySlots, device: wp.context.Device,
) -> None:
    """Find and fill the boundary slots of every particle (capture-safe)."""
    n = q.shape[0]
    if n == 0:
        return
    wp.launch(_find_slots_kernel, dim=n, inputs=[q, flags, phase, world, shapes, body_q, h, slots.shape, slots.phi,
                                                 slots.face], device=device)
    wp.launch(
        _fill_slots_kernel, dim=n,
        inputs=[q, world, shapes, body_q, body_qd, body_com, h, kind, slots.shape, slots.psi, slots.gpsi, slots.xb,
                slots.vb, slots.phi, slots.xbl, slots.nbl, slots.face, slots.bary],
        device=device,
    )
