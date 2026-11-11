# SPDX-License-Identifier: Apache-2.0
"""Curvature-aware boundary volume for mesh walls (SCR-WP9-10, coordinator rulings 2026-11-04 20:19Z / 20:53Z).

Problem. A mesh wall used the flat half-space boundary volume: ``boundary.mean_curvature`` knows the analytic
primitives only. A round mesh bore of radius R is concave as seen from the liquid. At R/h 1.67 (SC10's 1.0 mm
capillary at r 0.075 mm) the first liquid layer needs about 10 % more boundary volume than a flat wall, and about 5 %
more at R/h 3.5. A convex mesh rod needs less.

Curvature of a mesh at a wall point. Closest-point probes on the mesh BVH, at x_b +- l t along two orthogonal
tangents t of the surface point x_b (outward normal n), l = h / 2:

    kappa(t) = Menger curvature of (c-, x_b, c+), c+- = the mesh points closest to x_b +- l t

This is the circle through the three surface points. It is exact for a circle at any l, and on a 64-gon it is off by
the polygon's sagitta only: about 3 % of 1/R at R/h 1.67 with l = h / 2. The normal gives only its sign:
kappa > 0 where the wall bends toward the liquid (concave). One-sided probes (the circle through x_b and c+ tangent
to n) are biased instead by the facet normal, up to pi / 64 off the smooth one: up to 30 % at R/h 1.67, which made
Psi vary by +-3 % from particle to particle along the bore. Facet normals step by 2 pi / 64 from face to face, and
SC10's lathe ring has 30 mm long faces, so per-vertex curvature is unusable there. H = kappa(t1) + kappa(t2) is the
``halfspace_volume`` convention: H = -(k1 + k2), negative for convex solids. Each kappa is clamped to
|kappa| <= 2/h (``kernels.clamp_principal_curvature``), as for the primitives. At a sharp mesh edge the probes see
the other face, and the clamp bounds the result.

Volume. The fitted flat volume ``A Psi_0(phi + delta)`` (kernels.boundary_fit) carries a lattice correction: the
first layer of a cubic lattice has the deficit 0.150, while the integral gives 0.188. Curvature multiplies it by
the order-1 ratio at the particle's own distance:

    Psi = A Psi_0(phi + delta) * Psi_H(phi) / Psi_0(phi)

The values below are what a conforming ring fill needs at its first layer (``boundary_curvature_fill``), against
this form and the two alternatives (cubic, h = 2 spacings):

| wall          | R/h  | needed | this form | flat   | H at phi + delta (primitives today) |
|---------------|------|--------|-----------|--------|-------------------------------------|
| concave bore  | 1.67 | 0.1663 | 0.1673    | 0.1497 | 0.1817                              |
| concave bore  | 3.5  | 0.1585 | 0.1581    | 0.1497 | 0.1650                              |
| convex rod    | 1.67 | 0.1372 | 0.1320    | 0.1497 | 0.1176                              |
| convex rod    | 3.5  | 0.1445 | 0.1413    | 0.1497 | 0.1344                              |

Scope. Only ``GeoType.MESH`` shapes with a wp.Mesh change, and only when ``boundary_curvature_order`` is 1 and the
slot has a nonzero volume (phi < taper1). Planes, boxes and the analytic primitives keep their path bit for bit, and
so does a flat mesh, where H is exactly 0. Cost: four ray queries per mesh slot within 1.5 spacings of the wall
(five for a mesh read through a baked texture SDF), each bounded to 2 l. Measured on the R/h 3.5 bore, 5,082
particles, CPU, where every wall particle is next to a curved mesh (the worst case): slot search + fill 5.5 -> 8.7 ms
per step, 5.7 % -> 9.5 % of the step.
"""

from __future__ import annotations

import warp as wp

from .kernels import clamp_principal_curvature, halfspace_volume

PROBE_FRACTION = wp.constant(0.5)  # probe length l = PROBE_FRACTION * h
THIN_ASPECT = wp.constant(0.05)  # a face whose altitude is below 5 % of its longest edge is thin (face_closest)
EDGE_COS = wp.constant(0.5)  # a probe face turned more than 60 deg from the wall normal is an edge (kappa 0)


@wp.func
def _tangent(n: wp.vec3) -> wp.vec3:
    """A unit vector perpendicular to the unit vector n (any one: the sum over two orthogonal tangents is the trace)."""
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.6:
        a = wp.vec3(0.0, 1.0, 0.0)
    return wp.normalize(wp.cross(n, a))


@wp.func
def _edge_dist(c: wp.vec3, a: wp.vec3, b: wp.vec3, nf: wp.vec3) -> float:
    """In-plane signed distance of c from edge a -> b of a triangle wound counter-clockwise about nf (>= 0 inside)."""
    e = b - a
    le = wp.length(e)
    if le <= 0.0:
        return 0.0
    return wp.dot(c - a, wp.cross(nf, e)) / le


@wp.func
def _face_foot(mesh: wp.uint64, face: int, x: wp.vec3, scale: wp.vec3, slack: float):
    """(thin, ok, foot) of a mesh face (shape frame, scaled).
    - thin: altitude below THIN_ASPECT x its longest edge.
    - foot: the foot of x on the face's plane.
    - ok: the foot lies inside the face or within ``slack`` [m] of it in its plane. The edge distances use the
      edges' in-plane normals, so they stay accurate on long, thin faces."""
    p0 = wp.cw_mul(wp.mesh_eval_position(mesh, face, 1.0, 0.0), scale)
    p1 = wp.cw_mul(wp.mesh_eval_position(mesh, face, 0.0, 1.0), scale)
    p2 = wp.cw_mul(wp.mesh_eval_position(mesh, face, 0.0, 0.0), scale)
    nn = wp.cross(p1 - p0, p2 - p0)
    ln = wp.length(nn)
    lmax = wp.max(wp.max(wp.length(p1 - p0), wp.length(p2 - p1)), wp.length(p0 - p2))
    if ln <= 0.0 or ln >= THIN_ASPECT * lmax * lmax:
        return False, False, x
    nf = nn / ln
    cp = x - wp.dot(x - p0, nf) * nf
    dmin = wp.min(wp.min(_edge_dist(cp, p0, p1, nf), _edge_dist(cp, p1, p2, nf)), _edge_dist(cp, p2, p0, nf))
    return True, dmin >= -slack, cp


@wp.func
def face_closest(mesh: wp.uint64, face: int, u: float, v: float, x: wp.vec3, scale: wp.vec3, slack: float) -> wp.vec3:
    """Closest point to x on the mesh (shape frame, scaled) given the BVH's answer (face, u, v).
    - Why: on long, thin faces Warp's float32 closest-point query is unreliable.
      - SC10's lathe ring has 30 mm x 0.05 mm bore faces. There the BVH point lies up to 19 um off the plane foot,
        and the wall normal tilts up to 15 deg (measured at r 0.075 mm).
      - Tilted normals put a net axial force on a bore column, one that no boundary volume explains.
      - The query also returns the coplanar twin of the right triangle, even when asked from a point inside the
        right one.
    - On a thin face (altitude < THIN_ASPECT x longest edge) the plane foot is used when it lies inside the face or
      within ``slack`` of it in its plane:
      - past the long diagonal of a thin quad it belongs to the twin and is exact;
      - past an edge to a neighbour at angle theta, the distance changes by at most slack sin(theta).
    - Any other face keeps the BVH point bit for bit (mesh boxes, well-shaped meshes). The real fix for a mesh like
      SC10's is to split its long faces; the SPH layout does not remesh."""
    c = wp.cw_mul(wp.mesh_eval_position(mesh, face, u, v), scale)
    thin, ok, cp = _face_foot(mesh, face, x, scale, slack)
    if thin and ok:
        return cp
    return c


@wp.func
def _ray_hit(mesh: wp.uint64, scale: wp.vec3, p: wp.vec3, n: wp.vec3, ell: float):
    """(found, wall point, that face's unit normal) on the line through p along n, within +- l of p (shape frame,
    scaled). The ray starts on the liquid side (p + l n) and runs toward the wall. Starting inside the solid (p - l n)
    would put the start past the back face of a wall thinner than l, so the first hit would be that back face. A ray
    hit is well conditioned on thin faces, where the float32 closest-point test is not (``face_closest``)."""
    start = p + ell * n
    q = wp.mesh_query_ray(mesh, wp.cw_div(start, scale), wp.cw_div(-2.0 * ell * n, scale), 1.0)
    if not q.result:
        return False, p, wp.vec3(0.0, 0.0, 0.0)
    nf = wp.normalize(wp.cw_div(wp.mesh_eval_face_normal(mesh, q.face), scale))
    return True, start - q.t * (2.0 * ell) * n, nf


@wp.func
def _probe_kappa(mesh: wp.uint64, scale: wp.vec3, xb: wp.vec3, n: wp.vec3, t: wp.vec3, ell: float, h: float) -> float:
    """Normal curvature of the mesh along t at x_b (shape frame, scaled).
    - The circle through x_b and the wall points c+, c- on the lines through x_b +- l t along n: Menger curvature
      2 |a x b| / (|a| |b| |a - b|), a = c+ - x_b, b = c- - x_b.
    - Positive when the chord midpoint lies on the liquid side of x_b (along n). The normal only gives the sign and
      the probe direction, so a facet's normal (up to pi / segments off the smooth one) does not bias it.
    - 0 when a probe misses the wall within +- l, or when the circle is flat to noise. The threshold is
      |kappa| < 1e-3 / h, which keeps flat meshes bit for bit; ray-hit noise is about ulp(|x_local|) / l^2, so 1 /m
      for a 0.3 m mesh at r 0.075 mm, against 3.3 /m.
    - 0 as well when a probe hits a face whose normal is more than 60 deg from n, or one facing away from the
      liquid. That is a sharp edge (a mesh box's edge, a bore mouth), an edge other shapes may hide, or the far side
      of a thin wall or gap."""
    ok_a, ca, na = _ray_hit(mesh, scale, xb + ell * t, n, ell)
    ok_b, cb, nb = _ray_hit(mesh, scale, xb - ell * t, n, ell)
    if not (ok_a and ok_b):
        return 0.0
    if wp.dot(na, n) < EDGE_COS or wp.dot(nb, n) < EDGE_COS:  # an edge, or a face turned away (another wall)
        return 0.0
    a = ca - xb
    b = cb - xb
    den = wp.length(a) * wp.length(b) * wp.length(a - b)
    if den <= 1.0e-6 * ell * ell * ell:
        return 0.0
    k = 2.0 * wp.length(wp.cross(a, b)) / den
    if k < 1.0e-3 / h:
        return 0.0
    if wp.dot(a + b, n) < 0.0:
        k = -k
    return clamp_principal_curvature(k, h)


@wp.func
def mesh_mean_curvature(mesh: wp.uint64, scale: wp.vec3, xb: wp.vec3, n: wp.vec3, h: float, snap: int) -> float:
    """H = kappa(t1) + kappa(t2) of the mesh at surface point x_b with outward unit normal n (shape frame, scaled):
    positive for a concave wall, negative for a convex one, 0 on a flat one (``halfspace_volume`` convention).
    snap != 0 (a mesh read through a baked texture SDF): x_b is first moved onto the mesh along n by one more ray,
    because the SDF's surface point is off the mesh by its interpolation error. Through the BVH it is on the mesh."""
    if mesh == wp.uint64(0):
        return 0.0
    ell = PROBE_FRACTION * h
    x0 = xb
    if snap != 0:
        ok, xs, _ns = _ray_hit(mesh, scale, xb, n, ell)
        if not ok:
            return 0.0
        x0 = xs
    t1 = _tangent(n)
    t2 = wp.cross(n, t1)
    return _probe_kappa(mesh, scale, x0, n, t1, ell, h) + _probe_kappa(mesh, scale, x0, n, t2, ell, h)


@wp.func
def curved_volume(phi: float, offset: float, amplitude: float, h: float, H: float, kind: int) -> wp.vec2:
    """(Psi, dPsi/dphi) of a curved wall: the fitted flat volume A Psi_0(phi + delta) times the order-1 ratio
    Psi_H(phi) / Psi_0(phi) at the particle's own distance (module doc). H = 0 gives the flat volume exactly."""
    f = halfspace_volume(phi + offset, h, 0.0, kind) * amplitude
    if H == 0.0 or f[0] == 0.0:
        return f
    c = halfspace_volume(phi, h, H, kind)
    z = halfspace_volume(phi, h, 0.0, kind)
    if z[0] <= 0.0:
        return f
    ratio = c[0] / z[0]
    dratio = (c[1] * z[0] - c[0] * z[1]) / (z[0] * z[0])
    if ratio < 0.0:
        ratio = 0.0
        dratio = 0.0
    return wp.vec2(f[0] * ratio, f[1] * ratio + f[0] * dratio)


__all__ = ["EDGE_COS", "PROBE_FRACTION", "curved_volume", "face_closest", "mesh_mean_curvature"]
