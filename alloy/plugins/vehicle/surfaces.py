# SPDX-License-Identifier: Apache-2.0
"""Drivable surfaces of the vehicle plug-in (SCR-WP8-12 F1b): the top faces of named box shapes (a ramp, a dock
floor, a leveler deck and lip, a trailer floor), static or on moving bodies, plus the flat ground plane.

A wheel is a disc (centre, axle, radius). Against a face (normal ``n``, the box's +z) it touches at its lowest point
along ``n`` when that point lies over the face; past the face's edge it touches the edge (a crest, a step down): the
nearest point of the face's rectangle, at the disc's distance from it in the wheel's plane, with the normal from that
point to the centre.

A wheel keeps its two deepest contacts, which together bear ``max(d) + (1 - c)·min(d)`` in depth.
- Each bears its own depth times ``1 - c·s``, with ``s = r^q/2`` for the deeper and ``1 - r^(q-1)/2`` for the other
  (``r = d_min/d_max``, q = 8). At r = 1 both shares are 1/2, so the load and its direction stay continuous when the
  deeper contact changes, as over a leveler's deck-lip hinge (WP13, F4).
- Far from that point (r → 0) the deeper bears all of max(d), so a flat seam's edge contact, whose normal leans by the
  edge's offset, takes little load. A d²-proportional split, or WP13's ``1 - c·d_j/(d_a + d_b)``, put 40 % on it, and
  its radial damper then read the rolling speed as a 4 % load swing; the latter also bears 17 % less than max(d) at
  r = 0.41.
``c`` measures how far the two are one surface, continuously (WP13: no switch, so a wheel parked over a seam between
moving bodies, a sagging trailer floor or a leveler deck changing angle, sees its load change smoothly):
- the two faces are coplanar: normals within 3° and plane offsets at the contacts within 1 mm give 1, and 6° or 3 mm
  give 0, linear between. A seam between two decks, or a deck on the ground, is one surface; a step is not;
- or the two contacts share a normal (a crest's edge seen from both faces), on the same 3-6° ramp.
A valley (the ground and a ramp's foot) keeps both contacts in full. Each candidate face is evaluated once per wheel.
Slip is measured relative to the surface's body, and the tyre's reaction goes on that body (one thread per world: no
atomics).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError

SAME_NORMAL = wp.constant(0.9986)        # cos 3°: two faces or contacts this close in normal are one surface
APART_NORMAL = wp.constant(0.9945)       # cos 6°: faces this far apart in normal are never one surface
SAME_PLANE = wp.constant(1.0e-3)         # m: faces with plane offsets within this are one surface
APART_PLANE = wp.constant(3.0e-3)        # m: and at this, never


@wp.struct
class Surfaces:
    shape: wp.array2d[wp.int32]          # [W, S] drivable box shapes per world (S may be 0)
    shape_body: wp.array[wp.int32]       # model.shape_body
    shape_xf: wp.array[wp.transform]     # model.shape_transform (body frame; world frame for static shapes)
    shape_scale: wp.array[wp.vec3]       # box half extents
    ground: wp.float32                   # ground plane height
    plane: wp.int32                      # 1: the ground plane is a surface


@wp.struct
class Hit:
    depth: wp.float32                    # > 0 in contact [m]
    point: wp.vec3                       # contact point on the surface, world
    normal: wp.vec3                      # surface normal there, toward the wheel
    body: wp.int32                       # the surface's body (-1: static)
    rim: wp.int32                        # 1: on the face's edge
    low: wp.vec3                         # the disc's lowest point along the face normal (an edge contact's test point)
    face: wp.vec3                        # the face's normal (an edge contact's normal is not)
    origin: wp.vec3                      # a point of the face's plane
    weight: wp.float32                   # share of the load it carries (an edge blended into a seam: < 1)


@wp.struct
class Pair:
    a: Hit                               # deepest contact
    b: Hit                               # deepest one with another normal (depth <= 0: none)


@wp.func
def no_hit() -> Hit:
    h = Hit()
    h.depth = wp.float32(-1.0)
    h.body = -1
    h.weight = wp.float32(1.0)
    return h


@wp.func
def lowest(centre: wp.vec3, axis: wp.vec3, radius: wp.float32, n: wp.vec3) -> wp.vec3:
    """The disc's lowest point along ``-n`` (``centre`` if the disc lies flat on ``n``)."""
    up = n - wp.dot(n, axis) * axis
    if wp.length(up) < 1.0e-6:
        return centre
    return centre - radius * wp.normalize(up)


@wp.func
def plane_hit(centre: wp.vec3, axis: wp.vec3, radius: wp.float32, ground: wp.float32) -> Hit:
    h = no_hit()
    n = wp.vec3(0.0, 0.0, 1.0)
    low = lowest(centre, axis, radius, n)
    h.depth = ground - low[2]
    h.point = wp.vec3(low[0], low[1], ground)
    h.normal = n
    h.face = n
    h.origin = wp.vec3(0.0, 0.0, ground)
    return h


vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _clip(lo: wp.float32, hi: wp.float32, start: wp.float32, step: wp.float32, bound: wp.float32) -> wp.vec2:
    """Narrow the parameter range [lo, hi] of ``start + t·step`` to |value| <= bound."""
    if wp.abs(step) < 1.0e-9:
        if wp.abs(start) > bound:
            return wp.vec2(wp.float32(1.0), wp.float32(-1.0))
        return wp.vec2(lo, hi)
    ta = (-bound - start) / step
    tb = (bound - start) / step
    return wp.vec2(wp.max(lo, wp.min(ta, tb)), wp.min(hi, wp.max(ta, tb)))


@wp.func
def plane_cut(a1: wp.float32, a2: wp.float32, c0: wp.float32, half: wp.vec3) -> vec6:
    """Where the wheel's plane crosses a face, in its (s, u) coordinates: the line c0 + a1·s + a2·u = 0 as a point
    (s0, u0), a unit direction (ds, du) and the parameter range [lo, hi] inside the face's rectangle (lo > hi: the
    face lies wholly to one side of the wheel). A rim contact sits on this segment: projecting the face's nearest
    point into the wheel's plane instead made a face beside the wheel (a dock's floor beside a leveler pit) a contact
    under it."""
    out = vec6(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0),
               wp.float32(-1.0))
    aa = a1 * a1 + a2 * a2
    if aa < 1.0e-12:
        return out
    inv = 1.0 / wp.sqrt(aa)
    s0 = -c0 * a1 / aa
    u0 = -c0 * a2 / aa
    ds = -a2 * inv
    du = a1 * inv
    span = _clip(wp.float32(-1.0e30), wp.float32(1.0e30), s0, ds, half[0])
    span = _clip(span[0], span[1], u0, du, half[1])
    return vec6(s0, u0, ds, du, span[0], span[1])


@wp.func
def face_hit(centre: wp.vec3, axis: wp.vec3, radius: wp.float32, xf: wp.transform, half: wp.vec3,
             body: wp.int32) -> Hit:
    """The disc against the box's top face (+z in the box frame), or its rim."""
    h = no_hit()
    h.body = body
    n = wp.transform_vector(xf, wp.vec3(0.0, 0.0, 1.0))
    t1 = wp.transform_vector(xf, wp.vec3(1.0, 0.0, 0.0))
    t2 = wp.transform_vector(xf, wp.vec3(0.0, 1.0, 0.0))
    p0 = wp.transform_point(xf, wp.vec3(0.0, 0.0, half[2]))
    if wp.dot(centre - p0, n) <= 0.0:                    # the wheel centre is below the face
        return h
    low = lowest(centre, axis, radius, n)
    rel = low - p0
    l1 = wp.dot(rel, t1)
    l2 = wp.dot(rel, t2)
    h.low = low
    h.face = n
    h.origin = p0
    if wp.abs(l1) <= half[0] and wp.abs(l2) <= half[1]:
        h.depth = -wp.dot(rel, n)
        h.point = low + h.depth * n
        h.normal = n
        return h
    h.rim = 1
    seg = plane_cut(wp.dot(t1, axis), wp.dot(t2, axis), wp.dot(p0 - centre, axis), half)
    if seg[4] > seg[5]:                                  # the face does not reach the wheel's plane: no contact
        return h
    t = wp.clamp((l1 - seg[0]) * seg[2] + (l2 - seg[1]) * seg[3], seg[4], seg[5])
    q = p0 + (seg[0] + t * seg[2]) * t1 + (seg[1] + t * seg[3]) * t2
    d = centre - q
    r = wp.length(d)
    if r >= radius or r < 1.0e-9:
        return h
    h.depth = radius - r
    h.point = q
    h.normal = d / r
    return h



@wp.func
def face_xf(k: wp.int32, w: wp.int32, s: Surfaces, body_q: wp.array[wp.transform]) -> wp.transform:
    shape = s.shape[w, k]
    b = s.shape_body[shape]
    xf = s.shape_xf[shape]
    if b >= 0:
        xf = wp.transform_multiply(body_q[b], xf)
    return xf


@wp.func
def ramp(x: wp.float32, full: wp.float32, none: wp.float32) -> wp.float32:
    """1 at ``x <= full``, 0 at ``x >= none``, linear between."""
    return wp.clamp((none - x) / (none - full), 0.0, 1.0)


SPLIT_Q = wp.constant(8.0)   # sharpness of the two contacts' split away from equal depths (module notes)


@wp.func
def coplanar(a: Hit, b: Hit) -> wp.float32:
    """How far two contacts' faces are one plane (0..1): their normals, and their plane offsets at the contacts."""
    angle = ramp(1.0 - wp.dot(a.face, b.face), 1.0 - SAME_NORMAL, 1.0 - APART_NORMAL)
    offset = 0.5 * (wp.abs(wp.dot(b.point - a.origin, a.face)) + wp.abs(wp.dot(a.point - b.origin, b.face)))
    return angle * ramp(offset, SAME_PLANE, APART_PLANE)


@wp.func
def one_surface(a: Hit, b: Hit) -> wp.float32:
    """How far two contacts are one surface (0..1): coplanar faces, or a shared contact normal (a crest's edge seen
    from both faces)."""
    shared = ramp(1.0 - wp.dot(a.normal, b.normal), 1.0 - SAME_NORMAL, 1.0 - APART_NORMAL)
    return wp.max(coplanar(a, b), shared)


@wp.func
def toward_face(h: Hit, f: wp.float32) -> Hit:
    """``h`` with its normal turned toward its face's by ``f``: on coplanar faces a seam's edge is no real edge, so its
    contact acts along the surface's normal, not leaning toward the wheel centre (that lean read the rolling speed in
    the radial damper)."""
    out = h
    out.normal = wp.normalize(h.normal + f * (h.face - h.normal))
    return out


@wp.func
def surface_hit(k: wp.int32, w: wp.int32, centre: wp.vec3, axis: wp.vec3, radius: wp.float32, s: Surfaces,
                body_q: wp.array[wp.transform]) -> Hit:
    """Candidate ``k``: -1 the ground plane, else the world's k-th drivable box."""
    if k < 0:
        if s.plane == 0:
            return no_hit()
        return plane_hit(centre, axis, radius, s.ground)
    shape = s.shape[w, k]
    return face_hit(centre, axis, radius, face_xf(k, w, s, body_q), s.shape_scale[shape], s.shape_body[shape])


@wp.func
def wheel_hits(w: wp.int32, centre: wp.vec3, axis: wp.vec3, radius: wp.float32, s: Surfaces,
               body_q: wp.array[wp.transform]) -> Pair:
    """The wheel's two deepest contacts, weighted to bear ``max(d) + (1 - c)·min(d)`` with a split continuous where the
    deeper changes (c = ``one_surface``; see the module notes). One pass over the candidates (the top two tracked by
    index: no struct writes in the loop), then the two are rebuilt."""
    first = int(-2)
    second = int(-2)
    best = float(0.0)
    runner = float(0.0)
    for k in range(-1, s.shape.shape[1]):
        d = surface_hit(k, w, centre, axis, radius, s, body_q).depth
        if d > best:
            second = first
            runner = best
            first = k
            best = d
        elif d > runner:
            second = k
            runner = d
    p = Pair()
    p.a = no_hit()
    p.b = no_hit()
    if first == -2:
        return p
    p.a = surface_hit(first, w, centre, axis, radius, s, body_q)
    if second != -2:
        p.b = surface_hit(second, w, centre, axis, radius, s, body_q)
        c = one_surface(p.a, p.b)
        flat = coplanar(p.a, p.b)
        p.a = toward_face(p.a, flat)
        p.b = toward_face(p.b, flat)
        r = wp.max(p.b.depth, 0.0) / p.a.depth                                   # p.a is the deeper
        p.a.weight = 1.0 - c * 0.5 * wp.pow(r, SPLIT_Q)
        p.b.weight = 1.0 - c * (1.0 - 0.5 * wp.pow(r, SPLIT_Q - 1.0))
    return p


@wp.func
def surface_velocity(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                     body_com: wp.array[wp.vec3], b: wp.int32, point: wp.vec3) -> wp.vec3:
    if b < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    com = wp.transform_point(body_q[b], body_com[b])
    qd = body_qd[b]
    return wp.vec3(qd[0], qd[1], qd[2]) + wp.cross(wp.vec3(qd[3], qd[4], qd[5]), point - com)


# --- host side --------------------------------------------------------------------------------------------------------

def surface_index(ctx: Any, labels: tuple[str, ...]) -> np.ndarray:
    """``int32 [W, S]``: each named drivable shape in every world; every one must be a box."""
    model = ctx.model
    if not labels:
        return np.zeros((ctx.world_count, 0), np.int32)
    missing = [name for name in labels if not ctx.has(name, "shape")]
    if missing:
        raise PluginError(f"vehicle: no shape labelled {missing} for the drivable surfaces; shapes: "
                          f"{ctx.labels('shape')[:20]}")
    index = np.stack([ctx.shapes(name).numpy() for name in labels], axis=1).astype(np.int32)
    types = model.shape_type.numpy()
    bad = sorted({labels[k] for k in range(len(labels)) if (types[index[:, k]] != int(newton.GeoType.BOX)).any()})
    if bad:
        raise PluginError(f"vehicle: drivable surfaces must be boxes (their top face is driven on): {bad}")
    return index


def surfaces_struct(ctx: Any, index: np.ndarray, ground: float, plane: bool) -> Surfaces:
    model = ctx.model
    s = Surfaces()
    s.shape = ctx.array(index if index.size else np.zeros((ctx.world_count, 0), np.int32), wp.int32)
    s.shape_body, s.shape_xf, s.shape_scale = model.shape_body, model.shape_transform, model.shape_scale
    s.ground, s.plane = float(ground), int(bool(plane))
    return s


def surface_bodies(model: newton.Model, index: np.ndarray) -> list[int]:
    """The moving bodies that carry drivable surfaces (static faces excluded)."""
    bodies = model.shape_body.numpy()[index.reshape(-1)] if index.size else np.zeros(0, np.int32)
    return sorted({int(b) for b in bodies if b >= 0})


def _chain_mobility(model: newton.Model, b: int, point: np.ndarray) -> np.ndarray:
    """The 3x3 point mobility at ``point`` of body ``b`` up its joint chain: each revolute joint adds its hinge's own
    freedom ``g·gᵀ/I_e`` (g = e x (point - pin), I_e the child's inertia about the hinge: an upper bound, its own
    children's inertia left out); the chain ends at the world, or at a body on any other joint, whose free-body point
    mobility is added. A leveler lip on its deck on its hinge is then as mobile as its two hinges allow."""
    from alloy.plugins.vehicle.mobility import _hinge, _point, _world_body  # noqa: PLC0415

    kids, parents = model.joint_child.numpy(), model.joint_parent.numpy()
    mob = np.zeros((3, 3))
    while b >= 0:
        joints = np.flatnonzero(kids == b)
        parent = int(parents[joints[0]]) if joints.size else -1
        hinge = _hinge(model, b, parent) if joints.size else None
        if hinge is None:
            return mob + _point(*_world_body(model, b), point)
        axis, pin, i_e = hinge
        g = np.cross(axis, point - pin)
        mob += np.outer(g, g) / i_e
        b = parent
    return mob


def surface_mobility(model: newton.Model, index: np.ndarray) -> dict[int, tuple[float, float]]:
    """Each moving surface body's largest point mobility over the four corners of its drivable faces, at the build
    pose, up its hinges to the world or to a free body (``_chain_mobility``): (along the face's normal, the largest in
    its plane)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    shapes = index.reshape(-1)
    owner, xfs, half = model.shape_body.numpy(), model.shape_transform.numpy(), model.shape_scale.numpy()
    out: dict[int, tuple[float, float]] = {}
    for b in surface_bodies(model, index):
        q = np.asarray(model.body_q.numpy()[b], np.float64)
        body_rot = Rotation.from_quat(q[3:])
        normal, plane = 0.0, 0.0
        for s in {int(v) for v in shapes if int(owner[v]) == b}:
            xf = np.asarray(xfs[s], np.float64)
            face = body_rot * Rotation.from_quat(xf[3:])
            n, t1, t2 = face.apply(np.eye(3)[[2, 0, 1]])
            hx, hy, hz = (float(v) for v in half[s])
            for cx, cy in ((hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)):
                corner = q[:3] + body_rot.apply(xf[:3] + Rotation.from_quat(xf[3:]).apply((cx, cy, hz)))
                m = _chain_mobility(model, b, corner)
                tangent = np.stack([t1, t2])
                normal = max(normal, float(n @ m @ n))
                plane = max(plane, float(np.linalg.eigvalsh(tangent @ m @ tangent.T).max()))
        out[b] = (normal, plane)
    return out


def _per_body(body: np.ndarray, values: np.ndarray) -> float:
    """The largest per-body sum of per-wheel ``values`` (worlds alike: the largest over them)."""
    return float(max(values[w][body[w] == b].sum() for w in range(body.shape[0]) for b in np.unique(body[w])))


def surface_checks(mobility: dict[int, tuple[float, float]], body: np.ndarray, radial: tuple[np.ndarray, np.ndarray],
                   tread: tuple[float, np.ndarray], wheel: tuple[np.ndarray, np.ndarray],
                   dt: float) -> dict[str, float]:
    """The tyres' springs and dampers on the drivable surfaces' moving bodies, direction by direction (WP13: a contact
    acts on the relative motion, the wheel's mobility and the surface's together). ``radial`` = per-wheel (k, c),
    ``tread`` = (k, per-wheel low-speed damper c), ``wheel`` = per-wheel (w_zz, in-plane max) [W, N] each; ``body``
    [W, N] the wheels' bodies. Each wheel's own terms sum per vehicle body as in the tread check; every wheel may
    stand on the stiffest-coupled surface body at once, so the surface's terms sum over all wheels."""
    if not mobility:
        return {}
    w_n = max(v[0] for v in mobility.values())
    w_t = max(v[1] for v in mobility.values())
    (k_r, c_r), (k_t, c_t), (w_zz, w_ip) = radial, tread, wheel
    out = {"surface radial springs": np.sqrt(_per_body(body, k_r * w_zz) + float(k_r.sum()) * w_n) * dt,
           "surface radial damping": (_per_body(body, c_r * w_zz) + float(c_r.sum()) * w_n) * dt,
           "surface tread springs": np.sqrt(_per_body(body, k_t * w_ip) + k_t * w_ip.shape[1] * w_t) * dt,
           "surface tread damping": (_per_body(body, c_t * w_ip) + float(c_t.max(axis=0).sum()) * w_t) * dt}
    return {k: float(v) for k, v in out.items()}


__all__ = ["Hit", "Pair", "Surfaces", "surface_bodies", "surface_checks", "surface_mobility", "surface_index",
           "surface_velocity", "surfaces_struct", "wheel_hits"]
