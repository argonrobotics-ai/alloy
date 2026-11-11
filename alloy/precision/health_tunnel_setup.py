# SPDX-License-Identifier: Apache-2.0
"""Host tables of the run-time tunnel check (``health_tunnel.TunnelWatch``), built once from a Newton model.

Judged bodies (rule): ``inv_mass > 0``, not KINEMATIC, only COLLIDE shapes of supported types (SPHERE / BOX / CAPSULE /
CYLINDER / ELLIPSOID / CONVEX_MESH), and at least one matter point: the COM when it lies inside one of the body's own
primitive shapes (or the body has exactly one collide shape, a CONVEX_MESH), and the shapes' own points
(``matter_points``). A ring or cup made of boxes is judged through its boxes' points; a ring dropped over a static
bar takes none of them through the bar. Bodies with a decomposed (several convex meshes) shape have no point.

Judged static shapes: ``shape_body == -1``, COLLIDE_SHAPES, type PLANE / BOX / SPHERE / CAPSULE / CYLINDER, and a CONVEX_MESH
(what ``add_mesh(collision="convex")`` makes) whose hull fills >= 98 % of its axis-aligned bounding box, judged as that BOX
(other meshes and height fields are not). A world's list holds its own
shapes and the global ones (``shape_world == -1``).

A (body, static shape) pair is skipped when every collide shape of the body is collision-filtered against that static
shape (the broad phase's group rule, or an explicit filter pair): the pair cannot touch, so passing through is by
design. No contact gap is read: the travel limit of FAST_CONTACT is a constant (``health_tunnel_kernels.TRAVEL_LIMIT_M``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton

G = newton.GeoType
STATIC_TYPES = (int(G.PLANE), int(G.BOX), int(G.SPHERE), int(G.CAPSULE), int(G.CYLINDER))
PRIMITIVE_BODY_TYPES = (int(G.SPHERE), int(G.BOX), int(G.CAPSULE), int(G.CYLINDER), int(G.ELLIPSOID))
SHAPE_POINT_CAP, BODY_POINT_CAP = 3, 24
BOX_FILL = 0.98                  # static convex mesh judged as a box when its hull fills this much of its bounding box
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
KINEMATIC = int(newton.BodyFlags.KINEMATIC)


def _rot(q: np.ndarray) -> np.ndarray:
    """3x3 matrix of an xyzw quaternion."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _com_inside(gtype: int, half: np.ndarray, local: np.ndarray) -> bool:
    """Whether a point (the COM, in the shape frame) is inside a primitive of ``gtype`` and ``half`` = scale."""
    if gtype == int(G.SPHERE):
        return bool(np.linalg.norm(local) < half[0])
    if gtype == int(G.BOX):
        return bool(np.all(np.abs(local) < half))
    if gtype == int(G.ELLIPSOID):
        return bool(np.sum((local / np.maximum(half, 1e-12)) ** 2) < 1.0)
    radial = float(np.hypot(local[0], local[1]))
    if gtype == int(G.CYLINDER):
        return radial < half[0] and abs(local[2]) < half[1]
    return radial < half[0] and abs(local[2]) < half[1] + half[0] and (
        abs(local[2]) <= half[1] or float(np.hypot(radial, abs(local[2]) - half[1])) < half[0])     # capsule


@dataclass
class BodyShape:
    """One collide shape of a judged body, as a primitive about ``centre`` (a convex mesh: its bounding box)."""

    gtype: int
    half: np.ndarray
    xf: np.ndarray            # (7,) body-frame position + xyzw quaternion
    reach: float              # largest distance from the COM to a point of the shape
    from_mesh: bool = False   # a convex mesh's bounding box: no matter points (its hull may leave the box's corners)


@dataclass
class JudgedBody:
    body: int
    world: int
    shapes: list[BodyShape] = field(default_factory=list)
    shape_ids: list[int] = field(default_factory=list)
    has_com: bool = False                                   # the COM is a matter point (solid about its centre)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))     # other matter points, body frame
    dropped: int = 0                                        # matter points left out by the caps

    @property
    def r_out(self) -> float:
        return max((s.reach for s in self.shapes), default=0.0)


def _mesh_box(model: newton.Model, s: int, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(centre, half extents) of a convex mesh's bounding box in its shape frame."""
    src = model.shape_source[s] if s < len(model.shape_source) else None
    if src is None or not hasattr(src, "vertices") or len(src.vertices) == 0:
        return None
    pts = np.asarray(src.vertices, dtype=np.float64) * scale[:3]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def _reach(gtype: int, half: np.ndarray, centre_dist: float) -> float:
    ext = {int(G.SPHERE): half[0], int(G.CAPSULE): half[1] + half[0], int(G.CYLINDER): float(np.hypot(half[0], half[1])),
           int(G.BOX): float(np.linalg.norm(half)), int(G.ELLIPSOID): float(np.max(half))}[gtype]
    return centre_dist + float(ext)


def _body_shape(model: newton.Model, s: int, gtype: int, com: np.ndarray, tf: np.ndarray, scale: np.ndarray
                ) -> tuple[BodyShape, bool] | None:
    """(the shape as a primitive about its centre, whether the COM is inside it), None for unsupported types."""
    pos, rot = tf[:3].astype(np.float64), tf[3:7]
    if gtype == int(G.CONVEX_MESH):
        box = _mesh_box(model, s, scale)
        if box is None:
            return None
        centre, half = box
        return BodyShape(int(G.BOX), half, np.concatenate([pos + _rot(rot) @ centre, rot]),
                         _reach(int(G.BOX), half, float(np.linalg.norm(pos + _rot(rot) @ centre - com))), True), False
    if gtype not in PRIMITIVE_BODY_TYPES:
        return None
    inside = _com_inside(gtype, scale[:3].astype(np.float64), _rot(rot).T @ (com - pos))
    return BodyShape(gtype, scale[:3].astype(np.float64), np.concatenate([pos, rot]),
                     _reach(gtype, scale[:3].astype(np.float64), float(np.linalg.norm(pos - com)))), inside


def _shape_points(gtype: int, half: np.ndarray) -> list[np.ndarray]:
    """Matter points of one primitive in its own frame (module doc): centre plus up to two axis points."""
    zero = np.zeros(3)
    if gtype in (int(G.SPHERE), int(G.ELLIPSOID)):
        return [zero]
    if gtype in (int(G.CAPSULE), int(G.CYLINDER)):
        reach = half[1] if gtype == int(G.CAPSULE) else 0.9 * half[1]        # cylinder: inset, inside the solid
        return [zero] if reach < 1e-9 else [zero, np.array([0, 0, reach]), np.array([0, 0, -reach])]
    order = np.argsort(half)                                                  # box: along the longest axis
    off = float(half[order[2]] - half[order[1]])
    axis = np.zeros(3)
    axis[order[2]] = off
    return [zero] if off < 1e-6 else [zero, axis, -axis]


def matter_points(shapes: list[BodyShape], com: np.ndarray | None) -> tuple[np.ndarray, int]:
    """(body-frame matter points besides the COM, number dropped by the caps): at most ``SHAPE_POINT_CAP`` per shape
    and ``BODY_POINT_CAP`` per body (first shapes win); a point within 10 um of one kept (or of the COM) is dropped."""
    kept: list[np.ndarray] = [] if com is None else [com]
    out: list[np.ndarray] = []
    dropped = 0
    for sh in shapes:
        if sh.from_mesh:
            continue
        r = _rot(sh.xf[3:7])
        for q in _shape_points(sh.gtype, sh.half)[:SHAPE_POINT_CAP]:
            p = sh.xf[:3] + r @ q
            if any(np.linalg.norm(p - k) < 1e-5 for k in kept):
                continue
            if len(out) >= BODY_POINT_CAP:
                dropped += 1
                continue
            kept.append(p)
            out.append(p)
    return (np.array(out).reshape(-1, 3), dropped)


def judged_bodies(model: newton.Model) -> list[JudgedBody]:
    """The bodies the check judges (module doc), in body order."""
    if model.body_count == 0 or model.shape_count == 0:
        return []
    inv, flags = model.body_inv_mass.numpy(), model.body_flags.numpy()
    com, bworld = model.body_com.numpy(), model.body_world.numpy()
    sbody, sflags = model.shape_body.numpy(), model.shape_flags.numpy()
    tf, scale = model.shape_transform.numpy(), model.shape_scale.numpy()
    stype = model.shape_type.numpy()
    out = []
    shapes_of: dict[int, list[int]] = {}
    for s in np.flatnonzero((sbody >= 0) & ((sflags & COLLIDE) != 0)):
        shapes_of.setdefault(int(sbody[s]), []).append(int(s))
    for b in range(model.body_count):
        if not inv[b] > 0.0 or int(flags[b]) & KINEMATIC:
            continue
        mine = shapes_of.get(b, [])
        parts = [_body_shape(model, s, int(stype[s]), com[b].astype(np.float64), tf[s], scale[s]) for s in mine]
        if not mine or any(p is None for p in parts):
            continue
        has_com = any(p[1] for p in parts) or (len(mine) == 1 and int(stype[mine[0]]) == int(G.CONVEX_MESH))
        pts, dropped = matter_points([p[0] for p in parts], com[b].astype(np.float64) if has_com else None)
        if has_com or len(pts):
            out.append(JudgedBody(b, int(bworld[b]), [p[0] for p in parts], [int(s) for s in mine], has_com,
                                  pts, dropped))
    return out


def _box_like_hull(model: newton.Model, s: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(centre, half extents) in the shape frame of a convex mesh whose hull fills its axis-aligned bounding box
    (hull volume >= ``BOX_FILL`` x the box's), else None (also without scipy or with a degenerate hull)."""
    box = _mesh_box(model, s, model.shape_scale.numpy()[s].astype(np.float64))
    if box is None or float(np.min(box[1])) <= 0.0:
        return None
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415
        pts = np.asarray(model.shape_source[s].vertices, dtype=np.float64) * model.shape_scale.numpy()[s][:3]
        volume = float(ConvexHull(pts).volume)
    except Exception:  # noqa: BLE001 - no scipy, flat or coplanar points: not judged
        return None
    return box if volume >= BOX_FILL * 8.0 * float(np.prod(box[1])) else None


def static_shapes(model: newton.Model) -> np.ndarray:
    """Indices of the judged static shapes: the primitives plus the box-like static convex meshes (``box_overrides``)."""
    if model.shape_count == 0:
        return np.zeros(0, np.int64)
    ok = (model.shape_body.numpy() == -1) & ((model.shape_flags.numpy() & COLLIDE) != 0)
    types_ = model.shape_type.numpy()
    keep = ok & np.isin(types_, STATIC_TYPES)
    for s in np.flatnonzero(ok & (types_ == int(G.CONVEX_MESH))):
        keep[s] = _box_like_hull(model, int(s)) is not None
    return np.flatnonzero(keep)


def box_overrides(model: newton.Model, statics: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{position in ``statics``: (half extents, box centre in the shape frame)} of the static convex meshes judged as a
    BOX (the hull's bounding box; the watch composes it with the shape's live pose)."""
    types_ = model.shape_type.numpy()
    out = {}
    for pos, s in enumerate(statics):
        if int(types_[s]) != int(G.CONVEX_MESH):
            continue
        centre, half = _box_like_hull(model, int(s))
        out[pos] = (half, np.asarray(centre, np.float32))
    return out


def _groups_collide(a: np.ndarray, b: int) -> np.ndarray:
    """Newton's broad-phase group rule (``test_group_pair``) of each of ``a`` against ``b``."""
    both = (a != 0) & (b != 0)
    return both & np.where(a > 0, (a == b) | (b < 0), a != b)


def statics_by_world(model: newton.Model, statics: np.ndarray) -> dict[int, list[int]]:
    """{world: positions in ``statics``} of the judged static shapes each world sees (its own and the global ones)."""
    sw = model.shape_world.numpy()[statics]
    worlds = max(int(model.world_count), 1)
    global_pos = [i for i, w in enumerate(sw) if w < 0]
    out: dict[int, list[int]] = {w: list(global_pos) for w in range(worlds)}
    for i, w in enumerate(sw):
        if 0 <= w < worlds:
            out[int(w)].append(i)
    return {w: sorted(v) for w, v in out.items()}


def excluded_pairs(model: newton.Model, bodies: list[JudgedBody], statics: np.ndarray,
                   by_world: dict[int, list[int]]) -> dict[int, list[int]]:
    """{judged index: [positions in ``statics``]} where every collide shape of the body is filtered against that
    shape (``by_world``: ``statics_by_world``)."""
    group = model.shape_collision_group.numpy()
    out: dict[int, list[int]] = {}
    for j, jb in enumerate(bodies):
        ids = np.asarray(jb.shape_ids)
        for pos in by_world.get(jb.world, ()):
            s = int(statics[pos])
            can_hit = _groups_collide(group[ids], int(group[s]))
            for k, i in enumerate(ids):
                if can_hit[k] and model.shape_collision_filter_contains(min(int(i), s), max(int(i), s)):
                    can_hit[k] = False
            if not can_hit.any():
                out.setdefault(j, []).append(pos)
    return out


def csr(lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    start = np.zeros(len(lists) + 1, np.int32)
    start[1:] = np.cumsum([len(x) for x in lists])
    flat = np.array([v for x in lists for v in x], np.int32) if start[-1] else np.zeros(0, np.int32)
    return start, flat


def to_device(a: np.ndarray, dtype, device) -> wp.array:
    return wp.array(a, dtype=dtype, device=device)
