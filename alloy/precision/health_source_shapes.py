# SPDX-License-Identifier: Apache-2.0
"""Solids for the "half the thinner part" PEN_FAIL rule (coordinator rulings 2026-11-05 02:00Z and refinement, SB-7).

The rule's thickness is that of the solid the author declared, never that of a convex piece it was built from. A
piece boundary inside a body is not a surface anything can pass through. T12 reached SB-7 with three such cases:
- The exporter declares one 75 x 25 x 15 mm block as three side-by-side boxes ``<block>_box<i>``. Each box is
  8.33 mm thin, so WP7's 4.22-4.68 mm block-into-plywood contacts read as S1.
- The robot asset declares a finger tip mesh as ten convex pieces ``<side>_hull_linear_tip_right_<k>``, which
  ``alloy.tasks.split_scene`` may split again into ``_p<i>``. WP3's 7001:231 went 3.91 mm into plywood on a piece
  6.33 mm thin.

The rule:
- **Solid.** The colliding shapes of one rigid body form one solid. Static shapes (no body) are separate parts,
  so they are grouped only by their decomposition parent (``SOURCE_PART_RULES`` on the shape label).
- **Thickness.** The solid's width along the contact normal: ``max n.x - min n.x`` over the union of its shapes,
  from exact support functions (a pooled convex hull of the polytope members, closed forms for the round
  primitives). A shape without geometry to read (plane, height field) adds nothing; a solid with nothing to read
  has infinite width, as before.
- A real thin plate on its own body still fires.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.spatial import ConvexHull, QhullError

from alloy._vendor import newton

log = logging.getLogger("alloy.precision")

G = newton.GeoType
COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
_POLY = (int(G.MESH), int(G.CONVEX_MESH), int(G.BOX))
_ROUND = (int(G.SPHERE), int(G.CAPSULE), int(G.CYLINDER), int(G.CONE), int(G.ELLIPSOID))
_SPHERE, _CAPSULE, _CYLINDER, _CONE = (wp.constant(t) for t in _ROUND[:4])

# (pattern, replacement) applied in order to a static shape's label; a label any of them changes names a piece of
# the decomposition parent left after all of them
SOURCE_PART_RULES: tuple[tuple[str, str], ...] = (
    (r"_p\d+$", ""),                    # alloy.tasks.split_scene's pieces of one hull
    (r"_box\d+$", ""),                  # the exporter's multi-box parts
    (r"(hull_[^/]+)_\d+$", r"\1"),       # an asset's convex decomposition of one mesh, hull_<mesh>_<k>
)


@wp.struct
class SolidTable:
    shape_solid: wp.array[wp.int32]   # per shape: its solid (-1: none, infinite width)
    pt_begin: wp.array[wp.int32]      # per solid: range in pts (the polytope members' hull, in the solid's frame)
    pt_end: wp.array[wp.int32]
    pts: wp.array[wp.vec3]            # pooled: replicated worlds share one hull
    prim_begin: wp.array[wp.int32]    # per solid: range in prims (round primitive members)
    prim_end: wp.array[wp.int32]
    prims: wp.array[wp.int32]         # shape ids


@dataclass(frozen=True)
class SolidSummary:
    solids: int          # with at least one member
    multi: int           # with more than one member
    hull_points: int     # in the pool


def solid_ids(model: newton.Model, rules: Sequence[tuple[str, str]] = SOURCE_PART_RULES) -> np.ndarray:
    """int64 [shape_count]: per colliding shape, its solid's first shape id; -1 for shapes that never collide.
    One solid per rigid body; static shapes are grouped by ``rules`` on their labels, else alone."""
    n = int(model.shape_count)
    if n == 0:
        return np.zeros(0, np.int64)
    body = model.shape_body.numpy()
    out = np.where((model.shape_flags.numpy() & COLLIDE) != 0, np.arange(n), -1).astype(np.int64)
    labels = list(getattr(model, "shape_label", []) or [])
    compiled, first = [(re.compile(p), r) for p, r in rules], {}
    for s in np.flatnonzero(out >= 0):
        key = ("body", int(body[s])) if body[s] >= 0 else ("static", _stem(labels, int(s), compiled))
        out[s] = first.setdefault(key, int(s))
    return out


def _stem(labels: list, s: int, compiled: list) -> object:
    """A static shape's decomposition parent label, or the shape itself when no rule changes its label."""
    if s >= len(labels):
        return s
    stem, hits = str(labels[s]), 0
    for pattern, repl in compiled:
        stem, k = pattern.subn(repl, stem)
        hits += k
    return stem if hits else s


def build_solids(model: newton.Model, device=None) -> tuple[SolidTable, SolidSummary]:
    """The device table of every solid in ``model`` (host work once per Model; replicated worlds share hulls)."""
    dev = device if device is not None else model.device
    ids = solid_ids(model)
    n = len(ids)
    types = model.shape_type.numpy() if n else np.zeros(0, np.int32)
    scale = model.shape_scale.numpy()[:, :3].astype(np.float64) if n else np.zeros((0, 3))
    xform = model.shape_transform.numpy().astype(np.float64) if n else np.zeros((0, 7))
    roots = np.unique(ids[ids >= 0])
    solid_of = {int(r): i for i, r in enumerate(roots)}
    members: list[list[int]] = [[] for _ in roots]
    for s in np.flatnonzero(ids >= 0):
        members[solid_of[int(ids[s])]].append(int(s))
    pool, pool_at, ranges, prims = [], {}, [], []
    for mem in members:
        poly = [s for s in mem if int(types[s]) in _POLY]
        key = tuple((int(types[s]), id(_source(model, s)), scale[s].tobytes(), xform[s].tobytes()) for s in poly)
        if key not in pool_at:
            pts = _hull_points(model, poly, types, scale, xform)
            pool_at[key] = (sum(len(p) for p in pool), len(pts))
            pool.append(pts)
        at, k = pool_at[key]
        start = len(prims)
        prims += [s for s in mem if int(types[s]) in _ROUND]
        ranges.append((at, at + k, start, len(prims)))
    shape_solid = np.array([solid_of[int(i)] if i >= 0 else -1 for i in ids], dtype=np.int32)
    allpts = np.concatenate(pool) if pool else np.zeros((0, 3))
    cols = np.array(ranges, dtype=np.int32).reshape(-1, 4)
    t = SolidTable()
    t.shape_solid = wp.array(shape_solid, dtype=wp.int32, device=dev)
    t.pt_begin, t.pt_end, t.prim_begin, t.prim_end = (wp.array(np.ascontiguousarray(cols[:, i]), dtype=wp.int32,
                                                               device=dev) for i in range(4))
    t.pts = wp.array(allpts.astype(np.float32), dtype=wp.vec3, device=dev)
    t.prims = wp.array(np.array(prims, np.int32), dtype=wp.int32, device=dev)
    summary = SolidSummary(len(roots), sum(len(m) > 1 for m in members), len(allpts))
    log.info("health solids: %d solids (%d of several shapes), %d hull points", summary.solids, summary.multi,
             summary.hull_points)
    return t, summary


def _source(model: newton.Model, s: int) -> object | None:
    src = getattr(model, "shape_source", None)
    return src[s] if src is not None and s < len(src) else None


def _local_points(model: newton.Model, s: int, gtype: int, scale: np.ndarray) -> np.ndarray:
    if gtype == int(G.BOX):
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64) * scale
    src = _source(model, s)
    if src is None or not hasattr(src, "vertices"):
        return np.zeros((0, 3))
    return np.asarray(src.vertices, dtype=np.float64).reshape(-1, 3) * scale


def _hull_points(model: newton.Model, poly: list[int], types, scale, xform) -> np.ndarray:
    """Hull vertices of the polytope members' union, in the solid's frame (body frame; world for static)."""
    pts = [_rotate(xform[s, 3:], _local_points(model, s, int(types[s]), scale[s])) + xform[s, :3] for s in poly]
    allp = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(allp) < 5:
        return allp
    try:
        return allp[ConvexHull(allp).vertices]
    except QhullError:  # flat or degenerate: the points themselves bound the width exactly
        return allp


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate points ``v`` [k, 3] by the unit quaternion ``q`` (x, y, z, w; Warp's order)."""
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


@wp.func
def _round_support(gtype: int, scale: wp.vec3, u: wp.vec3) -> float:
    """Support of a round primitive along the unit ``u`` in its own frame (axis z, Newton's convention)."""
    r = scale[0]
    radial = wp.sqrt(wp.max(u[0] * u[0] + u[1] * u[1], 0.0))
    if gtype == _SPHERE:
        return r
    if gtype == _CAPSULE:
        return scale[1] * wp.abs(u[2]) + r
    if gtype == _CYLINDER:
        return scale[1] * wp.abs(u[2]) + r * radial
    if gtype == _CONE:  # base disc at -h, apex at +h
        return wp.max(scale[1] * u[2], -scale[1] * u[2] + r * radial)
    return wp.length(wp.cw_mul(scale, u))  # ellipsoid


@wp.func
def solid_width(tbl: SolidTable, shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
                shape_transform: wp.array[wp.transform], s: int, n: wp.vec3) -> float:
    """Width of shape ``s``'s solid along the unit ``n`` (in the solid's frame); inf with nothing to read."""
    if s < 0 or s >= tbl.shape_solid.shape[0]:
        return wp.inf
    j = tbl.shape_solid[s]
    if j < 0:
        return wp.inf
    hi = float(-wp.inf)
    lo = float(wp.inf)
    for i in range(tbl.pt_begin[j], tbl.pt_end[j]):
        d = wp.dot(n, tbl.pts[i])
        hi = wp.max(hi, d)
        lo = wp.min(lo, d)
    for i in range(tbl.prim_begin[j], tbl.prim_end[j]):
        m = tbl.prims[i]
        x = shape_transform[m]
        u = wp.quat_rotate_inv(wp.transform_get_rotation(x), n)
        c = wp.dot(n, wp.transform_get_translation(x))
        hi = wp.max(hi, c + _round_support(shape_type[m], shape_scale[m], u))
        lo = wp.min(lo, c - _round_support(shape_type[m], shape_scale[m], -u))
    if hi < lo:
        return wp.inf
    return hi - lo


@wp.func
def contact_thickness(tbl: SolidTable, shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
                      shape_transform: wp.array[wp.transform], shape_body: wp.array[wp.int32],
                      shape_thick: wp.array[wp.float32], body_q: wp.array[wp.transform], s: int,
                      n_world: wp.vec3) -> float:
    """Thin-part thickness of shape ``s`` at a contact with world normal ``n_world``: its solid's width along the
    normal, or inf when the shape's own thickness is inf (no geometry, or relaxed by ``relax_thin_part_rule``)."""
    if s < 0 or s >= shape_thick.shape[0] or wp.isinf(shape_thick[s]):
        return wp.inf
    n = n_world
    b = shape_body[s]
    if b >= 0 and b < body_q.shape[0]:
        n = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), n_world)
    if wp.length(n) <= 0.0:
        return shape_thick[s]
    return solid_width(tbl, shape_type, shape_scale, shape_transform, s, wp.normalize(n))


__all__ = ["SOURCE_PART_RULES", "SolidSummary", "SolidTable", "build_solids", "contact_thickness", "solid_ids", "solid_width"]
