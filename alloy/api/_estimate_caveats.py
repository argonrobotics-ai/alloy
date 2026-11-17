# SPDX-License-Identifier: Apache-2.0
"""What ``alloy.estimate`` reads about contact from its one-world template, and what it knows it leaves out.

Resting parts (``resting_parts``, ``resting_on_shapes``): a free body comes to rest on the parts at its bottom, and
MuJoCo C's rk4 tests each against its support in each of its four stages, about 3x a ground plane's cost on a box,
mesh or body, so the step time charges them (``_estimate.COST["cpu"]["rest_s"]``, ``["rest_shape_s"]``): three YCB
scans in a bin, resting on 49 of their 96 convex parts, stepped 2.1-2.6x the estimate without that term
(``bench/results/estimate_calib/rest_rows.json``); a decomposed bowl rests on 3 parts upright and on 16 upside down.

Mesh-on-mesh contact: the step terms see shapes, not the contacts between them, and meshes resting on
meshes step slower than estimated (bench/results/estimate_calib/wp7_s1_rows.json), so the estimate says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TOUCHING_MESH = ("{n} pair(s) of mesh / convex-hull / cylinder shapes on different bodies touch at the start: the step "
                 "time does not count mesh-on-mesh contact. On one L4 (Alloy main e65d95254) 16 bowls stacked 4 high "
                 "stepped 1.9-2.5x the estimate (decomposed) and 1.6-2.1x (one hull each) at 1-256 worlds, 2 towers of "
                 "8 decomposed bowls 2.9x (1 world) and 28x (64 worlds); spaced apart the same bowls stepped 0.5-1.0x "
                 "it. measure=True times a real run")
CCD_GEO = ("MESH", "CONVEX_MESH", "CYLINDER", "CONE", "ELLIPSOID")      # MuJoCo's convex (CCD) path


def caveats(load: dict[str, Any]) -> list[str]:
    """What the estimate knows it does not model for this scene (``TOUCHING_MESH``), as sentences."""
    n = int(load.get("touching_mesh_pairs", 0))
    return [TOUCHING_MESH.format(n=n)] if n else []


def touching_mesh_pairs(model: Any) -> int:
    """Pairs of colliding CCD-path shapes (``CCD_GEO``) on different bodies, one of them moving, whose bounding
    spheres (``shape_collision_radius``, plus both margins) overlap at the start pose: resting mesh-on-mesh contact.
    One shape against all later ones in numpy. Example: ``touching_mesh_pairs(built.model)``."""
    from alloy._vendor import newton  # noqa: PLC0415

    if not model.shape_count:
        return 0
    kinds, flags = model.shape_type.numpy(), model.shape_flags.numpy()
    ccd = np.isin(kinds, [int(getattr(newton.GeoType, g)) for g in CCD_GEO])
    ccd &= (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    idx = np.flatnonzero(ccd)
    if idx.size < 2:
        return 0
    body = model.shape_body.numpy()[idx]
    xf = model.shape_transform.numpy()[idx]
    pos = xf[:, :3].astype(np.float64)
    if model.body_count:
        bq = model.body_q.numpy().astype(np.float64)
        moving = body >= 0
        pos[moving] = bq[body[moving], :3] + _rotate(bq[body[moving], 3:7], pos[moving])
    mass = np.where(body >= 0, model.body_mass.numpy()[np.maximum(body, 0)] if model.body_count else 0.0, 0.0)
    reach = model.shape_collision_radius.numpy()[idx].astype(np.float64) + model.shape_margin.numpy()[idx]
    n = 0
    for i in range(idx.size - 1):
        j = np.arange(i + 1, idx.size)
        near = np.linalg.norm(pos[j] - pos[i], axis=1) <= reach[j] + reach[i]
        n += int(np.count_nonzero(near & (body[j] != body[i]) & ((mass[j] > 0) | (mass[i] > 0))))
    return n


def resting_parts(model: Any) -> np.ndarray:
    """How many parts each free body (a free joint, not kinematic, not a robot's links) rests on: its colliding
    shapes whose lowest point at the start pose is within the shape's contact reach (``shape_gap`` + ``shape_margin``,
    5 mm by default) of the body's lowest point, as it lands when it falls as it starts (a box, a ball, one hull: 1; a
    decomposed bowl upright 3, upside down on its rim 16; MuJoCo touched the same counts at rest). One count per such
    body with a colliding shape, from the shapes' local bounding boxes. Example: ``resting_parts(built.model)``."""
    return _resting(model)[0]


def resting_on_shapes(model: Any, beyond_first: bool = False) -> int:
    """How many of the resting parts (``resting_parts``) rest on a shape other than a ground plane: a static box or
    mesh (a table, a bin's floor) or another body, the highest one under the body's resting parts at the start pose
    that is not above the ground plane. MuJoCo tests those on its convex path, about 3x a plane's cost (a body of 16
    cubes, a settled bowl and three YCB scans, each on the ground and on a box table). ``beyond_first``: count each
    body's parts beyond its first only. Example: ``resting_on_shapes(built.model)``."""
    counts, owners, lo, hi = _resting(model)
    if not counts.size:
        return 0
    from alloy._vendor import newton  # noqa: PLC0415

    kinds, body = model.shape_type.numpy(), model.shape_body.numpy()
    collide = np.flatnonzero((model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)
    planes = collide[kinds[collide] == int(newton.GeoType.PLANE)]
    ground = max((float(model.shape_transform.numpy()[s][2]) for s in planes), default=-np.inf)
    others = collide[kinds[collide] != int(newton.GeoType.PLANE)]
    olo, ohi = _world_boxes(model, others)
    on_shape = 0
    for b, n, (blo, bhi) in zip(owners, counts, zip(lo, hi)):
        x, y = 0.5 * (blo[0] + bhi[0]), 0.5 * (blo[1] + bhi[1])
        under = ((body[others] != b) & (olo[:, 0] <= x) & (x <= ohi[:, 0]) & (olo[:, 1] <= y) & (y <= ohi[:, 1])
                 & (ohi[:, 2] <= blo[2] + 0.01))
        if np.any(under) and ohi[under, 2].max() > ground + 1e-3:
            on_shape += int(n) - int(beyond_first)
    return on_shape


def _resting(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(resting parts per free body, those bodies, the world box of each body's resting parts: lows, highs)."""
    from alloy._vendor import newton  # noqa: PLC0415

    none = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3)))
    if not (model.shape_count and model.joint_count) or model.shape_collision_aabb_lower is None:
        return none
    jt, child = model.joint_type.numpy(), model.joint_child.numpy()
    free = np.unique(child[jt == int(newton.JointType.FREE)])
    if model.body_flags is not None and free.size:          # driven (kinematic) bodies do not come to rest
        free = free[(model.body_flags.numpy()[free] & int(newton.BodyFlags.KINEMATIC)) == 0]
    body = model.shape_body.numpy()
    shapes = np.flatnonzero(((model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0)
                            & np.isin(body, free))
    if not shapes.size:
        return none
    lo, hi = _world_boxes(model, shapes)
    owner, reach = body[shapes], model.shape_gap.numpy()[shapes] + model.shape_margin.numpy()[shapes]
    bodies, slot = np.unique(owner, return_inverse=True)
    bottom = np.full(bodies.size, np.inf)
    np.minimum.at(bottom, slot, lo[:, 2])
    rest = lo[:, 2] - bottom[slot] <= reach + 1e-9
    counts = np.bincount(slot, weights=rest, minlength=bodies.size).astype(np.int64)
    blo, bhi = np.full((bodies.size, 3), np.inf), np.full((bodies.size, 3), -np.inf)
    np.minimum.at(blo, slot[rest], lo[rest])
    np.maximum.at(bhi, slot[rest], hi[rest])
    return counts, bodies, blo, bhi


def _world_boxes(model: Any, shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-space bounds (lows, highs) of each shape's local bounding box (``shape_collision_aabb_*``) at the start
    pose; a static shape's transform is already in the world frame."""
    lo, hi = model.shape_collision_aabb_lower.numpy()[shapes], model.shape_collision_aabb_upper.numpy()[shapes]
    pick = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])               # the 8 corners
    corners = np.where(pick[None, :, :] == 0, lo[:, None, :], hi[:, None, :]).reshape(-1, 3).astype(np.float64)
    xf = np.repeat(model.shape_transform.numpy()[shapes].astype(np.float64), 8, axis=0)
    world = _rotate(xf[:, 3:7], corners) + xf[:, :3]
    owner = np.repeat(model.shape_body.numpy()[shapes], 8)
    moving = owner >= 0
    if np.any(moving):
        bq = model.body_q.numpy()[owner[moving]].astype(np.float64)
        world[moving] = _rotate(bq[:, 3:7], world[moving]) + bq[:, :3]
    pts = world.reshape(-1, 8, 3)
    return pts.min(axis=1), pts.max(axis=1)


def _rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows of ``v`` rotated by the rows of the unit quaternions ``q_xyzw``."""
    u, w = q_xyzw[:, :3], q_xyzw[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)
