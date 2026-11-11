# SPDX-License-Identifier: Apache-2.0
"""Extents of MJCF-sourced objects, read through Newton's own MJCF importer (``ModelBuilder.add_mjcf``,
newton@ac7f7830:newton/_src/sim/builder.py:4952) into a throwaway builder at the model's default pose.

The result is a point cloud on the collision shapes (box corners, dense samples on spheres/ellipsoids and capsule caps,
circumscribed rings on cylinders/cones, mesh vertices), in the frame of the selected body (or the world frame for the whole file), so
``geometry.obb_extents`` gives the object's l1 >= l2 >= l3.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.cards")


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate rows of ``v`` by the unit quaternion q = (x, y, z, w) (Warp order)."""
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _sphere_dirs(n: int = 1500) -> np.ndarray:
    """Fibonacci directions pushed out to 1/cos(0.6 · spacing), so a sampled sphere covers the true one in every
    direction (extents high by ~0.2 %, never low)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(1.0 - z * z)
    phi = k * np.pi * (3.0 - np.sqrt(5.0))
    grow = 1.0 / np.cos(0.6 * np.sqrt(4.0 * np.pi / n))
    return grow * np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _ring(r: float, z: float, n: int = 64) -> np.ndarray:
    """A circumscribed n-gon (vertices at r / cos(pi/n)), so a rotated disc is never under-sized."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rr = r / np.cos(np.pi / n)
    return np.stack([rr * np.cos(a), rr * np.sin(a), np.full(n, z)], axis=1)


def _local_points(kind: str, scale: tuple[float, float, float], src: object) -> np.ndarray:
    sx, sy, sz = scale
    if kind == "BOX":
        return np.array(list(itertools.product((-sx, sx), (-sy, sy), (-sz, sz))), dtype=np.float64)
    if kind in ("SPHERE", "ELLIPSOID"):
        r = np.array((sx, sx, sx) if kind == "SPHERE" else (sx, sy, sz))
        return _sphere_dirs() * r
    if kind in ("CYLINDER", "CONE"):
        return np.concatenate([_ring(sx, -sy), _ring(sx if kind == "CYLINDER" else 0.0, sy)])
    if kind == "CAPSULE":
        cap = _sphere_dirs() * sx
        return np.concatenate([cap + [0.0, 0.0, sy], cap - [0.0, 0.0, sy]])
    if kind in ("MESH", "CONVEX_MESH") and src is not None:
        return np.asarray(src.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    return np.zeros((0, 3))


def _subtree(builder: object, root: int) -> set[int]:
    bodies, grew = {root}, True
    while grew:
        grew = False
        for parent, child in zip(builder.joint_parent, builder.joint_child):
            if parent in bodies and child not in bodies:
                bodies.add(child)
                grew = True
    return bodies


def _body_index(labels: list[str], body: str, path: Path) -> int:
    """Index of ``body``: an exact label, or the unique label whose last path segment is ``body`` (Newton labels
    MJCF bodies ``<model>/worldbody/<parent>/<body>``)."""
    if body in labels:
        return labels.index(body)
    hits = [i for i, lab in enumerate(labels) if lab.rsplit("/", 1)[-1] == body]
    if len(hits) != 1:
        raise KeyError(f"MJCF {path}: body {body!r} matches {len(hits)} bodies (labels: {labels})")
    return hits[0]


def mjcf_points(path: Path, body: str | None = None) -> np.ndarray:
    """Collision-shape sample points of an MJCF file (or of one body's subtree, in that body's frame)."""
    import warp as wp  # noqa: PLC0415 - heavy import, only for MJCF sources

    from alloy._vendor import newton  # noqa: PLC0415

    wp.config.quiet = True
    b = newton.ModelBuilder()
    b.add_mjcf(str(path), parse_visuals=False)
    keep = set(range(len(b.body_q)))
    frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if body is not None:
        root = _body_index(b.body_label, body, path)
        keep, frame = _subtree(b, root), np.array(b.body_q[root], dtype=np.float64)
    pts = []
    for s, kind in enumerate(b.shape_type):
        bi = b.shape_body[s]
        if (bi < 0 and body is not None) or (bi >= 0 and bi not in keep):
            continue
        local = _local_points(newton.GeoType(int(kind)).name, tuple(b.shape_scale[s]), b.shape_source[s])
        xs = np.array(b.shape_transform[s], dtype=np.float64)
        p = _quat_rotate(xs[3:], local) + xs[:3]
        if bi >= 0:
            xb = np.array(b.body_q[bi], dtype=np.float64)
            p = _quat_rotate(xb[3:], p) + xb[:3]
        pts.append(p)
    if not pts:
        return np.zeros((0, 3))
    world = np.concatenate(pts)
    qinv = np.array([-frame[3], -frame[4], -frame[5], frame[6]])
    return _quat_rotate(qinv, world - frame[:3])
