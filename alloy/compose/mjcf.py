# SPDX-License-Identifier: Apache-2.0
"""Collision geometry of an MJCF object card as one triangle mesh: the convex hull of each collision shape, joined.

MuJoCo collides every mesh geom as its convex hull, so the union of per-shape hulls is exactly what the engine
collides with: a bowl modelled as CoACD pieces keeps its hollow. Needs only the collision meshes on disk (visual
meshes and textures are not read).

Example::

    >>> v, f = mjcf_mesh(Path("mjcf/tray/robocasa_tray_001/model.xml"), "robocasa_tray_001")    # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mjcf_mesh(path: Path, body: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) in the body's frame (or the world frame without ``body``); raises if a mesh is missing."""
    parts = mjcf_parts(path, body)
    verts, faces, n = [], [], 0
    for pts in parts:
        hull = _hull(pts)
        if hull is None:
            continue
        v, f = hull
        verts.append(v)
        faces.append(f + n)
        n += len(v)
    if not verts:
        raise ValueError(f"MJCF {path}: no collision shapes{' under body ' + body if body else ''}")
    return np.vstack(verts), np.vstack(faces)


def mjcf_parts(path: Path, body: str | None = None) -> list[np.ndarray]:
    """Sample points of each collision shape (``alloy.cards.mjcf``'s sampling), in the body's frame."""
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.cards import mjcf as M  # noqa: PLC0415

    wp.config.quiet = True
    b = newton.ModelBuilder()
    b.add_mjcf(str(path), parse_visuals=False)
    keep = set(range(len(b.body_q)))
    frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if body is not None:
        root = M._body_index(b.body_label, body, path)
        keep, frame = M._subtree(b, root), np.array(b.body_q[root], dtype=np.float64)
    qinv = np.array([-frame[3], -frame[4], -frame[5], frame[6]])
    out = []
    for s, kind in enumerate(b.shape_type):
        bi = b.shape_body[s]
        if (bi < 0 and body is not None) or (bi >= 0 and bi not in keep):
            continue
        local = M._local_points(newton.GeoType(int(kind)).name, tuple(b.shape_scale[s]), b.shape_source[s])
        if not len(local):
            continue
        xs = np.array(b.shape_transform[s], dtype=np.float64)
        p = M._quat_rotate(xs[3:], local) + xs[:3]
        if bi >= 0:
            xb = np.array(b.body_q[bi], dtype=np.float64)
            p = M._quat_rotate(xb[3:], p) + xb[:3]
        out.append(M._quat_rotate(qinv, p - frame[:3]))
    return out


def _hull(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Convex hull as an outward-wound triangle mesh (Qhull's simplices have no consistent winding)."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    try:
        h = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    tri = h.simplices.copy()
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), h.equations[:, :3]) < 0
    tri[flip] = tri[flip][:, ::-1]
    idx = np.unique(tri)
    remap = np.full(len(pts), -1)
    remap[idx] = np.arange(len(idx))
    return pts[idx], remap[tri]


__all__ = ["mjcf_mesh", "mjcf_parts"]
