# SPDX-License-Identifier: Apache-2.0
"""Host-side set-up of the ``Bonds`` plug-in (numpy, once, at attach): vertex areas and spacing, which vertices a
grip holds or start bonded, and each vertex's rest distance to the nearest grip vertex."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton

from alloy.plugins.errors import PluginError

FAR = 1.0e30


def vertex_areas(x: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """A third of the incident triangles' rest areas per vertex [m²]."""
    a = 0.5 * np.linalg.norm(np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]]), axis=1)
    out = np.zeros(len(x))
    for k in range(3):
        np.add.at(out, tris[:, k], a / 3.0)
    return out


def vertex_spacing(x: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Mean rest length of the edges at each vertex [m] (0 for a vertex on no triangle)."""
    e = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]]), axis=1), axis=0)
    length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
    total, count = np.zeros(len(x)), np.zeros(len(x))
    for k in range(2):
        np.add.at(total, e[:, k], length)
        np.add.at(count, e[:, k], 1.0)
    return np.where(count > 0, total / np.maximum(count, 1.0), 0.0)


def film_cells(area: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per vertex: its film's area per vertex (the film = the vertex's connected triangle component), the cell a
    uniform press loads every vertex of that film with."""
    n = len(area)
    parent = np.arange(n)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in tris:
        ra, rb, rc = root(a), root(b), root(c)
        parent[rb] = ra
        parent[rc] = ra
    roots = np.array([root(i) for i in range(n)])
    total = np.bincount(roots, weights=area, minlength=n)
    count = np.bincount(roots, minlength=n).astype(float)
    return np.where(count[roots] > 0, total[roots] / np.maximum(count[roots], 1.0), 0.0)


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def shape_frame(model: newton.Model, body_q: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """World rotation matrix and origin of shape ``s`` (body pose, then the shape's local transform)."""
    tf = model.shape_transform.numpy()[s]
    r, p = _quat_matrix(tf[3:7]), tf[:3]
    b = int(model.shape_body.numpy()[s])
    if b < 0:
        return r, p
    bq = body_q[b]
    rb = _quat_matrix(bq[3:7])
    return rb @ r, rb @ p + bq[:3]


def box_distance(model: newton.Model, body_q: np.ndarray, s: int, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of ``pts`` to box shape ``s`` and the outward normal of its nearest face (world)."""
    if int(model.shape_type.numpy()[s]) != int(newton.GeoType.BOX):
        raise PluginError(f"bonds: shape {s} ({model.shape_label[s]!r}) is not a box; grips and pre-bonded "
                          f"substrates are boxes in v1 (bonds formed under pressure work on any shape)")
    r, o = shape_frame(model, body_q, s)
    half = model.shape_scale.numpy()[s]
    loc = (pts - o) @ r                                   # into the shape frame
    out = np.abs(loc) - half
    dist = np.where(np.all(out <= 0, axis=1), out.max(axis=1), np.linalg.norm(np.maximum(out, 0.0), axis=1))
    axis = np.argmax(out, axis=1)
    n_loc = np.zeros_like(loc)
    n_loc[np.arange(len(loc)), axis] = np.sign(loc[np.arange(len(loc)), axis])
    return dist, n_loc @ r.T


def shape_distance(model: newton.Model, body_q: np.ndarray, s: int, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of ``pts`` to shape ``s`` (a box, or a cylinder or capsule along its local z) and the outward
    normal there (world): the pre-bond test. Grips stay boxes (``box_distance``)."""
    kind = int(model.shape_type.numpy()[s])
    if kind == int(newton.GeoType.BOX):
        return box_distance(model, body_q, s, pts)
    if kind not in (int(newton.GeoType.CYLINDER), int(newton.GeoType.CAPSULE)):
        raise PluginError(f"bonds: shape {s} ({model.shape_label[s]!r}) is not a box, cylinder or capsule; pre-bonded "
                          f"substrates are those (bonds formed under pressure work on any shape)")
    r, o = shape_frame(model, body_q, s)
    rad, hh = model.shape_scale.numpy()[s][:2]
    loc = (pts - o) @ r
    if kind == int(newton.GeoType.CAPSULE):
        v = loc - np.stack([np.zeros(len(loc)), np.zeros(len(loc)), np.clip(loc[:, 2], -hh, hh)], 1)
        d = np.linalg.norm(v, axis=1)
        return d - rad, (v / np.maximum(d, 1e-30)[:, None]) @ r.T
    rr = np.hypot(loc[:, 0], loc[:, 1])
    radial = np.stack([loc[:, 0], loc[:, 1], np.zeros(len(loc))], 1) / np.maximum(rr, 1e-30)[:, None]
    axial = np.stack([np.zeros(len(loc)), np.zeros(len(loc)), np.sign(loc[:, 2])], 1)
    dr, dz = rr - rad, np.abs(loc[:, 2]) - hh
    inside = (dr <= 0) & (dz <= 0)
    a, b = np.maximum(dr, 0.0), np.maximum(dz, 0.0)
    out = np.hypot(a, b)
    n = np.where(inside[:, None], np.where((dr > dz)[:, None], radial, axial),
                 (radial * a[:, None] + axial * b[:, None]) / np.maximum(out, 1e-30)[:, None])
    return np.where(inside, np.maximum(dr, dz), out), n @ r.T


def unfold(tris: np.ndarray, poses: np.ndarray, n: int) -> np.ndarray:
    """Flat rest positions [n, 3] (z = 0) of a mesh built flat, from its triangles' rest frames (Newton's tri_poses:
    the inverse of [q - p, r - p] in a frame with q - p along x): each triangle is laid next to a placed neighbour
    across their shared edge. A label posed wrapped on a tube keeps its flat material distances this way (one rigid
    placement per connected film)."""
    local = np.linalg.inv(np.asarray(poses, np.float64))                 # [T, 2, 2]: columns q - p, r - p
    loc = np.stack([np.zeros((len(tris), 2)), local[:, :, 0], local[:, :, 1]], 1)    # [T, 3, 2]
    by_edge: dict[tuple[int, int], list[int]] = {}
    for t, tri in enumerate(tris):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            by_edge.setdefault(tuple(sorted((int(tri[a]), int(tri[b])))), []).append(t)
    x = np.full((n, 2), np.nan)
    done = np.zeros(len(tris), bool)
    for seed in range(len(tris)):
        if done[seed]:
            continue
        x[tris[seed]] = np.where(np.isnan(x[tris[seed]]), loc[seed], x[tris[seed]])
        done[seed] = True
        queue = [seed]
        while queue:
            t = queue.pop()
            for a, b in ((0, 1), (1, 2), (2, 0)):
                for u in by_edge[tuple(sorted((int(tris[t][a]), int(tris[t][b]))))]:
                    if not done[u]:
                        _place(x, tris[u], loc[u])
                        done[u] = True
                        queue.append(u)
    return np.concatenate([np.nan_to_num(x), np.zeros((n, 1))], axis=1)


def _place(x: np.ndarray, tri: np.ndarray, loc: np.ndarray) -> None:
    """Rigidly place a triangle (local 2D ``loc``) onto two of its vertices already placed in ``x``."""
    have = [k for k in range(3) if not np.isnan(x[tri[k], 0])]
    if len(have) == 3:
        return
    i, j = have[0], have[1]
    d_loc, d_x = loc[j] - loc[i], x[tri[j]] - x[tri[i]]
    th = np.arctan2(d_x[1], d_x[0]) - np.arctan2(d_loc[1], d_loc[0])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for k in range(3):
        if np.isnan(x[tri[k], 0]):
            x[tri[k]] = x[tri[i]] + rot @ (loc[k] - loc[i])


def flat_layout(x: np.ndarray, tris: np.ndarray, poses: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The rest layout for material quantities (areas, spacing, distances to the grip): the positions themselves when
    every world's film lies in a plane (bit for bit as before), else the unfolded rest mesh (a label posed wrapped)."""
    for w in range(len(start) - 1):
        pts = x[start[w]:start[w + 1]]
        if len(pts) > 2 and np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)[-1] > 1e-9 * np.ptp(pts):
            return unfold(tris, poses, len(x))
    return x


def world_ranges(particle_world: np.ndarray, worlds: int) -> np.ndarray:
    """[W + 1] start of each world's particles; the particles of a world must be contiguous."""
    if np.any(np.diff(particle_world) < 0) or particle_world.min(initial=0) < 0:
        raise PluginError("bonds: film particles must be grouped by world, all in a world (replicated scenes are)")
    return np.searchsorted(particle_world, np.arange(worlds + 1)).astype(np.int32)


def nearest_grip(x: np.ndarray, grip: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per vertex: rest distance to the nearest grip vertex of its world and that vertex (FAR, -1 without one)."""
    d, g = np.full(len(x), FAR), np.full(len(x), -1, np.int32)
    for w in range(len(start) - 1):
        lo, hi = int(start[w]), int(start[w + 1])
        gids = lo + np.flatnonzero(grip[lo:hi])
        if gids.size == 0:
            continue
        dist = np.linalg.norm(x[lo:hi, None, :] - x[None, gids, :], axis=2)
        k = np.argmin(dist, axis=1)
        d[lo:hi], g[lo:hi] = dist[np.arange(hi - lo), k], gids[k]
    d[grip] = 0.0
    return d, g


def rest_directions(x: np.ndarray, gnear: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unit rest direction from each vertex's nearest grip vertex (zero without one)."""
    out = np.zeros_like(x)
    ok = (gnear >= 0) & (d > 0.0) & (d < FAR)
    out[ok] = (x[ok] - x[gnear[ok]]) / d[ok, None]
    return out


def to_body(model: newton.Model, body_q: np.ndarray, s: np.ndarray, v: np.ndarray, point: bool) -> np.ndarray:
    """World points (``point``) or directions ``v`` into the body frame of shapes ``s`` (static shapes: world)."""
    out = v.copy()
    body = model.shape_body.numpy()
    for i in np.flatnonzero(s >= 0):
        b = int(body[s[i]])
        if b < 0:
            continue
        r = _quat_matrix(body_q[b, 3:7])
        out[i] = (v[i] - body_q[b, :3]) @ r if point else v[i] @ r
    return out


__all__ = ["FAR", "box_distance", "film_cells", "flat_layout", "nearest_grip", "rest_directions", "shape_distance",
           "shape_frame", "to_body", "unfold", "vertex_areas", "vertex_spacing", "world_ranges"]
