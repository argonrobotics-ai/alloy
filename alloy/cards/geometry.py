# SPDX-License-Identifier: Apache-2.0
"""Geometry facts about card shapes: extents, smallest member, mesh I/O, volume, watertightness, wall thickness.

Primitive sizes follow Newton's builder (newton@ac7f7830:newton/_src/sim/builder.py): box half-extents (8922),
sphere radius, capsule (radius, half_height of the cylinder part only, caps excluded; 8976-8998), cylinder and cone
(radius, half_height, axis Z; 9033-9118), ellipsoid semi-axes (8844), plane (width, length) with 0 = infinite (8696).
Newton has no OBJ/STL reader without trimesh (geometry/utils.py:369-420), so a small one lives here.
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy.cards import types as ct

log = logging.getLogger("alloy.cards")

PRIMITIVE_ARITY: dict[str, int] = {"box": 3, "sphere": 1, "capsule": 2, "cylinder": 2, "cone": 2, "plane": 2,
                                   "ellipsoid": 3}
ROUND_KINDS = frozenset({"sphere", "capsule", "cylinder", "ellipsoid"})


@dataclass(frozen=True)
class MeshData:
    """A triangle mesh in metres; ``groups`` maps an OBJ group/object name to its face indices."""

    vertices: np.ndarray                        # (N, 3) float64
    faces: np.ndarray                           # (M, 3) int64
    groups: dict[str, np.ndarray] = field(default_factory=dict)

    def scaled(self, scale: tuple[float, float, float]) -> MeshData:
        return MeshData(self.vertices * np.asarray(scale, dtype=np.float64), self.faces, self.groups)

    def subset(self, name: str) -> MeshData:
        return MeshData(self.vertices, self.faces[self.groups[name]])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- primitives


def primitive_extents(p: ct.Primitive) -> tuple[float, float, float]:
    """Full OBB lengths l1 >= l2 >= l3 [m]; an infinite plane gives inf."""
    s = p.size
    if p.kind == "box":
        dims = [2 * s[0], 2 * s[1], 2 * s[2]]
    elif p.kind == "sphere":
        dims = [2 * s[0]] * 3
    elif p.kind == "capsule":
        dims = [2 * s[1] + 2 * s[0], 2 * s[0], 2 * s[0]]
    elif p.kind in ("cylinder", "cone"):
        dims = [2 * s[1], 2 * s[0], 2 * s[0]]
    elif p.kind == "ellipsoid":
        dims = [2 * s[0], 2 * s[1], 2 * s[2]]
    else:  # plane
        dims = [s[0] if s[0] > 0 else math.inf, s[1] if s[1] > 0 else math.inf, 0.0]
    a, b, c = sorted(dims, reverse=True)
    return (a, b, c)


def primitive_min_member(p: ct.Primitive) -> float:
    """Smallest solid dimension [m] (the GOTCHAS:413 thin-member check); planes are infinitely thick."""
    if p.kind == "plane":
        return math.inf
    if p.kind == "cone":
        return 2 * min(p.size[0], p.size[1])
    return primitive_extents(p)[2]


def curve_length(c: ct.CurveSource) -> float:
    pts = np.asarray(c.points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def curve_extents(c: ct.CurveSource) -> tuple[float, float, float]:
    """A rod's (arc length, diameter, diameter): the router treats any curve as a rod."""
    d = 2 * c.radius
    return (max(curve_length(c), d), d, d)


def box_extents(lo: ct.Vec3, hi: ct.Vec3) -> tuple[float, float, float]:
    a, b, c = sorted((abs(hi[i] - lo[i]) for i in range(3)), reverse=True)
    return (a, b, c)


# --------------------------------------------------------------------------------------------- mesh I/O


def _parse_obj_face(tokens: list[str], nverts: int) -> list[int]:
    idx = []
    for tok in tokens:
        i = int(tok.split("/")[0])
        idx.append(i - 1 if i > 0 else nverts + i)
    return idx


def read_obj(path: Path) -> MeshData:
    """Vertices, triangulated faces (fan) and ``g``/``o`` groups of a Wavefront OBJ file."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if not tok or tok[0].startswith("#"):
            continue
        if tok[0] == "v":
            verts.append([float(x) for x in tok[1:4]])
        elif tok[0] == "f":
            poly = _parse_obj_face(tok[1:], len(verts))
            for k in range(1, len(poly) - 1):
                if current is not None:
                    groups[current].append(len(faces))
                faces.append([poly[0], poly[k], poly[k + 1]])
        elif tok[0] in ("g", "o") and len(tok) > 1:
            current = tok[1]
            groups.setdefault(current, [])
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if f.size and (f.min() < 0 or f.max() >= len(verts)):
        raise ValueError(f"{path}: face index out of range (mesh has {len(verts)} vertices)")
    return MeshData(np.asarray(verts, dtype=np.float64).reshape(-1, 3), f,
                    {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if v})


def write_obj(path: Path, mesh: MeshData, header: str = "") -> None:
    """Deterministic OBJ text (``repr`` floats), groups written in the order of their first face."""
    lines = [f"# {h}" for h in header.splitlines()]
    lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in mesh.vertices.tolist()]
    owner = np.full(len(mesh.faces), -1)
    names = sorted(mesh.groups, key=lambda g: int(mesh.groups[g].min()))
    for gi, g in enumerate(names):
        owner[mesh.groups[g]] = gi
    last = -2
    for fi, (a, b, c) in enumerate(mesh.faces.tolist()):
        if owner[fi] != last and owner[fi] >= 0:
            lines.append(f"g {names[owner[fi]]}")
        last = owner[fi]
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_stl(path: Path) -> MeshData:
    """Binary or ASCII STL; coincident vertices are merged exactly."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84 and len(raw) == 84 + 50 * struct.unpack_from("<I", raw, 80)[0]:
        n = struct.unpack_from("<I", raw, 80)[0]
        rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
        tri = rec["v"].astype(np.float64)
    else:
        nums = [[float(x) for x in ln.split()[1:4]] for ln in raw.decode("utf-8").splitlines() if ln.strip().startswith("vertex")]
        tri = np.asarray(nums, dtype=np.float64).reshape(-1, 3, 3)
    verts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return MeshData(verts, inv.reshape(-1, 3).astype(np.int64))


def read_mesh(path: Path) -> MeshData:
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return read_obj(path)
    if suffix == ".stl":
        return read_stl(path)
    raise ValueError(f"unsupported mesh format {suffix!r} (obj, stl)")


# --------------------------------------------------------------------------------------------- mesh facts


def signed_volume(mesh: MeshData) -> float:
    """Divergence-theorem volume [m^3]; positive for outward-wound closed meshes."""
    v = mesh.vertices[mesh.faces]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)


def _hull_frames(pts: np.ndarray, cap: int = 512) -> list[np.ndarray]:
    """Rotation frames from convex-hull facets (facet normal + one facet edge): the minimal box of a polytope has a
    face flush with a hull facet, so these catch rotated boxes that PCA misses (isotropic covariance)."""
    try:
        from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return []
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return []
    frames, seen = [], set()
    step = max(1, len(hull.simplices) // cap)
    for simplex, eq in zip(hull.simplices[::step], hull.equations[::step]):
        n = eq[:3] / np.linalg.norm(eq[:3])
        e = pts[simplex[1]] - pts[simplex[0]]
        e = e - n * (e @ n)
        if np.linalg.norm(e) < 1e-15:
            continue
        e /= np.linalg.norm(e)
        key = tuple(np.round(np.concatenate([np.abs(n), np.abs(e)]), 6))
        if key not in seen:
            seen.add(key)
            frames.append(np.stack([n, e, np.cross(n, e)], axis=1))
    return frames


_DERIVED: dict[tuple, object] = {}             # (function, array digest, args) -> result, for meshes seen in this process
_DERIVED_MAX = 512


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _memo(name: str, arrays: tuple[np.ndarray, ...], args: tuple, compute):
    """``compute()`` once per process for the same mesh content: libraries are rebuilt often (each test, each scene
    randomization end) and re-validate the same meshes; the geometry checks dominate that cost (wall_thickness ~1.5 s
    for a 10k-triangle mesh). Results are immutable (floats, bools, tuples)."""
    key = (name, _digest(*arrays), args)
    if key not in _DERIVED:
        if len(_DERIVED) >= _DERIVED_MAX:
            _DERIVED.pop(next(iter(_DERIVED)))
        _DERIVED[key] = compute()
    return _DERIVED[key]


def obb_extents(vertices: np.ndarray) -> tuple[float, float, float]:
    """OBB lengths l1 >= l2 >= l3: the smallest-volume box over axis-aligned, PCA and hull-facet frames."""
    pts = np.asarray(vertices, dtype=np.float64)
    return _memo("obb", (pts,), (), lambda: _obb_extents(pts))


def _obb_extents(pts: np.ndarray) -> tuple[float, float, float]:
    if len(pts) == 0:
        return (0.0, 0.0, 0.0)
    centered = pts - pts.mean(axis=0)
    frames = [np.eye(3)]
    if len(pts) >= 4:
        frames.append(np.linalg.eigh(centered.T @ centered)[1])
        frames += _hull_frames(centered)
    cands = [(centered @ f).max(axis=0) - (centered @ f).min(axis=0) for f in frames]
    best = min(cands, key=lambda d: float(np.prod(np.maximum(d, 1e-12))))
    a, b, c = sorted(best.tolist(), reverse=True)
    return (a, b, c)


def edge_report(mesh: MeshData) -> tuple[int, int]:
    """(edges not shared by exactly two faces, directed edges used twice): (0, 0) = closed and consistently wound."""
    f = mesh.faces
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    _, dcounts = np.unique(directed, axis=0, return_counts=True)
    return int((counts != 2).sum()), int((dcounts > 1).sum())


def is_closed(mesh: MeshData) -> bool:
    """Every edge shared by exactly two faces with a consistent winding (either orientation)."""
    return _memo("closed", (mesh.vertices, mesh.faces), (), lambda: _is_closed(mesh))


def _is_closed(mesh: MeshData) -> bool:
    if len(mesh.faces) == 0:
        return False
    open_edges, flipped = edge_report(_welded(mesh))
    return open_edges == 0 and flipped == 0


def is_watertight(mesh: MeshData) -> bool:
    """Newton's rule (every edge shared by exactly two triangles, newton/_src/geometry/types.py:1522) plus a
    consistent, outward winding (no directed edge used twice, positive signed volume), so the SDF sign and the
    inward rays of ``wall_thickness`` are meaningful. An inside-out mesh is not watertight here."""
    return is_closed(mesh) and signed_volume(mesh) > 0


def _welded(mesh: MeshData, tol: float = 1e-7) -> MeshData:
    """Merge vertices closer than ``tol`` (Newton quantises at 1e-7 m for the same check)."""
    keys = np.round(mesh.vertices / tol).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return MeshData(mesh.vertices[first], inv.reshape(-1)[mesh.faces])


_COINCIDENT = 1e-7          # m; hits closer than this are touching shells (e.g. a shaft against a grip), not walls


def _ray_hits(origins: np.ndarray, dirs: np.ndarray, tri: np.ndarray, skip: np.ndarray) -> np.ndarray:
    """Nearest exit distance along each ray (Möller-Trumbore against every triangle); inf if none."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(dirs[:, None, :], e2[None, :, :])
    det = np.einsum("rtk,tk->rt", p, e1)
    ok = np.abs(det) > 1e-18
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origins[:, None, :] - tri[None, :, 0, :]
    u = np.einsum("rtk,rtk->rt", s, p) * inv
    qv = np.cross(s, e1[None, :, :])
    v = np.einsum("rk,rtk->rt", dirs, qv) * inv
    t = np.einsum("tk,rtk->rt", e2, qv) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > _COINCIDENT)
    hit[np.arange(len(origins)), skip] = False
    return np.where(hit, t, np.inf).min(axis=1)


def wall_thickness(mesh: MeshData, max_rays: int = 1024, chunk: int = 128, within: float = math.inf) -> float:
    """Smallest material thickness [m]: rays from face centroids along the inward normal to the next surface.

    Faces are subsampled evenly to ``max_rays``. Returns inf for an empty or open-ray mesh. With a finite ``within``,
    only walls thinner than that are measured (exactly) and a thicker one reads inf: each ray tests only triangles
    whose box comes within ``within`` of it (the card validator: 38 s -> 3 s on an 86k-triangle jar)."""
    return _memo("wall", (mesh.vertices, mesh.faces), (max_rays, within),
                 lambda: _wall_thickness(mesh, max_rays, chunk, within))


def _near_walls(origins: np.ndarray, dirs: np.ndarray, tri: np.ndarray, pick: np.ndarray, within: float,
                chunk: int = 32) -> float:
    """``_wall_thickness`` below ``within``: rays sorted into compact groups; a hit nearer than ``within`` lies on a
    triangle whose box reaches the group's origin box grown by ``within``, so the other triangles are skipped."""
    cells = np.floor((origins - origins.min(axis=0)) / (4.0 * max(within, 1e-9))).astype(np.int64)
    order = np.lexsort((cells[:, 0], cells[:, 1], cells[:, 2]))
    tri_lo, tri_hi = tri.min(axis=1), tri.max(axis=1)
    best = math.inf
    for k in range(0, len(order), chunk):
        idx = order[k:k + chunk]
        lo, hi = origins[idx].min(axis=0) - within, origins[idx].max(axis=0) + within
        near = np.nonzero(np.all(tri_hi >= lo, axis=1) & np.all(tri_lo <= hi, axis=1))[0]
        hits = _ray_hits(origins[idx], dirs[idx], tri[near], np.searchsorted(near, pick[idx]))
        best = min(best, float(hits.min()))
    return best if best < within else math.inf


def _wall_thickness(mesh: MeshData, max_rays: int, chunk: int, within: float = math.inf) -> float:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    good = np.nonzero(norm > 1e-20)[0]
    if len(good) == 0:
        return math.inf
    pick = good[np.linspace(0, len(good) - 1, min(max_rays, len(good))).astype(np.int64)]
    origins = tri[pick].mean(axis=1)
    inward = -1.0 if signed_volume(mesh) >= 0 else 1.0      # an inside-out mesh's normals point inward
    dirs = inward * normals[pick] / norm[pick, None]
    if math.isfinite(within):
        return _near_walls(origins, dirs, tri, pick, within)
    best = math.inf
    for k in range(0, len(pick), chunk):
        d = _ray_hits(origins[k:k + chunk], dirs[k:k + chunk], tri, pick[k:k + chunk])
        best = min(best, float(d.min()))
    return best
