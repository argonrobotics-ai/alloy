# SPDX-License-Identifier: Apache-2.0
"""Static extraction from a Newton Model: worlds, shapes, meshes, replicas (no per-frame data)."""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass

import numpy as np

from .bundle import BundleError

log = logging.getLogger("alloy.render")

GEO_NAMES = {1: "plane", 3: "sphere", 4: "capsule", 5: "ellipsoid", 6: "cylinder", 7: "box", 8: "mesh",
             9: "cone", 10: "mesh"}
GEO_HFIELD = 2
FLAG_VISIBLE, FLAG_COLLIDE, FLAG_SITE = 1, 2, 8
DEFAULT_COLOR = [0.7, 0.7, 0.7]


def to_np(arr, default=None) -> np.ndarray | None:
    """Host numpy view of a Warp array, list or None."""
    if arr is None:
        return default
    return arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)


@dataclass
class WorldPlan:
    """Exported worlds, their grid offsets and the hero world."""

    worlds: list[int]
    offsets: dict[int, np.ndarray]
    hero: int

    def offset_of(self, world_ids: np.ndarray) -> np.ndarray:
        """Per-item offsets [N, 3] float32 (zeros for world -1)."""
        out = np.zeros((len(world_ids), 3), np.float32)
        for w, off in self.offsets.items():
            out[world_ids == w] = off
        return out

    def selects(self, world_ids: np.ndarray) -> np.ndarray:
        """Boolean mask of items belonging to an exported world or to the global world (-1)."""
        return (world_ids < 0) | np.isin(world_ids, self.worlds)


def model_worlds(model) -> dict[str, np.ndarray]:
    """World id per body / shape / particle (``-1`` for global or unset)."""
    sizes = {"body": len(model.body_label), "shape": len(model.shape_label), "particle": int(model.particle_count)}
    out = {}
    for key, n in sizes.items():
        w = to_np(getattr(model, f"{key}_world", None))
        out[key] = np.full(n, -1, np.int64) if w is None or len(w) != n else w.astype(np.int64)
    return out


def plan_worlds(world_arrays: dict[str, np.ndarray], worlds, grid_m: float | None) -> WorldPlan:
    """Resolve ``worlds`` ("all" or ints) and grid offsets."""
    if isinstance(worlds, str):
        if worlds != "all":
            raise BundleError(f"worlds must be 'all' or a list of ints, got {worlds!r}")
        present = np.unique(np.concatenate(list(world_arrays.values())))
        ids = [int(w) for w in present if w >= 0] or [0]
    else:
        ids = [int(w) for w in worlds]
        if not ids or len(set(ids)) != len(ids):
            raise BundleError(f"worlds must be a non-empty list of distinct ints, got {worlds!r}")
    n = max(1, math.ceil(math.sqrt(len(ids))))
    step = float(grid_m) if grid_m else 0.0
    offsets = {w: np.array([(i % n) * step, (i // n) * step, 0.0], np.float32) for i, w in enumerate(ids)}
    return WorldPlan(ids, offsets, ids[0])


class MeshStore:
    """Mesh arrays keyed by content hash, so identical meshes are stored once."""

    def __init__(self) -> None:
        self.meshes: dict[str, dict] = {}
        self._by_hash: dict[str, str] = {}

    def add(self, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None) -> str:
        v = np.ascontiguousarray(vertices, np.float32).reshape(-1, 3)
        f = np.ascontiguousarray(faces, np.int32).reshape(-1, 3)
        u = None if uv is None or len(uv) != len(v) else np.ascontiguousarray(uv, np.float32).reshape(-1, 2)
        h = hashlib.sha1(v.tobytes() + f.tobytes() + (u.tobytes() if u is not None else b"")).hexdigest()
        if h in self._by_hash:
            return self._by_hash[h]
        mid = f"mesh_{len(self.meshes):03d}"
        self.meshes[mid] = {"vertices": v, "faces": f, "uv": u}
        self._by_hash[h] = mid
        return mid

    def scene_entries(self) -> dict[str, dict]:
        return {mid: {"file": f"meshes/{mid}.npz", "verts": len(m["vertices"]), "faces": len(m["faces"]),
                      "uv": m["uv"] is not None} for mid, m in self.meshes.items()}


def _heightfield_mesh(hf):
    from alloy._vendor import newton

    heights = hf.min_z + hf.data * (hf.max_z - hf.min_z)
    mesh = newton.Mesh.create_heightfield(heightfield=heights.T, extent_x=hf.hx * 2.0, extent_y=hf.hy * 2.0,
                                          ground_z=hf.min_z, compute_inertia=False)
    return mesh


def _source_mesh_id(model, i: int, geo: int, store: MeshStore) -> str:
    src = model.shape_source[i]
    if src is None:
        raise BundleError(f"shape {i} ({model.shape_label[i]!r}): mesh shape without shape_source")
    mesh = _heightfield_mesh(src) if geo == GEO_HFIELD else src
    uv = getattr(mesh, "uvs", None)
    return store.add(np.asarray(mesh.vertices), np.asarray(mesh.indices), uv)


def _size_for(kind: str, scale: np.ndarray) -> list[float]:
    count = {"sphere": 1, "capsule": 2, "cylinder": 2, "cone": 2, "plane": 2}.get(kind, 3)
    return [float(x) for x in scale[:count]]


def shape_geometry(model, i: int, store: MeshStore, host: dict) -> dict:
    """Geometry record of model shape ``i`` (type, size, scale, mesh, xform in the body frame)."""
    geo = int(host["type"][i])
    if geo != GEO_HFIELD and geo not in GEO_NAMES:
        raise BundleError(f"shape {i} ({model.shape_label[i]!r}): unsupported GeoType {geo}")
    kind = "mesh" if geo == GEO_HFIELD else GEO_NAMES[geo]
    scale = host["scale"][i].astype(np.float64)
    mesh_id = None
    if kind == "mesh":
        mesh_id = _source_mesh_id(model, i, geo, store)
        scale = np.ones(3) if geo == GEO_HFIELD else scale
    return {"type": kind, "size": [float(s) for s in scale] if kind == "mesh" else _size_for(kind, scale),
            "scale": [float(s) for s in scale], "mesh": mesh_id,
            "xform": [float(x) for x in host["xform"][i]]}


def host_shape_arrays(model) -> dict:
    """Host copies of the per-shape arrays used by extraction."""
    n = len(model.shape_label)
    color = to_np(getattr(model, "shape_color", None))
    opacity = to_np(getattr(model, "shape_opacity", None))
    return {"type": to_np(model.shape_type), "scale": to_np(model.shape_scale), "xform": to_np(model.shape_transform),
            "body": to_np(model.shape_body).astype(np.int64), "flags": to_np(model.shape_flags),
            "color": color if color is not None and len(color) == n else None,
            "opacity": opacity if opacity is not None and len(opacity) == n else None,
            "labels": [str(x) for x in model.shape_label]}


def visible_shapes(host: dict) -> np.ndarray:
    """Mask of shapes to render: VISIBLE and not SITE, plus collision-only stand-ins (``collision_fallback``)."""
    flags = host["flags"]
    shown = ((flags & FLAG_VISIBLE) != 0) & ((flags & FLAG_SITE) == 0)
    return shown | collision_fallback(host, shown)


def collision_fallback(host: dict, shown: np.ndarray) -> np.ndarray:
    """Collision shapes of an object (label parent path) that has no visible shape.

    MJCF import with ``parse_visuals=True`` hides every collision geom, so a table that exists only as a collision
    box would vanish from the render. Objects that do have a visual geom keep showing only that one.
    """
    labels = host.get("labels")
    if not labels:
        return np.zeros_like(shown)
    keys = [_object_key(lab, int(b)) for lab, b in zip(labels, host["body"])]
    with_visual = {k for k, s in zip(keys, shown) if s and k is not None}
    flags = host["flags"]
    collide = ((flags & FLAG_COLLIDE) != 0) & ((flags & FLAG_SITE) == 0) & ~shown
    return np.array([bool(c) and k is not None and k not in with_visual for c, k in zip(collide, keys)], dtype=bool)


def _object_key(label: str, body: int):
    """The object a shape belongs to: its label's parent path (MJCF/USD import), else its body; None if unknown."""
    if "/" in label:
        return label.rsplit("/", 1)[0]
    return ("body", body) if body >= 0 else None


def extract_shapes(model, host: dict, world_of: np.ndarray, body_col: dict[int, int], plan: WorldPlan,
                   store: MeshStore, skip_bodies: set[int]) -> list[dict]:
    """Scene ``shapes[]`` for every exported, visible, non-site shape."""
    keep = visible_shapes(host) & plan.selects(world_of)
    entries: list[dict] = []
    for i in np.flatnonzero(keep):
        body = int(host["body"][i])
        if body in skip_bodies:
            continue
        rec = shape_geometry(model, int(i), store, host)
        world = int(world_of[i])
        if body < 0 and world in plan.offsets:
            rec["xform"][:3] = [float(x) for x in np.asarray(rec["xform"][:3], np.float32) + plan.offsets[world]]
        color = host["color"][i] if host["color"] is not None else DEFAULT_COLOR
        rec.update(id=len(entries), label=str(model.shape_label[i]), body=body_col.get(body, -1), world=world,
                   color=[float(c) for c in color],
                   opacity=float(host["opacity"][i]) if host["opacity"] is not None else 1.0,
                   replica_of=None, model_index=int(i))
        if not host["flags"][i] & FLAG_VISIBLE:
            rec["collision_fallback"] = True
        entries.append(rec)
    return entries


def mark_replicas(entries: list[dict], world_of: np.ndarray, plan: WorldPlan) -> None:
    """Set ``replica_of`` for shapes that repeat a hero-world shape at the same rank within their world."""
    if len(plan.worlds) < 2:
        return
    members = {w: np.flatnonzero(world_of == w) for w in plan.worlds}
    rank = {int(i): r for w in plan.worlds for r, i in enumerate(members[w])}
    by_index = {e["model_index"]: e for e in entries}
    hero = members[plan.hero]
    for e in entries:
        if e["world"] in (plan.hero, -1) or e["world"] not in members:
            continue
        r = rank[e["model_index"]]
        twin = by_index.get(int(hero[r])) if r < len(hero) else None
        same = twin is not None and all(twin[k] == e[k] for k in ("type", "mesh", "scale"))
        e["replica_of"] = twin["id"] if same else None
