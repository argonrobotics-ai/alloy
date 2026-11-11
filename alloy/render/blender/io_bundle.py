# SPDX-License-Identifier: Apache-2.0
"""ARB v1 bundle reader (docs/RENDER.md section 2). Pure numpy, no bpy.

Frame arrays are memory-mapped, so a 2 GB liquid bundle costs nothing until a frame is read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
MMAP_MIN_BYTES = 64 * 2**20   # arrays below this are read into RAM (no descriptor kept)

SHAPE_TYPES = ("box", "sphere", "capsule", "cylinder", "cone", "ellipsoid", "plane", "mesh")
ENTITY_KINDS = {"surfaces": "surf", "points": "pts", "curves": "curve", "instancers": "inst"}


class BundleError(Exception):
    """The bundle does not follow ARB v1; the message names the field and the bad value."""


def _require(obj: dict, key: str, where: str):
    if key not in obj:
        raise BundleError(f"{where}: missing field '{key}'")
    return obj[key]


def _world_offsets(scene: dict) -> dict[int, np.ndarray]:
    """World id -> offset. Accepts `worlds` as a list of ids (+ top-level `world_offsets`) or as a dict."""
    worlds = scene.get("worlds", [])
    offsets = scene.get("world_offsets")
    if isinstance(worlds, dict):
        offsets = worlds.get("world_offsets", worlds.get("offsets", offsets))
        worlds = worlds.get("ids", [])
    if offsets is None:
        offsets = [[0.0, 0.0, 0.0]] * len(worlds)
    if len(offsets) != len(worlds):
        raise BundleError(f"scene.json: {len(worlds)} worlds but {len(offsets)} world_offsets")
    out = {}
    for w, off in zip(worlds, offsets):
        arr = np.asarray(off, dtype=np.float64)
        if arr.shape != (3,):
            raise BundleError(f"scene.json: world_offsets for world {w} is {off!r}, expected [x, y, z]")
        out[int(w)] = arr
    return out


@dataclass
class Bundle:
    """A loaded ARB v1 bundle: scene description plus lazily memory-mapped frame arrays."""

    root: Path
    scene: dict
    t: np.ndarray
    fps: float
    offsets: dict[int, np.ndarray]
    body_q: np.ndarray | None = None
    _arrays: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.t.shape[0])

    def offset(self, world: int) -> np.ndarray:
        """Shift the recorder already applied to exported world `world` (informational; poses include it)."""
        return self.offsets.get(int(world), np.zeros(3))

    def time_at(self, fr: float) -> float:
        """Bundle time [s] at fractional frame fr, extrapolated with fps outside the recording."""
        if self.count == 0:
            return fr / self.fps
        if fr <= 0:
            return float(self.t[0] + fr / self.fps)
        if fr >= self.count - 1:
            return float(self.t[-1] + (fr - (self.count - 1)) / self.fps)
        i0 = int(np.floor(fr))
        a = fr - i0
        return float(self.t[i0] * (1 - a) + self.t[i0 + 1] * a)

    def frame_at_time(self, t: float) -> float:
        """Fractional bundle frame at bundle time t [s] (inverse of time_at; linear outside the recording)."""
        if self.count < 2:
            return float(t) * self.fps
        if t <= self.t[0]:
            return float((t - self.t[0]) * self.fps)
        if t >= self.t[-1]:
            return float(self.count - 1 + (t - self.t[-1]) * self.fps)
        return float(np.interp(t, self.t, np.arange(self.count)))

    def array(self, prefix: str, name: str) -> np.ndarray:
        """frames/<prefix>_<name>.npy: memory-mapped when large, read into RAM when small.

        Each mmap holds a file descriptor, and a 1,024-world grid has thousands of small per-world arrays (4,096
        cable curves in DC5's grid), which exhausts the default 1,024-descriptor limit.
        """
        key = f"{prefix}_{name}"
        if key not in self._arrays:
            path = self.root / "frames" / f"{key}.npy"
            if not path.exists():
                raise BundleError(f"missing frame array {path}")
            big = path.stat().st_size >= MMAP_MIN_BYTES
            self._arrays[key] = np.load(path, mmap_mode="r") if big else np.load(path)
        return self._arrays[key]

    def mesh(self, mesh_id: str) -> dict:
        """vertices [V,3] f32, faces [F,3] i32 and optional uv [V,2] of meshes/<id>.npz."""
        entry = self.scene.get("meshes", {}).get(str(mesh_id))
        if entry is None:
            raise BundleError(f"scene.json meshes: unknown mesh id {mesh_id!r}")
        path = self.root / entry.get("file", f"meshes/{mesh_id}.npz")
        with np.load(path) as z:
            out = {"vertices": np.asarray(z["vertices"], np.float32), "faces": np.asarray(z["faces"], np.int32)}
            if "uv" in z.files:
                out["uv"] = np.asarray(z["uv"], np.float32)
        if out["faces"].size and out["faces"].max() >= len(out["vertices"]):
            raise BundleError(f"mesh {mesh_id}: face index {out['faces'].max()} >= {len(out['vertices'])} vertices")
        return out

    def surface_topology(self, surf: dict) -> dict:
        """faces, src, face_tag and twins of surfaces/<name>.npz."""
        path = self.root / surf.get("file", f"surfaces/{surf['name']}.npz")
        with np.load(path) as z:
            faces = np.asarray(z["faces"], np.int32)
            tags = z["face_tag"] if "face_tag" in z.files else np.zeros(len(faces))
            twins = z["twins"] if "twins" in z.files else np.zeros((0, 2))
            out = {"faces": faces, "face_tag": np.asarray(tags, np.uint8),
                   "twins": np.asarray(twins, np.int64).reshape(-1, 2)}
        return out

    def entities(self, kind: str) -> list[dict]:
        return list(self.scene.get(kind, []))


def _check_frames(b: Bundle) -> None:
    if b.body_q is None:
        return
    shp = b.body_q.shape
    if len(shp) != 3 or shp[0] != b.count or shp[2] != 7:
        raise BundleError(f"frames/body_q.npy has shape {shp}, expected [{b.count}, B, 7]")
    nb = len(b.scene.get("bodies", []))
    if shp[1] < nb:
        raise BundleError(f"frames/body_q.npy has {shp[1]} bodies but scene.json lists {nb}")


def _check_shapes(b: Bundle) -> None:
    nb = b.body_q.shape[1] if b.body_q is not None else 0
    meshes = b.scene.get("meshes", {})
    for s in b.scene.get("shapes", []):
        where = f"shapes[id={s.get('id')}, label={s.get('label')!r}]"
        typ = _require(s, "type", where)
        if typ not in SHAPE_TYPES:
            raise BundleError(f"{where}: type {typ!r} not in {SHAPE_TYPES}")
        body = int(s.get("body", -1))
        if body >= nb:
            raise BundleError(f"{where}: body {body} but body_q has {nb} bodies")
        if typ == "mesh" and str(s.get("mesh")) not in meshes:
            raise BundleError(f"{where}: mesh {s.get('mesh')!r} not in scene.json meshes")
        if len(s.get("xform", [0] * 7)) != 7:
            raise BundleError(f"{where}: xform {s.get('xform')!r} is not [px py pz qx qy qz qw]")


def _check_entities(b: Bundle) -> None:
    for kind, prefix in ENTITY_KINDS.items():
        for e in b.entities(kind):
            name = _require(e, "name", kind)
            arr = b.array(prefix, name)
            n = int(e.get("count", e.get("verts", arr.shape[1] if arr.ndim > 1 else 0)))
            width = 7 if prefix == "inst" else 3
            if arr.ndim != 3 or arr.shape[0] != b.count or arr.shape[2] != width or arr.shape[1] != n:
                raise BundleError(f"frames/{prefix}_{name}.npy shape {arr.shape}, expected [{b.count}, {n}, {width}]")
            if kind == "points":
                vel = b.array("vel", name)
                if vel.shape != arr.shape:
                    raise BundleError(f"frames/vel_{name}.npy shape {vel.shape} != positions {arr.shape}")


def _check_surfaces(b: Bundle) -> None:
    for s in b.entities("surfaces"):
        topo = b.surface_topology(s)
        nv = b.array("surf", s["name"]).shape[1]
        faces, twins = topo["faces"], topo["twins"]
        if faces.size and (faces.min() < 0 or faces.max() >= nv):
            raise BundleError(f"surfaces/{s['name']}: face index out of [0, {nv})")
        if len(topo["face_tag"]) != len(faces):
            raise BundleError(f"surfaces/{s['name']}: face_tag has {len(topo['face_tag'])} for {len(faces)} faces")
        if twins.size and (twins.min() < 0 or twins.max() >= nv):
            raise BundleError(f"surfaces/{s['name']}: twins must be local vertex ids in [0, {nv})")


def load_bundle(root: str | Path) -> Bundle:
    """Read and validate a bundle; raises BundleError naming the bad field."""
    root = Path(root)
    path = root / "scene.json"
    if not path.exists():
        raise BundleError(f"no scene.json in {root}")
    scene = json.loads(path.read_text())
    if scene.get("arb") != 1:
        raise BundleError(f"scene.json: arb is {scene.get('arb')!r}, this renderer reads ARB 1")
    frames = _require(scene, "frames", "scene.json")
    fps = float(_require(frames, "fps", "scene.json frames"))
    if fps <= 0:
        raise BundleError(f"scene.json frames.fps is {fps}, must be > 0")
    t = np.load(root / "frames" / "t.npy")
    if t.shape != (int(frames["count"]),):
        raise BundleError(f"frames/t.npy has shape {t.shape}, frames.count is {frames['count']}")
    bq_path = root / "frames" / "body_q.npy"
    body_q = np.load(bq_path, mmap_mode="r") if bq_path.exists() else None
    b = Bundle(root=root, scene=scene, t=t, fps=fps, offsets=_world_offsets(scene), body_q=body_q)
    _check_frames(b)
    _check_shapes(b)
    _check_entities(b)
    _check_surfaces(b)
    log.info("bundle %s: %d frames at %.1f fps, %d shapes, %d bodies", root, b.count, fps,
             len(scene.get("shapes", [])), len(scene.get("bodies", [])))
    return b
