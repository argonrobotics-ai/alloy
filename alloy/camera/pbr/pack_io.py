# SPDX-License-Identifier: Apache-2.0
"""Read scene packs (v1 and v2) into plain Python/numpy, no Blender needed. See PACK_V2.md.

    pack = load_pack(dir)                       # Pack: .version, .static (dict), .frames (list of int)
    fr = pack.frame(420)                        # Frame: .camera, .lights, .meshes (list of Mesh)
    mesh.v (V,3) f32 world positions, mesh.tri (T,3) i32, mesh.n (T,3,3) f32 world shading normals per triangle
    corner, mesh.mat (T,) i32 slot index, mesh.uv[name] (T,3,2), mesh.attr[name] (T,3,k), mesh.meta (dict).

A v1 pack (scene.json + meshes/*.ply: flat colour per mesh) is presented through the same objects: a Diffuse material
per mesh, a SUN light, a uniform world colour and a vertical-FOV camera (`static["v1"] = True`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    meta: dict
    v: np.ndarray
    tri: np.ndarray
    n: np.ndarray
    mat: np.ndarray
    uv: dict = field(default_factory=dict)
    attr: dict = field(default_factory=dict)


@dataclass
class Frame:
    frame: int
    camera: dict
    lights: list
    meshes: list


class Pack:
    def __init__(self, root: Path, static: dict, version: int):
        self.root, self.static, self.version = root, static, version
        self.frames = list(static.get("frames", [0]))
        self._v1_cache: Frame | None = None

    def frame(self, f: int | None = None) -> Frame:
        if self.version == 1:
            return self._frame_v1()
        f = self.frames[0] if f is None else f
        doc = json.loads((self.root / "frames" / str(f) / "frame.json").read_text())
        return Frame(f, doc["camera"], doc["lights"], [self._load_mesh(m) for m in doc["meshes"]])

    def _load_mesh(self, meta: dict) -> Mesh:
        z = np.load(self.root / meta["file"])
        uv = {k[3:]: z[k] for k in z.files if k.startswith("uv:")}
        attr = {k[5:]: z[k] for k in z.files if k.startswith("attr:")}
        return Mesh(meta, z["v"], z["tri"], z["n"], z["mat"], uv, attr)

    def _frame_v1(self) -> Frame:
        if self._v1_cache is None:
            self._v1_cache = _v1_frame(self.root, self.static)
        return self._v1_cache


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian PLY: float x y z vertices, `list uchar int` faces (as written by pack v1)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode().splitlines()
    nv = int(next(l.split()[-1] for l in head if l.startswith("element vertex")))
    nf = int(next(l.split()[-1] for l in head if l.startswith("element face")))
    nprop = sum(1 for l in head if l.startswith("property float"))
    v = np.frombuffer(raw, "<f4", nv * nprop, end).reshape(nv, nprop)[:, :3]
    rec = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    f = np.frombuffer(raw, rec, nf, end + nv * nprop * 4)["i"]
    return v.astype(np.float32), f.astype(np.int32)


def _flat_normals(v: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = v[tri]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return np.repeat(n[:, None, :], 3, axis=1).astype(np.float32)


def _v1_frame(root: Path, doc: dict) -> Frame:
    meshes = []
    for k, m in enumerate(doc["meshes"]):
        v, tri = read_ply(root / m["file"])
        meta = {"name": m["name"], "file": m["file"], "tris": len(tri), "materials": [f"v1_mesh{k}"],
                "object_to_world": np.eye(4).tolist(), "v1_color": m["color"]}
        meshes.append(Mesh(meta, v, tri, _flat_normals(v, tri), np.zeros(len(tri), np.int32)))
    cam = doc["camera"]
    lt = doc["light"]
    return Frame(0, {"v1": cam}, [{"type": "SUN", "euler_xyz": lt["sun_euler_xyz"], "energy": lt["sun_energy"]}], meshes)


def load_pack(root: str | Path) -> Pack:
    root = Path(root)
    if (root / "pack.json").exists():
        static = json.loads((root / "pack.json").read_text())
        if static.get("pack_version") != 2:
            raise ValueError(f"unsupported pack_version {static.get('pack_version')}")
        return Pack(root, static, 2)
    if (root / "scene.json").exists():
        doc = json.loads((root / "scene.json").read_text())
        static = {**doc, "v1": True, "frames": [0], "world": {"uniform_rgb": doc["light"]["sky_rgb"]},
                  "materials": {f"v1_mesh{k}": {"diffuse_rgb": m["color"]} for k, m in enumerate(doc["meshes"])}}
        return Pack(root, static, 1)
    raise FileNotFoundError(f"{root}: neither pack.json (v2) nor scene.json (v1)")
