# SPDX-License-Identifier: Apache-2.0
"""Scene pack v1 (``scene.json`` + PLY meshes in world space, one linear RGB colour per mesh, a sun, a constant sky
and a camera) -> :class:`~alloy.camera.scene.SceneData` for the diffuse renderer.

Example::

    from alloy.camera.pack_v1 import scene_from_pack_v1
    sd = scene_from_pack_v1("t12_pack", num_worlds=1)      # sd.camera holds the pack's view
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .errors import CameraError
from .scene import IDENTITY_XF, Lighting, MeshGeom, SceneData, Sun, matrix_to_quat


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian PLY with float xyz vertices and uchar-count int triangle faces."""
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    head = data[:end].decode().split("\n")
    nv = int(next(h for h in head if h.startswith("element vertex")).split()[-1])
    nf = int(next(h for h in head if h.startswith("element face")).split()[-1])
    nprop = sum(1 for h in head if h.startswith("property float"))
    if nprop != 3:
        raise CameraError(f"{path}: {nprop} float vertex properties", "write x, y, z only")
    v = np.frombuffer(data, "<f4", nv * 3, end).reshape(nv, 3)
    rec = np.frombuffer(data, dtype=[("n", "u1"), ("i", "<i4", 3)], count=nf, offset=end + nv * 12)
    if not np.all(rec["n"] == 3):
        raise CameraError(f"{path}: faces that are not triangles", "triangulate the mesh")
    return v.astype(np.float32), rec["i"].astype(np.int32)


def scene_from_pack_v1(pack_dir, num_worlds: int = 1, static_names=None) -> SceneData:
    """Read a v1 pack. Meshes named by ``static_names`` (prefixes) are shared by all worlds; others are per world."""
    root = Path(pack_dir).expanduser()
    doc = json.loads((root / "scene.json").read_text())
    meshes, colors, static = [], [], []
    for m in doc["meshes"]:
        meshes.append(MeshGeom(m["name"], *read_ply(root / m["file"])))
        colors.append(m.get("color", (0.8, 0.8, 0.8)))
        static.append(bool(static_names) and any(m["name"].startswith(s) for s in static_names))
    rows = [(k, -1) for k, s in enumerate(static) if s]
    rows += [(k, w) for w in range(num_worlds) for k, s in enumerate(static) if not s]
    idx, grp = (np.array(x) for x in zip(*rows))
    lt = doc["light"]
    sun = Sun.from_euler(lt["sun_euler_xyz"], float(lt["sun_energy"]), angle=lt.get("sun_angle"))
    sd = SceneData(meshes, idx, grp, np.full(len(idx), -1), np.tile(IDENTITY_XF, (len(idx), 1)),
                   np.asarray(colors, np.float32)[idx], num_worlds, [meshes[k].name for k in idx],
                   Lighting([sun], tuple(float(c) for c in lt["sky_rgb"]), 1.0))
    cam = doc["camera"]
    rot = np.asarray(cam["rot"], np.float64)
    sd.camera = {"pose": np.concatenate([np.asarray(cam["eye"], np.float64), matrix_to_quat(rot)]),
                 "fov_deg": float(np.degrees(cam["fov"])), "fit": cam.get("sensor_fit", "VERTICAL"),
                 "clip": (float(cam.get("clip_start", 0.1)), float(cam.get("clip_end", 1000.0)))}
    return sd
