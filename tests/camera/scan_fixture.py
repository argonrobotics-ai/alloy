# SPDX-License-Identifier: Apache-2.0
"""Test fixture for scan textures: a small object card whose ``visual.glb`` carries a known image (no network, no
fetched scans). Shared by the loader, fast-tracer, full-tracer and law tests (SCAN-TEXTURES parts A-D).

``tile_card(root)`` writes a library root with one card, ``scan_tile``: a 10 x 10 x 1 cm box (bottom centre at the
body origin), its physics mesh and hull sidecar, and a visual.glb of the same box whose TOP face (+z) shows the whole
image upright when seen from above with +y up in the picture (image row 0 at +y, column 0 at -x). The side and bottom
faces show the image's centre texel. Use it as ``alloy.Scene(roots=[root]).add_object("tile", "scan_tile", ...)``.

Example::

    root = tile_card(tmp_path, image=quadrants(64))
    scene = alloy.Scene(roots=[root]); scene.add_object("tile", TILE, pos=(0, 0, 0))
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np

from alloy.camera.glb import write_glb

TILE = "scan_tile"
SIZE = (0.10, 0.10, 0.01)  # full edges, metres
QUADRANT_RGB = ((230, 40, 40), (40, 200, 60), (40, 70, 230), (240, 240, 240))  # TL, TR, BL, BR (sRGB)


def quadrants(n: int = 64) -> np.ndarray:
    """uint8 (n, n, 3) image: red top-left, green top-right, blue bottom-left, white bottom-right."""
    img = np.zeros((n, n, 3), np.uint8)
    h = n // 2
    for (r0, c0), rgb in zip(((0, 0), (0, h), (h, 0), (h, h)), QUADRANT_RGB):
        img[r0:r0 + h, c0:c0 + h] = rgb
    return img


def png_bytes(img: np.ndarray) -> bytes:
    """A PNG file of a uint8 (H, W, 3) image (no Pillow needed to write it)."""
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\0" + img[r].tobytes() for r in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def box_mesh(size=SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points (24, 3), tris (12, 3), uv (24, 2)) of a box in the BODY frame (z up, bottom centre at the origin), four
    vertices per face, wound outward; the top face maps the whole image as the module docstring says."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2]
    faces = {  # corners counter-clockwise seen from outside
        "top": [(-hx, hy, hz), (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz)],
        "bottom": [(-hx, hy, 0), (hx, hy, 0), (hx, -hy, 0), (-hx, -hy, 0)],
        "front": [(-hx, -hy, 0), (hx, -hy, 0), (hx, -hy, hz), (-hx, -hy, hz)],
        "back": [(hx, hy, 0), (-hx, hy, 0), (-hx, hy, hz), (hx, hy, hz)],
        "left": [(-hx, hy, 0), (-hx, -hy, 0), (-hx, -hy, hz), (-hx, hy, hz)],
        "right": [(hx, -hy, 0), (hx, hy, 0), (hx, hy, hz), (hx, -hy, hz)],
    }
    pts, tris, uv = [], [], []
    for name, quad in faces.items():
        k = len(pts)
        pts += quad
        tris += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
        uv += [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)] if name == "top" else [(0.5, 0.5)] * 4
    return np.array(pts, np.float32), np.array(tris, np.int32), np.array(uv, np.float32)


def to_gltf(points: np.ndarray) -> np.ndarray:
    """Body-frame points (z up) in glTF's +Y-up frame: (x, z, -y) (the scan importers' turn)."""
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1).astype(np.float32)


def _obj(points: np.ndarray, tris: np.ndarray, header: str = "", group: str | None = None) -> str:
    lines = [header] if header else []
    lines += [f"o {group}"] if group else []
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in tris]
    return "\n".join(lines) + "\n"


def tile_card(root, image: np.ndarray | None = None, name: str = TILE, factor=None, glb: bool = True) -> Path:
    """A library root under ``root`` holding card ``name`` (module docstring); returns the root.
    ``factor``: the glb's baseColorFactor (RGBA); ``glb=False`` leaves the visual.glb out."""
    root = Path(root)
    mesh_dir = root / "meshes" / name
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(parents=True, exist_ok=True)
    pts, tris, uv = box_mesh()
    corners = np.array([[x, y, z] for x in (-SIZE[0] / 2, SIZE[0] / 2) for y in (-SIZE[1] / 2, SIZE[1] / 2)
                        for z in (0.0, SIZE[2])], np.float32)
    hull_tris = _box_tris()
    physics = _obj(corners, hull_tris)
    (mesh_dir / "physics.obj").write_text(physics)
    sha = hashlib.sha256(physics.encode()).hexdigest()
    (mesh_dir / "physics.hulls.obj").write_text(_obj(corners, hull_tris, f"# source_sha256: {sha}", "hull_0"))
    if glb:
        part = {"points": to_gltf(pts), "tris": tris, "uv": uv, "image": png_bytes(quadrants() if image is None
                                                                                       else image),
                "mime": "image/png", "roughness": 0.5, "metallic": 0.0}
        if factor is not None:
            part["factor"] = tuple(factor)
        write_glb(mesh_dir / "visual.glb", [part])
    (root / "objects" / f"{name}.yaml").write_text(
        f"schema: alloy.object/1\nname: {name}\nshape:\n  mesh:\n    path: meshes/{name}/physics.obj\n"
        f"    sha256: {sha}\n    collision: decompose\nmaterial: abs\n")
    return root


def _box_tris() -> np.ndarray:
    """Outward triangles of the 8 corners ordered x-major, then y, then z (index = 4 ix + 2 iy + iz)."""
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    return np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int32)


BALL = "scan_ball"
BALL_RADIUS = 0.04


def checker(width: int = 256, height: int = 128, cols: int = 8, rows: int = 4) -> np.ndarray:
    """uint8 (height, width, 3) checker of ``cols`` x ``rows`` squares, red-orange and cream."""
    y, x = np.mgrid[0:height, 0:width]
    odd = ((x * cols // width) + (y * rows // height)) % 2 == 1
    img = np.empty((height, width, 3), np.uint8)
    img[odd], img[~odd] = (220, 60, 40), (240, 230, 200)
    return img


def icosphere(subdiv: int, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(points, outward tris) of an icosphere of ``radius`` at the origin."""
    t = (1.0 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    pts = [np.asarray(p, np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        mid, nf = {}, []

        def m(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = pts[a] + pts[b]
                pts.append(p / np.linalg.norm(p))
                mid[key] = len(pts) - 1
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return (np.array(pts) * radius).astype(np.float32), np.array(f, np.int32)


def uv_sphere(radius: float, lon: int = 64, lat: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, outward tris, uv) of a latitude-longitude sphere at the origin: u = longitude / 2 pi from +x, v =
    polar angle / pi from +z (v down the image from the north pole); the u = 0 / 1 meridian is a seam."""
    th = np.linspace(0.0, np.pi, lat + 1)
    ph = np.linspace(0.0, 2 * np.pi, lon + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    pts = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3) * radius
    uv = np.stack([P / (2 * np.pi), T / np.pi], -1).reshape(-1, 2)
    idx = np.arange((lat + 1) * (lon + 1)).reshape(lat + 1, lon + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    keep = np.ptp(pts[tris], axis=1).max(axis=1) > 1e-9 * radius  # drop the poles' zero-area triangles
    return pts.astype(np.float32), tris[keep].astype(np.int32), uv.astype(np.float32)


def ball_card(root, image: np.ndarray | None = None, name: str = BALL, material: str = "silicone_40a",
              radius: float = BALL_RADIUS) -> Path:
    """A library root under ``root`` with card ``name``: a sphere of ``radius`` resting on the body origin (centre
    at (0, 0, radius)) whose visual.glb is a latitude-longitude sphere showing ``image`` (default: ``checker()``).
    ``material`` silicone_40a makes it a soft body; ``add_object(..., rigid=True)`` keeps it rigid."""
    root = Path(root)
    mesh_dir = root / "meshes" / name
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(parents=True, exist_ok=True)
    lift = np.array([0.0, 0.0, radius], np.float32)
    phys, ptris = icosphere(3, radius)
    physics = _obj(phys + lift, ptris)
    (mesh_dir / "physics.obj").write_text(physics)
    sha = hashlib.sha256(physics.encode()).hexdigest()
    hull, htris = icosphere(1, radius)
    (mesh_dir / "physics.hulls.obj").write_text(_obj(hull + lift, htris, f"# source_sha256: {sha}", "hull_0"))
    pts, tris, uv = uv_sphere(radius)
    write_glb(mesh_dir / "visual.glb", [{"points": to_gltf(pts + lift), "tris": tris, "uv": uv, "mime": "image/png",
                                         "image": png_bytes(checker() if image is None else image),
                                         "roughness": 0.5, "metallic": 0.0}])
    (root / "objects" / f"{name}.yaml").write_text(
        f"schema: alloy.object/1\nname: {name}\nshape:\n  mesh:\n    path: meshes/{name}/physics.obj\n"
        f"    sha256: {sha}\n    collision: decompose\nmaterial: {material}\n")
    return root
