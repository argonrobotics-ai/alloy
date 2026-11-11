# SPDX-License-Identifier: Apache-2.0
"""An independent reference for the scan-texture law: a scan's glb mesh drawn with its own texture by a numpy
rasterizer through the camera's pinhole model, and the error measures the law reads.

Nothing here imports the camera's renderers. The rasterizer follows the camera's conventions: ``look_at`` axes
(x right, y up, z back), the vertical field of view, pixel centres at ``(k + 0.5) / size`` and image row 0 at the
top. It also follows the scan interface: glTF UVs (v down from the image's top row), bilinear filtering on the sRGB
values with texel centres at ``(x + 0.5) / W`` and repeat wrap, then sRGB decoding. Samples are 4 x 4 per pixel
with perspective-correct barycentrics, averaged as a box filter.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRIMESH_DEFAULT = (0.4, 0.4, 0.4)  # trimesh's baseColorFactor for an MTL without Kd: the camera reads it as 1


@dataclass
class ScanMesh:
    """A glb primitive in the body frame (+Z up): points (V, 3), tris (T, 3), glTF uv (V, 2), sRGB image (H, W, 3)."""

    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray
    image: np.ndarray
    factor: tuple[float, float, float]


@dataclass
class View:
    """A pinhole camera: eye, target, up (z by default), vertical field of view in degrees, image size."""

    eye: tuple
    target: tuple
    width: int
    height: int
    fov: float = 60.0
    up: tuple = (0.0, 0.0, 1.0)


@dataclass
class Reference:
    """Linear RGB (H, W, 3) of the textured mesh, the share of each pixel's samples that hit it (H, W), and the
    mean cosine between the surface normal and the view direction there (H, W)."""

    rgb: np.ndarray
    coverage: np.ndarray
    facing: np.ndarray


def load_scan(path: Path) -> ScanMesh:
    """The first primitive of a glb file, turned from glTF (+Y up) into the body frame: (x, y, z) -> (x, -z, y)."""
    from PIL import Image
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    binary = b[20 + n + 8:]

    def view_bytes(i: int) -> bytes:
        v = j["bufferViews"][i]
        return binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]

    def accessor(i: int) -> np.ndarray:
        a = j["accessors"][i]
        dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        raw = view_bytes(a["bufferView"])[a.get("byteOffset", 0):]
        return np.frombuffer(raw, dtype, a["count"] * width).reshape(a["count"], width)

    prim = j["meshes"][0]["primitives"][0]
    p = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
    pbr = j["materials"][prim["material"]]["pbrMetallicRoughness"]
    img = view_bytes(j["images"][j["textures"][pbr["baseColorTexture"]["index"]]["source"]]["bufferView"])
    return ScanMesh(np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1), accessor(prim["indices"]).reshape(-1, 3),
                    accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float64),
                    np.asarray(Image.open(io.BytesIO(img)).convert("RGB")),
                    tuple(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))[:3]))


def quat_xyzw_matrix(q) -> np.ndarray:
    x, y, z, w = (float(c) for c in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb8(c: np.ndarray) -> np.ndarray:
    """Linear RGB -> sRGB-encoded values on a 0-255 scale (float: the law's error unit)."""
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def sample_texture(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup of glTF ``uv`` (N, 2) in an sRGB uint8 image (repeat wrap), decoded to linear RGB (N, 3)."""
    h, w = image.shape[:2]
    x, y = uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    img = image.astype(np.float64) / 255.0

    def at(xi, yi):
        return img[np.mod(yi, h), np.mod(xi, w)]

    top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
    bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
    return srgb_to_linear(top * (1 - fy) + bottom * fy)


def _camera_axes(v: View) -> np.ndarray:
    """Columns: camera x (right), y (up), z (back) in world, as ``alloy.camera.pose.look_at_matrix``."""
    eye, target, up = (np.asarray(a, np.float64) for a in (v.eye, v.target, v.up))
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), -f], axis=1)


def _raster(points: np.ndarray, v: View, ss: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sample column, sample row, depth) of each point: sample k's centre is at k; row 0 at the top."""
    c = (points - np.asarray(v.eye, np.float64)) @ _camera_axes(v)
    depth = -c[:, 2]
    ey = np.tan(0.5 * np.radians(v.fov))
    ex = ey * v.width / v.height
    sx, sy = c[:, 0] / depth / ex, c[:, 1] / depth / ey
    return (sx + 1) * 0.5 * v.width * ss - 0.5, (1 - sy) * 0.5 * v.height * ss - 0.5, depth


def _hits(mesh: ScanMesh, points: np.ndarray, v: View, ss: int) -> tuple[np.ndarray, np.ndarray]:
    """Per sample: the nearest triangle (-1 = none) and its perspective-correct barycentrics."""
    X, Y, depth = _raster(points, v, ss)
    shape = (v.height * ss, v.width * ss)
    zbuf, tri_id, bary = np.full(shape, np.inf), np.full(shape, -1, np.int64), np.zeros(shape + (3,))
    for t, (a, b, c) in enumerate(mesh.tris):
        if min(depth[a], depth[b], depth[c]) <= 1e-6:
            continue
        x0, x1 = int(np.ceil(min(X[a], X[b], X[c]))), int(np.floor(max(X[a], X[b], X[c])))
        y0, y1 = int(np.ceil(min(Y[a], Y[b], Y[c]))), int(np.floor(max(Y[a], Y[b], Y[c])))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, shape[1] - 1), min(y1, shape[0] - 1)
        if x1 < x0 or y1 < y0:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        area = (X[b] - X[a]) * (Y[c] - Y[a]) - (X[c] - X[a]) * (Y[b] - Y[a])
        if abs(area) < 1e-12:
            continue
        la = ((X[b] - gx) * (Y[c] - gy) - (X[c] - gx) * (Y[b] - gy)) / area
        lb = ((X[c] - gx) * (Y[a] - gy) - (X[a] - gx) * (Y[c] - gy)) / area
        lc = 1.0 - la - lb
        inside = (la >= 0) & (lb >= 0) & (lc >= 0)
        if not inside.any():
            continue
        w = np.stack([la / depth[a], lb / depth[b], lc / depth[c]], axis=-1)
        z = 1.0 / w.sum(-1)
        win = inside & (z < zbuf[y0:y1 + 1, x0:x1 + 1])
        zbuf[y0:y1 + 1, x0:x1 + 1][win] = z[win]
        tri_id[y0:y1 + 1, x0:x1 + 1][win] = t
        bary[y0:y1 + 1, x0:x1 + 1][win] = w[win] * z[win][:, None]
    return tri_id, bary


def render_reference(mesh: ScanMesh, pos, quat_xyzw, v: View, ss: int = 4) -> Reference:
    """The mesh at body pose (pos, quat) seen from ``v``: linear RGB of its texture (factor 1), coverage, facing."""
    points = mesh.points @ quat_xyzw_matrix(quat_xyzw).T + np.asarray(pos, np.float64)
    tri_id, bary = _hits(mesh, points, v, ss)
    hit = tri_id >= 0
    corners = mesh.tris[tri_id[hit]]
    uv = np.einsum("nk,nkj->nj", bary[hit], mesh.uv[corners])
    n = np.cross(points[mesh.tris[:, 1]] - points[mesh.tris[:, 0]], points[mesh.tris[:, 2]] - points[mesh.tris[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    centre = points[corners].mean(1)
    to_eye = np.asarray(v.eye, np.float64) - centre
    cos = np.abs(np.einsum("nj,nj->n", n[tri_id[hit]], to_eye / np.linalg.norm(to_eye, axis=1, keepdims=True)))
    rgb, facing = np.zeros(hit.shape + (3,)), np.zeros(hit.shape)
    rgb[hit], facing[hit] = sample_texture(mesh.image, uv), cos

    def pool(a):
        return a.reshape(v.height, ss, v.width, ss, *a.shape[2:]).mean(axis=(1, 3))

    return Reference(pool(rgb), pool(hit.astype(np.float64)), pool(facing))


def law_mask(ref: Reference, min_facing: float = 0.8, erode: int = 1) -> np.ndarray:
    """Pixels fully on the mesh, at least ``erode`` pixels from its edge, on surfaces facing the camera."""
    m = (ref.coverage >= 1.0) & (ref.facing >= min_facing)
    for _ in range(erode):
        m[1:] &= m[:-1].copy()
        m[:-1] &= m[1:].copy()
        m[:, 1:] &= m[:, :-1].copy()
        m[:, :-1] &= m[:, 1:].copy()
    return m


def affine_fit(img: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per channel (gain, offset) of img ~ gain * ref + offset over the mask (the light on the surface)."""
    g, o = np.zeros(3), np.zeros(3)
    for ch in range(3):
        a = np.stack([ref[..., ch][mask], np.ones(int(mask.sum()))], axis=1)
        g[ch], o[ch] = np.linalg.lstsq(a, img[..., ch][mask], rcond=None)[0]
    return g, o


def region_errors(a8: np.ndarray, b8: np.ndarray, mask: np.ndarray, grid: int = 4, min_pixels: int = 8) -> np.ndarray:
    """Largest per-channel difference of mean colour in each cell of a ``grid`` x ``grid`` split of the mask's
    bounding box (cells with fewer than ``min_pixels`` mask pixels are skipped)."""
    rows, cols = np.nonzero(mask)
    ys = np.linspace(rows.min(), rows.max() + 1, grid + 1).astype(int)
    xs = np.linspace(cols.min(), cols.max() + 1, grid + 1).astype(int)
    out = []
    for i in range(grid):
        for k in range(grid):
            m = np.zeros_like(mask)
            m[ys[i]:ys[i + 1], xs[k]:xs[k + 1]] = True
            m &= mask
            if m.sum() >= min_pixels:
                out.append(np.abs(a8[m].mean(0) - b8[m].mean(0)).max())
    return np.asarray(out)


def write_scan(path: Path, mesh: ScanMesh) -> None:
    """``mesh`` as a one-primitive glb in the glTF frame (+Y up), its image embedded as PNG, the way the importer lays
    out a scan: indices, POSITION, TEXCOORD_0, one material whose baseColorTexture is the image."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(mesh.image, np.uint8)).save(buf, "PNG")
    p = mesh.points
    gltf_points = np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1).astype(np.float32)
    blobs = [mesh.tris.astype(np.uint32).tobytes(), gltf_points.tobytes(), mesh.uv.astype(np.float32).tobytes(),
             buf.getvalue()]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\0" * (-len(blob) % 4)
    material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "baseColorFactor": [*mesh.factor, 1.0],
                                         "roughnessFactor": 0.9, "metallicFactor": 0.0}}
    j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
         "meshes": [{"primitives": [{"attributes": {"POSITION": 1, "TEXCOORD_0": 2}, "indices": 0, "material": 0}]}],
         "materials": [material], "textures": [{"source": 0}], "images": [{"bufferView": 3, "mimeType": "image/png"}],
         "accessors": [{"bufferView": 0, "componentType": 5125, "count": int(mesh.tris.size), "type": "SCALAR"},
                       {"bufferView": 1, "componentType": 5126, "count": len(p), "type": "VEC3",
                        "min": gltf_points.min(0).tolist(), "max": gltf_points.max(0).tolist()},
                       {"bufferView": 2, "componentType": 5126, "count": len(p), "type": "VEC2"}],
         "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    text = json.dumps(j).encode()
    text += b" " * (-len(text) % 4)
    head = struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(binary))
    Path(path).write_bytes(head + struct.pack("<II", len(text), 0x4E4F534A) + text
                           + struct.pack("<II", len(binary), 0x004E4942) + binary)
