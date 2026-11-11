# SPDX-License-Identifier: Apache-2.0
"""Visual outputs: ``visual.glb`` (textured, PBR) and a ``thumb.png`` preview (needs trimesh and Pillow).

``visual.glb`` follows glTF 2.0: +Y up, metres. The body frame (+Z up) maps to it by a -90 degree turn about X
(x, y, z) -> (x, z, -y); glTF loaders that convert to Z-up (Blender) give back the body frame exactly. Base-colour
textures are re-encoded as JPEG (quality ``JPEG_QUALITY``) unless they carry alpha. The thumbnail is a small
software render (orthographic three-quarter view, two-sided Lambert, 2x supersampled) so it needs no GPU or OpenGL.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

from alloy.importers.common.errors import need

log = logging.getLogger("alloy.importers")

JPEG_QUALITY = 90
Z_UP_TO_GLTF = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])
GREY = np.array([170, 170, 170], dtype=np.float64)


def _jpeg_image(img):
    image_mod = need("PIL.Image")
    if img is None or img.mode in ("RGBA", "LA", "P"):
        return img
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    return image_mod.open(buf)


def base_image(mesh):
    """The base-colour texture image of a mesh (OBJ ``map_Kd`` or glTF ``baseColorTexture``), or None.

    Example::

        base_image(load_source(Path("textured.obj"))).size   # (2048, 2048)
    """
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    img = getattr(mat, "image", None)
    return img if img is not None else getattr(mat, "baseColorTexture", None)


def _pbr(mat):
    """The material as glTF PBR with a JPEG base-colour texture (alpha textures stay PNG)."""
    pbr = mat.to_pbr() if hasattr(mat, "to_pbr") else mat
    if getattr(pbr, "baseColorTexture", None) is not None:
        pbr.baseColorTexture = _jpeg_image(pbr.baseColorTexture)
    pbr.metallicFactor = 0.0 if pbr.metallicFactor is None else pbr.metallicFactor
    pbr.roughnessFactor = 0.8 if pbr.roughnessFactor is None else pbr.roughnessFactor
    return pbr


def export_glb(scene_or_mesh, path: Path, to_body: np.ndarray | None = None) -> None:
    """Write a mesh or a whole ``trimesh.Scene`` as glTF binary in the +Y-up glTF frame, every texture kept.

    ``to_body`` (4x4) takes the source to the body frame (metres, +Z up), e.g. from ``meshops.frame_matrix``; pass
    None when the input is already in the body frame. Exporting the source scene (not the concatenated mesh) keeps
    one material per part.

    Example::

        export_glb(trimesh.load("chair.glb"), Path("visual.glb"), to_body=frame4(0.01, "y"))
    """
    s = scene_or_mesh.copy()
    s.apply_transform(Z_UP_TO_GLTF @ (np.eye(4) if to_body is None else to_body))
    geoms = s.geometry.values() if hasattr(s, "geometry") else [s]
    for g in geoms:
        mat = getattr(getattr(g, "visual", None), "material", None)
        if mat is not None:
            g.visual.material = _pbr(mat)
    Path(path).write_bytes(s.export(file_type="glb"))


# ------------------------------------------------------------------------------------------------ thumbnail


def _camera(vertices: np.ndarray, az_deg: float, el_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows (right, up, toward-camera) and the bounding-box centre."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    back = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    centre = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return np.stack([right, up, back]), centre


def _project(vertices: np.ndarray, n: int, az: float, el: float) -> tuple[np.ndarray, np.ndarray]:
    rot, centre = _camera(vertices, az, el)
    cam = (vertices - centre) @ rot.T
    radius = float(np.max(np.linalg.norm(cam[:, :2], axis=1))) or 1.0
    s = 0.46 * n / radius
    xy = np.stack([n / 2 + s * cam[:, 0], n / 2 - s * cam[:, 1]], axis=1)
    return xy, cam[:, 2], rot


def _raster(xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel face id (-1 = background) and barycentrics, by z-buffer over every triangle."""
    zbuf = np.full((n, n), -np.inf)
    fid = np.full((n, n), -1, dtype=np.int64)
    bary = np.zeros((n, n, 3))
    for i, (a, b, c) in enumerate(faces):
        p = xy[[a, b, c]]
        x0, y0 = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(p.max(axis=0)).astype(int), n - 1)
        if x1 < x0 or y1 < y0:
            continue
        det = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(det) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((gx - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (gy - p[0, 1])) / det
        w2 = ((p[1, 0] - p[0, 0]) * (gy - p[0, 1]) - (gx - p[0, 0]) * (p[1, 1] - p[0, 1])) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
        win = inside & (z > zbuf[y0:y1 + 1, x0:x1 + 1])
        if win.any():
            zbuf[y0:y1 + 1, x0:x1 + 1][win] = z[win]
            fid[y0:y1 + 1, x0:x1 + 1][win] = i
            bary[y0:y1 + 1, x0:x1 + 1][win] = np.stack([w0, w1, w2], axis=-1)[win]
    return fid, bary


def _albedo(mesh, faces: np.ndarray, fid: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """RGB per covered pixel: the base-colour texture at the interpolated UV, else face colours, else grey."""
    hit = fid >= 0
    vis = mesh.visual
    uv = getattr(vis, "uv", None)
    img = base_image(mesh)
    if uv is not None and img is not None and len(uv) == len(mesh.vertices):
        tex = np.asarray(img.convert("RGB"), dtype=np.float64)
        th, tw = tex.shape[:2]
        f = faces[fid[hit]]
        st = np.einsum("pk,pkj->pj", bary[hit], np.asarray(uv, np.float64)[f])
        u = np.clip((st[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
        v = np.clip((1.0 - st[:, 1] % 1.0) * th, 0, th - 1).astype(int)
        return tex[v, u]
    colors = getattr(vis, "face_colors", None)
    if colors is not None and len(colors) == len(faces):
        return np.asarray(colors, np.float64)[fid[hit], :3]
    return np.tile(GREY, (int(hit.sum()), 1))


def render_thumbnail(mesh, size: int = 256, az: float = 35.0, el: float = 25.0, supersample: int = 2) -> np.ndarray:
    """RGBA uint8 image (``size`` x ``size``) of a textured mesh, transparent background.

    Example::

        rgba = render_thumbnail(load_source(Path("textured.obj")), size=256)
    """
    n = size * supersample
    verts = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    xy, depth, rot = _project(verts, n, az, el)
    fid, bary = _raster(xy, depth, faces, n)
    hit = fid >= 0
    normals = np.asarray(mesh.face_normals, np.float64)[fid[hit]]
    light = rot.T @ np.array([-0.4, 0.6, 0.7])
    shade = 0.35 + 0.65 * np.abs(normals @ (light / np.linalg.norm(light)))
    rgba = np.zeros((n, n, 4))
    rgba[hit, :3] = _albedo(mesh, faces, fid, bary) * shade[:, None]
    rgba[hit, 3] = 255.0
    small = rgba.reshape(size, supersample, size, supersample, 4).mean(axis=(1, 3))
    return np.clip(small, 0, 255).astype(np.uint8)


def write_png(rgba: np.ndarray, path: Path) -> None:
    """Save an RGBA uint8 array as PNG (deterministic bytes: no timestamp chunks).

    Example::

        write_png(render_thumbnail(mesh), Path("thumb.png"))
    """
    image_mod = need("PIL.Image")
    image_mod.fromarray(rgba, mode="RGBA").save(path, format="PNG", optimize=True)
