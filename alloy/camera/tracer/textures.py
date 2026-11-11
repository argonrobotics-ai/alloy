# SPDX-License-Identifier: Apache-2.0
"""Image textures of the fast tracer: per-vertex UVs and base-colour images of a ``SceneData``, packed for the
kernels, and the lookup they run at a hit.

A hit on instance i of mesh m is drawn ``inst_color[i] * srgb_to_linear(bilinear(textures[m.texture], uv))`` when
the mesh has a texture and UVs and ``inst_textured[i]`` is set, else ``inst_color[i]`` as before. The blend is on
the sRGB values, then decoded (as Cycles does for 8-bit sRGB images); texel centres sit at ((x + 0.5) / W,
(y + 0.5) / H); UVs wrap (repeat).

Layout on the device (fields of ``integrator.Mats``, one entry per instance unless noted):

- ``tex``: every texture's texels in one uint32 array, row 0 = top of the image, packed sRGB8 (r | g << 8 | b << 16);
- ``uv``: every textured mesh's per-vertex UVs in one vec2 array (u right, v down, (0, 0) = the image's top-left);
- ``tex_w``: texture width in texels, 0 = this instance is drawn flat;
- ``tex_h``, ``tex_off`` (first texel in ``tex``), ``uv_off`` (first UV of the instance's mesh in ``uv``).

Example::

    packed = pack_textures(sd, order)            # numpy arrays; packed.tex_w[i] == 0 for a flat instance
    rgb = sample_host(packed, 0, 0.25, 0.5)      # the same lookup on the host: linear RGB, before inst_color
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..errors import CameraError


@dataclass
class PackedTextures:
    """Host arrays in the layout above (instance arrays in device-scene instance order)."""

    tex: np.ndarray
    uv: np.ndarray
    tex_w: np.ndarray
    tex_h: np.ndarray
    tex_off: np.ndarray
    uv_off: np.ndarray

    @property
    def any(self) -> bool:
        return bool((self.tex_w > 0).any())


def srgb_to_linear(c):
    """Linear value of sRGB-encoded ``c`` in 0..1 (IEC 61966-2-1); the kernels' ``srgb_decode`` on the host."""
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def pack_image(img: np.ndarray) -> np.ndarray:
    """One uint8 (H, W, 3) sRGB texture as packed texels, uint32 (H * W,), row 0 first."""
    a = np.asarray(img)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] < 3 or a.shape[0] < 1 or a.shape[1] < 1:
        raise CameraError(f"texture of dtype {a.dtype}, shape {a.shape}", "give textures as uint8 (H, W, 3) sRGB")
    rgb = a[..., :3].astype(np.uint32)
    return (rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)).reshape(-1)


def _textured_meshes(sd) -> list[int]:
    """Meshes of ``sd`` that have both UVs and a texture, checked (CameraError names the field to fix)."""
    out = []
    for m, mesh in enumerate(sd.meshes):
        uv, t = getattr(mesh, "uv", None), int(getattr(mesh, "texture", -1))
        if uv is None or t < 0:
            continue
        if t >= len(sd.textures):
            raise CameraError(f"mesh {mesh.name!r} names texture {t}",
                              f"use an index into textures (0..{len(sd.textures) - 1})")
        if np.asarray(uv).size != 2 * len(mesh.points):
            raise CameraError(f"mesh {mesh.name!r}: {np.asarray(uv).size // 2} UVs for {len(mesh.points)} vertices",
                              "give one UV per vertex, in the mesh's vertex order")
        out.append(m)
    return out


def pack_textures(sd, order: np.ndarray) -> PackedTextures:
    """The textures and UVs of ``sd`` (``MeshGeom.uv`` / ``.texture``, ``textures``, ``inst_textured``; all optional)
    for instances in ``order``."""
    n = len(order)
    ints = (np.zeros(n, np.int32) for _ in range(4))
    out = PackedTextures(np.zeros(1, np.uint32), np.zeros((1, 2), np.float32), *ints)
    used = _textured_meshes(sd) if getattr(sd, "textures", None) else []
    if not used:
        return out
    texels, t_off, pos = [], {}, 0
    for t in sorted({int(sd.meshes[m].texture) for m in used}):
        packed = pack_image(sd.textures[t])
        t_off[t] = pos
        texels.append(packed)
        pos += len(packed)
    uvs, u_off, pos = [], {}, 0
    for m in used:
        uv = np.asarray(sd.meshes[m].uv, np.float32).reshape(-1, 2)
        u_off[m] = pos
        uvs.append(uv)
        pos += len(uv)
    out.tex, out.uv = np.concatenate(texels), np.concatenate(uvs)
    textured = getattr(sd, "inst_textured", None)
    textured = np.ones(len(sd.inst_mesh), bool) if textured is None else np.asarray(textured, bool)
    for i, (m, on) in enumerate(zip(np.asarray(sd.inst_mesh)[order], textured[order])):
        if on and int(m) in u_off:
            t = int(sd.meshes[int(m)].texture)
            out.tex_h[i], out.tex_w[i] = np.asarray(sd.textures[t]).shape[:2]
            out.tex_off[i], out.uv_off[i] = t_off[t], u_off[int(m)]
    return out


def fill_mats(mats, packed: PackedTextures, device) -> None:
    """Upload ``packed`` into the texture fields of a ``Mats`` struct."""
    mats.tex = wp.array(packed.tex, dtype=wp.uint32, device=device)
    mats.uv = wp.array(packed.uv, dtype=wp.vec2, device=device)
    for name in ("tex_w", "tex_h", "tex_off", "uv_off"):
        setattr(mats, name, wp.array(getattr(packed, name), dtype=wp.int32, device=device))


def sample_host(packed: PackedTextures, inst: int, u: float, v: float) -> np.ndarray:
    """The kernels' lookup on the host: linear RGB of instance ``inst``'s texture at UV (u, v)."""
    w, h, off = int(packed.tex_w[inst]), int(packed.tex_h[inst]), int(packed.tex_off[inst])
    x, y = (u - np.floor(u)) * w - 0.5, (v - np.floor(v)) * h - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    out = np.zeros(3)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            p = int(packed.tex[off + ((y0 + dy) % h) * w + (x0 + dx) % w])
            out += wx * wy * np.array([p & 255, (p >> 8) & 255, (p >> 16) & 255])
    return srgb_to_linear(out / 255.0)
