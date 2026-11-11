# SPDX-License-Identifier: Apache-2.0
"""Linear radiance -> display images: Blender's "Standard" view (sRGB OETF, 8-bit) or AgX (``agx``: no hard clip),
on device or on the host, plus a dependency-free PNG writer and the clipped-pixel guard.

Example::

    import numpy as np
    from alloy.camera.view import standard_u8, write_png
    img = standard_u8(np.full((4, 4, 3), 0.18, np.float32))     # mid grey -> 118
    assert img.dtype == np.uint8 and int(img[0, 0, 0]) == 118
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

from dataclasses import dataclass

import numpy as np
import warp as wp

from .agx import agx, contrast_of


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    """Linear -> sRGB-encoded floats (the piecewise sRGB curve), negatives clipped to 0."""
    x = np.maximum(np.asarray(x, np.float32), 0.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def standard_u8(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) linear -> uint8 (Standard view, exposure 0, gamma 1, no dither), rounding as Blender does."""
    f = srgb_oetf(rgb)
    out = np.where(f <= 0.0, 0.0, np.where(f > 1.0 - 0.5 / 255.0, 255.0, 255.0 * f + 0.5))
    return out.astype(np.uint8)


@wp.func
def _oetf(x: float) -> float:
    v = wp.max(x, 0.0)
    if v <= 0.0031308:
        return 12.92 * v
    return 1.055 * wp.pow(v, 1.0 / 2.4) - 0.055


@wp.func
def _to_byte(f: float) -> wp.uint8:
    if f <= 0.0:
        return wp.uint8(0)
    if f > 1.0 - 0.5 / 255.0:
        return wp.uint8(255)
    return wp.uint8(int(255.0 * f + 0.5))


@wp.kernel
def standard_u8_kernel(rgb: wp.array(dtype=wp.vec3), exposure_scale: float, out: wp.array2d(dtype=wp.uint8)):
    """One pixel: linear * 2^exposure -> sRGB bytes (out is (pixels, 3))."""
    i = wp.tid()
    c = rgb[i] * exposure_scale
    for k in range(3):
        out[i, k] = _to_byte(_oetf(c[k]))


@wp.kernel
def agx_u8_kernel(rgb: wp.array(dtype=wp.vec3), exposure_scale: float, contrast: float,
                  out: wp.array2d(dtype=wp.uint8)):
    """One pixel: linear * 2^exposure -> AgX -> sRGB bytes (out is (pixels, 3))."""
    i = wp.tid()
    c = agx(rgb[i] * exposure_scale, contrast)
    for k in range(3):
        out[i, k] = _to_byte(_oetf(c[k]))


@dataclass(frozen=True)
class View:
    """A display transform: ``transform`` "standard" or "agx", ``exposure`` in stops, AgX ``contrast``."""

    transform: str = "standard"
    exposure: float = 0.0
    contrast: float = 1.0


STANDARD = View()


def view_from_settings(view_settings: dict | None) -> View:
    """The View of Blender view settings ({"view_transform", "look", "exposure"}): AgX for "AgX" (and "Filmic", whose
    role AgX took in Blender 4), else Standard. Example: ``view_from_settings({"view_transform": "AgX"})``."""
    vs = view_settings or {}
    kind = "agx" if str(vs.get("view_transform", "Standard")) in ("AgX", "Filmic") else "standard"
    return View(kind, float(vs.get("exposure", 0.0) or 0.0), contrast_of(vs.get("look")) if kind == "agx" else 1.0)


def to_display(rgb: wp.array, exposure: float = 0.0, out: wp.array | None = None, view: View = STANDARD) -> wp.array:
    """Device linear (N, H, W) vec3 -> uint8 (N, H, W, 3) with ``view``; ``exposure`` (stops) adds to the view's."""
    shape = tuple(rgb.shape)
    n = int(np.prod(shape))
    if out is None or out.shape != (n, 3):
        out = wp.empty((n, 3), dtype=wp.uint8, device=rgb.device)
    scale = float(2.0 ** (exposure + view.exposure))
    if view.transform == "agx":
        wp.launch(agx_u8_kernel, dim=n, inputs=[rgb.reshape((n,)), scale, float(view.contrast)], outputs=[out],
                  device=rgb.device)
    else:
        wp.launch(standard_u8_kernel, dim=n, inputs=[rgb.reshape((n,)), scale], outputs=[out], device=rgb.device)
    return out.reshape((*shape, 3))


def clipped_fraction(img: np.ndarray) -> float:
    """Share of pixels with a channel at 255 (blown out) in an (..., H, W, 3) uint8 image."""
    a = np.asarray(img)
    return float((a[..., :3].max(axis=-1) >= 255).mean()) if a.size else 0.0


def write_png(path: str | Path, img: np.ndarray) -> Path:
    """Write an (H, W, 3) or (H, W) uint8 image as PNG (no extra packages). Returns the path."""
    a = np.ascontiguousarray(img)
    if a.dtype != np.uint8:
        a = standard_u8(a)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)
    return p
