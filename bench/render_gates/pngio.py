# SPDX-License-Identifier: Apache-2.0
"""Minimal 8-bit PNG reader and writer (numpy + zlib), used when Pillow is not installed.

Reads non-interlaced 8-bit grey, grey+alpha, RGB and RGBA with every PNG row filter; writes RGB with filter 0. Pillow
is faster and is used by :func:`bench.render_gates.metrics.load_rgb` when present.

Example:
    >>> import numpy as np, tempfile, pathlib
    >>> p = pathlib.Path(tempfile.mkdtemp()) / "a.png"
    >>> a = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    >>> write_png(p, a)
    >>> bool((read_png(p) == a).all())
    True
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

SIG = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


class PNGError(ValueError):
    """The file is not a PNG this reader supports; the message says what to convert it to."""


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an ``(H, W, 3) uint8`` array as an RGB PNG.

    Example:
        >>> write_png(Path("x.png"), np.zeros((4, 4, 3), np.uint8))  # doctest: +SKIP
    """
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise PNGError(f"write_png needs (H, W, 3) uint8, got {a.shape}")
    h, w, _ = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)], axis=1).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    Path(path).write_bytes(SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))


def _chunks(data: bytes):
    pos = len(SIG)
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        yield data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        pos += 12 + n


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def _unfilter_row(ftype: int, row: np.ndarray, prev: np.ndarray, bpp: int) -> np.ndarray:
    if ftype == 0:
        return row
    if ftype == 2:
        return (row.astype(np.uint16) + prev).astype(np.uint8)
    out = row.astype(np.int32)
    if ftype == 1:
        for c in range(bpp):
            out[c::bpp] = np.cumsum(out[c::bpp]) & 0xFF
        return out.astype(np.uint8)
    up = prev.astype(np.int32)
    for i in range(len(out)):
        left = out[i - bpp] if i >= bpp else 0
        if ftype == 3:
            out[i] = (out[i] + ((left + up[i]) >> 1)) & 0xFF
        elif ftype == 4:
            out[i] = (out[i] + _paeth(left, up[i], up[i - bpp] if i >= bpp else 0)) & 0xFF
        else:
            raise PNGError(f"unknown PNG filter type {ftype}")
    return out.astype(np.uint8)


def read_png(path: Path) -> np.ndarray:
    """Read a PNG as ``(H, W, 3) uint8`` (grey is repeated, alpha dropped).

    Example:
        >>> read_png(Path("ref.png")).dtype  # doctest: +SKIP
        dtype('uint8')
    """
    data = Path(path).read_bytes()
    if not data.startswith(SIG):
        raise PNGError(f"{path} is not a PNG")
    idat, hdr = b"", None
    for tag, body in _chunks(data):
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
    if hdr is None or hdr[2] != 8 or hdr[3] not in CHANNELS or hdr[6] != 0:
        raise PNGError(f"{path}: only 8-bit non-interlaced grey/RGB(A) PNGs are supported; install Pillow for others")
    w, h, ch = hdr[0], hdr[1], CHANNELS[hdr[3]]
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * ch)
    img = np.empty((h, w * ch), np.uint8)
    prev = np.zeros(w * ch, np.uint8)
    for y in range(h):
        prev = img[y] = _unfilter_row(int(raw[y, 0]), raw[y, 1:], prev, ch)
    img = img.reshape(h, w, ch)
    return np.repeat(img[..., :1], 3, axis=2) if ch < 3 else img[..., :3].copy()
