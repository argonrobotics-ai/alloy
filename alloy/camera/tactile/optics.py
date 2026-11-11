# SPDX-License-Identifier: Apache-2.0
"""Gel optics on the GPU, batched over worlds: Taxim's gel spreading (a pyramid of Gaussian blurs that keeps the
pressed core) and the slope -> colour look-up table.

Taxim (Si and Yuan 2022) models the gel around a pressed object with blurs of decreasing width, re-imposing the
object's shape where it presses deepest after each blur, then reads the colour change of each pixel from a table
indexed by the gel's slope (magnitude, direction) whose entries are quadratic polynomials over the image position.

Example (one 2 mm-deep spherical press, CPU)::

    import numpy as np, warp as wp
    from alloy.camera.tactile.presets import load_preset
    from alloy.camera.tactile.optics import GelOptics
    p = load_preset("gelsight")
    optics = GelOptics(p, height=120, width=160, num_worlds=1, device="cpu")
    yy, xx = np.mgrid[0:120, 0:160]
    r2 = ((xx - 80) ** 2 + (yy - 60) ** 2) * optics.mm_per_px ** 2
    press = np.clip(np.sqrt(np.clip(4.0 - r2, 0, None)) - 1.0, 0, None)[None].astype(np.float32)   # R 2 mm, 1 mm deep
    rgb = optics.shade(optics.spread(wp.array(press, device="cpu")))   # (1, 120, 160) vec3, 0..255 colour
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .presets import TactilePreset
from .stats import PressStats

MAX_TAPS_PER_SIDE = 64


@wp.func
def reflect101(i: int, n: int) -> int:
    """OpenCV's BORDER_REFLECT_101 index (the edge pixel is not repeated)."""
    if n == 1:
        return 0
    period = 2 * (n - 1)
    k = i % period
    if k < 0:
        k += period
    if k >= n:
        k = period - k
    return k


@wp.func
def outside(box: wp.vec4i, pad: int, scale: int, r: int, c: int) -> bool:
    """True when pixel (r, c) of a grid ``scale`` times coarser than the image is farther than pad image pixels from
    the world's pressed box (row0, row1, col0, col1); an empty box is outside everywhere."""
    rr = r * scale
    cc = c * scale
    return (box[1] < box[0] or rr + scale <= box[0] - pad or rr > box[1] + pad or cc + scale <= box[2] - pad
            or cc > box[3] + pad)


@wp.kernel
def blur_rows(src: wp.array3d(dtype=float), weights: wp.array(dtype=float), radius: int,
              box: wp.array(dtype=wp.vec4i), pad: int, scale: int, dst: wp.array3d(dtype=float)):
    """Horizontal pass of a separable Gaussian; pixels beyond ``pad`` of the world's pressed box stay zero."""
    n, r, c = wp.tid()
    if outside(box[n], pad, scale, r, c):
        dst[n, r, c] = 0.0
        return
    w = src.shape[2]
    acc = float(0.0)
    for k in range(-radius, radius + 1):
        acc += weights[k + radius] * src[n, r, reflect101(c + k, w)]
    dst[n, r, c] = acc


@wp.kernel
def blur_cols_keep(src: wp.array3d(dtype=float), weights: wp.array(dtype=float), radius: int,
                   box: wp.array(dtype=wp.vec4i), pad: int, scale: int, maxp: wp.array(dtype=float),
                   press: wp.array3d(dtype=float), keep_frac: float, dst: wp.array3d(dtype=float)):
    """Vertical pass, then the pressed core (press > keep_frac x deepest press of the world) keeps its shape
    (keep_frac < 0: no core, e.g. on a coarse grid)."""
    n, r, c = wp.tid()
    if outside(box[n], pad, scale, r, c):
        dst[n, r, c] = 0.0
        return
    h = src.shape[1]
    acc = float(0.0)
    for k in range(-radius, radius + 1):
        acc += weights[k + radius] * src[n, reflect101(r + k, h), c]
    if keep_frac >= 0.0:
        p0 = press[n, r, c]
        if p0 > keep_frac * maxp[n]:
            acc = p0
    dst[n, r, c] = acc


@wp.kernel
def downsample(src: wp.array3d(dtype=float), f: int, box: wp.array(dtype=wp.vec4i), pad: int,
               dst: wp.array3d(dtype=float)):
    """Mean of each f x f block (partial blocks at the edges average what they hold)."""
    n, r, c = wp.tid()
    if outside(box[n], pad, f, r, c):
        dst[n, r, c] = 0.0
        return
    h = src.shape[1]
    w = src.shape[2]
    acc = float(0.0)
    cnt = int(0)
    for i in range(f):
        for j in range(f):
            rr = r * f + i
            cc = c * f + j
            if rr < h and cc < w:
                acc += src[n, rr, cc]
                cnt += 1
    dst[n, r, c] = acc / float(wp.max(cnt, 1))


@wp.kernel
def upsample_keep(src: wp.array3d(dtype=float), f: int, box: wp.array(dtype=wp.vec4i), pad: int,
                  maxp: wp.array(dtype=float), press: wp.array3d(dtype=float), keep_frac: float,
                  dst: wp.array3d(dtype=float)):
    """Bilinear upsampling of a grid f times coarser, then the pressed core keeps its shape."""
    n, r, c = wp.tid()
    if outside(box[n], pad, 1, r, c):
        dst[n, r, c] = 0.0
        return
    hc = src.shape[1]
    wc = src.shape[2]
    y = wp.clamp((float(r) + 0.5) / float(f) - 0.5, 0.0, float(hc - 1))
    x = wp.clamp((float(c) + 0.5) / float(f) - 0.5, 0.0, float(wc - 1))
    r0 = int(y)
    c0 = int(x)
    r1 = wp.min(r0 + 1, hc - 1)
    c1 = wp.min(c0 + 1, wc - 1)
    fy = y - float(r0)
    fx = x - float(c0)
    top = src[n, r0, c0] * (1.0 - fx) + src[n, r0, c1] * fx
    bot = src[n, r1, c0] * (1.0 - fx) + src[n, r1, c1] * fx
    acc = top * (1.0 - fy) + bot * fy
    p0 = press[n, r, c]
    if p0 > keep_frac * maxp[n]:
        acc = p0
    dst[n, r, c] = acc


@wp.func
def poly6(table: wp.array(dtype=float), base: int, u: float, v: float) -> float:
    """Taxim's quadratic polynomial (u^2, v^2, uv, u, v, 1) stored at table[base:base + 6]."""
    return (table[base] * u * u + table[base + 1] * v * v + table[base + 2] * u * v + table[base + 3] * u
            + table[base + 4] * v + table[base + 5])


@wp.func
def slope_at(p: wp.array3d(dtype=float), n: int, r: int, c: int, mm_per_px: float) -> wp.vec2:
    """(d press / d row, d press / d col), dimensionless, by central differences (one-sided at the border)."""
    h = p.shape[1]
    w = p.shape[2]
    r0 = wp.max(r - 1, 0)
    r1 = wp.min(r + 1, h - 1)
    c0 = wp.max(c - 1, 0)
    c1 = wp.min(c + 1, w - 1)
    sr = float(0.0)
    sc = float(0.0)
    if r1 > r0:
        sr = (p[n, r1, c] - p[n, r0, c]) / (float(r1 - r0) * mm_per_px)
    if c1 > c0:
        sc = (p[n, r, c1] - p[n, r, c0]) / (float(c1 - c0) * mm_per_px)
    return wp.vec2(sr, sc)


@wp.func
def lut_delta(table: wp.array(dtype=float), bins: int, slope: wp.vec2, u: float, v: float,
              fade_rad: float) -> wp.vec3:
    """Colour change (0..255 units, RGB) of a gel pixel with this slope at normalised position (u, v)."""
    t = wp.sqrt(slope[0] * slope[0] + slope[1] * slope[1])
    if t <= 0.0:
        return wp.vec3(0.0)
    mag = wp.atan(t)
    ang = wp.atan2(slope[0], slope[1])
    bm = float(bins - 1)
    ix = wp.clamp(int(mag / (0.5 * wp.pi / bm)), 0, bins - 1)
    iy = wp.clamp(int((ang + wp.pi) / (2.0 * wp.pi / bm)), 0, bins - 1)
    plane = bins * bins * 6
    base = (ix * bins + iy) * 6
    d = wp.vec3(poly6(table, base, u, v), poly6(table, plane + base, u, v), poly6(table, 2 * plane + base, u, v))
    return d * wp.min(mag / fade_rad, 1.0)


@wp.kernel
def shade_kernel(p: wp.array3d(dtype=float), mm_per_px: float, table: wp.array(dtype=float), bins: int,
                 uv_map: wp.vec4, uv_box: wp.vec4, fade_rad: float, out: wp.array3d(dtype=wp.vec3)):
    """Colour change of every pixel; uv_map = (ku, bu, kv, bv) maps a column / row to normalised (u, v)."""
    n, r, c = wp.tid()
    u = wp.clamp(uv_map[0] * float(c) + uv_map[1], uv_box[0], uv_box[2])
    v = wp.clamp(uv_map[2] * float(r) + uv_map[3], uv_box[1], uv_box[3])
    out[n, r, c] = lut_delta(table, bins, slope_at(p, n, r, c, mm_per_px), u, v, fade_rad)


def gaussian_weights(sigma_px: float) -> np.ndarray:
    """Normalised 1-D Gaussian taps (radius ~3.2 sigma, OpenCV-like), or [1] for a negligible blur."""
    if sigma_px < 0.25:
        return np.ones(1, np.float32)
    radius = min(MAX_TAPS_PER_SIDE, max(1, int(math.ceil(3.2 * sigma_px))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma_px) ** 2)
    return (w / w.sum()).astype(np.float32)


@dataclass
class BlurLevel:
    """One Gaussian of the spreading pyramid: on the image grid (f = 1) or on a grid f times coarser."""

    f: int
    taps: wp.array
    radius: int
    coarse: tuple[wp.array, wp.array, wp.array] | None = None  # downsampled, row-blurred, blurred


def coarse_factor(sigma_px: float, enabled: bool = True) -> int:
    """Grid factor for a blur of sigma_px image pixels: the largest power of 2 (<= 8) leaving >= 2 coarse pixels of
    sigma; 1 (the image grid) for narrow blurs. Example: ``coarse_factor(15.5) == 4``."""
    f = 1
    while enabled and f < 8 and sigma_px / (2 * f) >= 2.0:
        f *= 2
    return f


class GelOptics:
    """Device buffers and launches of the optical model for one preset at one image size and world count.

    ``coarse=True`` (default) runs the wide blurs of the spreading pyramid on coarser grids (exact Gaussian there,
    bilinear back up, sigma compensated for the resampling): same result to a few micrometres, several times faster.
    """

    def __init__(self, preset: TactilePreset, height: int, width: int, num_worlds: int, device: str = "cuda:0",
                 coarse: bool = True):
        self.preset, self.device = preset, str(device)
        self.shape = (int(num_worlds), int(height), int(width))
        ch, cw = preset.calib_size
        self.mm_per_px = preset.mm_per_px * cw / width  # gel millimetres per output pixel
        self.table = wp.array(np.ascontiguousarray(preset.table, np.float32).reshape(-1), dtype=float, device=device)
        self.bins = preset.bins
        self.uv_map, self.uv_box = self._uv_maps(height, width)
        self.levels = [self._level(sg / self.mm_per_px, coarse) for sg in preset.smooth_sigmas_mm]
        self.final = self._level(preset.final_sigma_mm / self.mm_per_px, False)
        self.stats = PressStats(*self.shape, device)
        reach = [lv.radius * lv.f + lv.f for lv in self.levels]
        self.pad = sum(reach) + self.final.radius + 1  # how far the gel spreads around a press [px]
        self._a = wp.zeros(self.shape, dtype=float, device=device)
        self._b = wp.zeros(self.shape, dtype=float, device=device)
        self._c = wp.zeros(self.shape, dtype=float, device=device)
        self.delta = wp.zeros(self.shape, dtype=wp.vec3, device=device)

    def _level(self, sigma_px: float, coarse: bool) -> BlurLevel:
        f = coarse_factor(sigma_px, coarse)
        sigma = math.sqrt(max(sigma_px ** 2 - f * f / 4.0, 0.25)) / f if f > 1 else sigma_px
        w = gaussian_weights(sigma)
        lv = BlurLevel(f, wp.array(w, dtype=float, device=self.device), (len(w) - 1) // 2)
        if f > 1:
            n, h, wd = self.shape
            cs = (n, -(-h // f), -(-wd // f))
            lv.coarse = tuple(wp.zeros(cs, dtype=float, device=self.device) for _ in range(3))
        return lv

    def _uv_maps(self, height: int, width: int) -> tuple[wp.vec4, wp.vec4]:
        ch, cw = self.preset.calib_size
        cx, cy, s = (cw - 1) / 2.0, (ch - 1) / 2.0, (cw - 1) / 2.0
        kx, ky = cw / width, ch / height  # output pixel -> calibration pixel: x = (c + 0.5) kx - 0.5
        uv_map = wp.vec4(kx / s, (0.5 * kx - 0.5 - cx) / s, ky / s, (0.5 * ky - 0.5 - cy) / s)
        x0, y0, x1, y1 = self.preset.fit_box
        return uv_map, wp.vec4((x0 - cx) / s, (y0 - cy) / s, (x1 - cx) / s, (y1 - cy) / s)

    def spread(self, press: wp.array, touched: wp.array | None = None) -> wp.array:
        """Taxim's gel spreading of a press map (worlds, H, W) [mm]; returns an internal buffer (same shape).

        Also fills ``stats`` (deepest press, patch sums, bounding box; ``touched`` gives the shape at the deepest
        pixel). Only pixels within the spreading reach of a world's pressed box are computed."""
        self.stats.compute(press, touched, self.mm_per_px)
        src = press
        for lv in self.levels:
            self._run(lv, src, press, self.preset.contact_scale, self._b)
            src = self._b
        self._run(self.final, src, press, -1.0, self._c)
        return self._c

    def _run(self, lv: BlurLevel, src: wp.array, press: wp.array, keep: float, dst: wp.array) -> None:
        box, pad, maxp, dev = self.stats.box, self.pad, self.stats.maxp, self.device
        if lv.coarse is None:
            wp.launch(blur_rows, dim=self.shape, inputs=[src, lv.taps, lv.radius, box, pad, 1], outputs=[self._a],
                      device=dev)
            wp.launch(blur_cols_keep, dim=self.shape, inputs=[self._a, lv.taps, lv.radius, box, pad, 1, maxp, press,
                                                               keep], outputs=[dst], device=dev)
            return
        down, rows, blurred = lv.coarse
        cs = tuple(down.shape)
        wp.launch(downsample, dim=cs, inputs=[src, lv.f, box, pad], outputs=[down], device=dev)
        wp.launch(blur_rows, dim=cs, inputs=[down, lv.taps, lv.radius, box, pad, lv.f], outputs=[rows], device=dev)
        wp.launch(blur_cols_keep, dim=cs, inputs=[rows, lv.taps, lv.radius, box, pad, lv.f, maxp, press, -1.0],
                  outputs=[blurred], device=dev)
        wp.launch(upsample_keep, dim=self.shape, inputs=[blurred, lv.f, box, pad, maxp, press, keep], outputs=[dst],
                  device=dev)

    def shade(self, gel: wp.array) -> wp.array:
        """Colour change (worlds, H, W) vec3 [0..255 units, RGB] of the spread gel surface ``gel`` [mm]."""
        fade = math.radians(max(self.preset.flat_fade_deg, 1e-3))
        wp.launch(shade_kernel, dim=self.shape, inputs=[gel, self.mm_per_px, self.table, self.bins, self.uv_map,
                                                        self.uv_box, fade], outputs=[self.delta], device=self.device)
        return self.delta
