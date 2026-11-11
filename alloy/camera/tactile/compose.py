# SPDX-License-Identifier: Apache-2.0
"""Final tactile image per pixel: unpressed gel + colour change, marker dots, wear, tone, noise -> uint8 RGB.

Example: used by :class:`alloy.camera.tactile.renderer.TactileRenderer`; one launch writes every world's image::

    wp.launch(compose_kernel, dim=(worlds, H, W), inputs=[delta, bg, aug, draw, mm_per_px, seed], outputs=[rgb])
"""
from __future__ import annotations

import warp as wp


@wp.struct
class AugArrays:
    """Per-world augmentation values on the device (see alloy.camera.tactile.augment)."""

    light: wp.array(dtype=wp.vec3)
    bg_gain: wp.array(dtype=wp.vec3)
    tone: wp.array(dtype=wp.vec4)  # contrast, brightness, noise std, unused
    shift: wp.array(dtype=wp.vec2)  # background offset [px]
    spots: wp.array2d(dtype=wp.vec4)  # x, y, radius [mm], darkening
    marker_alpha: wp.array2d(dtype=float)


@wp.struct
class MarkerDraw:
    """Marker dots to draw: displaced centres [mm] per world and the rest grid that indexes them."""

    pos: wp.array2d(dtype=wp.vec2)
    x0: float
    y0: float
    pitch_x: float
    pitch_y: float
    radius: float
    darkness: float
    rows: int
    cols: int
    enabled: int


@wp.func
def sample_bg(bg: wp.array2d(dtype=wp.vec3), y: float, x: float) -> wp.vec3:
    """Bilinear background sample at pixel coordinates (row y, column x), clamped to the image."""
    h = bg.shape[0]
    w = bg.shape[1]
    yc = wp.clamp(y, 0.0, float(h - 1))
    xc = wp.clamp(x, 0.0, float(w - 1))
    r0 = int(yc)
    c0 = int(xc)
    r1 = wp.min(r0 + 1, h - 1)
    c1 = wp.min(c0 + 1, w - 1)
    fy = yc - float(r0)
    fx = xc - float(c0)
    top = bg[r0, c0] * (1.0 - fx) + bg[r0, c1] * fx
    bot = bg[r1, c0] * (1.0 - fx) + bg[r1, c1] * fx
    return top * (1.0 - fy) + bot * fy


@wp.func
def marker_cover(md: MarkerDraw, alpha: wp.array2d(dtype=float), n: int, p: wp.vec2, px_mm: float) -> float:
    """How much of the pixel at p [mm] a marker dot covers (0..1, one-pixel soft edge)."""
    ci = int(wp.round((p[0] - md.x0) / md.pitch_x))
    ri = int(wp.round((p[1] - md.y0) / md.pitch_y))
    cover = float(0.0)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r = ri + dr
            c = ci + dc
            if r >= 0 and r < md.rows and c >= 0 and c < md.cols:
                j = r * md.cols + c
                dist = wp.length(md.pos[n, j] - p)
                a = wp.clamp((md.radius - dist) / px_mm + 0.5, 0.0, 1.0) * alpha[n, j]
                cover = wp.max(cover, a)
    return cover


@wp.func
def wear_factor(spots: wp.array2d(dtype=wp.vec4), n: int, p: wp.vec2) -> float:
    f = float(1.0)
    for k in range(spots.shape[1]):
        s = spots[n, k]
        if s[3] > 0.0:
            d = p - wp.vec2(s[0], s[1])
            f *= 1.0 - s[3] * wp.exp(-0.5 * wp.dot(d, d) / (s[2] * s[2]))
    return f


@wp.kernel
def compose_kernel(delta: wp.array3d(dtype=wp.vec3), bg: wp.array2d(dtype=wp.vec3), aug: AugArrays, md: MarkerDraw,
                   mm_per_px: float, seed: int, out: wp.array4d(dtype=wp.uint8)):
    """uint8 RGB of pixel (r, c) of world n."""
    n, r, c = wp.tid()
    h = delta.shape[1]
    w = delta.shape[2]
    sh = aug.shift[n]
    base = sample_bg(bg, float(r) + sh[1], float(c) + sh[0])
    col = wp.cw_mul(base, aug.bg_gain[n]) + wp.cw_mul(delta[n, r, c], aug.light[n])
    p = wp.vec2((float(c) + 0.5 - 0.5 * float(w)) * mm_per_px, (float(r) + 0.5 - 0.5 * float(h)) * mm_per_px)
    if md.enabled != 0:
        col = col * (1.0 - md.darkness * marker_cover(md, aug.marker_alpha, n, p, mm_per_px))
    col = col * wear_factor(aug.spots, n, p)
    tone = aug.tone[n]
    col = (col - wp.vec3(127.5)) * (1.0 + tone[0]) + wp.vec3(127.5 + tone[1])
    if tone[2] > 0.0:
        rng = wp.rand_init(seed, ((n * h + r) * w + c) * 3)
        col += wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * tone[2]
    for k in range(3):
        out[n, r, c, k] = wp.uint8(wp.clamp(wp.round(col[k]), 0.0, 255.0))
