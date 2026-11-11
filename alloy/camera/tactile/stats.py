# SPDX-License-Identifier: Apache-2.0
"""Per-world summary of a press map, without atomics: deepest press, press-weighted patch sums, the touched shape at
the deepest pixel and the bounding box of pressed pixels. Two passes (rows, then worlds), so the result does not
depend on thread order and no world's pixels contend for one memory word.

Example::

    from alloy.camera.tactile.stats import PressStats
    st = PressStats(num_images=4, height=120, width=160, device="cpu")
    st.compute(press, touched, mm_per_px)      # st.maxp (4,), st.sums (4,) vec3, st.lead (4,), st.box (4,) vec4i
"""
from __future__ import annotations

import warp as wp


@wp.kernel
def row_stats(press: wp.array3d(dtype=float), touched: wp.array3d(dtype=wp.int32), has_touched: int,
              mm_per_px: float, rmax: wp.array2d(dtype=float), rsum: wp.array2d(dtype=wp.vec3), rlead: wp.array2d(dtype=wp.int32),
              rcols: wp.array2d(dtype=wp.vec2i)):
    """One thread per image row: max, (sum p, sum p x, sum p y), shape at the row max, first / last pressed col."""
    n, r = wp.tid()
    w = press.shape[2]
    y = (float(r) + 0.5 - 0.5 * float(press.shape[1])) * mm_per_px
    m = float(0.0)
    s = wp.vec3(0.0)
    lead = int(-1)
    c0 = int(w)
    c1 = int(-1)
    for c in range(w):
        p = press[n, r, c]
        if p > 0.0:
            x = (float(c) + 0.5 - 0.5 * float(w)) * mm_per_px
            s += wp.vec3(p, p * x, p * y)
            c0 = wp.min(c0, c)
            c1 = c
            if p > m:
                m = p
                if has_touched != 0:
                    lead = touched[n, r, c]
    rmax[n, r] = m
    rsum[n, r] = s
    rlead[n, r] = lead
    rcols[n, r] = wp.vec2i(c0, c1)


@wp.kernel
def world_stats(rmax: wp.array2d(dtype=float), rsum: wp.array2d(dtype=wp.vec3), rlead: wp.array2d(dtype=wp.int32),
                rcols: wp.array2d(dtype=wp.vec2i), maxp: wp.array(dtype=float), sums: wp.array(dtype=wp.vec3),
                lead: wp.array(dtype=wp.int32), box: wp.array(dtype=wp.vec4i)):
    """One thread per image: fold the rows. box = (row0, row1, col0, col1) of pressed pixels, row1 < row0 if none."""
    n = wp.tid()
    h = rmax.shape[1]
    m = float(0.0)
    s = wp.vec3(0.0)
    ld = int(-1)
    r0 = int(h)
    r1 = int(-1)
    c0 = int(1 << 30)
    c1 = int(-1)
    for r in range(h):
        cols = rcols[n, r]
        if cols[1] >= 0:
            r0 = wp.min(r0, r)
            r1 = r
            c0 = wp.min(c0, cols[0])
            c1 = wp.max(c1, cols[1])
            s += rsum[n, r]
            if rmax[n, r] > m:
                m = rmax[n, r]
                ld = rlead[n, r]
    maxp[n] = m
    sums[n] = s
    lead[n] = ld
    box[n] = wp.vec4i(r0, r1, c0, c1)


class PressStats:
    """Buffers and launches of the two-pass press summary for (images, H, W) press maps."""

    def __init__(self, num_images: int, height: int, width: int, device: str):
        n, h, dev = int(num_images), int(height), str(device)
        self.device = dev
        self._rmax = wp.zeros((n, h), dtype=float, device=dev)
        self._rsum = wp.zeros((n, h), dtype=wp.vec3, device=dev)
        self._rlead = wp.zeros((n, h), dtype=wp.int32, device=dev)
        self._rcols = wp.zeros((n, h), dtype=wp.vec2i, device=dev)
        self.maxp = wp.zeros(n, dtype=float, device=dev)
        self.sums = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.lead = wp.zeros(n, dtype=wp.int32, device=dev)
        self.box = wp.zeros(n, dtype=wp.vec4i, device=dev)
        self._no_touch = wp.zeros((1, 1, 1), dtype=wp.int32, device=dev)

    def compute(self, press: wp.array, touched: wp.array | None, mm_per_px: float) -> None:
        """Fill maxp, sums, lead (touched shape at the deepest pixel, -1 if none / unknown) and box."""
        n, h, _w = press.shape
        has = 0 if touched is None else 1
        wp.launch(row_stats, dim=(n, h), inputs=[press, touched if has else self._no_touch, has, mm_per_px],
                  outputs=[self._rmax, self._rsum, self._rlead, self._rcols], device=self.device)
        wp.launch(world_stats, dim=n, inputs=[self._rmax, self._rsum, self._rlead, self._rcols],
                  outputs=[self.maxp, self.sums, self.lead, self.box], device=self.device)
