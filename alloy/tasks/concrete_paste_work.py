# SPDX-License-Identifier: Apache-2.0
"""Device pieces of the concrete process layer (``concrete_paste``) that companion layers import (SOFT-2's sealant
contact): the sole's footprint and touch test, its slide, the advance gate, the bearing count, extent and pad
pressure, and the blister lift.

- **Footprint.** A cell is under the working tool's sole when its centre, carried along the chart's normal onto the
  sole's plane, lies within the sole's half extents about its centre, or GA's separable cover there exceeds 1/4
  (their ``footprint``, ``paste.py:572-600``). The sole's frame is its applicator's: centre ``center``, facing
  ``normal``, half extents along ``axis`` and ``normal`` x ``axis`` (body frame). The cover's cell, and the pass's
  added cell, is sqrt(cell area) (GA's cells are square; on a non-square chart, their mean size whichever way the sole
  slides).
- **Touch.** It bears when also the lowest tool underside there is within ``BEAR_BAND`` of the paste surface
  (``tf.wceil`` <= base + h + BEAR_BAND: surface flow's work fields for every tool, store tools included, so a
  curved nose bears by its own shape; ``coupling.py:53, 1001-1003``). Their bearing also asks for paste thicker
  than 10 µm; the layer's ``paste`` input marks where paste is (it may lie in the base, with no film over it).
- **Slide.** The sole centre's velocity, rotation included, in the chart's plane (``coupling.py:479-485``), on a
  static chart (a chart riding a body: its own motion is not taken off; every concrete scene's chart is static).
- **Advance.** A sole works only while its twice low-passed slide exceeds ``MIN_ADVANCE`` (``coupling.py:57, 60,
  641``).
- **Pressure.** The working tool's vertical load over its bearing cells' count x the cell area (``p = f/(n·cell²)``,
  ``coupling.py:1003-1007``), per world; the count is an integer sum, so it is the same on every run and device.
- **Passes.** The bearing patch's extent along the slide plus a cell (``coupling.py:1015-1021``), from the bearing
  cells' least and greatest projection on the slide's direction (min / max: order-independent).
- **Blisters.** Trapped water above BLISTER_Q lifts the surface by 4 x the excess, box-blurred over +-7.5 mm along
  each axis of the plane chart and capped at 3 mm; a blister never sinks back (their ``paste.py`` _blisters, run with
  the bleed). Their np.roll wraps around the region, which their pad's 40 mm non-paste margin keeps from carrying a
  blister across (``layout.py:224-231``): it reads zeros past the paste. Our charts end at the paste, so the blur
  reads zeros past the chart's edges (the window's sum over 2k+1, as theirs with its margin). Their update skips rises
  under 1e-5 m; this one applies every rise.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.geometry import chart_xform
from alloy.tasks import concrete_materials as CM

BEAR_BAND = 3.0e-4          # m: a sole within this of the paste surface bears on it (coupling.py:53)
MIN_ADVANCE = 1.0e-3        # m/s: slower soles do no work (coupling.py:641)
FAR = 1.0e30                # an empty extent's bounds


@wp.struct
class Sole:
    """Where the working sole bears, in the flow's one chart's frame: the cells' centres and the chart's normal and
    cell size, the chart's pose per world (body, local), each tool's sole centre in its body frame, its slide [W, T]
    and its bearing patch's bounds along the slide [W]."""

    pos: wp.array[wp.vec3]
    n: wp.vec3
    cell: wp.float32
    cbody: wp.array2d[wp.int32]
    clocal: wp.array2d[wp.transform]
    center: wp.array[wp.vec3]
    ex: wp.array[wp.vec3]         # each tool's sole axes in its body frame: along its half extents, and its back
    ey: wp.array[wp.vec3]
    ez: wp.array[wp.vec3]
    slid: wp.array2d[wp.vec3]
    lo: wp.array[wp.float32]
    hi: wp.array[wp.float32]


@wp.func
def in_chart(body_q: wp.array[wp.transform], g: Sole, w: int, b: int) -> wp.transform:
    """Body ``b``'s pose in world ``w``'s chart frame."""
    X = chart_xform(body_q, g.cbody[w, 0], g.clocal[w, 0])
    return wp.transform_multiply(wp.transform_inverse(X), body_q[b])


@wp.func
def inside_sole(xf: wp.transform, g: Sole, k: int, half: wp.vec2, p: wp.vec3) -> bool:
    """The cell centre ``p`` is under tool ``k``'s sole: within its half extents (sole axes ex, ey) about its centre
    once carried along the chart normal onto the sole's plane, or GA's cover there over 1/4 (paste.py:590-598)."""
    o = wp.transform_point(xf, g.center[k])
    zs = wp.transform_vector(xf, g.ez[k])
    dn = wp.dot(g.n, zs)
    if dn < 0.2:                                        # a sole standing on edge has no footprint (theirs: < 0.2)
        return False
    rel = p - o
    q = wp.quat_rotate_inv(wp.transform_get_rotation(xf), rel - g.n * (wp.dot(rel, zs) / dn))
    x = wp.abs(wp.dot(q, g.ex[k]))
    y = wp.abs(wp.dot(q, g.ey[k]))
    cell = float(g.cell)
    if x <= half[0] and y <= half[1]:
        return True
    cover = wp.clamp((half[0] - x) / cell + 0.5, 0.0, 1.0) * wp.clamp((half[1] - y) / cell + 0.5, 0.0, 1.0)
    return cover > 0.25


@wp.func
def bears(inside: bool, ceil: float, base: float, h: wp.float64) -> bool:
    """A cell under the sole bears on its paste: the underside within BEAR_BAND of the surface."""
    return inside and float(ceil) <= base + float(h) + BEAR_BAND


@wp.func
def slide(body_q: wp.array[wp.transform], qd: wp.spatial_vector, com: wp.vec3, g: Sole, w: int, b: int,
          center: wp.vec3) -> wp.vec3:
    """The sole centre's velocity in the chart's plane, chart frame [m/s]: the centre of mass's plus the rotation's
    (world ``body_qd``, linear first)."""
    xf = body_q[b]
    lin = wp.vec3(qd[0], qd[1], qd[2])
    ang = wp.vec3(qd[3], qd[4], qd[5])
    v = lin + wp.cross(ang, wp.transform_point(xf, center) - wp.transform_point(xf, com))
    X = chart_xform(body_q, g.cbody[w, 0], g.clocal[w, 0])
    vc = wp.quat_rotate_inv(wp.transform_get_rotation(X), v)
    return vc - g.n * wp.dot(vc, g.n)


@wp.func
def advancing(v: wp.vec3) -> bool:
    """The sole slides: its low-passed advance exceeds MIN_ADVANCE."""
    return wp.length(v) > MIN_ADVANCE


@wp.func
def bearing(body_q: wp.array[wp.transform], g: Sole, tool: wp.array2d[wp.int32], half: wp.array[wp.vec2], w: int,
            k: int, c: int, ceil: float, base: float, h: wp.float64) -> bool:
    """Cell ``c`` is under tool ``k``'s sole (in the chart frame) and bears on its paste."""
    xf = in_chart(body_q, g, w, tool[w, k])
    return bears(inside_sole(xf, g, k, half[k], g.pos[c]), ceil, base, h)


@wp.kernel
def bear_count_kernel(paste: wp.array2d[wp.int32], h: wp.array2d[wp.float64], base: wp.array2d[wp.float32],
                      ceil: wp.array2d[wp.float32], g: Sole, active: wp.array[wp.int32], tool: wp.array2d[wp.int32],
                      body_q: wp.array[wp.transform], half: wp.array[wp.vec2], count: wp.array[wp.int32]):
    """Per world: the working tool's bearing paste cells, counted, and their extent along its slide (into g.lo,
    g.hi)."""
    w, c = wp.tid()
    k = active[w]
    if paste[w, c] == 0 or k < 0:
        return
    if not bearing(body_q, g, tool, half, w, k, c, ceil[w, c], base[w, c], h[w, c]):
        return
    wp.atomic_add(count, w, 1)
    v = g.slid[w, k]
    if wp.length(v) > 1.0e-9:
        along = wp.dot(g.pos[c], wp.normalize(v))
        wp.atomic_min(g.lo, w, along)
        wp.atomic_max(g.hi, w, along)


@wp.kernel
def pressure_kernel(load: wp.array2d[wp.float32], active: wp.array[wp.int32], count: wp.array[wp.int32],
                    cell_area: float, pressure: wp.array[wp.float32]):
    """Per world: the working tool's load over its bearing cells' area [Pa] (0 when it bears on nothing)."""
    w = wp.tid()
    k = active[w]
    p = float(0.0)
    if k >= 0 and count[w] > 0:
        p = wp.max(load[w, k], 0.0) / (float(count[w]) * cell_area)
    pressure[w] = p


@wp.func
def patch_along(lo: float, hi: float, cell: float, half: wp.vec2, moving: bool) -> float:
    """The bearing patch's extent along the slide plus a cell (coupling.py:1015-1021); at rest the sole's width."""
    if moving and hi >= lo:
        return hi - lo + cell
    return wp.max(2.0 * wp.min(half[0], half[1]), cell)


@wp.kernel
def lift_kernel(q: wp.array2d[wp.float32], out: wp.array2d[wp.float32]):
    """The blister lift before its blur: BLISTER_GAIN x the trapped water above BLISTER_Q."""
    w, c = wp.tid()
    out[w, c] = CM.BLISTER_GAIN * wp.max(q[w, c] - CM.BLISTER_Q, 0.0)


@wp.kernel
def blur_kernel(src: wp.array2d[wp.float32], nx: int, ny: int, k: int, along_i: int, dst: wp.array2d[wp.float32]):
    """A box mean over +-k cells along one axis of a plane chart (c = i·ny + j), zero past its edges."""
    w, c = wp.tid()
    i = c / ny
    j = c - i * ny
    acc = float(0.0)
    for s in range(-k, k + 1):
        if along_i == 1:
            if i + s >= 0 and i + s < nx:
                acc += src[w, (i + s) * ny + j]
        elif j + s >= 0 and j + s < ny:
            acc += src[w, i * ny + j + s]
    dst[w, c] = acc / float(2 * k + 1)


@wp.kernel
def blister_kernel(lift: wp.array2d[wp.float32], bl: wp.array2d[wp.float32]):
    """bl <- min(max(bl, lift), BLISTER_CAP): a blister never sinks back."""
    w, c = wp.tid()
    bl[w, c] = wp.min(wp.max(bl[w, c], lift[w, c]), CM.BLISTER_CAP)


__all__ = ["BEAR_BAND", "FAR", "MIN_ADVANCE", "Sole", "advancing", "bear_count_kernel", "bearing", "bears",
           "blister_kernel", "blur_kernel", "in_chart", "inside_sole", "lift_kernel", "patch_along",
           "pressure_kernel", "slide"]
