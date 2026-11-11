# SPDX-License-Identifier: Apache-2.0
"""Which hole and which stage a drill is in: the lookups the drill's front makes in a layer's hole table (device).

- ``crossing``: where the bit's axis crosses the face (the point a hole's entry is matched against).
- ``nearest_hole``: the hole whose entry lies within its own widest radius (or the drill's, if wider) of that point.
  So a small drill off the centre of a big hole is still in that hole, not starting a new one.
- ``stage_of``: the stage of the drill's diameter, or why there is none (inside a wider stage's void, or every stage
  slot used), and the bottoms the front needs in those cases.
- ``hole_radius``: the hole's radius at an axial position (the union of its stages); ``landing``: where an end of
  a given radius stops in it.
- ``open_hole``: the table writes when a drill starts a hole or a stage.
"""

from __future__ import annotations

import warp as wp

MIN_CROSS = wp.constant(0.2)     # the axis must cross the face plane at least this steeply to use the crossing
SAME_D = wp.constant(1.0e-5)     # m: a drill this close to a stage's diameter continues that stage


@wp.struct
class StageInfo:
    stage: wp.int32              # the stage of this diameter, or -1 (none yet)
    formed: wp.int32             # 1 when some stage has its full diameter at the face (the hole guides the bit)
    wide_b: wp.float64           # deepest bottom of the stages at least this wide (-1e30: none)
    deep_b: wp.float64           # deepest bottom of all stages


@wp.func
def crossing(lt: wp.vec3, la: wp.vec3) -> wp.vec2:
    """Where the bit's axis crosses the face plane (face frame); the tip's footprint if the axis is too shallow."""
    if la[2] < -MIN_CROSS:
        p = lt - (lt[2] / la[2]) * la
        return wp.vec2(p[0], p[1])
    return wp.vec2(lt[0], lt[1])


@wp.func
def widest(stage_d: wp.array2d[wp.float32], w: int, base: int, n: int) -> wp.float32:
    r = wp.float32(0.0)
    for k in range(n):
        r = wp.max(r, 0.5 * stage_d[w, base + k])
    return r


@wp.func
def nearest_hole(hole_entry: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
                 w: int, first: int, n: int, stages: int, c: wp.vec2, radius: wp.float32) -> wp.int32:
    """The layer's hole whose entry is nearest ``c`` within max(``radius``, its widest stage radius); -1: none."""
    best = int(-1)
    dist = wp.float32(1.0e30)
    for h in range(n):
        slot = first + h
        dx = c[0] - hole_entry[w, slot, 0]
        dy = c[1] - hole_entry[w, slot, 1]
        r = wp.sqrt(dx * dx + dy * dy)
        if r <= wp.max(radius, widest(stage_d, w, slot * stages, stage_n[w, slot])) and r < dist:
            dist = r
            best = h
    return best


@wp.func
def stage_of(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
             d: wp.float32, tan_half: wp.float64) -> StageInfo:
    st = StageInfo()
    st.stage = -1
    st.wide_b = wp.float64(-1.0e30)
    st.deep_b = wp.float64(-1.0e30)
    for k in range(n):
        dk = stage_d[w, base + k]
        bk = stage_b[w, base + k]
        if wp.abs(dk - d) <= SAME_D:
            st.stage = k
        elif dk > d:
            st.wide_b = wp.max(st.wide_b, bk)
        if bk >= wp.float64(0.5 * dk) / tan_half:
            st.formed = 1
        st.deep_b = wp.max(st.deep_b, bk)
    return st


@wp.func
def hole_radius(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
                u: wp.float32, t: wp.float32) -> wp.float32:
    """The hole's radius at axial position ``u``: the union of its stages' drill-point solids."""
    r = wp.float32(0.0)
    for k in range(n):
        rk = 0.5 * stage_d[w, base + k]
        r = wp.max(r, wp.clamp((wp.float32(stage_b[w, base + k]) - u) * t, 0.0, rk))
    return r


@wp.func
def landing(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
            r: wp.float32, t: wp.float32) -> wp.float32:
    """The deepest axial position where the hole is at least ``r`` wide: where an end of that radius lands (on the
    face's rim, 0, if no stage is that wide; on a stepped hole's shoulder, inside a wider stage's point cone)."""
    u = wp.float32(0.0)
    for k in range(n):
        if 0.5 * stage_d[w, base + k] >= r:
            u = wp.max(u, wp.float32(stage_b[w, base + k]) - r / t)
    return u


@wp.func
def open_hole(hole: int, stage: int, cross: wp.vec2, axis: wp.vec3, bottom: wp.float64, w: int, i: int,
              d: wp.float32, holes: int, stages: int, hole_n: wp.array2d[wp.int32], hole_entry: wp.array3d[wp.float32],
              hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
              stage_b: wp.array2d[wp.float64]) -> wp.vec2i:
    """(hole, stage) a drill cuts in, adding a hole (``hole`` < 0: at ``cross`` along ``axis``) and a stage
    (``stage`` < 0: diameter ``d`` from ``bottom``) where needed."""
    h = hole
    if h < 0:
        h = hole_n[w, i]
        slot = i * holes + h
        hole_n[w, i] = h + 1
        hole_entry[w, slot, 0] = cross[0]
        hole_entry[w, slot, 1] = cross[1]
        for c in range(3):
            hole_axis[w, slot, c] = axis[c]
        stage_n[w, slot] = 0
    slot = i * holes + h
    k = stage
    if k < 0:
        k = stage_n[w, slot]
        stage_d[w, slot * stages + k] = d
        stage_b[w, slot * stages + k] = bottom
        stage_n[w, slot] = k + 1
    return wp.vec2i(h, k)


__all__ = ["MIN_CROSS", "SAME_D", "StageInfo", "crossing", "hole_radius", "landing", "nearest_hole", "open_hole",
           "stage_of", "widest"]
