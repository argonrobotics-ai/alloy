# SPDX-License-Identifier: Apache-2.0
"""Box-box face contacts reduced to 4 points, for the opt-in fast rigid mode only (SPEED-S1; WP7's mode, WP3's part).

MuJoCo's ``mjc_BoxBox`` (MuJoCo Warp's ``box_box``, on the primitive collider since BUL-0011) clips the incident face
against the reference face and keeps every vertex of the overlap polygon: up to 8 contacts per face pair, as for a
yawed square resting on a square (an octagon). MuJoCo Warp's convex path, which stock MuJoCo Warp and Newton run,
emits 4. On WP7's 3,072-world box stack (L4, 10 ms) Alloy has 175 contacts per world against stock's 108, and the
contact-sized kernels cost about 3.5 ms of a 19.3 ms period.

``box_box_reduce_wrapper`` is ``mjw_boxbox_line.box_box_line_wrapper`` (MuJoCo's contacts plus the BUL-0014 line
rule) with one change. A pair with more than ``KEEP`` contacts writes the ``KEEP`` whose convex hull is largest
(``pick4``: every 4-subset of at most 8 points). Each keeps its own depth and position, and all keep the pair's normal.
- A tilted pair keeps its deepest contact: its depths span more than ``DEPTH_TIE``.
- A level pair's depths differ by noise, so its deepest point is not forced in. On a small yaw that point is a
  mid-edge vertex, and a greedy pick starting from it lost up to two thirds of a corner's support.
- Four points cannot span an octagon. Over the overlap of two equal squares, the 4 keep 98 % of its area and support
  at 2 degrees of yaw, 85 % at 20 and 71 % at 45: a regular octagon's alternate vertices. Support along a direction
  sets the box's tipping onset over that edge. Stock MuJoCo Warp's convex path also emits 4.
- Pairs with 4 or fewer contacts are unchanged: an aligned face pair already gives 4.

Scope: only an entry whose route sets ``REDUCE_KEY`` (``entries.mujoco_overrides`` under ``RIGID_FAST_EFFECT``, the
fast rigid mode). ``mjw_boxbox_line.enable_line_boxbox(solver, reduce4=True)`` puts this wrapper in MuJoCo Warp's
primitive table around that solver's steps instead of the line wrapper. An entry with line contacts off, or on the
MuJoCo C route (``mjc_BoxBox``), keeps MuJoCo's points. MuJoCo's source is not changed.
"""

from __future__ import annotations

import warp as wp
from alloy._vendor.mujoco_warp._src.collision_core import Geom, write_contact
from alloy._vendor.mujoco_warp._src.collision_primitive_core import box_box, mat83f, vec8f
from alloy._vendor.mujoco_warp._src.math import make_frame
from alloy._vendor.mujoco_warp._src.types import vec5

from alloy.contact.mjw_boxbox_line import box_box_line

REDUCE_KEY = "alloy:boxbox_reduce4"   # MuJoCo entry kwarg: True reduces box-box face pairs to KEEP contacts
KEEP = 4
DEPTH_TIE = 1.0e-5   # [m] a face pair whose depths span more than this is tilted: its deepest contact is kept
_DEPTH_TIE = wp.constant(DEPTH_TIE)


@wp.func
def _area2(a: wp.vec3, b: wp.vec3, c: wp.vec3, n: wp.vec3) -> float:
    """Twice the signed area of triangle abc seen along n."""
    return wp.dot(wp.cross(b - a, c - a), n)


@wp.func
def _hull4(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, n: wp.vec3) -> float:
    """Twice the area of the convex hull of four coplanar points (the largest of the three quad orders and the four
    triangles: a quad order through the hull is the hull, a point inside leaves a triangle)."""
    q1 = wp.abs(_area2(a, b, c, n) + _area2(a, c, d, n))
    q2 = wp.abs(_area2(a, b, d, n) + _area2(a, d, c, n))
    q3 = wp.abs(_area2(a, c, b, n) + _area2(a, b, d, n))
    t = wp.max(wp.max(wp.abs(_area2(a, b, c, n)), wp.abs(_area2(a, b, d, n))),
               wp.max(wp.abs(_area2(a, c, d, n)), wp.abs(_area2(b, c, d, n))))
    return wp.max(wp.max(q1, q2), wp.max(q3, t))


@wp.func
def pick4(ncon: int, dist: vec8f, pos: mat83f, n: wp.vec3) -> wp.vec4i:
    """The four of ``ncon`` (5-8) contacts whose convex hull is largest, over all 4-subsets (at most 70). When the
    pair is not level, the deepest one is among them: its depth is more than ``DEPTH_TIE`` below the shallowest's."""
    i0 = int(0)
    top = dist[0]
    for i in range(1, ncon):
        if dist[i] < dist[i0]:
            i0 = i
        top = wp.max(top, dist[i])
    force = dist[i0] < top - _DEPTH_TIE
    best = float(-1.0)
    keep = wp.vec4i(0, 1, 2, 3)
    for a in range(ncon):
        for b in range(a + 1, ncon):
            for c in range(b + 1, ncon):
                for d in range(c + 1, ncon):
                    if force and a != i0 and b != i0 and c != i0 and d != i0:
                        continue
                    h = _hull4(pos[a], pos[b], pos[c], pos[d], n)
                    if h > best:
                        best = h
                        keep = wp.vec4i(a, b, c, d)
    return keep


@wp.func
def box_box_reduce_wrapper(
    # Data in:
    naconmax_in: int,
    # In:
    box1: Geom,
    box2: Geom,
    worldid: int,
    margin: float,
    gap: float,
    condim: int,
    friction: vec5,
    solref: wp.vec2,
    solreffriction: wp.vec2,
    solimp: vec5,
    adhesion: float,
    geoms: wp.vec2i,
    pairid: wp.vec2i,
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    contact_adhesion_out: wp.array[float],
    nacon_out: wp.array[int],
):
    """``box_box_line_wrapper`` (same arguments, same writer) with a face pair's contacts reduced to ``KEEP``."""
    ncon, dist, pos, normal = box_box(box1.pos, box1.rot, box1.size, box2.pos, box2.rot, box2.size, margin + gap)
    if ncon == 0:
        return
    if ncon == 1:
        n_line, d0, d1, p0, p1, n = box_box_line(box1.pos, box1.rot, box1.size, box2.pos, box2.rot, box2.size,
                                                 normal, margin + gap)
        if n_line == 2:
            ncon = 2
            dist[0] = d0
            dist[1] = d1
            pos[0] = p0
            pos[1] = p1
            normal = n
    frame = make_frame(normal)
    if ncon > KEEP:
        keep = pick4(ncon, dist, pos, normal)
        for k in range(KEEP):
            j = keep[k]
            write_contact(
                naconmax_in, k, dist[j], pos[j], frame, margin, gap, condim, friction, solref, solreffriction, solimp,
                adhesion, geoms, pairid, worldid, contact_dist_out, contact_pos_out, contact_frame_out,
                contact_includemargin_out, contact_friction_out, contact_solref_out, contact_solreffriction_out,
                contact_solimp_out, contact_dim_out, contact_geom_out, contact_efc_address_out, contact_worldid_out,
                contact_type_out, contact_geomcollisionid_out, contact_adhesion_out, nacon_out,
            )
        return
    for i in range(ncon):
        write_contact(
            naconmax_in, i, dist[i], pos[i], frame, margin, gap, condim, friction, solref, solreffriction, solimp,
            adhesion, geoms, pairid, worldid, contact_dist_out, contact_pos_out, contact_frame_out,
            contact_includemargin_out, contact_friction_out, contact_solref_out, contact_solreffriction_out,
            contact_solimp_out, contact_dim_out, contact_geom_out, contact_efc_address_out, contact_worldid_out,
            contact_type_out, contact_geomcollisionid_out, contact_adhesion_out, nacon_out,
        )


__all__ = ["DEPTH_TIE", "KEEP", "REDUCE_KEY", "box_box_reduce_wrapper", "pick4"]
