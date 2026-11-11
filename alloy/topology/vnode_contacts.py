# SPDX-License-Identifier: Apache-2.0
"""Blade contacts for virtual-node copies (WP10 item 9 step 3; used by ``vnode_cut.VNodeCutter.filter_contacts``).

- ``mute_copy_contacts``: copies collide only once they are free far-side copies (``vnode_split.weld_copies`` gives them
  a side). A welded copy's hub collides for it, and a copy without mass is not material. The record is moved ``FAR`` behind its body surface along its normal, so the solver sees no
  penetration.
- ``rewrite_hub_contacts``: a hub whose side is fixed (or a free far-side copy) meets its blade on that side's face,
  as ``BladeContact`` does for active twins. Under the ``work`` rule (``vnode_work``) a hub not yet cut keeps its stock contact and presses
  the knife, unless that contact would push it across the cut. Its normal becomes the signed face normal and the record's plane sits one particle radius inside
  the face, so the hub's centre, a point of the cut face, comes to rest on the blade face. With the radius as a
  collision skin, a 1 mm blade pushed cut-face material 1.5 mm aside per side, and the cut faces of tets split
  later sat that far off the swept surface.
  - Behind the edge (on the spine side), the face takes the hub at full depth, half - sgn d.
  - Ahead of it, the depth is capped at half + edge_radius - (distance to the edge), so the face eases the hub
    aside as the edge arrives (step 1's ramp).
  - A record with no allowed depth is muted under the ``lead`` rule: the hub's tet is already cut, and nothing in
    front of the edge holds. Under ``work`` it keeps the stock contact, as ``BladeContact`` does for twins: the knife
    pressed the hub before cutting it, and muted, a cut hub still under the edge sprang up at 1.3 m/s.
  With stock contacts, hubs near an oblique cut plane were driven down by the blade's bottom face (1 m/s). A
  full-depth face kicked them aside at 0.3 m/s; the ramp keeps them at 0.17 m/s or below.
"""

from __future__ import annotations

import warp as wp

from alloy.topology.blade_contact_kernels import segment_distance

FAR = wp.constant(1.0)     # [m] how far a muted contact record moves behind its body surface


@wp.struct
class CutterBlades:
    of_world: wp.array(dtype=wp.int32)        # [W] blade index, -1 = none
    body: wp.array(dtype=wp.int32)
    edge_a: wp.array(dtype=wp.vec3)           # body frame
    edge_b: wp.array(dtype=wp.vec3)
    axis: wp.array(dtype=wp.vec3)             # face normal, body frame
    center: wp.array(dtype=wp.vec3)           # a point on the mid-plane, body frame
    half: wp.array(dtype=wp.float32)          # half-thickness [m]
    edge_radius: wp.array(dtype=wp.float32)   # [m]
    spine: wp.array(dtype=wp.vec3)            # body frame, edge -> spine (zero: no spine, all behind)
    softness: wp.array(dtype=wp.float32)      # Blade.softness (``vnode_work``'s threshold)
    cut_ke: wp.array(dtype=wp.float32)        # Blade.cut_spring_ke (``vnode_work``'s bond while cutting)
    shape_blade: wp.array(dtype=wp.int32)     # [shape_count] blade of a shape on a blade body, -1 = none


@wp.kernel
def mute_copy_contacts(is_copy: wp.array(dtype=wp.int8), particle_side: wp.array(dtype=wp.int8),
                       body_q: wp.array(dtype=wp.transform),
                       shape_body: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32), max_contacts: int,
                       c_indices: wp.array(dtype=wp.vec3i), c_shape: wp.array(dtype=int),
                       c_body_pos: wp.array(dtype=wp.vec3), c_normal: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    if k >= wp.min(count[0], max_contacts):
        return
    idx = c_indices[k]
    hit = False
    for slot in range(3):
        if idx[slot] >= 0 and is_copy[idx[slot]] != wp.int8(0) and particle_side[idx[slot]] < wp.int8(0):
            hit = True
    if not hit or c_shape[k] < 0:
        return
    b = shape_body[c_shape[k]]
    nrm = c_normal[k]
    if b >= 0:
        nrm = wp.transform_vector(wp.transform_inverse(body_q[b]), nrm)
    c_body_pos[k] = c_body_pos[k] - FAR * nrm


@wp.func
def behind(pl: wp.vec3, ea: wp.vec3, eb: wp.vec3, spine: wp.vec3) -> bool:
    """Body frame: is pl level with or behind the edge along the spine?"""
    if wp.length_sq(spine) == 0.0:
        return True
    ab = eb - ea
    t = wp.clamp(wp.dot(pl - ea, ab) / wp.max(wp.dot(ab, ab), 1.0e-30), 0.0, 1.0)
    return wp.dot(pl - (ea + t * ab), spine) >= 0.0


@wp.kernel
def rewrite_hub_contacts(bl: CutterBlades, particle_side: wp.array(dtype=wp.int8),
                         released: wp.array(dtype=wp.int8), body_q: wp.array(dtype=wp.transform), x: wp.array(dtype=wp.vec3),
                         radius: wp.array(dtype=wp.float32), count: wp.array(dtype=wp.int32), max_contacts: int,
                         c_indices: wp.array(dtype=wp.vec3i), c_shape: wp.array(dtype=int),
                         c_body_pos: wp.array(dtype=wp.vec3), c_body_vel: wp.array(dtype=wp.vec3),
                         c_normal: wp.array(dtype=wp.vec3), keep_ahead: int):
    k = wp.tid()
    if k >= wp.min(count[0], max_contacts):
        return
    s = c_shape[k]
    idx = c_indices[k]
    if s < 0 or idx[0] < 0 or idx[1] >= 0:
        return
    j = bl.shape_blade[s]
    if j < 0 or particle_side[idx[0]] < wp.int8(0):
        return
    sgn = 1.0
    if particle_side[idx[0]] == wp.int8(1):
        sgn = -1.0
    xf = body_q[bl.body[j]]
    a = bl.axis[j]
    if released[idx[0]] == wp.int8(0) and wp.dot(c_normal[k], sgn * wp.transform_vector(xf, a)) >= 0.0:
        return                                   # uncut (work rule): keeps pressing, never across the cut
    pl = wp.transform_point(wp.transform_inverse(xf), x[idx[0]])
    d = wp.dot(pl - bl.center[j], a)
    r = radius[idx[0]]
    shift = r                                    # the hub's centre on the face: penetration = half - sgn d
    if not behind(pl, bl.edge_a[j], bl.edge_b[j], bl.spine[j]):
        allowed = bl.half[j] + bl.edge_radius[j] - segment_distance(pl, bl.edge_a[j], bl.edge_b[j])
        if allowed <= 0.0:
            if keep_ahead != 0:
                return                           # ``work``: still under the edge, it presses it (as BladeContact)
            c_body_pos[k] = c_body_pos[k] - FAR * wp.transform_vector(wp.transform_inverse(xf), c_normal[k])
            return
        shift = r + wp.max(0.0, (bl.half[j] - sgn * d) - allowed)
    c_body_pos[k] = pl - (d - sgn * (bl.half[j] - shift)) * a
    c_normal[k] = sgn * wp.transform_vector(xf, a)
    c_body_vel[k] = wp.vec3(0.0)
