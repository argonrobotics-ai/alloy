# SPDX-License-Identifier: Apache-2.0
"""Wedge anchors (drilling v2, phase C): a listed pin with ``anchor`` > 0 is an expansion anchor set in the stack's
last layer (the wall).

- **Embed**: the depth of the anchor's tip below the last layer's face (``pin_embed``), from the hole geometry: an
  anchor pushed into a blind hole lands on its point cone (``pin_kernels``), so the embed is the hole's.
- **Expansion**: it engages (``pin_anchored``) once the anchor's tip is in a hole of the last layer at least
  ``anchor`` deep and no wider than ``anchor_hole`` (the maker's bit tolerance) and its nut is torqued (per-world
  input ``nut_torque`` [W, P], N m). From then on the clip holds the tip at the deepest it has reached against
  pull-out (a one-sided spring along the hole axis; driven deeper, it follows), until the shank leaves the layer.
- **Clamp load**: the nut torque T over the nut factor, F = T / (K d) with d the shank's diameter (``Drill.read()``
  ``anchor_clamp``). K defaults to 0.2: NASA RP-1228, Fastener Design Manual (1990), p. 21: "The commonly assumed
  value for K is 0.2", with its Table VI's K = 0.189 to 0.250 at thread and head friction 0.15 to 0.20. The load is
  a readout: the anchor holds the clamped parts with the pin springs, not with F (a 27 kN preload on 2e5 N/m springs
  would sink the head 13 cm; the known gap says so). The clip has no pull-out capacity of its own: its hold is the
  pin spring, so a hard pull slides it out, whatever the concrete's strength.

One thread per world and anchor pin.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.holes.pin_geom import Holes, PinSpec, contact_of, push, relative_velocity, spring


@wp.func
def tip_axial(ct_tip: wp.vec3, w: int, slot: int, hole_entry: wp.array3d[wp.float32],
              hole_axis: wp.array3d[wp.float32]) -> wp.float32:
    """The tip's axial position along hole ``slot``'s axis from its entry (face frame)."""
    a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
    d = ct_tip - wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
    return wp.dot(d, a)


@wp.kernel
def anchor_update_kernel(q_out: wp.array[wp.transform], body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32],
                         pins: wp.array[PinSpec], face_body: wp.array2d[wp.int32],
                         face_local: wp.array2d[wp.transform], bit: Holes, thickness: wp.array[wp.float32],
                         half: wp.array[wp.vec2], layers: int, hole_entry: wp.array3d[wp.float32],
                         hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
                         stage_d: wp.array2d[wp.float32], pin_hole: wp.array2d[wp.int32],
                         nut_torque: wp.array2d[wp.float32], pin_embed: wp.array2d[wp.float32],
                         pin_anchored: wp.array2d[wp.int32], pin_anchor_u: wp.array2d[wp.float32]):
    """After the step: each anchor's embed in the last layer, and whether its expansion is engaged (and where)."""
    w, p = wp.tid()
    pin = pins[p]
    if pin.anchor <= 0.0:
        return
    base = layers - 1
    j = p * layers + base
    ct = contact_of(q_out, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers)
    h = pin_hole[w, j] - 1
    pin_embed[w, p] = wp.where(h >= 0, wp.max(-ct.tip[2], 0.0), 0.0)
    if h < 0:
        pin_anchored[w, p] = 0                                    # the shank left the wall
        return
    slot = base * bit.holes + h
    u = tip_axial(ct.tip, w, slot, hole_entry, hole_axis)
    widest = wp.float32(0.0)
    for k in range(stage_n[w, slot]):
        widest = wp.max(widest, stage_d[w, slot * bit.stages + k])
    grips = widest <= pin.anchor_hole                             # a hole too wide: the clip never bites
    if pin_anchored[w, p] == 0 and grips and pin_embed[w, p] >= pin.anchor and nut_torque[w, p] > 0.0:
        pin_anchored[w, p] = 1
        pin_anchor_u[w, p] = u
    elif pin_anchored[w, p] != 0:
        pin_anchor_u[w, p] = wp.max(pin_anchor_u[w, p], u)        # the clip slides in, never back


@wp.kernel
def anchor_force_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                        body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32], pins: wp.array[PinSpec],
                        face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], bit: Holes,
                        thickness: wp.array[wp.float32], half: wp.array[wp.vec2], layers: int, dt: wp.float32,
                        hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32],
                        pin_hole: wp.array2d[wp.int32], pin_anchored: wp.array2d[wp.int32],
                        pin_anchor_u: wp.array2d[wp.float32], body_f: wp.array[wp.spatial_vector]):
    """ADD the expansion clip's hold: an engaged anchor pulled back past where it engaged is pushed back in."""
    w, p = wp.tid()
    pin = pins[p]
    if pin.anchor <= 0.0 or pin_anchored[w, p] == 0:
        return
    base = layers - 1
    j = p * layers + base
    h = pin_hole[w, j] - 1
    if h < 0:
        return
    ct = contact_of(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers)
    slot = base * bit.holes + h
    u = tip_axial(ct.tip, w, slot, hole_entry, hole_axis)
    a = wp.transform_vector(ct.face, wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2]))
    tip_w = wp.transform_point(ct.face, ct.tip)
    push(body_f, ct, tip_w, spring(a, pin_anchor_u[w, p] - u, relative_velocity(ct, body_qd, tip_w), pin, dt))


__all__ = ["anchor_force_kernel", "anchor_update_kernel"]
