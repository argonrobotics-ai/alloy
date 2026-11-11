# SPDX-License-Identifier: Apache-2.0
"""Pin-through-hole contact (drilling v2, phase B): bolts and anchors listed as pins pass only through holes the
drill has cut.

A pin is a body with a shank (a segment from ``top``, just under its head, to ``tip``, radius r) and, optionally, a
head (a disk of radius r_h whose underside is at ``top``); its end has a 45 deg chamfer of width c (0: square). The
scene gives the pin no solver contact with the drilled parts; the drill plug-in owns it. Per pin and layer:

- **Wall**: where the shank is inside one of the layer's holes, a spring across the hole axis at each face the
  shank crosses and at its tip, by the penetration |offset| + r - rho(u) (r - c at the tip). rho(u) is the hole's
  radius at axial position u (the union of its stages; it never widens with depth).
- **Landing**: the deepest tip position u* at which the end fits the hole at its offset (``fit``): its flat end
  (reach |offset| + r - c), its chamfer and its full shank against the narrowing below. The narrowing is a hole's
  mouth (narrower than the shank, or off its axis by more than the clearance), the shoulder of a stepped hole (a
  stage's point cone over a narrower stage) or a blind hole's point cone. A tip that crosses u* lands there (kept as
  ``pin_land_u``) and rests on it at any depth, along the hole axis where the flat end binds and at 45 deg (toward the
  axis and back) where the chamfer binds, so a chamfer guides an offset pin in; the rest position is the deeper of
  the landed one and u* now. The landing ends when the tip is above it (it backed out, or moved where it fits). A tip
  that crosses a face where there is no hole lands on the face. So a narrower hole, a step, a blind bottom or two
  parts' holes that no single axis fits stop the pin at any push. The walls of a landed
  pin see it at its rest position (not the landing spring's own penetration).
- **Head**: a head's flat underside lands like the end: where it crosses the deepest position its disk fits in the
  pin's hole (the mouth when the head is wider than it, a counterbore's floor when narrower), or the face where there
  is no hole; it rests there at any depth until it lifts off, on four rim springs (it resists tilt as a disk).
- **Off the plate**: landings, seats and membership end when the pin leaves the layer's patch (a tip landing once
  its whole end is off). A tip that comes back over the plate under its face lands again (on the face, or at its
  hole's narrowing), and a shank already under the face joins a hole only where it fits (wall overlap at most 0.1
  mm), so no wall spring switches on deep.
- **Rigid walls**: a landing judges the fit with the shank moved back out of the other layers' walls it presses,
  along the line through its corrected face crossings there (``legal_shift``: a tilt is undone too), so a chamfer
  pressing a misaligned pin into one hole's wall springs cannot wedge it through another.
- **Membership**: the hole a pin is in is kept per pin and layer (``pin_hole``) while its shank is in the layer, so
  a side load past the wall grows the wall spring instead of losing the hole. A shank found crossing a face in solid
  material without having landed (embedded) sets ``pin_solid``.

Every spring acts only while it penetrates, is damped (Kelvin-Voigt time ``damping``) and carries regularized
Coulomb friction (``friction``, viscous below a slip speed that keeps the explicit force stable). Reactions go to the
layers that are bodies. Landings and seats are state, set after each step from the step's start and end poses
(``pin_member_kernel``); the forces read only state. One thread per world.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.holes.pin_geom import (Contact, Holes, PinSpec, TipGeo, contact_of, fit, head_geo, hole_at, over,
                                          over_by, push, relative_velocity, sample_point, spring, tip_geo, tip_geo_at)

LAND_NONE = wp.constant(0)
LAND_EDGE = wp.constant(1)      # resting on a mouth, shoulder or blind bottom of the pin's hole in this layer
LAND_FACE = wp.constant(2)      # resting on the layer's face (no hole there)
SIN45 = wp.constant(0.70710678)
REST_GAP = wp.constant(1.0e-5)  # m: a landed shank point this little above its end's rest is not in the hole
JOIN = wp.constant(1.0e-4)      # m: the most wall overlap a shank already under the face may join a hole with
SEAT_FACE = wp.constant(1)      # pin_seat: the head on the face (no hole under it); SEAT_HOLE + h: in hole h
SEAT_HOLE = wp.constant(2)


@wp.func
def rest_force(g: TipGeo, rest: wp.float32, ct: Contact, pin: PinSpec, v: wp.vec3, dt: wp.float32) -> wp.vec3:
    """A landed tip past its rest position: back along the axis where the flat end binds; at 45 deg toward the axis
    and back where the chamfer binds (along the axis if the tip is centred)."""
    out = wp.transform_vector(ct.face, g.out)
    if g.bind <= 0.0:
        return spring(out, g.u - rest, v, pin, dt)
    n_dir = wp.normalize(wp.transform_vector(ct.face, g.inward) + out)
    return spring(n_dir, SIN45 * (g.u - rest), v, pin, dt)


@wp.func
def wall_force(g: TipGeo, s: wp.float32, ct: Contact, pin: PinSpec, v: wp.vec3, dt: wp.float32) -> wp.vec3:
    """The wall across the axis at a shank point ``s`` above the tip (inside the chamfer the shank is narrower)."""
    pen = g.pen - wp.max(pin.chamfer - wp.max(s, 0.0), 0.0)
    return spring(wp.transform_vector(ct.face, g.inward), pen, v, pin, dt)


@wp.func
def legal_shift(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32],
                face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], pin: PinSpec,
                thickness: wp.array[wp.float32], half: wp.array[wp.vec2], w: int, j: int, layers: int, bit: Holes,
                hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
                stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], pin_hole: wp.array2d[wp.int32],
                inv: wp.transform, q: wp.vec3) -> wp.vec3:
    """How far shank point ``q`` (world) would move (face frame of pin-layer ``j``) if the other layers' walls it
    presses were rigid. In each other layer the shank's face crossings move back out of the wall they overlap, toward
    the hole axis; the rigid shank is the line through them, so a tilt is undone too (one crossing: a translation). A
    landing judges the fit there, so springy walls cannot let a misaligned or tilted pin wedge through."""
    p = j // layers
    total = wp.vec3(0.0, 0.0, 0.0)
    for k in range(layers):
        jk = p * layers + k
        m = pin_hole[w, jk]
        if jk == j or m == 0:
            continue
        ck = contact_of(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, w, jk, layers)
        n = int(0)
        p0 = wp.vec3(0.0, 0.0, 0.0)
        p1 = wp.vec3(0.0, 0.0, 0.0)
        d0 = wp.vec3(0.0, 0.0, 0.0)
        d1 = wp.vec3(0.0, 0.0, 0.0)
        for c in range(2):
            sp = sample_point(ck, c)
            if sp[3] == 0.0:
                continue
            qc = wp.vec3(sp[0], sp[1], sp[2])
            g = tip_geo(qc, m - 1, pin, w, k, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)
            back = wp.max(g.pen, 0.0) * wp.transform_vector(ck.face, g.inward)
            if n == 0:
                p0 = wp.transform_point(ck.face, qc)
                d0 = back
            else:
                p1 = wp.transform_point(ck.face, qc)
                d1 = back
            n = n + 1
        if n == 1:
            total = total + d0
        elif n == 2:
            seg = p1 - p0
            sq = wp.max(wp.dot(seg, seg), 1.0e-12)
            total = total + d0 + (wp.dot(q - p0, seg) / sq) * (d1 - d0)
    return wp.transform_vector(inv, total)


@wp.struct
class Shifts:
    tip: wp.vec3                 # the tip's rigid-wall shift (face frame)
    face: wp.vec3                # the shank's face crossing's


@wp.func
def legal_shifts(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32],
                 face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], pin: PinSpec,
                 thickness: wp.array[wp.float32], half: wp.array[wp.vec2], w: int, j: int, layers: int, bit: Holes,
                 hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
                 stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], pin_hole: wp.array2d[wp.int32],
                 ct: Contact) -> Shifts:
    """``legal_shift`` at the tip and at the shank's crossing of this layer's face."""
    out = Shifts()
    q = wp.transform_point(ct.face, ct.tip)
    out.tip = legal_shift(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers, bit,
                          hole_entry, hole_axis, stage_n, stage_d, stage_b, pin_hole, ct.inv, q)
    out.face = out.tip
    sp = sample_point(ct, 0)
    if sp[3] != 0.0:
        q = wp.transform_point(ct.face, wp.vec3(sp[0], sp[1], sp[2]))
        out.face = legal_shift(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers,
                               bit, hole_entry, hole_axis, stage_n, stage_d, stage_b, pin_hole, ct.inv, q)
    return out


@wp.func
def end_geo(ct: Contact, h: int, pin: PinSpec, w: int, i: int, bit: Holes, hole_entry: wp.array3d[wp.float32],
            hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
            stage_b: wp.array2d[wp.float64], shift: wp.vec3, face_shift: wp.vec3) -> TipGeo:
    """The tip against hole ``h`` at its ``shift``-ed position, its fit judged at the larger of the tip's and the
    shank's (``face_shift``-ed) face-crossing offsets (a tilted shank binds where it is farthest off the axis)."""
    g = tip_geo_at(ct.tip, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b, shift)
    sp = sample_point(ct, 0)
    if sp[3] != 0.0:
        g0 = tip_geo_at(wp.vec3(sp[0], sp[1], sp[2]), h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d,
                        stage_b, face_shift)
        if g0.off > g.off:
            slot = i * bit.holes + h
            us = fit(stage_d, stage_b, w, slot * bit.stages, stage_n[w, slot], g0.off + pin.radius, pin.chamfer,
                     bit.tan_half)
            g.rest = us[0]
            g.bind = us[1]
    return g


@wp.func
def seat_force(ct: Contact, q: wp.vec3, seat: int, seat_u: wp.float32, pin: PinSpec, w: int, i: int, bit: Holes,
               hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
               stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], v: wp.vec3,
               dt: wp.float32) -> wp.vec3:
    """One of the head's rim springs (a quarter of the pin's stiffness) at face-frame point ``q`` of its underside: on
    the face along its normal, or in a hole along its axis at the deeper of where it seated and where its disk fits
    now (a counterbore's floor, or the mouth of a hole narrower than the head)."""
    if seat == SEAT_FACE:
        return 0.25 * spring(ct.n, -q[2], v, pin, dt)
    h = seat - SEAT_HOLE
    hg = head_geo(ct.top, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)
    slot = i * bit.holes + h
    a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
    u = hg[0] + wp.dot(q - ct.top, a)
    return 0.25 * spring(-wp.transform_vector(ct.face, a), u - wp.max(seat_u, hg[1]), v, pin, dt)


@wp.func
def push_seat(body_f: wp.array[wp.spatial_vector], body_qd: wp.array[wp.spatial_vector], ct: Contact, seat: int,
              seat_u: wp.float32, pin: PinSpec, w: int, i: int, bit: Holes, hole_entry: wp.array3d[wp.float32],
              hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
              stage_b: wp.array2d[wp.float64], dt: wp.float32):
    """The seated head as a disk: four rim springs, so it resists tilt (k r_h^2 / 2 per axis) as well as the push."""
    ax = wp.normalize(ct.top - ct.tip)
    e1 = wp.cross(ax, wp.vec3(1.0, 0.0, 0.0))
    if wp.length(e1) < 0.1:
        e1 = wp.cross(ax, wp.vec3(0.0, 1.0, 0.0))
    e1 = wp.normalize(e1)
    e2 = wp.cross(ax, e1)
    for c in range(4):
        sgn = wp.where(c % 2 == 0, 1.0, -1.0)
        q = ct.top + pin.head * sgn * wp.where(c < 2, e1, e2)
        p = wp.transform_point(ct.face, q)
        push(body_f, ct, p, seat_force(ct, q, seat, seat_u, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d,
                                       stage_b, relative_velocity(ct, body_qd, p), dt))


@wp.kernel
def pin_force_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                     body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32], pins: wp.array[PinSpec],
                     n_pins: int, face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], bit: Holes,
                     thickness: wp.array[wp.float32], half: wp.array[wp.vec2], layers: int, dt: wp.float32,
                     hole_n: wp.array2d[wp.int32], hole_entry: wp.array3d[wp.float32],
                     hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
                     stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64],
                     pin_hole: wp.array2d[wp.int32], pin_land: wp.array2d[wp.int32],
                     pin_land_u: wp.array2d[wp.float32], pin_seat: wp.array2d[wp.int32],
                     pin_seat_u: wp.array2d[wp.float32], body_f: wp.array[wp.spatial_vector]):
    """ADD the walls', landings', faces' and heads' contact forces on every pin (reactions on layers that are
    bodies)."""
    w = wp.tid()
    for j in range(n_pins * layers):
        pin = pins[j // layers]
        i = j % layers
        ct = contact_of(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers)
        member = pin_hole[w, j]
        land = pin_land[w, j]
        tip_w = wp.transform_point(ct.face, ct.tip)
        v_tip = relative_velocity(ct, body_qd, tip_w)
        sunk = wp.float32(0.0)                                    # the landing spring's own penetration
        if land == LAND_FACE:
            push(body_f, ct, tip_w, spring(ct.n, -ct.tip[2], v_tip, pin, dt))
        elif land == LAND_EDGE and member > 0:
            sh = legal_shifts(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers,
                              bit, hole_entry, hole_axis, stage_n, stage_d, stage_b, pin_hole, ct)
            g = end_geo(ct, member - 1, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b, sh.tip,
                        sh.face)
            rest = wp.max(pin_land_u[w, j], g.rest)
            sunk = g.u - rest
            push(body_f, ct, tip_w, rest_force(g, rest, ct, pin, v_tip, dt))
        for c in range(3):
            sp = sample_point(ct, c)
            if sp[3] == 0.0 or land == LAND_FACE or (c == 2 and land != LAND_NONE):
                continue                                          # a landed tip rests (on a face: nothing is in)
            q = wp.vec3(sp[0], sp[1], sp[2])
            s = wp.length(ct.tip - q) - sunk                      # height above the tip at rest
            if land != LAND_NONE and s < REST_GAP:
                continue                                          # at or below the landed end: not in the hole
            p = wp.transform_point(ct.face, q)
            v = relative_velocity(ct, body_qd, p)
            h = member - 1
            if member == 0:
                h = hole_at(q, w, i, bit, hole_n, hole_entry, hole_axis, stage_n, stage_d, stage_b)
            f = wp.vec3(0.0, 0.0, 0.0)
            if h >= 0:
                g = tip_geo(q, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)
                if member > 0 or g.pen - wp.where(c == 2, pin.chamfer, 0.0) <= JOIN:
                    f = wall_force(g, s, ct, pin, v, dt)                  # not yet in it: never switched on deep
            elif c == 2:                                          # in solid without having landed (seeded)
                f = spring(ct.n, -ct.tip[2], v, pin, dt)
            push(body_f, ct, p, f)
        if pin_seat[w, j] != 0:
            push_seat(body_f, body_qd, ct, pin_seat[w, j], pin_seat_u[w, j], pin, w, i, bit, hole_entry, hole_axis,
                      stage_n, stage_d, stage_b, dt)


@wp.struct
class Landing:
    kind: wp.int32               # LAND_*
    u: wp.float32                # where an edge landing rests (axial)


@wp.func
def landed_at(kind: int, u: wp.float32) -> Landing:
    out = Landing()
    out.kind = kind
    out.u = u
    return out


@wp.func
def landing_of(before: Contact, now: Contact, h: int, was: Landing, pin: PinSpec, w: int, i: int, bit: Holes,
               hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
               stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], shift: Shifts) -> Landing:
    """The tip's landing after the step: kept on a face while below it (moved over a hole, it rests on that hole's
    narrowing if it does not fit there), kept on an edge while past its rest; a free tip that entered the layer during
    the step lands where it crossed a face with no hole under it, or u*. A free tip that comes back over the plate
    below its face (from beside it) lands again: on the face, or at its hole's narrowing. It is let go once its whole
    end is off the plate. The fit is judged with the shank at its ``shift``-ed (rigid-wall) position."""
    end = pin.radius - pin.chamfer                                # the flat end's radius
    if (was.kind == LAND_FACE and now.tip[2] >= 0.0) or not over_by(now, now.tip, end):
        return landed_at(LAND_NONE, 0.0)                          # backed off the face, or slid off the plate
    entered = now.tip[2] < 0.0 and before.tip[2] > -now.t        # in the layer during the step
    back = was.kind == LAND_NONE and entered and not over_by(before, before.tip, end)   # came back from beside it
    if h < 0:
        if was.kind == LAND_NONE and ((before.tip[2] >= 0.0 and entered) or back):
            return landed_at(LAND_FACE, 0.0)                      # crossed the face where there is no hole
        return landed_at(wp.where(was.kind == LAND_FACE, LAND_FACE, LAND_NONE), 0.0)   # an edge: left the layer
    g = end_geo(now, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b, shift.tip, shift.face)
    if was.kind == LAND_EDGE:
        if g.u > wp.max(was.u, g.rest):
            return was                                            # still past its rest
        return landed_at(LAND_NONE, 0.0)                          # backed out, or fits
    if was.kind == LAND_FACE or back or (entered and tip_geo(before.tip, h, pin, w, i, bit, hole_entry, hole_axis,
                                                             stage_n, stage_d, stage_b).u <= g.rest):
        if g.u > g.rest:
            return landed_at(LAND_EDGE, g.rest)                   # crossed u* (or slid over a hole it does not fit)
    return landed_at(LAND_NONE, 0.0)


@wp.func
def seat_of(before: Contact, now: Contact, h: int, seat: int, seat_u: wp.float32, pin: PinSpec, w: int, i: int,
            bit: Holes, hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32],
            stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64]) -> wp.vec2:
    """(seat, seat_u) after the step: a head seats where its underside crosses the deepest position its disk fits in
    the pin's hole (the mouth when the head is wider), or the face where there is no hole; it lifts off above it."""
    if seat != 0 and not over(now, now.top):
        return wp.vec2(0.0, 0.0)                                  # slid off the plate
    if seat == SEAT_FACE:
        return wp.vec2(wp.where(now.top[2] >= 0.0, 0.0, 1.0), 0.0)
    if seat >= SEAT_HOLE:
        hg = head_geo(now.top, seat - SEAT_HOLE, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)
        if hg[0] <= wp.max(seat_u, hg[1]):
            return wp.vec2(0.0, 0.0)                              # lifted off
        return wp.vec2(wp.float32(seat), seat_u)
    if pin.head <= 0.0:
        return wp.vec2(0.0, 0.0)
    if h >= 0:
        hg = head_geo(now.top, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)
        u0 = head_geo(before.top, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)[0]
        if u0 <= hg[1] and hg[0] > hg[1]:
            return wp.vec2(wp.float32(SEAT_HOLE + h), hg[1])
    elif before.top[2] >= 0.0 and now.top[2] < 0.0 and over(now, now.top):
        return wp.vec2(wp.float32(SEAT_FACE), 0.0)
    return wp.vec2(0.0, 0.0)


@wp.kernel
def pin_member_kernel(q_in: wp.array[wp.transform], q_out: wp.array[wp.transform], body_com: wp.array[wp.vec3],
                      pin_body: wp.array2d[wp.int32], pins: wp.array[PinSpec], n_pins: int,
                      face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], bit: Holes,
                      thickness: wp.array[wp.float32], half: wp.array[wp.vec2], layers: int,
                      hole_n: wp.array2d[wp.int32], hole_entry: wp.array3d[wp.float32],
                      hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
                      stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64],
                      pin_hole: wp.array2d[wp.int32], pin_land: wp.array2d[wp.int32],
                      pin_land_u: wp.array2d[wp.float32], pin_seat: wp.array2d[wp.int32],
                      pin_seat_u: wp.array2d[wp.float32], pin_solid: wp.array[wp.int32]):
    """After the step: each pin's hole per layer (kept while its shank is in the layer), its tip's landing, its
    head's seat, and the embedded flag."""
    w = wp.tid()
    for j in range(n_pins * layers):
        pin = pins[j // layers]
        i = j % layers
        now = contact_of(q_out, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers)
        before = contact_of(q_in, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers)
        inside = wp.max(now.top[2], now.tip[2]) > -now.t and wp.min(now.top[2], now.tip[2]) < 0.0
        inside = inside and (sample_point(now, 0)[3] != 0.0 or sample_point(now, 1)[3] != 0.0 or
                             sample_point(now, 2)[3] != 0.0)              # and over the plate
        if not inside:
            pin_hole[w, j] = 0
        elif pin_hole[w, j] == 0:
            sp = sample_point(now, 2)
            at_tip = sp[3] != 0.0
            if not at_tip:
                sp = sample_point(now, 0)
            if sp[3] != 0.0:
                q = wp.vec3(sp[0], sp[1], sp[2])
                hc = hole_at(q, w, i, bit, hole_n, hole_entry, hole_axis, stage_n, stage_d, stage_b)
                if hc >= 0 and before.tip[2] < 0.0:              # already under the face: joins only where it fits
                    pen = tip_geo(q, hc, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b).pen
                    if pen - wp.where(at_tip, pin.chamfer, 0.0) > JOIN:
                        hc = -1
                pin_hole[w, j] = hc + 1
        h = pin_hole[w, j] - 1
        was = landed_at(pin_land[w, j], pin_land_u[w, j])
        if h >= 0 and was.kind == LAND_EDGE and was.u <= pin.chamfer + JOIN and sample_point(now, 2)[3] != 0.0:
            g0 = tip_geo(now.tip, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b)
            if g0.pen >= 2.0 * pin.radius - pin.chamfer:          # slid on from a mouth it rested on: on the face
                pin_hole[w, j] = 0
                h = -1
                was = landed_at(LAND_FACE, 0.0)
        shift = legal_shifts(q_out, body_com, pin_body, face_body, face_local, pin, thickness, half, w, j, layers, bit,
                             hole_entry, hole_axis, stage_n, stage_d, stage_b, pin_hole, now)
        land = landing_of(before, now, h, was, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b,
                          shift)
        pin_land[w, j] = land.kind
        pin_land_u[w, j] = land.u
        seat = seat_of(before, now, h, pin_seat[w, j], pin_seat_u[w, j], pin, w, i, bit, hole_entry, hole_axis,
                       stage_n, stage_d, stage_b)
        pin_seat[w, j] = int(seat[0])
        pin_seat_u[w, j] = seat[1]
        embedded = pin_land[w, j] == LAND_NONE and h < 0 and sample_point(now, 0)[3] != 0.0
        if embedded and -now.tip[2] > pin.chamfer + 1.0e-3 and pin_solid[w] == 0:
            wp.printf("pin %d in world %d crosses layer %d's face in solid material (embedded)\n", j // layers, w, i)
            pin_solid[w] = 1


__all__ = ["Contact", "Holes", "LAND_EDGE", "LAND_FACE", "LAND_NONE", "Landing", "PinSpec", "SEAT_FACE", "SEAT_HOLE",
           "TipGeo", "contact_of", "hole_at", "legal_shift", "pin_force_kernel", "pin_member_kernel", "tip_geo"]
