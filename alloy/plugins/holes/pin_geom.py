# SPDX-License-Identifier: Apache-2.0
"""Geometry of the pin contact (drilling v2, phase B): the pin and hole specs, one pin in one layer, a shank point
against a hole (its wall overlap and where its end fits: ``fit``), and the damped contact spring
(``pin_kernels`` uses them).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.holes.match import hole_radius, landing
from alloy.plugins.removal.surface import patch_frame


@wp.struct
class Holes:
    holes: wp.int32              # hole slots per layer (the drill's max_holes)
    stages: wp.int32             # stage slots per hole
    tan_half: wp.float32         # tan of the drills' point half-angle


@wp.struct
class PinSpec:
    top: wp.vec3                 # shank top (under the head), pin body frame [m]
    tip: wp.vec3                 # shank tip, pin body frame [m]
    radius: wp.float32           # shank radius [m]
    head: wp.float32             # head radius [m]; 0: no head
    stiffness: wp.float32        # N/m per contact spring
    damping: wp.float32          # s: its Kelvin-Voigt time
    friction: wp.float32         # Coulomb coefficient
    chamfer: wp.float32          # m: width of the pin end's 45 deg chamfer
    slip: wp.float32             # m/s: below this the friction is viscous (raised in the kernel for stability)
    mobility: wp.float32         # worst point mobility of the pin and a floating layer [1/kg] (stability)
    anchor: wp.float32           # m: an anchor's least embed for its expansion to engage (0: not an anchor)
    nut: wp.float32              # an anchor's nut factor K (T = K F d)
    anchor_hole: wp.float32      # m: the widest hole an anchor's expansion grips


@wp.struct
class Contact:
    """One pin in one layer: the frames and points the contact springs act at."""
    face: wp.transform
    inv: wp.transform
    n: wp.vec3                   # face normal, world
    top: wp.vec3                 # shank top, face frame
    tip: wp.vec3                 # shank tip, face frame
    pin_com: wp.vec3
    face_com: wp.vec3
    pb: wp.int32
    fb: wp.int32
    t: wp.float32                # layer thickness
    half: wp.vec2                # half extent of the layer's patch


@wp.struct
class TipGeo:
    """A shank point against hole ``h``: its axial position, the way to the axis, its lateral overlap with the wall,
    and where an end at its offset lands."""
    u: wp.float32
    inward: wp.vec3              # toward the axis (face frame; 0 when centred)
    out: wp.vec3                 # back along the hole axis (face frame)
    pen: wp.float32              # |offset| + r - rho(u)
    off: wp.float32              # |offset|: the point's distance from the axis
    rest: wp.float32             # u*: the deepest tip position at which the end fits
    bind: wp.float32             # where it binds there, above the tip: 0 the flat end, (0, c] the chamfer


@wp.func
def narrower(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
             r: wp.float32, t: wp.float32) -> wp.float32:
    """``landing`` for a reach just over ``r``: the deepest position where the hole is wider than r (0: nowhere)."""
    u = wp.float32(0.0)
    for k in range(n):
        if 0.5 * stage_d[w, base + k] > r:
            u = wp.max(u, wp.float32(stage_b[w, base + k]) - r / t)
    return u


@wp.func
def fit(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int,
        reach: wp.float32, c: wp.float32, t: wp.float32) -> wp.vec2:
    """(u*, s*): the deepest tip position at which an end of full ``reach`` (offset + radius) with a 45 deg chamfer
    of width c fits the hole, and the height above the tip where it binds. The end at height s reaches
    reach - c + min(s, c); since the hole never widens with depth, u* = min over s of landing(that reach) + s, which
    is linear between the stages' radii: so only s = 0, s = c and s just past each stage radius can bind."""
    end = reach - c
    u = landing(stage_d, stage_b, w, base, n, end, t)
    s = wp.float32(0.0)
    full = landing(stage_d, stage_b, w, base, n, reach, t) + c
    if full < u:
        u = full
        s = c
    for k in range(n):
        sk = 0.5 * stage_d[w, base + k] - end
        if sk >= 0.0 and sk < c:
            uk = narrower(stage_d, stage_b, w, base, n, 0.5 * stage_d[w, base + k], t) + sk
            if uk < u:
                u = uk
                s = sk
    return wp.vec2(u, s)


@wp.func
def hole_at(q: wp.vec3, w: int, i: int, bit: Holes, hole_n: wp.array2d[wp.int32], hole_entry: wp.array3d[wp.float32],
            hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
            stage_b: wp.array2d[wp.float64]) -> wp.int32:
    """The layer's hole whose cut contains face-frame point ``q`` (most centrally), or -1."""
    best = int(-1)
    share = wp.float32(1.0)
    for h in range(hole_n[w, i]):
        slot = i * bit.holes + h
        d = q - wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
        a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
        u = wp.dot(d, a)
        rho = hole_radius(stage_d, stage_b, w, slot * bit.stages, stage_n[w, slot], u, bit.tan_half)
        if rho > 0.0:
            off = wp.length(d - u * a) / rho
            if off < share:
                share = off
                best = h
    return best


@wp.func
def tip_geo(q: wp.vec3, h: int, pin: PinSpec, w: int, i: int, bit: Holes, hole_entry: wp.array3d[wp.float32],
            hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
            stage_b: wp.array2d[wp.float64]) -> TipGeo:
    return tip_geo_at(q, h, pin, w, i, bit, hole_entry, hole_axis, stage_n, stage_d, stage_b, wp.vec3(0.0, 0.0, 0.0))


@wp.func
def tip_geo_at(q: wp.vec3, h: int, pin: PinSpec, w: int, i: int, bit: Holes, hole_entry: wp.array3d[wp.float32],
               hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
               stage_b: wp.array2d[wp.float64], shift: wp.vec3) -> TipGeo:
    """``tip_geo`` with the shank moved across the axis by ``shift`` (face frame) first: where the end would fit if
    the other layers' walls were rigid (``pin_kernels.legal_shift``)."""
    slot = i * bit.holes + h
    base = slot * bit.stages
    n = stage_n[w, slot]
    a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
    d = q + shift - wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
    g = TipGeo()
    g.u = wp.dot(d, a)
    off = d - g.u * a
    r_off = wp.length(off)
    g.off = r_off
    g.inward = wp.vec3(0.0, 0.0, 0.0)
    if r_off > 1.0e-9:
        g.inward = -off / r_off
    g.out = -a
    g.pen = r_off + pin.radius - hole_radius(stage_d, stage_b, w, base, n, g.u, bit.tan_half)
    us = fit(stage_d, stage_b, w, base, n, r_off + pin.radius, pin.chamfer, bit.tan_half)
    g.rest = us[0]
    g.bind = us[1]
    return g


@wp.func
def head_geo(top: wp.vec3, h: int, pin: PinSpec, w: int, i: int, bit: Holes, hole_entry: wp.array3d[wp.float32],
             hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
             stage_b: wp.array2d[wp.float64]) -> wp.vec2:
    """(u, rest) of the head's underside in hole ``h``: its axial position, and where its flat disk (radius
    ``pin.head`` at the shank's offset) fits no deeper (0, the mouth, when the hole is nowhere that wide)."""
    slot = i * bit.holes + h
    a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
    d = top - wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
    u = wp.dot(d, a)
    reach = wp.length(d - u * a) + pin.head
    return wp.vec2(u, landing(stage_d, stage_b, w, slot * bit.stages, stage_n[w, slot], reach, bit.tan_half))


@wp.func
def spring(n_dir: wp.vec3, pen: wp.float32, v: wp.vec3, pin: PinSpec, dt: wp.float32) -> wp.vec3:
    """A damped contact spring along unit ``n_dir`` (pushing the pin) with regularized Coulomb friction, while it
    penetrates; ``v`` is the pin's velocity relative to the part at the contact."""
    if pen <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    fn = wp.max(pin.stiffness * (pen - pin.damping * wp.dot(v, n_dir)), 0.0)
    vt = v - wp.dot(v, n_dir) * n_dir
    eps = wp.max(pin.slip, 2.0 * pin.friction * fn * dt * pin.mobility)   # the explicit viscous slope stays stable
    return fn * n_dir - (pin.friction * fn / wp.max(wp.length(vt), eps)) * vt


@wp.func
def push(body_f: wp.array[wp.spatial_vector], ct: Contact, p: wp.vec3, f: wp.vec3):
    add_wrench(body_f, ct.pb, wrench_at(f, p, ct.pin_com))
    if ct.fb >= 0:
        add_wrench(body_f, ct.fb, wrench_at(-f, p, ct.face_com))


@wp.func
def contact_of(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32],
               face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], pin: PinSpec,
               thickness: wp.array[wp.float32], half: wp.array[wp.vec2], w: int, j: int, layers: int) -> Contact:
    """Pin ``j // layers`` against layer ``j % layers`` in world ``w``."""
    i = j % layers
    ct = Contact()
    ct.pb = pin_body[w, j // layers]
    ct.fb = face_body[w, i]
    ct.pin_com = com_world(body_q, body_com, ct.pb)
    ct.face = patch_frame(body_q, ct.fb, face_local[w, i])
    ct.inv = wp.transform_inverse(ct.face)
    ct.n = wp.transform_vector(ct.face, wp.vec3(0.0, 0.0, 1.0))
    ct.face_com = wp.vec3(0.0, 0.0, 0.0)
    if ct.fb >= 0:
        ct.face_com = com_world(body_q, body_com, ct.fb)
    ct.top = wp.transform_point(ct.inv, wp.transform_point(body_q[ct.pb], pin.top))
    ct.tip = wp.transform_point(ct.inv, wp.transform_point(body_q[ct.pb], pin.tip))
    ct.t = thickness[i]
    ct.half = half[i]
    return ct


@wp.func
def over(ct: Contact, q: wp.vec3) -> wp.bool:
    return wp.abs(q[0]) <= ct.half[0] and wp.abs(q[1]) <= ct.half[1]


@wp.func
def over_by(ct: Contact, q: wp.vec3, margin: wp.float32) -> wp.bool:
    """A disk of radius ``margin`` at ``q`` still overlaps the patch (a pin's end on the plate's edge)."""
    return wp.abs(q[0]) <= ct.half[0] + margin and wp.abs(q[1]) <= ct.half[1] + margin


@wp.func
def in_layer(ct: Contact, q: wp.vec3) -> wp.bool:
    return q[2] < 0.0 and q[2] > -ct.t and over(ct, q)


@wp.func
def sample_point(ct: Contact, c: int) -> wp.vec4:
    """Sample ``c`` of the shank in the layer (face frame, w = 1), or w = 0: its crossing of the face (c = 0) or of
    the back face (c = 1), or its tip if the tip is inside the layer (c = 2); only over the patch."""
    q = ct.tip
    ok = int(1)
    if c < 2:
        z = wp.where(c == 0, 0.0, -ct.t)
        if (ct.top[2] - z) * (ct.tip[2] - z) >= 0.0:
            ok = 0                                                # the shank does not cross this face
        else:
            q = ct.top + ((z - ct.top[2]) / (ct.tip[2] - ct.top[2])) * (ct.tip - ct.top)
    elif not in_layer(ct, q):
        ok = 0
    if not over(ct, q):
        ok = 0
    return wp.vec4(q[0], q[1], q[2], wp.float32(ok))


@wp.func
def relative_velocity(ct: Contact, body_qd: wp.array[wp.spatial_vector], p: wp.vec3) -> wp.vec3:
    v = point_velocity(body_qd, ct.pin_com, ct.pb, p)
    if ct.fb >= 0:
        v = v - point_velocity(body_qd, ct.face_com, ct.fb, p)
    return v


__all__ = ["Contact", "Holes", "PinSpec", "TipGeo", "contact_of", "fit", "head_geo", "hole_at", "in_layer", "narrower",
           "over", "push", "relative_velocity", "sample_point", "spring", "tip_geo", "tip_geo_at"]
