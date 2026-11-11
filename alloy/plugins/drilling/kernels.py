# SPDX-License-Identifier: Apache-2.0
"""Cutting-front and hole kernels of the drilling plug-in.

One thread per world: each world owns its drill, its holes and its ledger, so every element has one writer and no
atomics are needed. Both kernels evaluate the same fronts (``layer_front``, ``drive``) from the pre-solve state, so
the cut in ``cut_kernel`` is exactly the one whose forces ``drill_force_kernel`` applied.

Geometry (v2). A twist drill of diameter d and point half-angle kappa sits on the tool body (``Bit.tip``,
``Bit.axis`` in the tool frame). The workpiece is one or more layers, each a flat face (``removal.surface`` patch) of
a plate of thickness t: material fills face-frame z in [-t, 0] over the patch. Each layer keeps a table of holes
(``alloy.plugins.holes``): an entry point on its face, an axis, and the stages (one per drill diameter) that cut it.

- A hole runs along the drill's own axis. It starts where the bit's axis crosses the face, along the bit's axis, and
  follows the bit while the point enters (until the full diameter is in). After that the hole guides the bit, through
  its wall springs.
- A re-entry whose axis crosses the face within a drill radius of a hole's entry continues that hole: peck drilling,
  or, with another diameter, a new stage (enlarging a pilot). Farther away it starts a new hole.
- A drill tilted past MIN_FACING does not cut: pressed into a layer, the front holds it like a rigid face.

Cutting front (rigid-plastic). With the tip at axial distance s from a hole's entry, it presses e = s - b into uncut
material (b is the stage's bottom). The deepest press over the layers sets the front force T = k (e + tau de/dt),
never negative. Every layer the edges cut yields at the same feed per revolution f, solved from T = the sum over the
layers of Shaw and Oxford's thrust (Machinery's Handbook form, SI):

    T = sum_i (phi_lips,i C_T,i f^0.8 d^0.8 + phi_chisel,i C_J,i d^2)       M = sum_i phi_m,i C_M,i f^0.8 d^1.8

- Each stage's bottom advances by f |omega_i| dt / 2 pi, never past the tip.
- The phi are the fractions of the lips and of the chisel edge in a layer's material. They ramp in on entry and out at
  breakthrough, and leave out the radii a pilot has cut; phi_m weights the lips by their torque arm.
- Each layer takes its share of the thrust and its own torque.
- Hole wall springs hold the bit's tip and its crossing of each face on the hole axis.

Ledger: a layer's removed volume is the change in the union volume of its hole's stages
(``holes.geometry.union_volume``, float64). So the removed mass is density V(final stages) to rounding (enlarging a
through pilot books density pi (R^2 - r^2) t), and every gram goes to the chip sink or stays on the part.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, torque_wrench, wrench_at
from alloy.plugins.holes.bars import JAM_PROBE, Bars, bar_limit, blocking_bar, layer_clearance
from alloy.plugins.holes.front_types import (CUT, FREE, IN_FULL, IN_NEW, IN_NONE, JAM, MIN_FACING, NONE, RIGID,
                                             ST_FULL, ST_INSIDE, TWO_PI, Bit, Drive, LayerFront, Shaw, Tool, Work,
                                             tool_state)
from alloy.plugins.holes.percussive import hammer_feed, section
from alloy.plugins.holes.geometry import engaged_edges, lateral, stage_start, union_volume, void_radius
from alloy.plugins.holes.match import MIN_CROSS, crossing, hole_radius, nearest_hole, open_hole, stage_of
from alloy.plugins.removal.surface import patch_frame


@wp.func
def layer_front(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                fb: wp.int32, fl: wp.transform, tool: Tool, bit: Bit, work: Work, d: wp.float32, w: int, i: int,
                hole_n: wp.array2d[wp.int32], hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32],
                stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64],
                cur: wp.array2d[wp.int32], front: wp.float32, bars: Bars) -> LayerFront:
    """The drill against one layer. A new hole or stage starts at ``front`` behind the tip when other layers are
    already cutting (one cut front through a stack), else where its point first meets material."""
    lf = LayerFront()
    lf.face = patch_frame(body_q, fb, fl)
    inv = wp.transform_inverse(lf.face)
    lf.n = wp.transform_vector(lf.face, wp.vec3(0.0, 0.0, 1.0))
    lt = wp.transform_point(inv, tool.tip)
    lf.bit_l = wp.transform_vector(inv, tool.axis)
    lf.tip_z = lt[2]
    lf.s_n = -lt[2]
    over = wp.abs(lt[0]) <= work.half[0] and wp.abs(lt[1]) <= work.half[1]
    lf.over = wp.where(over, 1, 0)
    facing = -lf.bit_l[2] >= MIN_FACING
    lf.com = wp.vec3(0.0, 0.0, 0.0)
    v = tool.v
    omega = tool.omega
    if fb >= 0:
        lf.com = com_world(body_q, body_com, fb)
        v = v - point_velocity(body_qd, lf.com, fb, tool.tip)
        qf = body_qd[fb]
        omega = omega - wp.vec3(qf[3], qf[4], qf[5])
    lf.v = v
    lf.spin = wp.dot(omega, tool.axis)
    radius = 0.5 * d
    lf.cross = crossing(lt, lf.bit_l)
    nh = bit.holes
    lf.hole = cur[w, i] - 1
    if lf.hole < 0:
        lf.hole = IN_NONE
        if lt[2] < 0.0 and over:
            lf.hole = nearest_hole(hole_entry, stage_n, stage_d, w, i * nh, hole_n[w, i], bit.stages, lf.cross,
                                   radius)
            if lf.hole < 0:
                lf.hole = wp.where(hole_n[w, i] < nh, IN_NEW, IN_FULL)
    lf.entry_l = lf.cross
    lf.a_l = lf.bit_l
    lf.b = wp.float64(0.0)
    lf.stage = -1
    lf.nst = 0
    lf.base = -1
    lf.formed = 0
    tan64 = wp.float64(bit.tan_half)
    if lf.hole >= 0:
        slot = i * nh + lf.hole
        lf.entry_l = wp.vec2(hole_entry[w, slot, 0], hole_entry[w, slot, 1])
        lf.a_l = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
        lf.base = slot * bit.stages
        lf.nst = stage_n[w, slot]
        st = stage_of(stage_d, stage_b, w, lf.base, lf.nst, d, tan64)
        lf.stage = st.stage
        lf.formed = st.formed
        if lf.stage >= 0:
            lf.b = stage_b[w, lf.base + lf.stage]
        elif st.wide_b > wp.float64(0.0):                 # inside a wider stage: cuts nothing until past its bottom
            lf.stage = ST_INSIDE
            lf.b = st.wide_b
        elif lf.nst < bit.stages:
            lf.b = stage_start(stage_d, stage_b, w, lf.base, lf.nst, wp.float64(radius), tan64)
        else:
            lf.stage = ST_FULL                            # held where its point meets uncut material
            lf.b = stage_start(stage_d, stage_b, w, lf.base, lf.nst, wp.float64(radius), tan64)
    lf.a = wp.transform_vector(lf.face, lf.a_l)
    lf.entry = wp.transform_point(lf.face, wp.vec3(lf.entry_l[0], lf.entry_l[1], 0.0))
    lf.s = wp.dot(lt - wp.vec3(lf.entry_l[0], lf.entry_l[1], 0.0), lf.a_l)
    lf.span = work.thickness / wp.max(-lf.a_l[2], 1.0e-6)
    if lf.stage == ST_INSIDE and lf.s > wp.float32(lf.b):    # past the wider stage's bottom: a stage of its own
        lf.stage = wp.where(lf.nst < bit.stages, -1, ST_FULL)
        lf.b = stage_start(stage_d, stage_b, w, lf.base, lf.nst, wp.float64(radius), tan64)
    if (lf.hole == IN_NEW or lf.stage == -1) and front >= 0.0:
        lf.b = wp.float64(lf.s - front)                   # where the stack's cut front is
    lf.e = lf.s - wp.float32(lf.b)
    void = wp.float32(0.0)
    if lf.hole >= 0:
        void = void_radius(stage_d, stage_b, w, lf.base, lf.nst, lf.stage, lf.b, radius)
    lf.r_void = void
    edges = engaged_edges(lf.s, lf.span, radius, bit.chisel * radius, bit.tan_half, void)
    lf.lips = edges[0]
    lf.moment = edges[1]
    lf.chisel = edges[2]
    lf.mode = NONE
    if lf.s_n <= 0.0 or not over:
        return lf
    if lf.stage == ST_INSIDE:
        return lf                                          # in the void: only the walls act
    if not facing or lf.hole == IN_FULL or lf.stage == ST_FULL:
        lf.e = lf.s_n - wp.float32(lf.b) * wp.max(-lf.a_l[2], 0.0)
        if lf.s_n < work.thickness and lf.e > 0.0:
            lf.mode = RIGID
        return lf
    if lf.lips + lf.chisel > 0.0:
        if lf.e > 0.0:
            lf.mode = CUT
            probe = wp.float32(lf.b) + JAM_PROBE
            entry = wp.vec3(lf.entry_l[0], lf.entry_l[1], 0.0)
            bottom = wp.float32(lf.b)
            if bars.n[i] > 0 and layer_clearance(bars, i, entry, lf.a_l, probe, radius, bit.tan_half, bottom) < 0.0:
                lf.mode = JAM                                     # on a bar it has not cut through
    else:
        lf.mode = FREE
    return lf


@wp.func
def cut_feed(thrust: wp.float32, d: wp.float32, lips: wp.float32, chisel: wp.float32, bit: Bit) -> wp.float32:
    """The feed per revolution at which the engaged edges' summed Shaw-Oxford thrust equals ``thrust`` (capped);
    ``lips`` and ``chisel`` are the sums of phi C_T and phi C_J over the cutting layers."""
    net = thrust - chisel * d * d
    if net <= 0.0:
        return wp.float32(0.0)
    full = lips * wp.pow(d, 0.8)
    if full <= 0.0:
        return bit.feed_max
    return wp.min(wp.pow(net / full, 1.25), bit.feed_max)


@wp.func
def drive(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
          tool: Tool, face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], bit: Bit,
          shaws: wp.array[Shaw], works: wp.array[Work], layers: int, d: wp.float32, dt: wp.float32, w: int,
          hole_n: wp.array2d[wp.int32], hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32],
          stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64],
          cur: wp.array2d[wp.int32], power: wp.float32, bars: Bars) -> Drive:
    """The front force from the deepest press over the cutting layers, and the one feed every layer yields at (no
    more than a rotary hammer's blows cut in a percussive layer). The layers already cutting (their stage exists) set
    the front a new stage in another layer starts at."""
    dr = Drive()
    dr.front = wp.float32(-1.0)
    for i in range(layers):
        lf = layer_front(body_q, body_qd, body_com, face_body[w, i], face_local[w, i], tool, bit, works[i], d, w, i,
                         hole_n, hole_entry, hole_axis, stage_n, stage_d, stage_b, cur, wp.float32(-1.0), bars)
        if lf.mode == CUT and lf.stage >= 0:
            dr.front = wp.max(dr.front, lf.e)
    e_max = wp.float32(0.0)
    closing = wp.float32(0.0)
    stopped = int(0)
    cap = wp.float32(1.0e30)
    blows = wp.float32(1.0e30)
    for i in range(layers):
        lf = layer_front(body_q, body_qd, body_com, face_body[w, i], face_local[w, i], tool, bit, works[i], d, w, i,
                         hole_n, hole_entry, hole_axis, stage_n, stage_d, stage_b, cur, dr.front, bars)
        if lf.mode == CUT:
            if works[i].ev_n > 0:
                dr.hammer = dr.hammer + 1
                area = section(wp.float32(lf.b), lf.span, 0.5 * d, bit.tan_half, lf.r_void)
                blows = wp.min(blows, hammer_feed(power, works[i], d, lf.spin, area))
            sh = shaws[i]
            dr.lips = dr.lips + lf.lips * sh.lips
            dr.chisel = dr.chisel + lf.chisel * sh.chisel
            if dr.count == 0 or lf.e > e_max:
                e_max = lf.e
                closing = wp.dot(lf.v, lf.a)
            dr.count = dr.count + 1
            turn = wp.abs(lf.spin) * dt / TWO_PI                    # revolutions this step
            if turn > 0.0:
                cap = wp.min(cap, lf.e / turn)                     # never past the tip in any layer
            else:
                stopped = 1
    if dr.count != 0:
        dr.thrust = wp.max(bit.stiffness * (e_max + bit.damping * closing), 0.0)
        if stopped == 0:
            shaw = cut_feed(dr.thrust, d, dr.lips, dr.chisel, bit)
            if dr.hammer > 0 and dr.lips + dr.chisel <= 0.0 and dr.thrust > 0.0:
                shaw = wp.float32(1.0e30)                          # only blows cut: no flute limit
            dr.feed = wp.min(wp.min(shaw, cap), blows)
    return dr


@wp.func
def wall_force(lf: LayerFront, p: wp.vec3, tool: Tool, fb: wp.int32, body_qd: wp.array[wp.spatial_vector],
               bit: Bit, gap: wp.float32) -> wp.vec3:
    """Hole wall force on the bit at world point ``p``: back toward the hole axis, across it, once the bit is more
    than ``gap`` (the hole's radius there less the drill's: 0 in its own hole) off the axis."""
    off = lateral(p, lf.entry, lf.a)
    r = wp.length(off)
    if r <= gap:
        return wp.vec3(0.0, 0.0, 0.0)
    v = point_velocity(body_qd, tool.com, tool.body, p)
    if fb >= 0:
        v = v - point_velocity(body_qd, lf.com, fb, p)
    v_lat = v - wp.dot(v, lf.a) * lf.a
    return -bit.radial * (off * ((r - gap) / r) + bit.radial_damping * v_lat)


@wp.func
def layer_thrust(lf: LayerFront, sh: Shaw, dr: Drive, d: wp.float32, pf8: wp.float32, blows: wp.bool) -> wp.float32:
    """This layer's share of the front force: its Shaw-Oxford thrust at the solved feed over the sum's. With rotary
    hammer layers cutting, the Shaw layers take their thrust at the feed and the percussive ones (``blows``) share the
    rest."""
    law = dr.lips * wp.pow(d, 0.8) * pf8 + dr.chisel * d * d
    mine = lf.lips * sh.lips * wp.pow(d, 0.8) * pf8 + lf.chisel * sh.chisel * d * d
    if dr.hammer > 0:
        shaw = wp.min(law, dr.thrust)
        if blows:
            return (dr.thrust - shaw) / wp.float32(dr.hammer)
        return wp.where(law > 0.0, mine * (shaw / wp.max(law, 1.0e-30)), 0.0)
    if law > 0.0:
        return dr.thrust * (mine / law)
    if dr.lips + dr.chisel > 0.0:
        return dr.thrust * ((lf.lips * sh.lips + lf.chisel * sh.chisel) / (dr.lips + dr.chisel))
    return dr.thrust / wp.float32(wp.max(dr.count, 1))                  # no edge law (chisel_j = 0 at entry)


@wp.func
def jam_force(lf: LayerFront, bit: Bit) -> wp.float32:
    """The front holding a bit jammed on a bar, along the hole axis."""
    return wp.max(bit.stiffness * (lf.e + bit.damping * wp.dot(lf.v, lf.a)), 0.0)


@wp.kernel
def drill_force_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                       body_com: wp.array[wp.vec3], tool: wp.array[wp.int32], face_body: wp.array2d[wp.int32],
                       face_local: wp.array2d[wp.transform], bit: Bit, shaws: wp.array[Shaw], works: wp.array[Work],
                       layers: int, diameter: wp.array[wp.float32], dt: wp.float32, hole_n: wp.array2d[wp.int32],
                       hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32],
                       stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
                       stage_b: wp.array2d[wp.float64], cur: wp.array2d[wp.int32],
                       blow_energy: wp.array[wp.float32], blow_rate: wp.array[wp.float32], bars: Bars,
                       body_f: wp.array[wp.spatial_vector]):
    """ADD each layer's share of the front, its cutting torque and its hole wall springs to the tool (reactions on
    the layer if it is a body)."""
    w = wp.tid()
    tl = tool_state(body_q, body_qd, body_com, tool[w], bit)
    d = diameter[w]
    dr = drive(body_q, body_qd, body_com, tl, face_body, face_local, bit, shaws, works, layers, d, dt, w, hole_n,
               hole_entry, hole_axis, stage_n, stage_d, stage_b, cur, blow_energy[w] * blow_rate[w], bars)
    pf8 = wp.pow(dr.feed, 0.8)
    for i in range(layers):
        fb = face_body[w, i]
        work = works[i]
        lf = layer_front(body_q, body_qd, body_com, fb, face_local[w, i], tl, bit, work, d, w, i, hole_n,
                         hole_entry, hole_axis, stage_n, stage_d, stage_b, cur, dr.front, bars)
        if lf.s_n <= 0.0 or lf.over == 0:
            continue
        f_tip = wp.vec3(0.0, 0.0, 0.0)
        tau = wp.vec3(0.0, 0.0, 0.0)
        if lf.mode == CUT:
            sh = shaws[i]
            f_tip = -layer_thrust(lf, sh, dr, d, pf8, work.ev_n > 0) * lf.a
            torque = lf.moment * sh.torque * wp.pow(d, 1.8) * pf8
            if torque > 0.0:
                eps = wp.max(bit.spin_eps, 2.0 * torque * dt / bit.spin_inertia)   # explicit viscous slope stays stable
                tau = -(torque * lf.spin / wp.max(wp.abs(lf.spin), eps)) * tl.axis
        elif lf.mode == RIGID:
            f_tip = wp.max(bit.stiffness * (lf.e - bit.damping * wp.dot(lf.v, lf.n)), 0.0) * lf.n
        elif lf.mode == JAM:
            f_tip = -jam_force(lf, bit) * lf.a
        f_wall = wp.vec3(0.0, 0.0, 0.0)
        f_cross = wp.vec3(0.0, 0.0, 0.0)
        p_cross = tl.tip
        if lf.hole >= 0:                                          # the walls of the hole the bit is in
            if lf.s_n <= work.thickness:                         # the tip is inside the layer
                gap_tip = hole_radius(stage_d, stage_b, w, lf.base, lf.nst, lf.s, bit.tan_half) - 0.5 * d
                f_wall = wall_force(lf, tl.tip, tl, fb, body_qd, bit, wp.max(gap_tip, 0.0))
            if lf.bit_l[2] <= -MIN_CROSS and lf.formed != 0:     # full diameter at the face
                p_cross = tl.tip - (lf.s_n / -lf.bit_l[2]) * tl.axis
                gap = hole_radius(stage_d, stage_b, w, lf.base, lf.nst, 0.0, bit.tan_half) - 0.5 * d
                f_cross = wall_force(lf, p_cross, tl, fb, body_qd, bit, wp.max(gap, 0.0))
        on_tool = wrench_at(f_tip + f_wall, tl.tip, tl.com) + wrench_at(f_cross, p_cross, tl.com)
        add_wrench(body_f, tl.body, on_tool + torque_wrench(tau))
        if fb >= 0:
            on_face = wrench_at(-(f_tip + f_wall), tl.tip, lf.com) + wrench_at(-f_cross, p_cross, lf.com)
            add_wrench(body_f, fb, on_face + torque_wrench(-tau))


@wp.func
def book(stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], w: int, base: int, n: int, k: int,
         new: wp.float64, span: wp.float64, tan64: wp.float64, work: Work, i: int, removed: wp.array[wp.float64],
         sink: wp.array[wp.float64], loose: wp.array[wp.float64], layer_removed: wp.array2d[wp.float64]):
    """Move stage ``k``'s bottom to ``new`` and book the union volume it adds at the layer's density, exactly."""
    v_old = union_volume(stage_d, stage_b, w, base, n, span, tan64)
    stage_b[w, base + k] = new
    dm = wp.float64(work.density) * (union_volume(stage_d, stage_b, w, base, n, span, tan64) - v_old)
    caught = dm * wp.float64(work.capture)
    removed[w] = removed[w] + dm
    sink[w] = sink[w] + caught
    loose[w] = loose[w] + (dm - caught)
    layer_removed[w, i] = layer_removed[w, i] + dm


@wp.func
def bar_cut_into(lf: LayerFront, bars: Bars, i: int, d: wp.float32, thrust: wp.float32, dt: wp.float32, bit: Bit,
                 w: int, hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32],
                 stage_b: wp.array2d[wp.float64]) -> wp.float64:
    """A jammed stage's new bottom: the bit turns into the bar at the Shaw-Oxford feed of the bar's work row under
    the jam thrust (an ESTIMATE: a carbide masonry bit cuts steel no better; its blows do nothing on steel)."""
    slot = i * bit.holes + lf.hole
    old = stage_b[w, lf.base + lf.stage]
    hole_e = wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
    hole_a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
    k = blocking_bar(bars, i, hole_e, hole_a, wp.float32(old) + JAM_PROBE, 0.5 * d, bit.tan_half, wp.float32(old))
    feed = cut_feed(thrust, d, bars.lips[k], bars.chisel[k], bit)
    advance = wp.min(feed * wp.abs(lf.spin) * dt / TWO_PI, wp.max(lf.e, 0.0))
    return old + wp.float64(advance)


@wp.kernel
def cut_kernel(q_in: wp.array[wp.transform], qd_in: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
               tool: wp.array[wp.int32], face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform],
               bit: Bit, shaws: wp.array[Shaw], works: wp.array[Work], layers: int,
               diameter: wp.array[wp.float32], dt: wp.float32, blow_energy: wp.array[wp.float32],
               blow_rate: wp.array[wp.float32], bars: Bars, hole_n: wp.array2d[wp.int32],
               hole_entry: wp.array3d[wp.float32], hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32],
               stage_d: wp.array2d[wp.float32], stage_b: wp.array2d[wp.float64], cur: wp.array2d[wp.int32],
               depth: wp.array[wp.float64], entry: wp.array2d[wp.float32], engaged: wp.array[wp.int32],
               holes: wp.array[wp.int32], full: wp.array[wp.int32], removed: wp.array[wp.float64],
               sink: wp.array[wp.float64], loose: wp.array[wp.float64], layer_removed: wp.array2d[wp.float64],
               thrust: wp.array[wp.float32], torque: wp.array[wp.float32], feed: wp.array[wp.float32],
               jammed: wp.array[wp.int32], bar_cut: wp.array[wp.float64]):
    """Advance each cut stage's bottom by the cut of this step and book the removed mass exactly.

    Nothing changes while the tip is above a layer's face or beside its patch. A drill tilted past MIN_FACING opens
    no hole where there is none; inside an existing hole it stays in that hole. A retract above a face leaves its
    hole; a re-entry decides between continuing a hole and starting a new one (``layer_front``)."""
    w = wp.tid()
    tl = tool_state(q_in, qd_in, body_com, tool[w], bit)
    d = diameter[w]
    dr = drive(q_in, qd_in, body_com, tl, face_body, face_local, bit, shaws, works, layers, d, dt, w, hole_n,
               hole_entry, hole_axis, stage_n, stage_d, stage_b, cur, blow_energy[w] * blow_rate[w], bars)
    pf8 = wp.pow(dr.feed, 0.8)
    tan64 = wp.float64(bit.tan_half)
    pushed = wp.float32(0.0)
    moment = wp.float32(0.0)
    count = int(0)
    for i in range(layers):
        work = works[i]
        lf = layer_front(q_in, qd_in, body_com, face_body[w, i], face_local[w, i], tl, bit, work, d, w, i, hole_n,
                         hole_entry, hole_axis, stage_n, stage_d, stage_b, cur, dr.front, bars)
        if lf.mode == CUT or lf.mode == FREE:
            hk = open_hole(lf.hole, lf.stage, lf.cross, lf.bit_l, lf.b, w, i, d, bit.holes, bit.stages, hole_n,
                           hole_entry, hole_axis, stage_n, stage_d, stage_b)
            slot = i * bit.holes + hk[0]
            base = slot * bit.stages
            n = stage_n[w, slot]
            old = stage_b[w, base + hk[1]]
            span_bit = wp.float64(work.thickness) / wp.float64(wp.max(-lf.bit_l[2], 1.0e-6))
            if hk[1] == 0 and old < wp.float64(0.5 * d) / tan64 and old < wp.min(wp.float64(lf.span), span_bit):
                hole_entry[w, slot, 0] = lf.cross[0]                    # the point is entering: the hole follows
                hole_entry[w, slot, 1] = lf.cross[1]                    # the bit (its volume does not depend on
                for c in range(3):                                      # the span until the tip is out)
                    hole_axis[w, slot, c] = lf.bit_l[c]
            span = wp.float64(work.thickness) / wp.float64(wp.max(-hole_axis[w, slot, 2], 1.0e-6))   # as Hole.span
            advance = wp.float64(dr.feed * wp.abs(lf.spin) * dt / TWO_PI)
            new = wp.max(old, wp.float64(lf.s))                         # FREE: follows the tip, or the front's cut
            if dr.front >= 0.0:
                new = wp.max(old, wp.float64(lf.s - dr.front) + advance)
            if lf.mode == CUT:
                new = old + advance
                moment = moment + lf.moment * shaws[i].torque * wp.pow(d, 1.8) * pf8
            if bars.n[i] > 0:                                           # never into a bar it cannot cut
                hole_e = wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
                hole_a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
                new = bar_limit(bars, i, hole_e, hole_a, old, new, 0.5 * d, bit.tan_half)
            book(stage_d, stage_b, w, base, n, hk[1], new, span, tan64, work, i, removed, sink, loose, layer_removed)
            cur[w, i] = hk[0] + 1
            if i == 0:
                depth[w] = new
                entry[w, 0] = hole_entry[w, slot, 0]
                entry[w, 1] = hole_entry[w, slot, 1]
        else:
            if lf.mode == RIGID:
                pushed = pushed + wp.max(bit.stiffness * (lf.e - bit.damping * wp.dot(lf.v, lf.n)), 0.0)
            elif lf.mode == JAM:
                tj = jam_force(lf, bit)
                pushed = pushed + tj
                jammed[w] = 1                                            # on a bar: a new hole's too
            if lf.mode == JAM and lf.hole >= 0 and lf.stage >= 0:
                before = wp.float32(stage_b[w, lf.base + lf.stage])
                new = bar_cut_into(lf, bars, i, d, jam_force(lf, bit), dt, bit, w, hole_entry, hole_axis, stage_b)
                slot = i * bit.holes + lf.hole
                span = wp.float64(work.thickness) / wp.float64(wp.max(-hole_axis[w, slot, 2], 1.0e-6))
                book(stage_d, stage_b, w, lf.base, lf.nst, lf.stage, new, span, tan64, work, i, removed, sink, loose,
                     layer_removed)
                hole_e = wp.vec3(hole_entry[w, slot, 0], hole_entry[w, slot, 1], 0.0)
                hole_a = wp.vec3(hole_axis[w, slot, 0], hole_axis[w, slot, 1], hole_axis[w, slot, 2])
                gap = layer_clearance(bars, i, hole_e, hole_a, wp.float32(new), 0.5 * d, bit.tan_half, before)
                bar_cut[w] = wp.max(bar_cut[w], wp.float64(wp.max(-gap, 0.0)))
                if i == 0:
                    depth[w] = new
            tableful = lf.hole == IN_FULL or lf.stage == ST_FULL
            if tableful and lf.over != 0 and lf.tip_z < 0.0 and full[w] == 0:  # once per world, when it fills
                wp.printf("drill: world %d layer %d: hole table full (max_holes %d, max_stages %d): held as by "
                          "a face, nothing cut; raise max_holes or max_stages\n", w, i, bit.holes, bit.stages)
                full[w] = 1
            if lf.hole >= 0 and lf.tip_z < 0.0 and lf.over != 0:
                cur[w, i] = lf.hole + 1                                  # still in that hole (not cutting)
        if lf.tip_z >= 0.0:
            cur[w, i] = 0                                                # retracted above the face
        count = count + hole_n[w, i]
    holes[w] = count
    engaged[w] = wp.where(cur[w, 0] != 0, 1, wp.where(hole_n[w, 0] > 0, 2, 0))
    thrust[w] = dr.thrust + pushed
    torque[w] = moment
    feed[w] = dr.feed


__all__ = ["Bit", "Drive", "LayerFront", "Shaw", "Tool", "Work", "cut_kernel", "drill_force_kernel", "drive",
           "jam_force", "layer_front", "tool_state"]
