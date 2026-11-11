# SPDX-License-Identifier: Apache-2.0
"""Device kernels of the run-time tunnel check (rules in ``health_tunnel.py``).

``static_sdf`` is the exact signed distance of a primitive static shape in its own frame with its outward unit
gradient; ``reach_along`` the support distance of a body primitive along a direction. ``tunnel_step`` judges one
body per thread against its world's static shapes for one rigid step (COM of ``state_in`` to COM of ``state_out``).
"""

from __future__ import annotations

import warp as wp

from alloy.core.defaults import RIGID_GAP_M

T_PLANE, T_SPHERE, T_CAPSULE, T_ELLIPSOID, T_CYLINDER, T_BOX = 1, 3, 4, 5, 6, 7
TRAVEL_LIMIT_M = RIGID_GAP_M                  # FAST_CONTACT: approach per rigid step above this, and "near" within it
HOLD_STEPS = 64                               # rigid steps a FAST_CONTACT event stays visible (health_tunnel docstring)
_LIMIT = wp.constant(wp.float32(TRAVEL_LIMIT_M))
_HOLD = wp.constant(wp.int32(HOLD_STEPS))
SEG_SAMPLES = 16
OPPOSITE = -0.5
SIDE_DEPTH_M = 0.002                          # a point this deep that leaves by another face passed the corner
SIDE_DOT = 0.5
DEPTH_TOL = 1.0e-6                            # m: float32 positions near 1 m; a point 2 mm in by construction counts
THROUGH_TIE = 1.0e-7                          # m: face pairs this close in distance are tied


@wp.func
def static_sdf(t: int, h: wp.vec3, p: wp.vec3) -> wp.vec4:
    """(distance, outward unit gradient) of a primitive at shape-frame point ``p``; planes are +z."""
    if t == T_PLANE:
        return wp.vec4(p[2], 0.0, 0.0, 1.0)
    if t == T_SPHERE:
        n = wp.normalize(p) if wp.length(p) > 1.0e-9 else wp.vec3(0.0, 0.0, 1.0)
        return wp.vec4(wp.length(p) - h[0], n[0], n[1], n[2])
    if t == T_CAPSULE:
        q = wp.vec3(p[0], p[1], p[2] - wp.clamp(p[2], -h[1], h[1]))
        n = wp.normalize(q) if wp.length(q) > 1.0e-9 else wp.vec3(1.0, 0.0, 0.0)
        return wp.vec4(wp.length(q) - h[0], n[0], n[1], n[2])
    if t == T_CYLINDER:
        r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
        dr = r - h[0]
        dz = wp.abs(p[2]) - h[1]
        ux = 1.0
        uy = 0.0
        if r > 1.0e-9:
            ux = p[0] / r
            uy = p[1] / r
        sz = 1.0
        if p[2] < 0.0:
            sz = -1.0
        if dr > 0.0 or dz > 0.0:
            v = wp.vec3(ux * wp.max(dr, 0.0), uy * wp.max(dr, 0.0), sz * wp.max(dz, 0.0))
            n = wp.normalize(v)
            return wp.vec4(wp.length(v), n[0], n[1], n[2])
        if dr > dz:
            return wp.vec4(dr, ux, uy, 0.0)
        return wp.vec4(dz, 0.0, 0.0, sz)
    q = wp.vec3(wp.abs(p[0]) - h[0], wp.abs(p[1]) - h[1], wp.abs(p[2]) - h[2])
    s = wp.vec3(1.0, 1.0, 1.0)
    for i in range(3):
        if p[i] < 0.0:
            s[i] = -1.0
    out = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    if wp.length(out) > 0.0:
        n = wp.cw_mul(wp.normalize(out), s)
        return wp.vec4(wp.length(out), n[0], n[1], n[2])
    k = 0
    if q[1] > q[k]:
        k = 1
    if q[2] > q[k]:
        k = 2
    n = wp.vec3(0.0, 0.0, 0.0)
    n[k] = s[k]
    return wp.vec4(q[k], n[0], n[1], n[2])


@wp.func
def world_sdf(t: int, h: wp.vec3, xf: wp.transform, p: wp.vec3) -> wp.vec4:
    """``static_sdf`` of a shape at pose ``xf`` for the world point ``p``; the gradient in world axes."""
    r = static_sdf(t, h, wp.transform_point(wp.transform_inverse(xf), p))
    n = wp.transform_vector(xf, wp.vec3(r[1], r[2], r[3]))
    return wp.vec4(r[0], n[0], n[1], n[2])


@wp.func
def reach_along(t: int, h: wp.vec3, d: wp.vec3) -> float:
    """Support distance of a body primitive centred at the origin along the unit shape-frame direction ``d``."""
    if t == T_SPHERE:
        return h[0]
    if t == T_CAPSULE:
        return h[0] + h[1] * wp.abs(d[2])
    if t == T_CYLINDER:
        return h[0] * wp.sqrt(d[0] * d[0] + d[1] * d[1]) + h[1] * wp.abs(d[2])
    if t == T_ELLIPSOID:
        return wp.sqrt(h[0] * h[0] * d[0] * d[0] + h[1] * h[1] * d[1] * d[1] + h[2] * h[2] * d[2] * d[2])
    return h[0] * wp.abs(d[0]) + h[1] * wp.abs(d[1]) + h[2] * wp.abs(d[2])


@wp.func
def can_be_passed(t: int, h: wp.vec3, xf: wp.transform, n0: wp.vec3, travel: float) -> bool:
    """Whether a body travelling ``travel`` toward a static shape (outward normal ``n0`` at the start) could pass
    through it in one step: never a plane (a half-space), else only when its extent along ``n0`` is under twice the
    travel (a thinner part than the step can be crossed without a contact; a thicker one is met first)."""
    if t == T_PLANE:
        return False
    d = wp.transform_vector(wp.transform_inverse(xf), n0)
    return 2.0 * reach_along(t, h, d) < 2.0 * travel


@wp.func
def box_segment_hit(h: wp.vec3, a: wp.vec3, b: wp.vec3) -> bool:
    """Exact slab test: does the segment a -> b (box frame) meet the box of half extents ``h``."""
    t0 = float(0.0)
    t1 = float(1.0)
    for i in range(3):
        d = b[i] - a[i]
        if wp.abs(d) < 1.0e-12:
            if wp.abs(a[i]) > h[i]:
                return False
        else:
            u = (-h[i] - a[i]) / d
            v = (h[i] - a[i]) / d
            t0 = wp.max(t0, wp.min(u, v))
            t1 = wp.min(t1, wp.max(u, v))
    return t0 <= t1


@wp.func
def box_segment_mid(h: wp.vec3, a: wp.vec3, b: wp.vec3) -> wp.vec3:
    """The middle of the segment a -> b's run through the box of half extents ``h`` (box frame; call after a hit)."""
    t0 = float(0.0)
    t1 = float(1.0)
    for i in range(3):
        d = b[i] - a[i]
        if wp.abs(d) >= 1.0e-12:
            u = (-h[i] - a[i]) / d
            v = (h[i] - a[i]) / d
            t0 = wp.max(t0, wp.min(u, v))
            t1 = wp.min(t1, wp.max(u, v))
    return a + (b - a) * (0.5 * (t0 + t1))


@wp.func
def pair_nearest(h: wp.vec3, p: wp.vec3, n: wp.vec3) -> int:
    """Whether the face pair along the box-frame normal ``n`` is the one nearest the box-frame point ``p``: 1 strictly
    (the point is in the part's thickness along ``n``), 0 when another pair is strictly nearer (near another face: a
    slide along it), -1 for a tie (a corner's diagonal, a cube's centre)."""
    k = 0
    if wp.abs(n[1]) > wp.abs(n[k]):
        k = 1
    if wp.abs(n[2]) > wp.abs(n[k]):
        k = 2
    q = wp.vec3(wp.abs(p[0]) - h[0], wp.abs(p[1]) - h[1], wp.abs(p[2]) - h[2])
    tie = int(0)
    for i in range(3):
        if i != k:
            if q[i] > q[k] + THROUGH_TIE:
                return 0
            if q[i] > q[k] - THROUGH_TIE:
                tie = 1
    if tie != 0:
        return -1
    return 1


@wp.func
def through_box(t: int, h: wp.vec3, xf: wp.transform, p: wp.vec3, n: wp.vec3) -> int:
    """``pair_nearest`` for a static box at pose ``xf`` (world point and normal); 1 for every other kind (a curved
    shape keeps the plain opposite-normal rule)."""
    if t != T_BOX:
        return 1
    inv = wp.transform_inverse(xf)
    return pair_nearest(h, wp.transform_point(inv, p), wp.transform_vector(inv, n))


@wp.func
def segment_passes(t: int, h: wp.vec3, xf: wp.transform, p0: wp.vec3, p1: wp.vec3, n0: wp.vec3, n1: wp.vec3) -> bool:
    """Whether a point with both ends of its step outside the static solid passed through it in that step. A box,
    exactly, judged at the middle of the run inside as the in-and-out rules judge a sampled point: out by the opposite
    side (``n1 . n0 < -0.5``) through the box's thickness along ``n0`` (that face pair nearest the middle; ties count),
    or out by any other side (``< 0.5``) with the middle ``SIDE_DEPTH_M`` or more inside (a run along a thin wall's
    plane or past an edge). A point sliding along a face while shallowly inside does neither. Other kinds: <= 16
    signed-distance samples and the opposite-side rule."""
    d = wp.dot(n1, n0)
    inv = wp.transform_inverse(xf)
    if t == T_BOX:
        a = wp.transform_point(inv, p0)
        b = wp.transform_point(inv, p1)
        if not box_segment_hit(h, a, b):
            return False
        mid = box_segment_mid(h, a, b)
        if d < OPPOSITE and pair_nearest(h, mid, wp.transform_vector(inv, n0)) != 0:
            return True
        return d < SIDE_DOT and -static_sdf(T_BOX, h, mid)[0] >= SIDE_DEPTH_M - DEPTH_TOL
    if d >= OPPOSITE:
        return False
    for k in range(SEG_SAMPLES):
        f = float(k + 1) / float(SEG_SAMPLES + 1)
        if static_sdf(t, h, wp.transform_point(inv, p0 + (p1 - p0) * f))[0] < 0.0:
            return True
    return False


@wp.struct
class StaticTable:
    kind: wp.array[wp.int32]
    half: wp.array[wp.vec3]
    shape: wp.array[wp.int32]            # the model's shape index of each entry
    local: wp.array[wp.transform]        # the entry's frame in its shape's frame (a box-like mesh: its box centre)
    xform: wp.array[wp.transform]        # the model's live shape_transform: a layout that moves static shapes after
                                         # the build (alloy.episode.layout) is seen at the next step
    world_start: wp.array[wp.int32]      # CSR by world into world_idx
    world_idx: wp.array[wp.int32]        # indices into the arrays above


@wp.struct
class BodyTable:
    body: wp.array[wp.int32]
    world: wp.array[wp.int32]
    local: wp.array[wp.int32]            # rank within its world (for the packed key)
    r_out: wp.array[wp.float32]
    ex_start: wp.array[wp.int32]
    ex_idx: wp.array[wp.int32]
    sh_start: wp.array[wp.int32]
    sh_kind: wp.array[wp.int32]
    sh_half: wp.array[wp.vec3]
    sh_pose: wp.array[wp.transform]


@wp.struct
class ItemTable:
    """Matter points: one per row (its body's judged index, body-frame point; the COM row reads ``body_com``)."""
    body: wp.array[wp.int32]
    point: wp.array[wp.vec3]
    is_com: wp.array[wp.int32]


@wp.struct
class Tracker:
    in_shape: wp.array[wp.int32]         # per matter point from here down to in_depth
    in_normal: wp.array[wp.vec3]
    in_depth: wp.array[wp.float32]       # deepest -d while inside (m)
    in_thru: wp.array[wp.int32]          # 1: at its deepest inside, the point was nearest the entry face pair
    fast: wp.array[wp.int32]             # [W] rigid steps left of the last FAST_CONTACT event (0: none)
    tunnel: wp.array[wp.int32]
    fast_key: wp.array[wp.uint64]        # [W] (bits(travel) << 32 | local body << 16 | local static)
    tunnel_key: wp.array[wp.uint64]
    seen_fast: wp.array[wp.uint64]       # [W] the latest nonzero key, in no snapshot: names survive a latch's restore
    seen_tunnel: wp.array[wp.uint64]


@wp.func
def static_pose(st: StaticTable, s: int) -> wp.transform:
    """Entry ``s``'s world pose now (a static shape's transform is in world frame)."""
    return wp.transform_multiply(st.xform[st.shape[s]], st.local[s])


@wp.func
def near_distance(bt: BodyTable, j: int, st: StaticTable, s: int, q1: wp.transform) -> float:
    """Distance from the body's near surface to static shape ``s`` at pose ``q1`` (min over the body's shapes)."""
    best = float(1.0e30)
    for i in range(bt.sh_start[j], bt.sh_start[j + 1]):
        xf = wp.transform_multiply(q1, bt.sh_pose[i])
        r = world_sdf(st.kind[s], st.half[s], static_pose(st, s), wp.transform_get_translation(xf))
        d_in = wp.transform_vector(wp.transform_inverse(xf), wp.vec3(-r[1], -r[2], -r[3]))
        best = wp.min(best, r[0] - reach_along(bt.sh_kind[i], bt.sh_half[i], d_in))
    return best


@wp.func
def key_of(travel: float, jl: int, sl: int) -> wp.uint64:
    return (wp.uint64(wp.cast(travel, wp.uint32)) << wp.uint64(32)) | (wp.uint64(wp.min(jl, 65535)) << wp.uint64(16)) \
        | wp.uint64(wp.min(sl, 65535))


@wp.struct
class Track:
    cur: int                 # the static shape the point is inside (-1: none)
    normal: wp.vec3          # outward normal of the face it entered by
    depth: float             # deepest penetration while inside
    thru: int                # 1: at the latest deepest sample that was not a tie, the entry face pair was nearest
    hit: int                 # 1: TUNNEL this step


@wp.func
def track_pair(r_out: float, st: StaticTable, s: int, p0: wp.vec3, p1: wp.vec3, r0: wp.vec4, r1: wp.vec4, cur: int,
               nin: wp.vec3, depth: float, thru: int) -> Track:
    """Entry tracking of one matter point against static shape ``s`` for one step (``health_tunnel`` rules); ``r_out``
    > 0 only for the COM (the whole-body-inside rule)."""
    out = Track()
    out.cur = cur
    out.normal = nin
    out.depth = depth
    out.thru = thru
    out.hit = 0
    n0 = wp.vec3(r0[1], r0[2], r0[3])
    n1 = wp.vec3(r1[1], r1[2], r1[3])
    if cur == s:
        if r1[0] >= 0.0:
            d = wp.dot(n1, nin)
            if (d < OPPOSITE and thru != 0) or (depth >= SIDE_DEPTH_M - DEPTH_TOL and d < SIDE_DOT):
                out.hit = 1
            out.cur = -1
            out.depth = 0.0
        else:
            if -r1[0] >= depth - THROUGH_TIE:            # as deep as ever: a plateau along a face updates too
                out.depth = wp.max(depth, -r1[0])
                v = through_box(st.kind[s], st.half[s], static_pose(st, s), p1, nin)
                if v >= 0:
                    out.thru = v
            if r_out > 0.0 and -r1[0] >= r_out:
                out.hit = 1
    elif cur == -1 and r0[0] >= 0.0:
        if r1[0] < 0.0:
            out.cur = s
            out.normal = n0
            out.depth = -r1[0]
            out.thru = wp.max(through_box(st.kind[s], st.half[s], static_pose(st, s), p1, n0), 0)
        elif segment_passes(st.kind[s], st.half[s], static_pose(st, s), p0, p1, n0, n1):
            out.hit = 1
    return out


@wp.func
def is_excluded(bt: BodyTable, j: int, s: int) -> bool:
    """Whether the pair (judged body j, static shape s) is collision-filtered (cannot touch)."""
    for e in range(bt.ex_start[j], bt.ex_start[j + 1]):
        if bt.ex_idx[e] == s:
            return True
    return False


@wp.kernel
def tunnel_step(q_in: wp.array[wp.transform], q_out: wp.array[wp.transform], com: wp.array[wp.vec3],
                bt: BodyTable, it: ItemTable, st: StaticTable, tr: Tracker):
    """One matter point for one rigid step (rules: ``health_tunnel``): FAST_CONTACT (COM only), entry tracking, TUNNEL."""
    i = wp.tid()
    j = it.body[i]
    b = bt.body[j]
    w = bt.world[j]
    is_com = it.is_com[i] != 0
    pt = it.point[i]
    r_out = float(0.0)
    if is_com:
        pt = com[b]
        r_out = bt.r_out[j]
    p0 = wp.transform_point(q_in[b], pt)
    p1 = wp.transform_point(q_out[b], pt)
    if not (wp.isfinite(p0[0] + p0[1] + p0[2] + p1[0] + p1[1] + p1[2])):
        return
    seg = wp.length(p1 - p0)
    cur = int(tr.in_shape[i])
    nin = wp.vec3(tr.in_normal[i])
    depth = float(tr.in_depth[i])
    thru = int(tr.in_thru[i])
    for k in range(st.world_start[w], st.world_start[w + 1]):
        s = st.world_idx[k]
        if is_excluded(bt, j, s):
            continue
        xs = static_pose(st, s)
        r0 = world_sdf(st.kind[s], st.half[s], xs, p0)
        r1 = world_sdf(st.kind[s], st.half[s], xs, p1)
        if wp.min(r0[0], r1[0]) > r_out + _LIMIT + seg and cur != s:
            continue
        sl = k - st.world_start[w]
        travel = r0[0] - r1[0]
        if is_com and travel > _LIMIT and can_be_passed(st.kind[s], st.half[s], xs,
                                                         wp.vec3(r0[1], r0[2], r0[3]), travel) \
                and near_distance(bt, j, st, s, q_out[b]) <= _LIMIT:
            wp.atomic_max(tr.fast, w, _HOLD)
            wp.atomic_max(tr.fast_key, w, key_of(travel, bt.local[j], sl))
        trk = track_pair(r_out, st, s, p0, p1, r0, r1, cur, nin, depth, thru)
        cur = trk.cur
        nin = trk.normal
        depth = trk.depth
        thru = trk.thru
        if trk.hit != 0:
            wp.atomic_max(tr.tunnel, w, 1)
            wp.atomic_max(tr.tunnel_key, w, key_of(1.0, bt.local[j], sl))
    tr.in_shape[i] = cur
    tr.in_normal[i] = nin
    tr.in_depth[i] = depth
    tr.in_thru[i] = thru


@wp.kernel
def clear_worlds(mask: wp.array[wp.int32], all_worlds: int, tr: Tracker, item_world: wp.array[wp.int32]):
    """Reset: forget the masked worlds' entry tracking per matter point (``mask`` [>= W], nonzero = reset)."""
    j = wp.tid()
    w = item_world[j]
    if all_worlds != 0 or mask[w] != 0:
        tr.in_shape[j] = -1
        tr.in_depth[j] = 0.0


@wp.kernel
def clear_flags(mask: wp.array[wp.int32], all_worlds: int, tr: Tracker):
    w = wp.tid()
    if all_worlds != 0 or mask[w] != 0:
        tr.fast[w] = 0
        tr.tunnel[w] = 0
        tr.fast_key[w] = wp.uint64(0)
        tr.tunnel_key[w] = wp.uint64(0)
        tr.seen_fast[w] = wp.uint64(0)
        tr.seen_tunnel[w] = wp.uint64(0)


@wp.kernel
def remember_keys(tr: Tracker):
    """After a step: copy a world's nonzero keys to the ``seen_*`` copies, which no snapshot restores (the latch puts
    a frozen world's tracker back every step; the text of why it froze must stay)."""
    w = wp.tid()
    if tr.fast_key[w] != wp.uint64(0):
        tr.seen_fast[w] = tr.fast_key[w]
    if tr.tunnel_key[w] != wp.uint64(0):
        tr.seen_tunnel[w] = tr.tunnel_key[w]


@wp.kernel
def age_fast(tr: Tracker):
    """Before each watched rigid step: count a world's FAST_CONTACT hold down; at zero the event's key goes too."""
    w = wp.tid()
    if tr.fast[w] > 0:
        tr.fast[w] = tr.fast[w] - 1
        if tr.fast[w] == 0:
            tr.fast_key[w] = wp.uint64(0)
