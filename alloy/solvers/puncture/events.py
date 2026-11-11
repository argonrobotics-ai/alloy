# SPDX-License-Identifier: Apache-2.0
"""Per-step anchor events on device (design note §2.3-2.4). One step, before the VBD solve:

1. ``update`` (one thread per world, needles and anchors in index order), at the start-of-step state (x0, X0):
   - TIP: axial force F = k_tip (tip − p)·a. If F ≥ fp of its tet, it is punctured: it becomes the track's first
     NEEDLE anchor (entry) or its exit hole. If F < 0, contact is lost and it is freed.
   - NEEDLE past the swage: hands over to the thread (THREAD), or is freed if there is none. Behind the tip
     (withdrawn): freed.
   - THREAD: the segment walk is advanced. Past the last particle it becomes STOPPER (knot) or is freed. Back before
     the first particle it returns to the needle.
   - Growth: on an inside track whose newest anchor is at least h_a behind the tip, the needle
     point h_a ahead of it is queued for location (one per step: after a puncture the shaft already inside fills at
     one anchor per step).
2. ``locate`` (one thread per tissue tet): the smallest tissue tet containing each queued point (atomic min on
   an int, so order-independent).
3. ``cross_param`` / ``cross_pick`` (one thread per surface face): the first face the tip's swept segment
   X0 -> X1 crosses with the right orientation (outside: entering; inside: leaving), as the smallest segment
   parameter, then the smallest face index at it. While the tip is within r_guard of the newest hole, faces near that
   hole are skipped (the tip sits on the faces the hole made; see ``_r_guard``).
4. ``create`` (one thread per world): the growth anchor, then the TIP anchor at the crossing point.
Slots are never reused (``used`` per world only grows), so the order of creation fixes the anchor index.
"""

from __future__ import annotations

import warp as wp

from . import needle as N
from .needle import Needles
from .terms import FREE, NEEDLE, STOPPER, THREAD, TIP, Anchors, Params, Tissue, material_point, stretch, thread_closest
from .topology import Surface

OUTSIDE = wp.constant(0)
TIP_ENTRY = wp.constant(1)
INSIDE = wp.constant(2)
TIP_EXIT = wp.constant(3)
NONE = wp.constant(2147483647)

F_OVERFLOW = wp.constant(1)        # anchor slots exhausted
F_LOST_TIP = wp.constant(2)        # inside track whose tip is in no tissue tet (an exit crossing was missed)
F_WRONG_SIDE = wp.constant(4)      # a crossing with the wrong orientation for the track state
F_LATE_EXIT = wp.constant(8)       # the surface slid past the tip within a step: exit hole placed without a TIP phase
F_LATE_ENTRY = wp.constant(32)     # the tip was already past a surface when its crossing was found: entry hole placed
F_TUNNEL = wp.constant(16)         # an outside tip found inside tissue (an entry was missed)


@wp.struct
class Tracks:
    state: wp.array[wp.int32]       # per needle: OUTSIDE / TIP_ENTRY / INSIDE / TIP_EXIT
    tip: wp.array[wp.int32]         # per needle: TIP anchor (-1)
    newest: wp.array[wp.int32]      # per needle: newest anchor of an inside track (-1)
    hole: wp.array[wp.int32]        # per needle: the last exit hole (-1, diagnostics)
    count: wp.array[wp.int32]       # per needle: entry punctures so far (the current track id)
    need_newest: wp.array[wp.int32]  # per needle: the newest anchor slid off the tip; pick the next or leave
    want: wp.array[wp.int32]        # per needle: 1 = locate the needle point at want_s this step
    want_s: wp.array[float]         # per needle: arc length of the growth point (h_a ahead of the newest anchor)
    loc_tet: wp.array[wp.int32]     # per needle: smallest tissue tet containing it (NONE)
    tip_tet: wp.array[wp.int32]     # per needle: smallest tissue tet containing the tip at the step start (NONE)
    seg_o: wp.array[wp.vec3]        # per needle: start of the crossing segment (newest anchor inside, else the tip)
    seg_t0: wp.array[float]         # per needle: parameter of the start-of-step tip on that segment
    best_t: wp.array[float]         # per needle: first crossing parameter (2 = none)
    best_f: wp.array[wp.int32]      # per needle: face at best_t (NONE)
    fax: wp.array[float]            # per needle: TIP axial force at the step start (0 without a TIP)
    guard_j: wp.array[wp.int32]     # per needle: the newest hole (entry or exit; -1), until the tip leaves its guard
    guard_v: wp.array2d[wp.int32]   # per needle: the 4 vertices of that hole's tet
    guard_on: wp.array[wp.int32]    # per needle: 1 = tip within r_guard of it: faces near the hole are not crossings
    fill: wp.array[wp.int32]        # per needle: after an exit, the anchor the track is filled on from toward the
                                    # exit hole (the shaft cut under the tent has none yet; -1 = done)
    tent_s: wp.array[float]         # per needle: under an exit tent, the newest anchor's arc length (else 0): the
                                    # shaft ahead of it is held by the TIP and that anchor (the untracked monitor)
    punctures: wp.array[wp.int32]   # per needle: count
    flags: wp.array[wp.int32]       # per needle: F_* bits (sticky)
    needle_start: wp.array[wp.int32]  # per world [W+1]
    used: wp.array[wp.int32]        # per world: anchor slots used
    capacity: int


@wp.func
def _r_guard(nd: Needles, pr: Params, i: int) -> float:
    """While the tip is within one growth step plus the look-ahead and the wire of a new hole, the faces touching
    the hole's tet (the faces the hole was made on, however the skin folds round the dimple) are not crossings, and an
    outside tip is not a tunnel (the tent round an exit hole relaxes over the wire, which has no contact of its own).
    The facing wall of an incision has its own vertices (the slit is split), so it is always found, however narrow
    the gap (the V-shaped incision is ~1 mm wide at the bite depth)."""
    return pr.h_a + pr.lookahead + nd.wire[i]


@wp.func
def _guard(sf: Surface, an: Anchors, ts: Tissue, nd: Needles, pr: Params, tr: Tracks, i: int, tip0: wp.vec3,
           x0: wp.array[wp.vec3]):
    tr.guard_on[i] = 0
    g = tr.guard_j[i]
    if g < 0:
        return
    if an.state[g] == FREE:
        tr.guard_j[i] = -1
        return
    if wp.length(tip0 - material_point(ts, an.tet[g], an.w[g], x0)) >= _r_guard(nd, pr, i):
        tr.guard_j[i] = -1                            # the tip has left this hole for good
        return
    for k in range(4):
        tr.guard_v[i, k] = ts.tet_indices[an.tet[g], k]
    tr.guard_on[i] = 1


@wp.func
def _fill_want(an: Anchors, ts: Tissue, nd: Needles, pr: Params, tr: Tracks, i: int, X: wp.transform,
               x0: wp.array[wp.vec3]):
    """Outside after an exit: queue the needle point h_a ahead of the fill anchor while the exit hole is more than
    1.5 h_a ahead of it (one per step, like growth)."""
    f = tr.fill[i]
    h = tr.hole[i]
    if f < 0:
        return
    if tr.state[i] != OUTSIDE or h < 0 or an.state[f] != NEEDLE or an.state[h] != NEEDLE:
        tr.fill[i] = -1
        return
    sf = N.project_world(nd, i, X, material_point(ts, an.tet[f], an.w[f], x0))
    sh = N.project_world(nd, i, X, material_point(ts, an.tet[h], an.w[h], x0))
    if sf - sh < 1.5 * pr.h_a:
        tr.fill[i] = -1
        return
    tr.want[i] = 1
    tr.want_s[i] = sf - pr.h_a


@wp.func
def _free_newest(an: Anchors, tr: Tracks, i: int):
    n = tr.newest[i]
    if n >= 0:
        an.cut[n] = 0
    tr.newest[i] = -1


@wp.func
def _tip_lost(nd: Needles, pr: Params, i: int, X: wp.transform, p: wp.vec3) -> bool:
    """A TIP contact not pressing is kept while its point lies within the look-ahead ahead of the tip and within two
    wire radii of the tip axis; otherwise the needle has moved away and it is freed."""
    c = N.point_world(nd, i, X, 0.0)
    a = N.tip_axis_world(nd, i, X)
    ahead = wp.dot(p - c, a)
    if ahead <= 0.0:
        return False
    lat = wp.length((p - c) - ahead * a)
    return ahead > 1.5 * pr.lookahead or lat > 2.0 * nd.wire[i]


@wp.func
def _next_newest(an: Anchors, ts: Tissue, nd: Needles, tr: Tracks, w: int, i: int, X: wp.transform,
                 x0: wp.array[wp.vec3]):
    """The newest anchor of the inside track slid off the tip (needle withdrawn): the next one of the same track,
    closest to the tip, takes over; with none left the needle is out of the tissue."""
    best = int(-1)
    best_s = float(1.0e30)
    for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
        if an.needle[j] == i and an.state[j] == NEEDLE and an.trk[j] == tr.count[i]:
            s = N.project_world(nd, i, X, material_point(ts, an.tet[j], an.w[j], x0))
            if s >= 0.0 and s < best_s:
                best = j
                best_s = s
    if best >= 0 and tr.state[i] == INSIDE:
        tr.newest[i] = best
        an.cut[best] = 1
    elif tr.state[i] == INSIDE:
        tr.state[i] = OUTSIDE


@wp.func
def _project_thread(j: int, i: int, an: Anchors, nd: Needles, p: wp.vec3, x0: wp.array[wp.vec3]) -> float:
    """Closest thread point to p at the step start: sets seg, alpha (clamped) and tdir; returns the unclamped
    fraction on the chosen segment (< 0 before the first particle, > 1 past the last)."""
    k, a = thread_closest(nd, i, an.seg[j], p, x0)
    d = x0[k + 1] - x0[k]
    an.seg[j] = k
    an.alpha[j] = a
    an.tdir[j] = wp.normalize(d)
    return wp.dot(p - x0[k], d) / wp.max(wp.dot(d, d), 1.0e-20)


@wp.func
def _update_one(j: int, i: int, an: Anchors, ts: Tissue, nd: Needles, tr: Tracks, X: wp.transform,
                x0: wp.array[wp.vec3]):
    """NEEDLE / THREAD transitions of anchor j (TIP is handled by ``update``)."""
    st = an.state[j]
    p = material_point(ts, an.tet[j], an.w[j], x0)
    if st == NEEDLE:
        s = N.project_world(nd, i, X, p)
        if s > nd.length[i]:
            if nd.thread_count[i] > 1:
                an.state[j] = THREAD
                an.seg[j] = nd.thread_start[i]
                _project_thread(j, i, an, nd, p, x0)
            else:
                an.state[j] = FREE
            if tr.newest[i] == j:
                _free_newest(an, tr, i)
        elif s < 0.0:
            an.state[j] = FREE
            if tr.newest[i] == j:
                _free_newest(an, tr, i)
                tr.need_newest[i] = 1
        return
    if st == THREAD:
        raw = _project_thread(j, i, an, nd, p, x0)
        k = an.seg[j]
        first = nd.thread_start[i]
        last = first + nd.thread_count[i] - 2
        if k == last and raw > 1.0:
            if nd.stopper[i] == 1:
                an.state[j] = STOPPER
            else:
                an.state[j] = FREE
        elif k == first and raw < 0.0:
            an.state[j] = NEEDLE


@wp.kernel(enable_backward=False)
def update(sf: Surface, an: Anchors, ts: Tissue, nd: Needles, pr: Params, tr: Tracks, x0: wp.array[wp.vec3],
           body_q0: wp.array[wp.transform], body_q1: wp.array[wp.transform]):
    w = wp.tid()
    for i in range(tr.needle_start[w], tr.needle_start[w + 1]):
        X = N.pose(nd, i, body_q0)
        tr.fax[i] = 0.0
        tr.want[i] = 0
        tr.loc_tet[i] = NONE
        tr.tip_tet[i] = NONE
        tr.best_t[i] = 2.0
        tr.best_f[i] = NONE
        for j in range(w * tr.capacity, w * tr.capacity + tr.used[w]):
            if an.needle[j] != i or an.state[j] == FREE:
                continue
            if an.state[j] == TIP:
                p = material_point(ts, an.tet[j], an.w[j], x0)
                fax = pr.k_tip * wp.dot(N.point_world(nd, i, X, 0.0) - p, N.tip_axis_world(nd, i, X))
                tr.fax[i] = fax
                if fax >= ts.fp[an.tet[j]]:
                    an.state[j] = NEEDLE
                    an.hole[j] = 1
                    # an entry hole also stands for the track ahead of the newest anchor (h_a); an exit hole closes a
                    # track filled up to it (``fill``): half a spacing, as late holes
                    half = wp.where(tr.state[i] == TIP_EXIT, 0.5, 1.0)
                    an.ell[j] = half * pr.h_a / stretch(ts, an.tet[j], x0, N.tangent_world(nd, i, X, 0.0))
                    tr.punctures[i] = tr.punctures[i] + 1
                    tr.tip[i] = -1
                    tr.guard_j[i] = j
                    tr.fill[i] = -1
                    if tr.state[i] == TIP_EXIT and tr.newest[i] >= 0:
                        if an.hole[tr.newest[i]] == 0:
                            tr.fill[i] = tr.newest[i]
                    _free_newest(an, tr, i)
                    if tr.state[i] == TIP_ENTRY:
                        tr.count[i] = tr.count[i] + 1
                        tr.state[i] = INSIDE
                        tr.newest[i] = j
                        an.cut[j] = 1
                    else:
                        tr.state[i] = OUTSIDE
                        tr.hole[i] = j
                    an.trk[j] = tr.count[i]
                elif _tip_lost(nd, pr, i, X, p):
                    an.state[j] = FREE
                    tr.tip[i] = -1
                    if tr.state[i] == TIP_ENTRY:
                        tr.state[i] = OUTSIDE
                    else:
                        tr.state[i] = INSIDE
            else:
                _update_one(j, i, an, ts, nd, tr, X, x0)
        if tr.need_newest[i] == 1:
            tr.need_newest[i] = 0
            _next_newest(an, ts, nd, tr, w, i, X, x0)
        n = tr.newest[i]
        tr.tent_s[i] = 0.0
        if tr.state[i] == TIP_EXIT and n >= 0:
            tr.tent_s[i] = N.project_world(nd, i, X, material_point(ts, an.tet[n], an.w[n], x0))
        _fill_want(an, ts, nd, pr, tr, i, X, x0)
        if tr.state[i] == INSIDE and n >= 0:
            p = material_point(ts, an.tet[n], an.w[n], x0)
            sn = N.project_world(nd, i, X, p)
            if sn >= pr.h_a:
                tr.want[i] = 1
                tr.want_s[i] = sn - pr.h_a
        tip0 = N.point_world(nd, i, X, 0.0)
        _guard(sf, an, ts, nd, pr, tr, i, tip0, x0)
        o = tip0
        if tr.state[i] == INSIDE and n >= 0:
            if an.hole[n] == 0 or tr.guard_on[i] == 1:   # a hole lies on the faces it made: start there only while
                o = material_point(ts, an.tet[n], an.w[n], x0)   # the guard skips them
        elif tr.state[i] == OUTSIDE and tr.guard_on[i] == 1:
            # just out through a hole: a facing surface the exit tent was pressed into (a narrow gap; the tent's apex
            # is a face point, which self-contact does not see) may already be behind the tip, and so is the hole
            # (held at the tip). Start r_guard back along the needle; such a surface is found as a late entry.
            o = N.point_world(nd, i, X, _r_guard(nd, pr, i))
        d = _seg_end(nd, pr, i, body_q1) - o
        tr.seg_o[i] = o
        tr.seg_t0[i] = wp.dot(tip0 - o, d) / wp.max(wp.dot(d, d), 1.0e-30)


@wp.func
def _tet_bary(ts: Tissue, tet: int, q: wp.vec3, x: wp.array[wp.vec3]) -> wp.vec4:
    a = x[ts.tet_indices[tet, 0]]
    m = wp.matrix_from_cols(x[ts.tet_indices[tet, 1]] - a, x[ts.tet_indices[tet, 2]] - a,
                            x[ts.tet_indices[tet, 3]] - a)
    if wp.abs(wp.determinant(m)) < 1.0e-30:
        return wp.vec4(-1.0)
    b = wp.inverse(m) * (q - a)
    return wp.vec4(1.0 - b[0] - b[1] - b[2], b[0], b[1], b[2])


@wp.kernel(enable_backward=False)
def locate(sf: Surface, ts: Tissue, nd: Needles, tr: Tracks, x0: wp.array[wp.vec3], tet_world: wp.array[wp.int32],
           body_q0: wp.array[wp.transform]):
    k = wp.tid()
    tet = sf.tets[k]
    w = tet_world[k]
    for i in range(tr.needle_start[w], tr.needle_start[w + 1]):
        X = N.pose(nd, i, body_q0)
        if tr.want[i] == 1:
            b = _tet_bary(ts, tet, N.point_world(nd, i, X, tr.want_s[i]), x0)
            if wp.min(wp.min(b[0], b[1]), wp.min(b[2], b[3])) >= -1.0e-6:
                wp.atomic_min(tr.loc_tet, i, tet)
        b = _tet_bary(ts, tet, N.point_world(nd, i, X, 0.0), x0)
        if wp.min(wp.min(b[0], b[1]), wp.min(b[2], b[3])) >= 1.0e-6:
            wp.atomic_min(tr.tip_tet, i, tet)


@wp.func
def _seg_tri(o: wp.vec3, d: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """(t, u, v) of segment o + t d against triangle abc (Möller-Trumbore); t = 2 if no hit in [0, 1]."""
    e1 = b - a
    e2 = c - a
    pv = wp.cross(d, e2)
    det = wp.dot(e1, pv)
    if wp.abs(det) < 1.0e-30:
        return wp.vec3(2.0, 0.0, 0.0)
    inv = 1.0 / det
    tv = o - a
    u = wp.dot(tv, pv) * inv
    qv = wp.cross(tv, e1)
    v = wp.dot(d, qv) * inv
    t = wp.dot(e2, qv) * inv
    eps = 1.0e-7
    if u < -eps or v < -eps or u + v > 1.0 + eps or t < 0.0 or t > 1.0:
        return wp.vec3(2.0, 0.0, 0.0)
    return wp.vec3(t, u, v)


@wp.func
def _seg_end(nd: Needles, pr: Params, i: int, body_q1: wp.array[wp.transform]) -> wp.vec3:
    """End of the crossing segment: the end-of-step tip, extended by the look-ahead along the tip axis."""
    X1 = N.pose(nd, i, body_q1)
    return N.point_world(nd, i, X1, 0.0) + pr.lookahead * N.tip_axis_world(nd, i, X1)


@wp.func
def _crossing(f: int, i: int, sf: Surface, nd: Needles, pr: Params, tr: Tracks, x0: wp.array[wp.vec3],
              body_q0: wp.array[wp.transform], body_q1: wp.array[wp.transform]) -> wp.vec3:
    st = tr.state[i]
    if st != OUTSIDE and st != INSIDE:
        return wp.vec3(2.0, 0.0, 0.0)
    o = tr.seg_o[i]
    d = _seg_end(nd, pr, i, body_q1) - o
    a = x0[sf.tri[f, 0]]
    b = x0[sf.tri[f, 1]]
    c = x0[sf.tri[f, 2]]
    if tr.guard_on[i] == 1:
        for e in range(3):
            for k in range(4):
                if sf.tri[f, e] == tr.guard_v[i, k]:
                    return wp.vec3(2.0, 0.0, 0.0)
    side = wp.dot(d, wp.cross(b - a, c - a))
    if (st == OUTSIDE and side >= 0.0) or (st == INSIDE and side <= 0.0):
        return wp.vec3(2.0, 0.0, 0.0)
    return _seg_tri(o, d, a, b, c)


@wp.kernel(enable_backward=False)
def cross_param(sf: Surface, nd: Needles, pr: Params, tr: Tracks, x0: wp.array[wp.vec3],
                face_world: wp.array[wp.int32], body_q0: wp.array[wp.transform], body_q1: wp.array[wp.transform]):
    f = wp.tid()
    w = face_world[f]
    for i in range(tr.needle_start[w], tr.needle_start[w + 1]):
        h = _crossing(f, i, sf, nd, pr, tr, x0, body_q0, body_q1)
        if h[0] <= 1.0:
            wp.atomic_min(tr.best_t, i, h[0])


@wp.kernel(enable_backward=False)
def cross_pick(sf: Surface, nd: Needles, pr: Params, tr: Tracks, x0: wp.array[wp.vec3],
               face_world: wp.array[wp.int32], body_q0: wp.array[wp.transform], body_q1: wp.array[wp.transform]):
    f = wp.tid()
    w = face_world[f]
    for i in range(tr.needle_start[w], tr.needle_start[w + 1]):
        h = _crossing(f, i, sf, nd, pr, tr, x0, body_q0, body_q1)
        if h[0] <= 1.0 and h[0] == tr.best_t[i]:
            wp.atomic_min(tr.best_f, i, f)


@wp.func
def _new_anchor(an: Anchors, tr: Tracks, w: int, i: int, tet: int, b: wp.vec4, state: int, ell: float) -> int:
    if tr.used[w] >= tr.capacity:
        tr.flags[i] = tr.flags[i] | F_OVERFLOW
        return -1
    j = w * tr.capacity + tr.used[w]
    tr.used[w] = tr.used[w] + 1
    an.state[j] = state
    an.needle[j] = i
    an.tet[j] = tet
    an.w[j] = b
    an.seg[j] = -1
    an.cut[j] = 0
    an.ell[j] = ell
    an.trk[j] = tr.count[i]
    an.hole[j] = 0
    return j


@wp.func
def _cross_anchor(sf: Surface, an: Anchors, ts: Tissue, nd: Needles, pr: Params, tr: Tracks, w: int, i: int, f: int,
                  x0: wp.array[wp.vec3], body_q1: wp.array[wp.transform]):
    """The first crossing of this step: a TIP anchor at it, or, when the surface is already behind the start-of-step
    tip, the hole itself (exit from inside, entry from outside; flagged late)."""
    o = tr.seg_o[i]
    d = _seg_end(nd, pr, i, body_q1) - o
    h = _seg_tri(o, d, x0[sf.tri[f, 0]], x0[sf.tri[f, 1]], x0[sf.tri[f, 2]])
    fb = wp.vec3(1.0 - h[1] - h[2], h[1], h[2])
    b = wp.vec4(0.0)
    for c in range(3):
        b[sf.slot[f, c]] = fb[c]
    if tr.state[i] == INSIDE and h[0] < tr.seg_t0[i]:
        jh = _new_anchor(an, tr, w, i, sf.tet[f], b, NEEDLE, 0.5 * pr.h_a)
        if jh >= 0:
            an.hole[jh] = 1
            tr.guard_j[i] = jh
        _free_newest(an, tr, i)
        tr.state[i] = OUTSIDE
        tr.hole[i] = jh
        tr.punctures[i] = tr.punctures[i] + 1
        tr.flags[i] = tr.flags[i] | F_LATE_EXIT
        return
    if tr.state[i] == OUTSIDE and h[0] < tr.seg_t0[i]:
        tr.count[i] = tr.count[i] + 1                     # the tip is already in: the entry hole itself
        jh = _new_anchor(an, tr, w, i, sf.tet[f], b, NEEDLE, 0.5 * pr.h_a)
        if jh >= 0:
            an.hole[jh] = 1
            tr.guard_j[i] = jh
            tr.state[i] = INSIDE
            tr.newest[i] = jh
            an.cut[jh] = 1
        tr.hole[i] = -1
        tr.punctures[i] = tr.punctures[i] + 1
        tr.flags[i] = tr.flags[i] | F_LATE_ENTRY
        return
    jt = _new_anchor(an, tr, w, i, sf.tet[f], b, TIP, 0.0)
    if jt >= 0:
        tr.tip[i] = jt
        if tr.state[i] == OUTSIDE:
            tr.state[i] = TIP_ENTRY
        else:
            tr.state[i] = TIP_EXIT


@wp.kernel(enable_backward=False)
def create(sf: Surface, an: Anchors, ts: Tissue, nd: Needles, pr: Params, tr: Tracks, x0: wp.array[wp.vec3],
           body_q0: wp.array[wp.transform], body_q1: wp.array[wp.transform]):
    w = wp.tid()
    for i in range(tr.needle_start[w], tr.needle_start[w + 1]):
        f = tr.best_f[i]
        if f != NONE:
            _cross_anchor(sf, an, ts, nd, pr, tr, w, i, f, x0, body_q1)
        elif tr.state[i] == OUTSIDE and tr.tip_tet[i] != NONE and tr.guard_on[i] == 0:
            tr.flags[i] = tr.flags[i] | F_TUNNEL
        if tr.want[i] == 1 and tr.state[i] == OUTSIDE and tr.fill[i] >= 0:
            if tr.loc_tet[i] == NONE:
                tr.fill[i] = -1
            else:
                X0 = N.pose(nd, i, body_q0)
                q = N.point_world(nd, i, X0, tr.want_s[i])
                tet = tr.loc_tet[i]
                ell = pr.h_a / stretch(ts, tet, x0, N.tangent_world(nd, i, X0, tr.want_s[i]))
                tr.fill[i] = _new_anchor(an, tr, w, i, tet, _tet_bary(ts, tet, q, x0), NEEDLE, ell)
        if tr.want[i] == 1 and tr.state[i] == INSIDE:
            if tr.loc_tet[i] == NONE:
                tr.flags[i] = tr.flags[i] | F_LOST_TIP
                _free_newest(an, tr, i)
                tr.state[i] = OUTSIDE
            else:
                tet = tr.loc_tet[i]
                X0 = N.pose(nd, i, body_q0)
                tip = N.point_world(nd, i, X0, tr.want_s[i])
                ell = pr.h_a / stretch(ts, tet, x0, N.tangent_world(nd, i, X0, tr.want_s[i]))
                j = _new_anchor(an, tr, w, i, tet, _tet_bary(ts, tet, tip, x0), NEEDLE, ell)
                if j >= 0:
                    _free_newest(an, tr, i)
                    tr.newest[i] = j
                    an.cut[j] = 1
