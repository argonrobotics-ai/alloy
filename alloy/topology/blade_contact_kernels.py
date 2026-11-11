# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of ``alloy.topology.blade_contact`` (side-aware blade contact for pre-split twins)."""

from __future__ import annotations

import warp as wp

SEAM_ALIGNMENT = 0.5   # a blade acts on a seam when |seam normal . blade face normal| >= this (60 deg): the blade
# contact opens it (``BladeContactConfig.min_alignment``) and, gated, the cutter cuts it (``cut_alignment``)
REENTRY_V_MIN = 1.0e-3  # [m/s] an edge re-entering an opened cut closes on a twin at least this fast (``edge_closing``)


@wp.struct
class BladeTable:
    """Per blade j (body frame geometry) and the shape -> blade map."""

    body: wp.array(dtype=wp.int32)
    world: wp.array(dtype=wp.int32)
    face_shape: wp.array(dtype=wp.int32)
    center: wp.array(dtype=wp.vec3)          # point on the blade mid-plane
    axis: wp.array(dtype=wp.vec3)            # unit face normal
    half: wp.array(dtype=wp.float32)         # half-thickness [m]
    edge_a: wp.array(dtype=wp.vec3)
    edge_b: wp.array(dtype=wp.vec3)
    edge_radius: wp.array(dtype=wp.float32)
    spine: wp.array(dtype=wp.vec3)           # unit, in the face plane, from the edge into the blade; 0 = unknown
    com: wp.array(dtype=wp.vec3)             # the blade body's centre of mass (body_qd's reference point)
    shape_blade: wp.array(dtype=wp.int32)    # [shape_count] blade index, -1 = not a blade shape


@wp.struct
class TwinTable:
    """TET_CUT twins: per particle CSR of its bonds, and the bond columns read at run time."""

    start: wp.array(dtype=wp.int32)          # [P + 1]
    bonds: wp.array(dtype=wp.int32)
    a: wp.array(dtype=wp.int32)              # [B] twin a (the -n side)
    b: wp.array(dtype=wp.int32)              # [B] twin b (the +n side)
    world: wp.array(dtype=wp.int32)
    normal: wp.array(dtype=wp.vec3)          # rest seam normal n0, a's region -> b's
    tet: wp.array(dtype=wp.int32)            # [B] a tet holding twin a, -1 = none
    tet_indices: wp.array2d(dtype=wp.int32)
    tet_poses: wp.array(dtype=wp.mat33)      # Dm^-1
    radius: wp.array(dtype=wp.float32)
    intact: wp.array(dtype=wp.int8)
    damage: wp.array(dtype=wp.float32)


@wp.func
def deformed_normal(n0: wp.vec3, t: int, tet_indices: wp.array2d(dtype=wp.int32),
                    tet_poses: wp.array(dtype=wp.mat33), x: wp.array(dtype=wp.vec3)) -> wp.vec3:
    """Current seam normal F^-T n0 from tet ``t`` (rest normal when there is none or F is singular). Shared by the
    blade contact and the cut damage's alignment gate (``cut.cut_pre_step``), so both judge a seam the same way."""
    if t < 0:
        return n0
    x0 = x[tet_indices[t, 0]]
    ds = wp.matrix_from_cols(x[tet_indices[t, 1]] - x0, x[tet_indices[t, 2]] - x0, x[tet_indices[t, 3]] - x0)
    f = ds * tet_poses[t]
    if wp.abs(wp.determinant(f)) < 1.0e-6:
        return n0
    n = wp.transpose(wp.inverse(f)) * n0
    ln = wp.length(n)
    if ln < 1.0e-12:
        return n0
    return n / ln


@wp.func
def seam_normal(tw: TwinTable, i: int, x: wp.array(dtype=wp.vec3)) -> wp.vec3:
    """Current seam normal of bond i from its side-a tet (``deformed_normal``)."""
    return deformed_normal(tw.normal[i], tw.tet[i], tw.tet_indices, tw.tet_poses, x)


@wp.func
def segment_distance(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-30), 0.0, 1.0)
    return wp.length(p - (a + t * ab))


@wp.func
def bond_active(tw: TwinTable, bl: BladeTable, i: int, j: int, xf: wp.transform, x: wp.array(dtype=wp.vec3),
                reach: float, edge_gate: int) -> bool:
    if tw.intact[i] == wp.int8(0) or tw.damage[i] > 0.0:
        return True
    if edge_gate == 0:
        return False
    pa = tw.a[i]
    pb = tw.b[i]
    mid = 0.5 * (x[pa] + x[pb])
    ea = wp.transform_point(xf, bl.edge_a[j])
    eb = wp.transform_point(xf, bl.edge_b[j])
    gate = bl.edge_radius[j] + reach + wp.max(tw.radius[pa], tw.radius[pb])
    return segment_distance(mid, ea, eb) <= gate


@wp.func
def edge_advancing(bl: BladeTable, j: int, xf: wp.transform, qd: wp.spatial_vector, xp: wp.vec3, vp: wp.vec3) -> bool:
    """SCR-JOB1-2: is blade j's edge moving into the material at xp (or holding still)? The particle's velocity
    relative to the blade point there, along the face-plane direction from the edge into the blade, is >= 0. A
    material point falling out past the edge, or a blade lifting out of it, is not being cut. ``qd`` is Newton's
    (v at the centre of mass, w). True when the blade has no spine (no box)."""
    s = wp.transform_vector(xf, bl.spine[j])
    if wp.length_sq(s) == 0.0:
        return True
    vb = wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), xp - wp.transform_point(xf, bl.com[j]))
    return wp.dot(vp - vb, s) >= 0.0


@wp.func
def edge_closing(bl: BladeTable, j: int, xf: wp.transform, qd: wp.spatial_vector, xp: wp.vec3, vp: wp.vec3,
                 nb: wp.vec3) -> bool:
    """Gap #9 re-entry (WP10's review): is blade j's edge coming down on xp? Its speed toward xp along the spine is
    over ``REENTRY_V_MIN`` and at least its speed across xp along the face normal ``nb``. A blade holding still, a
    face-first sweep or clear (JOB1) and a blade sawing in place do not count (``edge_advancing``'s >= 0 passes them
    on noise); a descent while sawing does, since travel along the edge is not weighed. False without a spine."""
    s = wp.transform_vector(xf, bl.spine[j])
    if wp.length_sq(s) == 0.0:
        return False
    rel = vp - (wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), xp - wp.transform_point(xf, bl.com[j])))
    along = wp.dot(rel, s)
    return along > REENTRY_V_MIN and along >= wp.abs(wp.dot(rel, nb))


@wp.func
def behind_edge(bl: BladeTable, j: int, xf: wp.transform, xp: wp.vec3) -> bool:
    """SCR-JOB1-2 problem 3: is xp level with or behind blade j's edge along its spine, where the faces exist? True
    when the blade has no spine (no box). The edge is taken as its segment, so a point past a tip and below the edge
    line counts as ahead too."""
    sp = wp.transform_vector(xf, bl.spine[j])
    if wp.length_sq(sp) == 0.0:
        return True
    ea = wp.transform_point(xf, bl.edge_a[j])
    eb = wp.transform_point(xf, bl.edge_b[j])
    ab = eb - ea
    t = wp.clamp(wp.dot(xp - ea, ab) / wp.max(wp.dot(ab, ab), 1.0e-30), 0.0, 1.0)
    return wp.dot(xp - (ea + t * ab), sp) >= 0.0


@wp.func
def in_cut_zone(tw: TwinTable, bl: BladeTable, i: int, j: int, xf: wp.transform, x: wp.array(dtype=wp.vec3),
                p: int, target: float, advancing: bool, closing: bool) -> int:
    """SCR-JOB1-2: may twin p of bond i be put on blade j's face ``target`` (+1 / -1)? 0: no; 1: yes; 2: yes, by the
    re-entry case. The own-side face anywhere on the blade; the far face (across the blade from p) only with p within
    the cut-front cap 2h + r of the edge segment (h the blade half-thickness, r the larger twin radius), the edge
    advancing into p (``edge_advancing``), and either
    - the blade cutting the pair: the pair still closer than 2h + r; or
    - re-entry (gap #9, 2026-11-10): p ahead of the edge (``behind_edge`` false), where no face stands between it and
      its own side yet, and the edge coming down on it (``closing``: ``edge_closing``). An edge coming down again
      into an opened cut, off its seam: the pieces have parted, and the stock record there is the blade's bottom
      pressing straight down on the twin (SB-7's stroke: 0.5 mm deep at the mid-plane as the edge re-entered
      0.1-0.9 mm off the seam).
    Beside the faces, or under an edge that sweeps face-first or holds still, an opened twin on the far side is never
    pulled across (JOB1's sweep and clear). How deep is ``rewrite_blade_contacts``' ramp."""
    pa = tw.a[i]
    pb = tw.b[i]
    cap = 2.0 * bl.half[j] + wp.max(tw.radius[pa], tw.radius[pb])
    ea = wp.transform_point(xf, bl.edge_a[j])
    eb = wp.transform_point(xf, bl.edge_b[j])
    nb = wp.transform_vector(xf, bl.axis[j])
    d = wp.dot(x[p] - wp.transform_point(xf, bl.center[j]), nb)
    if d * target > 0.0:
        return 1
    if segment_distance(x[p], ea, eb) > cap or not advancing:
        return 0
    if wp.length(x[pa] - x[pb]) < cap:
        return 1
    if closing and not behind_edge(bl, j, xf, x[p]):
        return 2
    return 0


@wp.func
def twin_sign(tw: TwinTable, bl: BladeTable, p: int, j: int, xf: wp.transform, nb: wp.vec3,
              x: wp.array(dtype=wp.vec3), reach: float, min_alignment: float, edge_gate: int,
              advancing: bool, closing: bool) -> float:
    """+1 / -1: the face side particle p belongs on (+2 / -2: by ``in_cut_zone``'s re-entry case); 0: p is not an
    active twin aligned with blade j, or the face is not one ``in_cut_zone`` allows."""
    if p + 1 >= tw.start.shape[0]:
        return 0.0
    best = int(-1)
    best_c = float(0.0)
    bw = bl.world[j]
    for e in range(tw.start[p], tw.start[p + 1]):
        i = tw.bonds[e]
        w = tw.world[i]
        if w >= 0 and bw >= 0 and w != bw:
            continue
        c = wp.dot(seam_normal(tw, i, x), nb)
        if wp.abs(c) > wp.abs(best_c):
            best = i
            best_c = c
    if best < 0 or wp.abs(best_c) < min_alignment:
        return 0.0
    if not bond_active(tw, bl, best, j, xf, x, reach, edge_gate):
        return 0.0
    side = 1.0
    if tw.a[best] == p:
        side = -1.0
    target = side * wp.sign(best_c)
    return target * float(in_cut_zone(tw, bl, best, j, xf, x, p, target, advancing, closing))


@wp.kernel
def seam_normals_kernel(tw: TwinTable, x: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = seam_normal(tw, i, x)


@wp.kernel
def rewrite_blade_contacts(bl: BladeTable, tw: TwinTable, body_q: wp.array(dtype=wp.transform),
                           body_qd: wp.array(dtype=wp.spatial_vector), x: wp.array(dtype=wp.vec3),
                           v: wp.array(dtype=wp.vec3), reach: float, min_alignment: float, edge_gate: int,
                           allow_reentry: int,
                           count: wp.array(dtype=wp.int32), max_contacts: int,
                           c_indices: wp.array(dtype=wp.vec3i), c_bary: wp.array(dtype=wp.vec3),
                           c_shape: wp.array(dtype=int), c_body_pos: wp.array(dtype=wp.vec3),
                           c_body_vel: wp.array(dtype=wp.vec3), c_normal: wp.array(dtype=wp.vec3),
                           c_rigid: wp.array(dtype=wp.vec3i), reentry_count: wp.array(dtype=wp.int32),
                           reentry_cross: wp.array(dtype=wp.float32)):
    k = wp.tid()
    if k >= wp.min(count[0], max_contacts):
        return
    s = c_shape[k]
    if s < 0:
        return
    j = bl.shape_blade[s]
    if j < 0:
        return
    xf = body_q[bl.body[j]]
    qd = body_qd[bl.body[j]]
    nb = wp.transform_vector(xf, bl.axis[j])
    idx = c_indices[k]
    bary = c_bary[k]
    sgn = float(0.0)
    pt = wp.vec3(0.0)
    r = float(0.0)
    ahead = int(0)
    reentry = int(0)
    for slot in range(3):
        p = idx[slot]
        if p >= 0:
            r = wp.max(r, tw.radius[p])
            if not behind_edge(bl, j, xf, x[p]):
                ahead = 1
            adv = edge_advancing(bl, j, xf, qd, x[p], v[p])
            closing = allow_reentry != 0 and edge_closing(bl, j, xf, qd, x[p], v[p], nb)
            sp = twin_sign(tw, bl, p, j, xf, nb, x, reach, min_alignment, edge_gate, adv, closing)
            if wp.abs(sp) > 1.5:
                reentry = 1
                sp = wp.sign(sp)
            if sp == 0.0 or (sgn != 0.0 and sp != sgn):
                return
            sgn = sp
            pt += bary[slot] * x[p]
    if sgn == 0.0:
        return
    a = bl.axis[j]
    pl = wp.transform_point(wp.transform_inverse(xf), pt)
    d = wp.dot(pl - bl.center[j], a)
    shift = float(0.0)
    if ahead != 0:                                           # SCR-JOB1-2 problem 3: ramp, not a jump
        ea = wp.transform_point(xf, bl.edge_a[j])
        eb = wp.transform_point(xf, bl.edge_b[j])
        allowed = wp.max(0.0, bl.half[j] + r + bl.edge_radius[j] - segment_distance(pt, ea, eb))
        if allowed <= 0.0:
            return                                           # the edge is not near yet: stock record
        # the full depth against the plane at s * (h - r) is h - s d: the ramp caps it at ``allowed`` (gap #9 slice 2
        # first kept the face plane's r + h - s d here, which capped it at allowed - r: in C9 (r 0.5 mm) a twin then
        # met the edge on the wrong side of an off-seam knife, and its bond broke with it inside the blade)
        shift = wp.max(0.0, (bl.half[j] - sgn * d) - allowed)
    if reentry != 0:                                         # diagnostics: records by the re-entry case, per world
        w = wp.max(bl.world[j], 0)
        wp.atomic_add(reentry_count, w, 1)
        wp.atomic_max(reentry_cross, w, -sgn * d)
    # gap #9 (WP15): a seam twin is a point of the cut surface, not a ball: the face plane moves in by its radius, so it
    # rests on the face itself and a blade opens a kerf of its own thickness (not 2 h + 2 r, whose pinch dragged the cut
    # faces along a sawing stroke). The plane may sit past the mid-plane (h < r + shift: every pre-split block here, h
    # 0.5 mm, r 1 mm); the normal still points to the twin's side, so it rests at s * (h - shift), never past its pair.
    c_body_pos[k] = pl - (d - sgn * (bl.half[j] - shift - r)) * a
    c_normal[k] = sgn * nb
    c_body_vel[k] = wp.vec3(0.0)
    c_rigid[k] = wp.vec3i(-1, -1, -1)
    if bl.face_shape[j] >= 0:
        c_shape[k] = bl.face_shape[j]
