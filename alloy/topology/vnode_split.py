# SPDX-License-Identifier: Apache-2.0
"""Split passes for virtual-node copies (WP10 item 9 step 3; ``vnode_cut.VNodeCutter.pre_step`` launches them, and
its module docstring states the rules).

- ``follow_dormant``: copies without mass stay on their hub.
- ``decide_split``: per whole cuttable tet, the cut plane's corner sides, the crack-front flag, and the split.
- ``apply_split``: T's material to its fragments.
- ``hand_over``: per hub, mass to its copies, its side at the first split, and the welds.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor import newton
from alloy.topology.geometry import segment_near_tet
from alloy.topology.tet_copies import HUB_KEEP, fraction_from_params, vec6
from alloy.topology.vnode_contacts import CutterBlades
from alloy.topology.vnode_work import PHASE_CUT, PHASE_WELD, RULE_WORK, WorkRule

SNAP_FRACTION = 0.05       # corner-to-plane snap, as a fraction of the hub's mean rest edge length
ON_PLANE = wp.constant(1.0e-6)  # a snapped corner sits ON_PLANE x its snap distance off the plane, on its side
FRONT = wp.constant(3.0)    # crack-front flags are refreshed for tets within FRONT x lead of the edge
SLIDE = wp.constant(0.2)   # |e x v| < SLIDE |e| |v|: the edge slides along itself; use the face normal
ADVANCE = wp.constant(0.5)  # the edge leads when v . (-spine) >= ADVANCE |v| (a 60 deg cone)
FOLD = 0.01                 # a split side under this volume fraction folds into the other (ruling: 1-2 %)
KEEP = wp.constant(HUB_KEEP)
MASS_FLOOR = wp.constant(0.1)  # beta: copies take (f + beta) / (1 + 2 beta) of a split corner's handed-over mass
ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))
FRAGMENT_DAMPING = 1.0e-3  # [s] fragment k_damp per unit (k_mu + k_lambda): strong-lean strokes ran virtual copies
                           # at 1.6-2.2 m/s undamped, 0.20-0.22 m/s at 1e-3 s (energy cost: to measure)


@wp.struct
class TetTable:
    world: wp.array(dtype=wp.int32)           # [C]
    tet: wp.array(dtype=wp.int32)             # [C]
    hub: wp.array2d(dtype=wp.int32)           # [C, 4] corner particles
    hub_row: wp.array2d(dtype=wp.int32)       # [C, 4] their hub-table rows
    tet_edge: wp.array2d(dtype=wp.int32)      # [C, 6] edge-table rows, LOCAL_EDGES order
    tet_edge_flip: wp.array2d(dtype=wp.int32)  # [C, 6] 1: the local pair runs high -> low hub row
    copy_tet: wp.array2d(dtype=wp.int32)      # [C, 2] T+, T-
    material: wp.array(dtype=wp.vec3)         # [C]
    mass: wp.array(dtype=wp.float32)          # [C]


@wp.struct
class HubTable:
    hub: wp.array(dtype=wp.int32)             # [H] particle
    node: wp.array2d(dtype=wp.int32)          # [H, 2] v+, v-
    weld: wp.array2d(dtype=wp.int32)          # [H, 2]
    weld_ke: wp.array(dtype=wp.float32)       # [H]
    start: wp.array(dtype=wp.int32)           # [H + 1] CSR over the hub's cuttable tets
    rows: wp.array(dtype=wp.int32)            # tet rows, ascending
    corner: wp.array(dtype=wp.int32)          # the hub's corner index in that row


@wp.func
def corner_side(s: float, eps: float, fixed: int) -> float:
    """Signed distance of a corner: within eps it is on the plane (just on the - side), and a hub whose side is
    fixed (0 = +, 1 = -) is put on the plane on that side when geometry disagrees. "On the plane" is ON_PLANE eps
    off it, so a tet whose other corners lie on one side splits into a full fragment and an empty one (folded)."""
    on = ON_PLANE * eps
    out = s
    if wp.abs(s) < eps:
        out = -on
    if fixed == 0 and out < 0.0:
        out = on
    elif fixed == 1 and out >= 0.0:
        out = -on
    return out


@wp.func
def cut_corner_side(s: float, eps: float, fixed: int, phase: int) -> float:
    """``corner_side``, but an unsided hub the knife is cutting (``work`` rule: phase past welded) is on the plane, on
    the side the edge's wedge pushed it: displaced 0.3-0.6 mm before its cut, it left slivers whose light copies ran
    at 1.3 m/s."""
    if fixed < 0 and phase != PHASE_WELD:
        on = ON_PLANE * eps
        return wp.where(s >= 0.0, on, -on)
    return corner_side(s, eps, fixed)


@wp.func
def sides_mask(s: wp.vec4) -> int:
    """Bit i set: corner i on the + side."""
    mask = int(0)
    for i in range(4):
        if s[i] >= 0.0:
            mask = mask | (1 << i)
    return mask


@wp.func
def cut_normal(xf: wp.transform, vel: wp.vec3, e: wp.vec3, axis: wp.vec3) -> wp.vec3:
    """Unit normal of the swept plane (edge direction x edge velocity), the face normal when the edge slides along
    itself; signed to the blade face normal's side."""
    face = wp.transform_vector(xf, axis)
    n = wp.cross(e, vel)
    if wp.length(n) < SLIDE * wp.length(e) * wp.length(vel) or wp.length(vel) < 1.0e-9:
        n = wp.cross(e, wp.cross(face, e))
    n = wp.normalize(n)
    if wp.dot(n, face) < 0.0:
        n = -n
    return n


@wp.func
def edge_leads(xf: wp.transform, vel: wp.vec3, spine: wp.vec3) -> bool:
    """Is the blade moving edge-first in its face plane (within the ``ADVANCE`` cone)? True with no spine."""
    sp = wp.transform_vector(xf, spine)
    if wp.length_sq(sp) == 0.0:
        return True
    speed = wp.length(vel)
    return speed > 1.0e-9 and -wp.dot(vel, sp) >= ADVANCE * speed


@wp.kernel
def follow_dormant(ht: HubTable, hub_side: wp.array(dtype=wp.int32), mass: wp.array(dtype=wp.float32),
                   x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3)):
    """Copies without mass (not yet given any by a split, or only on folded sides) stay on their hub."""
    r = wp.tid()
    h = ht.hub[r]
    for k in range(2):
        n = ht.node[r, k]
        if hub_side[r] < 0 or mass[n] == 0.0:
            x[n] = x[h]
            v[n] = v[h]


@wp.func
def edge_corners(k: int) -> wp.vec2i:
    if k < 3:
        return wp.vec2i(0, k + 1)
    if k < 5:
        return wp.vec2i(1, k - 1)
    return wp.vec2i(2, 3)


@wp.func
def crossing_params(tt: TetTable, c: int, s: wp.vec4, edge_t: wp.array(dtype=wp.float32)) -> vec6:
    """Per local edge crossed by the cut, its parameter low -> high local corner: the stored one if the edge was
    crossed before, else from s, stored for later tets. Every thread computes it in the edge's canonical order (low
    hub row -> high), so tets splitting together in one launch store and read identical bits (WP5 review: opposite
    corner orders differ by an ulp in float32, and the stored value then depended on thread timing)."""
    t6 = vec6()
    for k in range(6):
        ij = edge_corners(k)
        if (s[ij[0]] >= 0.0) != (s[ij[1]] >= 0.0):
            flip = tt.tet_edge_flip[c, k] != 0
            lo = wp.where(flip, ij[1], ij[0])
            hi = wp.where(flip, ij[0], ij[1])
            tc = s[lo] / (s[lo] - s[hi])
            e = tt.tet_edge[c, k]
            old = edge_t[e]
            if old >= 0.0:
                tc = old
            else:
                edge_t[e] = tc
            t6[k] = wp.where(flip, 1.0 - tc, tc)
    return t6


@wp.func
def near_corner_cut(wr: WorkRule, s: wp.vec4, r0: int, r1: int, r2: int, r3: int, ball: float) -> bool:
    """``work`` rule: has the knife cut a corner of this tet near the cut plane (the cut has reached the tet)?"""
    hit = int(0)
    for i in range(4):
        r = r0
        if i == 1:
            r = r1
        elif i == 2:
            r = r2
        elif i == 3:
            r = r3
        if wp.abs(s[i]) < ball and wr.cut[r] == PHASE_CUT:
            hit = 1
    return hit != 0


@wp.kernel
def decide_split(tt: TetTable, bl: CutterBlades, body_q: wp.array(dtype=wp.transform),
                 body_qd: wp.array(dtype=wp.spatial_vector), x: wp.array(dtype=wp.vec3), lead: float,
                 hub_side: wp.array(dtype=wp.int32), hub_eps: wp.array(dtype=wp.float32),
                 split: wp.array(dtype=wp.int32), split_now: wp.array(dtype=wp.int32),
                 side_mask: wp.array(dtype=wp.int32), frac: wp.array(dtype=wp.float32),
                 edge_t: wp.array(dtype=wp.float32), fold: float, front: wp.array(dtype=wp.int32), wr: WorkRule,
                 on_plane: wp.array(dtype=wp.int32)):
    c = wp.tid()
    split_now[c] = 0
    if split[c] != 0:
        return
    j = bl.of_world[tt.world[c]]
    if j < 0:
        return
    x0 = x[tt.hub[c, 0]]
    x1 = x[tt.hub[c, 1]]
    x2 = x[tt.hub[c, 2]]
    x3 = x[tt.hub[c, 3]]
    xf = body_q[bl.body[j]]
    ea = wp.transform_point(xf, bl.edge_a[j])
    eb = wp.transform_point(xf, bl.edge_b[j])
    vel = wp.spatial_top(body_qd[bl.body[j]])
    if not edge_leads(xf, vel, bl.spine[j]):
        return
    if not segment_near_tet(ea, eb, x0, x1, x2, x3, FRONT * lead):
        return
    n = cut_normal(xf, vel, eb - ea, bl.axis[j])
    r0 = tt.hub_row[c, 0]
    r1 = tt.hub_row[c, 1]
    r2 = tt.hub_row[c, 2]
    r3 = tt.hub_row[c, 3]
    s = wp.vec4(cut_corner_side(wp.dot(n, x0 - ea), hub_eps[r0], hub_side[r0], wr.cut[r0]),
                cut_corner_side(wp.dot(n, x1 - ea), hub_eps[r1], hub_side[r1], wr.cut[r1]),
                cut_corner_side(wp.dot(n, x2 - ea), hub_eps[r2], hub_side[r2], wr.cut[r2]),
                cut_corner_side(wp.dot(n, x3 - ea), hub_eps[r3], hub_side[r3], wr.cut[r3]))
    mask = sides_mask(s)
    front[c] = wp.where(mask == 0 or mask == 15, 0, 1)        # whole, across the cut: the crack front
    if front[c] == 0 or not segment_near_tet(ea, eb, x0, x1, x2, x3, lead):
        return
    if wr.rule == RULE_WORK and not near_corner_cut(wr, s, r0, r1, r2, r3, lead):
        return
    fa = wp.clamp(fraction_from_params(mask, crossing_params(tt, c, s, edge_t)), 0.0, 1.0)
    if fa < fold:
        fa = 0.0
    elif fa > 1.0 - fold:
        fa = 1.0
    side_mask[c] = mask
    frac[c] = fa
    split_now[c] = 1
    for i in range(4):                                       # corners the cut passes through (snapped onto it)
        r = tt.hub_row[c, i]
        if wp.abs(s[i]) <= ON_PLANE * hub_eps[r]:
            on_plane[r] = 1




@wp.kernel
def apply_split(tt: TetTable, split_now: wp.array(dtype=wp.int32), frac: wp.array(dtype=wp.float32),
                damping: float, split: wp.array(dtype=wp.int32), tet_materials: wp.array2d(dtype=wp.float32)):
    """T's material to 0, its fragments to f+ and f- of it; fragment damping k_damp + damping (k_mu + k_lambda)."""
    c = wp.tid()
    if split_now[c] == 0:
        return
    mat = tt.material[c]
    mat = wp.vec3(mat[0], mat[1], mat[2] + damping * (mat[0] + mat[1]))
    fa = frac[c]
    for m in range(3):
        tet_materials[tt.tet[c], m] = 0.0
        tet_materials[tt.copy_tet[c, 0], m] = fa * mat[m]
        tet_materials[tt.copy_tet[c, 1], m] = (1.0 - fa) * mat[m]
    split[c] = 1


@wp.func
def add_mass(n: int, dm: float, vel: wp.vec3, mass: wp.array(dtype=wp.float32),
             inv_mass: wp.array(dtype=wp.float32), v: wp.array(dtype=wp.vec3)):
    """Merge mass dm moving at vel into particle n, keeping momentum."""
    if dm <= 0.0:
        return
    m0 = mass[n]
    m1 = m0 + dm
    v[n] = (m0 * v[n] + dm * vel) / m1
    mass[n] = m1
    inv_mass[n] = 1.0 / m1


@wp.func
def weld_copies(ht: HubTable, r: int, side: int, split: wp.array(dtype=wp.int32), front: wp.array(dtype=wp.int32),
                wr: WorkRule, mass: wp.array(dtype=wp.float32), spring_ke: wp.array(dtype=wp.float32),
                particle_side: wp.array(dtype=wp.int8), on_plane: wp.array(dtype=wp.int32)):
    """A sided hub's welds; only a copy with mass is welded. SolverVBD leaves massless particles where they are, so a
    weld to one pinned its hub to the substep's start (a plane hub whose own-side copy got no mass, every split
    there folding to the far side, pressed the TOP-2 knife with 0.9 N at first touch instead of 0.07 N).
    The hub's own-side copy is always welded. The copy on the far side of the cut stays welded, so both sides'
    material still meets at the hub, while
    - the hub's tets still hold a whole one across the cut (the crack front has not passed the hub; with the copy
      free at once, the far side tore along the faces between its fragments and those whole tets: the TOP-2 twin's
      far top layer swung down as a flap hinged on one row of welds), or
    - under the ``work`` rule, the knife has not cut the hub yet: the weld is then the seam's twin bond
      (``vnode_work.bond_stiffness``).
    Under the ``work`` rule a far copy with mass whose hub the cut passes through (``on_plane``) is a node of the cut
    face, and meets the blade as its hub does, on its own side (``particle_side``): pressing it (stock contact) while
    held, as the seam's twins both press the knife until their bond breaks (muted, only the hub pressed and the knife
    wedged it aside); on its face once free (``released``). Every other far copy stays muted, its hub colliding for
    it: off the plane it is a virtual node beyond the cut (with blade contacts, an off-node stroke shoved material at
    0.39 m/s), and under the ``lead`` rule faces on on-plane far copies kicked oblique-stroke hubs to 0.49 m/s."""
    own = ht.node[r, side]
    spring_ke[ht.weld[r, side]] = wp.where(mass[own] > 0.0, ht.weld_ke[r], 0.0)
    ke = float(0.0)
    for e in range(ht.start[r], ht.start[r + 1]):
        c = ht.rows[e]
        if split[c] == 0 and front[c] != 0:
            ke = ht.weld_ke[r]
    if wr.rule == RULE_WORK and wr.cut[r] != PHASE_CUT:
        ke = wr.bond[r]
    far = ht.node[r, 1 - side]
    hold = ke > 0.0 or (wr.rule == RULE_WORK and wr.cut[r] != PHASE_CUT)
    if mass[far] == 0.0:
        ke = 0.0
    spring_ke[ht.weld[r, 1 - side]] = ke
    if mass[far] > 0.0 and on_plane[r] != 0 and wr.rule == RULE_WORK:
        particle_side[far] = wp.int8(1 - side)
    wr.released[far] = wp.where(hold, wp.int8(0), wp.int8(1))


@wp.kernel
def hand_over(ht: HubTable, tt: TetTable, split_now: wp.array(dtype=wp.int32), frac: wp.array(dtype=wp.float32),
              side_mask: wp.array(dtype=wp.int32), mass: wp.array(dtype=wp.float32),
              inv_mass: wp.array(dtype=wp.float32), flags: wp.array(dtype=wp.int32), v: wp.array(dtype=wp.vec3),
              spring_ke: wp.array(dtype=wp.float32), hub_side: wp.array(dtype=wp.int32),
              particle_side: wp.array(dtype=wp.int8), conflicts: wp.array(dtype=wp.int32),
              added: wp.array(dtype=wp.float32), split: wp.array(dtype=wp.int32), front: wp.array(dtype=wp.int32),
              wr: WorkRule, on_plane: wp.array(dtype=wp.int32)):
    """Per hub, in CSR order over its tets split this substep: mass to v+ / v-, and the hub's side at its first. A
    kinematic (clamped) hub keeps its zero mass, so the mass its copies take is new: booked in ``added`` (TOP-1)."""
    r = wp.tid()
    h = ht.hub[r]
    for e in range(ht.start[r], ht.start[r + 1]):
        c = ht.rows[e]
        if split_now[c] == 0:
            continue
        share = (1.0 - KEEP) * tt.mass[c] / 4.0
        fa = frac[c]
        vh = v[h]
        if inv_mass[h] != 0.0:
            mass[h] = mass[h] - share
            inv_mass[h] = 1.0 / mass[h]
        else:
            added[r] = added[r] + share
        wp_ = wp.where(fa > 0.0, fa + MASS_FLOOR, 0.0)        # a folded (empty) side takes no mass
        wm = wp.where(fa < 1.0, 1.0 - fa + MASS_FLOOR, 0.0)
        add_mass(ht.node[r, 0], share * wp_ / (wp_ + wm), vh, mass, inv_mass, v)
        add_mass(ht.node[r, 1], share * wm / (wp_ + wm), vh, mass, inv_mass, v)
        side = int(1)
        if (side_mask[c] >> ht.corner[e]) & 1 != 0:
            side = 0
        if hub_side[r] < 0:
            hub_side[r] = side
            particle_side[h] = wp.int8(side)
            for k in range(2):
                flags[ht.node[r, k]] = flags[ht.node[r, k]] | ACTIVE
        elif hub_side[r] != side:
            conflicts[r] = conflicts[r] + 1
    if hub_side[r] >= 0:
        weld_copies(ht, r, hub_side[r], split, front, wr, mass, spring_ke, particle_side, on_plane)
