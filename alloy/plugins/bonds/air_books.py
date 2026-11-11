# SPDX-License-Identifier: Apache-2.0
"""The relabel of the trapped-air pockets after each film step, with exact gas books (WP13 must-fixes 3 and 5).

Order (``air.py``): label the free vertices (``pockets.label``), take their gaps at the step's end, then
1. ``survivors``: per old slot, the volume of all its vertices and of those still free. A vertex that bonded takes
   its volume's share of the air with it, booked vanished: it bonded in contact, where its air is roughness air
   (``air_kernels.held``), which the adhesive takes up. So a pocket's pressure does not jump as its rim bonds, and
   bonds closing in from the edges do not sweep the label's roughness air into a false bubble. An enclosed pocket
   that holds air cannot bond (``block_trapped``);
2. ``gather``: each free vertex brings its old slot's air in proportion to its volume (a split shares the air at a
   common pressure, a merge sums it); a vertex new to every pocket draws P_atm·v from the atmosphere (``formed``).
   An open pocket's throat sums w_b·h³/(12·mu·s_b) over its edge vertices, h at least the pocket's mean gap;
3. ``select``: every pocket's area from the labels (the trapped-area reading and the overflow count use all of
   them); the ``SLOTS`` largest get a slot, the air of the rest is booked ``untracked``;
4. ``assign``: a pocket keeps the slot its root held (and its felt pressure), the others take free slots.
Books per world: sum of slot amounts + vented + vanished + untracked - formed = 0, exactly up to float64 rounding.
"""

from __future__ import annotations

import warp as wp

from .air_kernels import H_FLOOR, P_ATM, SLOTS, Air, Books, Scratch, held


@wp.kernel
def survivors(air: Air, sc: Scratch, bk: Books):
    """Per old slot: the volume of all its vertices (``vsum``) and of those still free (``csum``, reused). The air of
    the vertices that left (bonded) leaves with them: booked vanished (at contact it is roughness air)."""
    w = wp.tid()
    for k in range(SLOTS):
        air.vsum[w, k] = wp.float64(0.0)
        air.csum[w, k] = wp.float64(0.0)
    for i in range(air.start[w], air.start[w + 1]):
        k = air.slot[i]
        if k >= 0:
            v = wp.float64(air.area[i]) * wp.float64(air.gap[i])
            air.vsum[w, k] = air.vsum[w, k] + v
            if sc.parent[i] >= 0:
                air.csum[w, k] = air.csum[w, k] + v
    for k in range(SLOTS):
        if air.root[w, k] >= 0:
            gone = air.amount[w, k]
            if air.vsum[w, k] > wp.float64(0.0):
                gone = air.amount[w, k] * (air.vsum[w, k] - air.csum[w, k]) / air.vsum[w, k]
            bk.vanished[w] = bk.vanished[w] + gone


@wp.kernel
def clear_roots(sc: Scratch):
    i = wp.tid()
    sc.amount[i] = wp.float64(0.0)
    sc.area[i] = wp.float64(0.0)
    sc.vol[i] = wp.float64(0.0)
    sc.lam[i] = wp.float64(0.0)
    sc.open_[i] = 0
    sc.slot[i] = -1


@wp.kernel
def gather(air: Air, sc: Scratch, bk: Books):
    w = wp.tid()
    for i in range(air.start[w], air.start[w + 1]):
        r = sc.parent[i]
        if r < 0:
            continue
        a = wp.float64(air.area[i])
        h = wp.float64(air.gap[i])
        v = a * h
        k = air.slot[i]
        share = P_ATM * v
        if k >= 0 and air.vsum[w, k] > wp.float64(0.0):
            share = air.amount[w, k] * v / air.vsum[w, k]
        else:
            bk.formed[w] = bk.formed[w] + share
        sc.amount[r] = sc.amount[r] + share
        sc.area[r] = sc.area[r] + a
        sc.vol[r] = sc.vol[r] + v
        if air.throat[i] > wp.float64(0.0):
            sc.open_[r] = 1
    for i in range(air.start[w], air.start[w + 1]):         # the throats, at least the pocket's mean gap
        r = sc.parent[i]
        if r >= 0 and air.throat[i] > wp.float64(0.0):
            h = wp.max(wp.float64(air.gap[i]), sc.vol[r] / sc.area[r])
            sc.lam[r] = sc.lam[r] + air.throat[i] * h * h * h


@wp.func
def insert_top(sc: Scratch, w: int, r: int, a: wp.float64):
    """Insert pocket ``r`` of area ``a`` into world ``w``'s descending top-SLOTS list (ties: the earlier root)."""
    j = SLOTS
    for m in range(SLOTS):
        if j == SLOTS and a > sc.sel_area[w, m]:
            j = m
    if j == SLOTS:
        return
    for m in range(SLOTS - 1, j, -1):
        sc.sel_area[w, m] = sc.sel_area[w, m - 1]
        sc.sel_root[w, m] = sc.sel_root[w, m - 1]
    sc.sel_area[w, j] = a
    sc.sel_root[w, j] = r


@wp.func
def chosen(sc: Scratch, w: int, r: int) -> bool:
    for m in range(SLOTS):
        if sc.sel_root[w, m] == r:
            return True
    return False


@wp.kernel
def select(air: Air, sc: Scratch, bk: Books):
    w = wp.tid()
    for k in range(SLOTS):
        sc.sel_area[w, k] = wp.float64(-1.0)
        sc.sel_root[w, k] = -1
    count = wp.float64(0.0)
    trapped = wp.float64(0.0)
    for i in range(air.start[w], air.start[w + 1]):
        if sc.parent[i] != i:
            continue
        count = count + wp.float64(1.0)
        if sc.open_[i] == 0 and held(sc.amount[i], sc.area[i], air.rough) > wp.float64(0.0):
            trapped = wp.max(trapped, sc.area[i])
        insert_top(sc, w, i, sc.area[i])
    over = wp.float64(0.0)
    for i in range(air.start[w], air.start[w + 1]):
        if sc.parent[i] == i and not chosen(sc, w, i):
            over = over + wp.float64(1.0)
            bk.untracked[w] = bk.untracked[w] + sc.amount[i]
    bk.pockets[w] = count
    bk.trapped[w] = trapped
    bk.overflow[w] = over


@wp.func
def fill_slot(air: Air, sc: Scratch, w: int, k: int, lo: int):
    r = sc.new_root[w, k]
    if r < 0:
        air.root[w, k] = -1
        air.amount[w, k] = wp.float64(0.0)
        air.p[w, k] = wp.float64(0.0)
        air.area_k[w, k] = wp.float64(0.0)
        air.volume[w, k] = wp.float64(0.0)
        air.lam[w, k] = wp.float64(0.0)
        air.enclosed[w, k] = 0
        return
    v = wp.max(sc.vol[r], sc.area[r] * H_FLOOR)
    if air.root[w, k] != r - lo:
        air.p[w, k] = sc.amount[r] / v - P_ATM               # a new pocket starts at its Boyle pressure
    air.root[w, k] = r - lo
    air.amount[w, k] = sc.amount[r]
    air.area_k[w, k] = sc.area[r]
    air.volume[w, k] = v
    air.lam[w, k] = sc.lam[r]
    air.enclosed[w, k] = 1 - sc.open_[r]


@wp.kernel
def assign(air: Air, sc: Scratch):
    """Slots for the selected pockets: the slot its root held first, then free slots; then each slot's arrays."""
    w = wp.tid()
    lo = air.start[w]
    for k in range(SLOTS):
        sc.new_root[w, k] = -1
    for m in range(SLOTS):
        r = sc.sel_root[w, m]
        if r >= 0:
            for k in range(SLOTS):
                if sc.slot[r] < 0 and sc.new_root[w, k] < 0 and air.root[w, k] == r - lo:
                    sc.new_root[w, k] = r
                    sc.slot[r] = k
    for m in range(SLOTS):
        r = sc.sel_root[w, m]
        if r >= 0:
            for k in range(SLOTS):
                if sc.slot[r] < 0 and sc.new_root[w, k] < 0:
                    sc.new_root[w, k] = r
                    sc.slot[r] = k
    for k in range(SLOTS):
        fill_slot(air, sc, w, k, lo)


@wp.kernel
def vertex_slots(air: Air, sc: Scratch):
    i = wp.tid()
    r = sc.parent[i]
    air.slot[i] = -1
    if r >= 0:
        air.slot[i] = sc.slot[r]


__all__ = ["assign", "clear_roots", "gather", "select", "survivors", "vertex_slots"]
