# SPDX-License-Identifier: Apache-2.0
"""Boundary slot rows: insert a shape into a particle's phi-ascending row of NB = 4 slots (boundary.py's slot
search). Split from boundary.py."""

from __future__ import annotations

import warp as wp

COS_SAME = 0.85  # walls whose normals are within about 32 deg face the same way (one wall) when the slots are full
_COS_SAME = wp.constant(COS_SAME)


@wp.func
def _slot_insert(slots_shape: wp.array2d[wp.int32], slots_phi: wp.array2d[wp.float32],
                 slots_face: wp.array2d[wp.int32], i: int, n: int, s: int, phi: float, face: int) -> int:
    """Insert (s, phi, face) into the phi-ascending slot row; returns the slot index used, or -1 if dropped."""
    pos = n
    k = n - 1
    while k >= 0:
        if slots_phi[i, k] <= phi:
            break
        pos = k
        k -= 1
    if pos >= 4:
        return -1
    last = n
    if n >= 4:
        last = 3
    k = last
    while k > pos:
        slots_shape[i, k] = slots_shape[i, k - 1]
        slots_phi[i, k] = slots_phi[i, k - 1]
        slots_face[i, k] = slots_face[i, k - 1]
        k -= 1
    slots_shape[i, pos] = s
    slots_phi[i, pos] = phi
    slots_face[i, pos] = face
    return pos


@wp.func
def _slot_copy(slots_shape: wp.array2d[wp.int32], slots_phi: wp.array2d[wp.float32],
               slots_face: wp.array2d[wp.int32], slots_nrm: wp.array2d[wp.vec3], i: int, src: int, dst: int):
    slots_shape[i, dst] = slots_shape[i, src]
    slots_phi[i, dst] = slots_phi[i, src]
    slots_face[i, dst] = slots_face[i, src]
    slots_nrm[i, dst] = slots_nrm[i, src]


@wp.func
def _shape_insert(slots_shape: wp.array2d[wp.int32], slots_phi: wp.array2d[wp.float32],
                  slots_face: wp.array2d[wp.int32], slots_nrm: wp.array2d[wp.vec3], i: int, n: int, s: int,
                  phi: float, nrm: wp.vec3, near: float) -> int:
    """Insert a shape (s, phi, world normal nrm) into the phi-ascending row of n slots; 1 if kept (below 4 slots as
    ``_slot_insert``). A full row drops the farthest duplicate of its 4 slots and the new wall: one with a nearer wall
    facing its way (cos > COS_SAME), never one nearer than ``near`` (a CFL step); with none, the farthest of the five.
    A narrow cup's slabs put 5-6 within h: the nearest four lost a corner's floor, and dropping each new same-facing
    slab lost an inner vertex's second slab, which a particle then crossed."""
    if n < 4:
        r = _slot_insert(slots_shape, slots_phi, slots_face, i, n, s, phi, -1)
        if r >= 0:
            k = n
            while k > r:
                slots_nrm[i, k] = slots_nrm[i, k - 1]
                k -= 1
            slots_nrm[i, r] = nrm
            return 1
        return 0
    ev = int(-1)  # the farthest duplicate: 0-3 a slot, 4 the new wall
    ev_phi = float(-1.0)
    for m in range(4):
        if phi >= near and slots_phi[i, m] <= phi and wp.dot(slots_nrm[i, m], nrm) > _COS_SAME:
            ev = 4
            ev_phi = phi
    for k in range(4):
        pk = slots_phi[i, k]
        dup = int(0)
        for m in range(4):
            if m != k and slots_phi[i, m] <= pk and wp.dot(slots_nrm[i, m], slots_nrm[i, k]) > _COS_SAME:
                dup = 1
        if phi <= pk and wp.dot(nrm, slots_nrm[i, k]) > _COS_SAME:
            dup = 1
        if dup == 1 and pk >= near and pk > ev_phi:
            ev = k
            ev_phi = pk
    if ev < 0 and phi < slots_phi[i, 3]:
        ev = 3
    if ev < 0 or ev == 4:
        return 0
    k = ev
    while k < 3:
        _slot_copy(slots_shape, slots_phi, slots_face, slots_nrm, i, k + 1, k)
        k += 1
    pos = int(3)
    while pos > 0 and slots_phi[i, pos - 1] > phi:
        _slot_copy(slots_shape, slots_phi, slots_face, slots_nrm, i, pos - 1, pos)
        pos -= 1
    slots_shape[i, pos] = s
    slots_phi[i, pos] = phi
    slots_face[i, pos] = -1
    slots_nrm[i, pos] = nrm
    return 1


__all__ = ["COS_SAME", "_shape_insert", "_slot_insert"]
