# SPDX-License-Identifier: Apache-2.0
"""Verlet-skin reuse of the world-local neighbour lists (LIQ-9 L4a).

A superset row per particle holds every neighbour within ``h + s`` at its last build (``s`` = skin). It stays valid
while no particle of the world has moved more than ``s / 2`` since that build and the world's active set is unchanged:
then every pair now closer than ``h`` was closer than ``h + s`` at the build. Each step:

1. the world's flag is set when a particle moved more than ``s / 2`` (0.49 s, a rounding margin), changed group
   (poured, retired, reset) or its world was forced (below); the flagged worlds' superset rows are rebuilt from a
   grid at radius ``h + s`` (other worlds keep theirs). On CPU, ``wp.capture_if`` skips the grid and the build when no
   world is flagged. On CUDA the grid is rebuilt every step and the build kernel returns at once for unflagged
   worlds: under ``wp.capture_if`` the grid's sort (temporary allocations) made graph instantiation fail ("invalid
   argument", T4, r66);
2. a filter pass writes the rows the solver reads: the superset entries now closer than ``h``, in the row's ascending
   order, the first ``capacity`` of them, with the same distance test as a plain build (``d = q[j] - q[i]``,
   ``dot(d, d) < h * h``). So the rows, counts and overflow counts equal a plain build at these positions, bit for bit.

The superset capacity is sized from the rest fill (``skin_capacity``), not from ``neighbor_capacity``. A row that
overflows it is written as a plain radius-h row for this step (the filter skips it) and its world rebuilds next step,
so a crowded world loses the reuse, never exactness. The flags are per world, so a world's rows depend on its own
history only (D9).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from .neighbors import _get_local, _insert_sorted_local, _put_local


@wp.kernel(enable_backward=False)
def _seed_kernel(force: wp.array[wp.int32], flag: wp.array[wp.int32], any_flag: wp.array[wp.int32]):
    w = wp.tid()
    f = force[w]
    flag[w] = f
    force[w] = 0
    if f != 0:
        any_flag[0] = 1


@wp.kernel(enable_backward=False)
def _check_kernel(q: wp.array[wp.vec3], group: wp.array[wp.int32], world: wp.array[wp.int32],
                  q_ref: wp.array[wp.vec3], g_ref: wp.array[wp.int32], half2: float, flag: wp.array[wp.int32],
                  any_flag: wp.array[wp.int32]):
    i = wp.tid()
    w = world[i]
    if w < 0:
        return
    g = group[i]
    moved = g != g_ref[i]
    if not moved and g >= 0:
        d = q[i] - q_ref[i]
        moved = not (wp.dot(d, d) <= half2)  # NaN moves
    if moved:
        flag[w] = 1
        any_flag[0] = 1


@wp.kernel(enable_backward=False)
def _build_skin_kernel(
    grid: wp.uint64, q: wp.array[wp.vec3], group: wp.array[wp.int32], world: wp.array[wp.int32],
    world_start: wp.array[wp.int32], flag: wp.array[wp.int32], h: float, hs: float, cap: int,
    sup: wp.array2d[wp.uint16], sup_hi: wp.array2d[wp.uint8], wide: int, sup_count: wp.array[wp.int32],
    q_ref: wp.array[wp.vec3], g_ref: wp.array[wp.int32], force: wp.array[wp.int32], cap_row: int,
    nbr: wp.array2d[wp.uint16], hi: wp.array2d[wp.uint8], count: wp.array[wp.int32],
    overflow_world: wp.array[wp.int32],
):
    t = wp.tid()
    i = wp.hash_grid_point_id(grid, t)
    w = world[i]
    if w < 0 or flag[w] == 0:
        return
    g = group[i]
    xi = q[i]
    q_ref[i] = xi
    g_ref[i] = g
    if g < 0:
        sup_count[i] = 0
        return
    base = world_start[g]
    hs2 = hs * hs
    query = wp.hash_grid_query(grid, xi, hs, g)
    j = int(0)
    n = int(0)
    total = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            d = q[j] - xi
            if wp.dot(d, d) < hs2:
                total += 1
                n = _insert_sorted_local(sup, sup_hi, wide, i, n, cap, j - base)
    if total <= cap:
        sup_count[i] = n
        return
    # skin row too full: this step's row is a plain build (the filter skips it) and the world rebuilds next step
    h2 = h * h
    m = int(0)
    total_h = int(0)
    query = wp.hash_grid_query(grid, xi, h, g)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            d = q[j] - xi
            if wp.dot(d, d) < h2:
                total_h += 1
                m = _insert_sorted_local(nbr, hi, wide, i, m, cap_row, j - base)
    if total_h > cap_row:
        wp.atomic_add(overflow_world, g, 1)
    count[i] = m
    sup_count[i] = -1
    force[g] = 1


@wp.kernel(enable_backward=False)
def _filter_kernel(
    q: wp.array[wp.vec3], group: wp.array[wp.int32], world_start: wp.array[wp.int32], h: float, cap: int,
    sup: wp.array2d[wp.uint16], sup_hi: wp.array2d[wp.uint8], wide: int, sup_count: wp.array[wp.int32],
    nbr: wp.array2d[wp.uint16], hi: wp.array2d[wp.uint8], count: wp.array[wp.int32],
    overflow_world: wp.array[wp.int32],
):
    i = wp.tid()
    g = group[i]
    if g < 0:
        count[i] = 0
        return
    if sup_count[i] < 0:  # a plain row written by this step's rebuild
        return
    base = world_start[g]
    xi = q[i]
    h2 = h * h
    n = int(0)
    total = int(0)
    for k in range(sup_count[i]):
        jl = _get_local(sup, sup_hi, wide, k, i)
        d = q[base + jl] - xi
        if wp.dot(d, d) < h2:
            total += 1
            if n < cap:
                _put_local(nbr, hi, wide, n, i, jl)
                n += 1
    if total > cap:
        wp.atomic_add(overflow_world, g, 1)
    count[i] = n


SKIN_FILL_MARGIN = 1.75  # superset capacity / interior lattice neighbours within h + s (CPU scenes at skin 0.1: max fill 53 of 56)


def skin_capacity(h: float, spacing: float, skin: float) -> int:
    """Superset row capacity: 1.75 x the neighbours of an interior lattice particle within ``h + skin`` (the rest
    fill; a fuller row is built plain and its world rebuilds next step, so a small capacity costs reuse, never
    exactness), at least 16."""
    rad = (h + skin) / spacing
    n = int(math.ceil(rad)) + 1
    ax = np.arange(-n, n + 1, dtype=np.float64)
    d2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    fill = int(np.count_nonzero(d2 < rad * rad)) - 1
    return max(16, int(math.ceil(SKIN_FILL_MARGIN * fill)))


class SkinState:
    """Superset rows, reference positions and per-world flags of one ``NeighborLists`` (module docstring)."""

    def __init__(self, lists, skin: float, world_count: int, capacity: int):
        n, dev = max(lists.n, 1), lists.device
        self.skin = float(skin)
        self.capacity = int(capacity)
        self.radius = lists.h + self.skin
        self.half2 = (0.49 * self.skin) ** 2
        self.sup = wp.zeros((self.capacity, n), dtype=wp.uint16, device=dev)
        self.sup_hi = wp.zeros((self.capacity, n) if lists.wide else (1, 1), dtype=wp.uint8, device=dev)
        self.sup_count = wp.zeros(n, dtype=wp.int32, device=dev)
        self.q_ref = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.g_ref = wp.full(n, -2, dtype=wp.int32, device=dev)  # never built: every world builds first
        W = max(int(world_count), 1)
        self.flag = wp.zeros(W, dtype=wp.int32, device=dev)
        self.force = wp.zeros(W, dtype=wp.int32, device=dev)
        self.any = wp.zeros(1, dtype=wp.int32, device=dev)
        self.builds = 0  # host count of rebuild launches (CPU: rebuilds; CUDA: every update)

    def update(self, lists, q: wp.array, world: wp.array, overflow_world: wp.array) -> None:
        """Steps 1-2 of the module docstring; ``lists.group`` is current (capture-safe)."""
        dev, n = lists.device, lists.n
        self.any.zero_()
        wp.launch(_seed_kernel, dim=self.flag.shape[0], inputs=[self.force, self.flag, self.any], device=dev)
        wp.launch(_check_kernel, dim=n, inputs=[q, lists.group, world, self.q_ref, self.g_ref, self.half2, self.flag,
                                                self.any], device=dev)

        def rebuild():
            lists.desc.build(q, self.radius, groups=lists.group)
            wp.launch(_build_skin_kernel, dim=n,
                      inputs=[lists.grid.id, q, lists.group, world, lists.base, self.flag, lists.h, self.radius,
                              self.capacity, self.sup, self.sup_hi, lists.wide, self.sup_count, self.q_ref, self.g_ref,
                              self.force, lists.capacity, lists.nbr, lists.hi, lists.count, overflow_world], device=dev)
            self.builds += 1

        if wp.get_device(dev).is_cuda:
            rebuild()  # the grid sort allocates temporaries, which a conditional graph node cannot hold (r66)
        else:
            wp.capture_if(self.any, on_true=rebuild)
        wp.launch(_filter_kernel, dim=n,
                  inputs=[q, lists.group, lists.base, lists.h, lists.capacity, self.sup, self.sup_hi, lists.wide,
                          self.sup_count, lists.nbr, lists.hi, lists.count, overflow_world], device=dev)

    def rebuilt_worlds(self) -> np.ndarray:
        """Host copy of the last update's per-world rebuild flags (tests, diagnostics)."""
        return self.flag.numpy().astype(bool)
