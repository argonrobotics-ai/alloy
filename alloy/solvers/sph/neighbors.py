# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity neighbour lists from a world-grouped Warp hash grid.

Each active SPH particle gets up to ``capacity`` neighbours within the support radius ``h``, stored in ascending
particle-id order. Sorting makes every neighbour sum independent of the hash grid's internal visit order, and since
worlds keep their relative particle order when batched, a world's sums are the same alone or in a pack (D9). On
overflow the ``capacity`` lowest ids are kept and the world's overflow counter is incremented (integer atomics only).

Grouping: ``group = world`` for active SPH particles and -1 otherwise, so queries never cross worlds even though
every world sits at the origin (F1), and spares and foreign particles are invisible
(warp@1.18.0:warp/_src/types.py:8025-8090, HashGrid.build(groups=)).

World-local rows (LIQ-9 L4b, ``world_start`` given; SolverDFSPH's mode): a neighbour j of particle i is stored as its
offset from i's world start, ``j - world_start[world]``, in uint16, column-major ``nbr[k, i]`` (a warp's 32 threads
read 32 adjacent entries). Half the bytes of the int32 rows and coalesced on CUDA; same ids in the same order, so every
sum is unchanged bit for bit. A world over 65,536 particles adds a uint8 high byte per entry (``wide``; 24-bit
offsets). Readers decode with ``pressure.nbr_j``. Without ``world_start`` the lists stay global int32 rows
``nbr[i, k]`` (other users, e.g. alloy.plugins.thermal).
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from .grid_descriptor import GridDescriptor

log = logging.getLogger("alloy.sph")

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.kernel(enable_backward=False)
def _set_groups_kernel(
    flags: wp.array[wp.int32],
    phase: wp.array[wp.int8],
    world: wp.array[wp.int32],
    wall: wp.array[wp.int32],
    has_wall: int,
    group: wp.array[wp.int32],
):
    i = wp.tid()
    g = int(-1)
    if (flags[i] & _ACTIVE) != 0 and int(phase[i]) >= 1:
        g = world[i]
    elif has_wall != 0 and wall[i] != 0 and (flags[i] & _ACTIVE) != 0 and world[i] >= 0:
        g = -2 - world[i]  # an active particle wall (particle_walls.py): its own group, never a fluid neighbour
    group[i] = g


@wp.func
def _insert_sorted(nbr: wp.array2d[wp.int32], i: int, n: int, cap: int, j: int) -> int:
    """Insert j into the ascending row nbr[i, :n] (capacity cap); returns the new length (duplicates skipped)."""
    pos = n
    k = n - 1
    while k >= 0:
        v = nbr[i, k]
        if v == j:
            return n
        if v < j:
            break
        pos = k
        k -= 1
    if pos >= cap:
        return n
    last = n
    if n >= cap:
        last = cap - 1
    k = last
    while k > pos:
        nbr[i, k] = nbr[i, k - 1]
        k -= 1
    nbr[i, pos] = j
    if n < cap:
        return n + 1
    return n


@wp.kernel(enable_backward=False)
def _build_lists_kernel(
    grid: wp.uint64,
    q: wp.array[wp.vec3],
    group: wp.array[wp.int32],
    h: float,
    cap: int,
    nbr: wp.array2d[wp.int32],
    count: wp.array[wp.int32],
    overflow_world: wp.array[wp.int32],
):
    t = wp.tid()
    i = wp.hash_grid_point_id(grid, t)
    g = group[i]
    if g < 0:
        count[i] = 0
        return
    xi = q[i]
    h2 = h * h
    query = wp.hash_grid_query(grid, xi, h, g)
    j = int(0)
    n = int(0)
    total = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            d = q[j] - xi
            if wp.dot(d, d) < h2:
                total += 1
                n = _insert_sorted(nbr, i, n, cap, j)
    if total > cap:
        wp.atomic_add(overflow_world, g, 1)
    count[i] = n


@wp.func
def _get_local(nbr: wp.array2d[wp.uint16], hi: wp.array2d[wp.uint8], wide: int, k: int, i: int) -> int:
    v = wp.int32(nbr[k, i])
    if wide != 0:
        v = v + (wp.int32(hi[k, i]) << 16)
    return v


@wp.func
def _put_local(nbr: wp.array2d[wp.uint16], hi: wp.array2d[wp.uint8], wide: int, k: int, i: int, v: int):
    nbr[k, i] = wp.uint16(v & 65535)
    if wide != 0:
        hi[k, i] = wp.uint8(v >> 16)


@wp.func
def _insert_sorted_local(nbr: wp.array2d[wp.uint16], hi: wp.array2d[wp.uint8], wide: int, i: int, n: int, cap: int,
                         j: int) -> int:
    """``_insert_sorted`` on the column-major world-local rows."""
    pos = n
    k = n - 1
    while k >= 0:
        v = _get_local(nbr, hi, wide, k, i)
        if v == j:
            return n
        if v < j:
            break
        pos = k
        k -= 1
    if pos >= cap:
        return n
    last = n
    if n >= cap:
        last = cap - 1
    k = last
    while k > pos:
        _put_local(nbr, hi, wide, k, i, _get_local(nbr, hi, wide, k - 1, i))
        k -= 1
    _put_local(nbr, hi, wide, pos, i, j)
    if n < cap:
        return n + 1
    return n


@wp.kernel(enable_backward=False)
def _build_local_kernel(
    grid: wp.uint64,
    q: wp.array[wp.vec3],
    group: wp.array[wp.int32],
    world_start: wp.array[wp.int32],
    h: float,
    cap: int,
    nbr: wp.array2d[wp.uint16],
    hi: wp.array2d[wp.uint8],
    wide: int,
    count: wp.array[wp.int32],
    overflow_world: wp.array[wp.int32],
):
    t = wp.tid()
    i = wp.hash_grid_point_id(grid, t)
    g = group[i]
    if g < 0:
        count[i] = 0
        return
    base = world_start[g]
    xi = q[i]
    h2 = h * h
    query = wp.hash_grid_query(grid, xi, h, g)
    j = int(0)
    n = int(0)
    total = int(0)
    while wp.hash_grid_query_next(query, j):
        if j != i:
            d = q[j] - xi
            if wp.dot(d, d) < h2:
                total += 1
                n = _insert_sorted_local(nbr, hi, wide, i, n, cap, j - base)
    if total > cap:
        wp.atomic_add(overflow_world, g, 1)
    count[i] = n


LOCAL_LIMIT = 65536  # particles per world the uint16 offsets cover; above it the high byte is used


class NeighborLists:
    """Grid + sorted fixed-K neighbour lists for one pack of SPH particles. With ``world_start`` (W + 1 contiguous
    world ranges) the rows are world-local uint16, column-major ``nbr[k, i]`` (module docstring); without it, global
    int32 rows ``nbr[i, k]``. ``local_limit``: world size above which a uint8 high byte is added (tests lower it).
    ``skin`` > 0 (a length; world-local rows only): Verlet-skin reuse (``neighbors_skin``); ``spacing`` sizes its rows,
    ``skin_rows`` overrides that capacity (tests)."""

    def __init__(self, n_particles: int, capacity: int, h: float, device: wp.context.Device, grid_dim: int = 128,
                 world_start: np.ndarray | None = None, local_limit: int = LOCAL_LIMIT, skin: float = 0.0,
                 spacing: float | None = None, skin_rows: int | None = None):
        if capacity < 8:
            raise ValueError(f"neighbor_capacity must be >= 8, got {capacity}")
        self.h = float(h)
        self.capacity = int(capacity)
        self.device = device
        self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device)
        self.grid.reserve(max(n_particles, 1), with_groups=True)
        self.desc = GridDescriptor(self.grid)    # graph replays keep this grid's descriptor (grid_descriptor.py)
        rows = max(n_particles, 1)
        self.local = world_start is not None
        if self.local:
            ws = np.asarray(world_start, dtype=np.int64)
            self.wide = int(len(ws) > 1 and int(np.max(np.diff(ws))) > int(local_limit))
            self.nbr = wp.zeros((capacity, rows), dtype=wp.uint16, device=device)
            self.hi = wp.zeros((capacity, rows) if self.wide else (1, 1), dtype=wp.uint8, device=device)
            self.base = wp.array(ws[:-1].astype(np.int32) if len(ws) > 1 else np.zeros(1, np.int32), dtype=wp.int32,
                                 device=device)
            if self.wide:
                log.info("NeighborLists: a world has %d particles (> %d): 24-bit local offsets",
                         int(np.max(np.diff(ws))), local_limit)
        else:
            self.wide = 0
            self.nbr = wp.zeros((rows, capacity), dtype=wp.int32, device=device)
            self.hi = self.base = None
        self.count = wp.zeros(rows, dtype=wp.int32, device=device)
        self.group = wp.zeros(rows, dtype=wp.int32, device=device)
        self.wall = wp.zeros(1, dtype=wp.int32, device=device)  # set_walls: particle walls in group -2 - world
        self.has_wall = 0
        self.n = int(n_particles)
        self.skin = None  # neighbors_skin.SkinState: Verlet-skin reuse (L4a), world-local rows only
        if skin > 0.0:
            if not self.local:
                raise ValueError("NeighborLists: a skin needs world-local rows (world_start)")
            from .neighbors_skin import SkinState, skin_capacity  # noqa: PLC0415

            cap = skin_rows or skin_capacity(self.h, spacing if spacing else self.h / 2.0, skin)
            self.skin = SkinState(self, skin, len(world_start) - 1, cap)

    def set_walls(self, mask: wp.array) -> None:
        """Put the particles with ``mask`` 1 into the grid as particle walls (group -2 - world; particle_walls.py).
        The Verlet skin keeps a stale grid between rebuilds, so it is refused with walls."""
        if self.skin is not None:
            raise ValueError("SolverDFSPH: particle walls need the grid rebuilt every step (reneighbor_skin off)")
        self.wall, self.has_wall = mask, 1

    def build(
        self,
        q: wp.array,
        flags: wp.array,
        phase: wp.array,
        world: wp.array,
        overflow_world: wp.array,
    ) -> None:
        """Rebuild the grid and the lists from positions ``q`` (capture-safe; no host sync)."""
        if self.n == 0:
            return
        wp.launch(_set_groups_kernel, dim=self.n, inputs=[flags, phase, world, self.wall, self.has_wall, self.group],
                  device=self.device)
        if self.skin is not None:
            self.skin.update(self, q, world, overflow_world)
            return
        self.desc.build(q, self.h, groups=self.group)
        if self.local:
            ins = [self.grid.id, q, self.group, self.base, self.h, self.capacity, self.nbr, self.hi, self.wide,
                   self.count, overflow_world]
            wp.launch(_build_local_kernel, dim=self.n, inputs=ins, device=self.device)
            return
        wp.launch(
            _build_lists_kernel,
            dim=self.n,
            inputs=[self.grid.id, q, self.group, self.h, self.capacity, self.nbr, self.count, overflow_world],
            device=self.device,
        )

    def global_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Host copy (tests, diagnostics): global neighbour ids as [n, capacity] int64 rows (entries past
        ``count`` are undefined) and the counts."""
        cnt = self.count.numpy()[: self.n]
        if not self.local:
            return self.nbr.numpy()[: self.n].astype(np.int64), cnt
        off = self.nbr.numpy()[:, : self.n].T.astype(np.int64)
        if self.wide:
            off += self.hi.numpy()[:, : self.n].T.astype(np.int64) << 16
        base = self.base.numpy().astype(np.int64)
        return off + base[np.clip(self.group.numpy()[: self.n], 0, None)][:, None], cnt
