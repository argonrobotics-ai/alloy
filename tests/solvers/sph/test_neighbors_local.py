# SPDX-License-Identifier: Apache-2.0
"""World-local uint16 neighbour rows (LIQ-9 L4b): the same neighbours in the same order as the global int32 rows."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.solvers.sph.neighbors import NeighborLists


def _pack(rng, sizes: list[int], sp: float = 1.0) -> tuple[np.ndarray, ...]:
    """Worlds of jittered lattice points, all at the origin (they overlap in space), contiguous ids per world."""
    qs, ws = [], []
    for w, n in enumerate(sizes):
        side = int(np.ceil(n ** (1.0 / 3.0)))
        ax = np.arange(side) * sp
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)[:n]
        qs.append(g + rng.uniform(-0.2, 0.2, g.shape) * sp)
        ws.append(np.full(n, w, np.int32))
    q, world = np.concatenate(qs).astype(np.float32), np.concatenate(ws)
    flags = np.full(len(q), int(ParticleFlags.ACTIVE), np.int32)
    flags[::17] = 0                                   # some inactive particles
    start = np.concatenate([[0], np.cumsum(sizes)])
    return q, world, flags, np.ones(len(q), np.int8), start


def _build(device, q, world, flags, phase, cap, **kw):
    arr = lambda x, t: wp.array(x, dtype=t, device=device)  # noqa: E731
    nl = NeighborLists(len(q), cap, 1.9, wp.get_device(device), grid_dim=16, **kw)
    ovf = wp.zeros(int(world.max()) + 1, dtype=wp.int32, device=device)
    nl.build(arr(q, wp.vec3), arr(flags, wp.int32), arr(phase, wp.int8), arr(world, wp.int32), ovf)
    return nl, ovf.numpy()


@pytest.mark.parametrize("limit", [65536, 40])     # 40: worlds above it use the high byte (24-bit offsets)
@pytest.mark.parametrize("cap", [64, 8])            # 8: rows overflow and keep the lowest ids
def test_local_rows_equal_global_rows(device, rng, limit, cap):
    q, world, flags, phase, start = _pack(rng, [64, 27, 125])
    ref, ovf_ref = _build(device, q, world, flags, phase, cap)
    loc, ovf_loc = _build(device, q, world, flags, phase, cap, world_start=start, local_limit=limit)
    assert loc.nbr.dtype == wp.uint16 and loc.nbr.shape == (cap, len(q))       # column-major [k, i]
    assert loc.wide == int(limit < 125)
    r_ids, r_cnt = ref.global_rows()
    l_ids, l_cnt = loc.global_rows()
    assert np.array_equal(r_cnt, l_cnt) and np.array_equal(ovf_ref, ovf_loc)
    assert r_cnt.max() > 0 and (cap > 8 or ovf_ref.sum() > 0)
    for i in range(len(q)):
        assert np.array_equal(r_ids[i, : r_cnt[i]], l_ids[i, : l_cnt[i]])


def test_local_rows_halve_the_list_bytes(device):
    n, cap = 1000, 64
    glob = NeighborLists(n, cap, 1.0, wp.get_device(device), grid_dim=8)
    loc = NeighborLists(n, cap, 1.0, wp.get_device(device), grid_dim=8, world_start=np.array([0, 500, 1000]))
    assert loc.nbr.capacity * 2 == glob.nbr.capacity and loc.hi.size == 1
