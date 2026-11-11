# SPDX-License-Identifier: Apache-2.0
"""Neighbour lists (world grouping, order, overflow) and deterministic per-world reductions."""

from __future__ import annotations

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.solvers.sph.neighbors import NeighborLists
from alloy.solvers.sph.reduce import WorldReducer, chunk_table


def _lattice(n: int, sp: float) -> np.ndarray:
    ax = np.arange(n) * sp
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)


def test_neighbors_grouped_sorted_and_counted(device):
    sp, h = 1.0, 2.0
    g = _lattice(5, sp)
    q = np.concatenate([g, g])  # two worlds overlapping at the origin
    world = np.repeat([0, 1], len(g)).astype(np.int32)
    flags = np.full(len(q), int(ParticleFlags.ACTIVE), np.int32)
    flags[3] = 0  # an inactive particle in world 0
    phase = np.ones(len(q), np.int8)
    nl = NeighborLists(len(q), 64, h, wp.get_device(device), grid_dim=16)
    ovf = wp.zeros(2, dtype=wp.int32, device=device)
    arr = lambda x, t: wp.array(x, dtype=t, device=device)  # noqa: E731
    nl.build(arr(q, wp.vec3), arr(flags, wp.int32), arr(phase, wp.int8), arr(world, wp.int32), ovf)
    nbr, cnt = nl.nbr.numpy(), nl.count.numpy()
    for i in range(len(q)):
        row = nbr[i, : cnt[i]]
        assert np.all(np.diff(row) > 0)  # ascending, no duplicates
        assert np.all(world[row] == world[i]) and 3 not in row and i not in row
        d = np.linalg.norm(q[row] - q[i], axis=1)
        assert np.all(d < h)
    assert cnt[3] == 0
    brute = [(np.linalg.norm(g - g[i], axis=1) < h).sum() - 1 for i in range(len(g))]
    assert list(cnt[len(g):]) == brute  # world 1 untouched by world 0's inactive particle
    assert ovf.numpy().tolist() == [0, 0]
    small = NeighborLists(len(q), 8, h, wp.get_device(device), grid_dim=16)
    small.build(arr(q, wp.vec3), arr(flags, wp.int32), arr(phase, wp.int8), arr(world, wp.int32), ovf)
    assert ovf.numpy()[1] > 0 and small.count.numpy().max() == 8


def test_chunk_table_and_reducer_match_numpy(device, rng):
    start = np.array([0, 5, 5, 700, 1203])
    b, e, ws = chunk_table(start, 256)
    assert ws.tolist() == [0, 1, 1, 4, 6] and b[0] == 0 and e[-1] == 1203
    vals = rng.normal(size=1203).astype(np.float32)
    red = WorldReducer(start, wp.get_device(device))
    out = wp.zeros(4, dtype=wp.float64, device=device)
    red.sum(wp.array(vals, device=device), out)
    expect = [vals[start[w]:start[w + 1]].astype(np.float64).sum() for w in range(4)]
    assert np.allclose(out.numpy(), expect, rtol=1e-12, atol=1e-12)


def test_reducer_is_batch_invariant_bitwise(device, rng):
    # a world's sum depends only on its own values: alone or after another world, the bits are equal
    a = rng.normal(size=777).astype(np.float32)
    other = rng.normal(size=300).astype(np.float32)
    out1 = wp.zeros(1, dtype=wp.float64, device=device)
    WorldReducer(np.array([0, 777]), wp.get_device(device)).sum(wp.array(a, device=device), out1)
    out2 = wp.zeros(2, dtype=wp.float64, device=device)
    WorldReducer(np.array([0, 300, 1077]), wp.get_device(device)).sum(
        wp.array(np.concatenate([other, a]), device=device), out2)
    assert out1.numpy()[0].tobytes() == out2.numpy()[1].tobytes()


def test_reducer_max_and_sum2_match_numpy(device, rng):
    """sum / sum2 with ``out_max`` (SCR-WP8-7 L2): the per-world max (floored at 0; an empty world gives 0, 0), and a
    world over 256 chunks (the CUDA world block strides over them)."""
    start = np.array([0, 5, 5, 700, 1203, 1203 + 70_000])
    W = len(start) - 1
    vals = rng.normal(size=start[-1]).astype(np.float32)
    pair = rng.normal(size=(start[-1], 2)).astype(np.float32)
    red = WorldReducer(start, wp.get_device(device))
    out, mx = wp.zeros(W, dtype=wp.float64, device=device), wp.zeros(W, dtype=wp.float32, device=device)
    red.sum(wp.array(vals, device=device), out, mx)
    out2, mx2 = wp.zeros(W, dtype=wp.vec2d, device=device), wp.zeros(W, dtype=wp.float32, device=device)
    red.sum2(wp.array(pair, dtype=wp.vec2, device=device), out2, mx2)
    for w in range(W):
        v, q = vals[start[w]:start[w + 1]].astype(np.float64), pair[start[w]:start[w + 1]].astype(np.float64)
        assert np.isclose(out.numpy()[w], v.sum(), rtol=1e-12, atol=1e-12)
        assert mx.numpy()[w] == (max(float(v.max()), 0.0) if len(v) else 0.0)
        assert np.allclose(out2.numpy()[w], q.sum(axis=0) if len(q) else 0.0, rtol=1e-12, atol=1e-12)
        assert mx2.numpy()[w] == (max(float(q[:, 1].max()), 0.0) if len(q) else 0.0)


def test_sum2_is_batch_invariant_bitwise(device, rng):
    a = rng.normal(size=(777, 2)).astype(np.float32)
    other = rng.normal(size=(300, 2)).astype(np.float32)
    o1, m1 = wp.zeros(1, dtype=wp.vec2d, device=device), wp.zeros(1, dtype=wp.float32, device=device)
    WorldReducer(np.array([0, 777]), wp.get_device(device)).sum2(wp.array(a, dtype=wp.vec2, device=device), o1, m1)
    o2, m2 = wp.zeros(2, dtype=wp.vec2d, device=device), wp.zeros(2, dtype=wp.float32, device=device)
    WorldReducer(np.array([0, 300, 1077]), wp.get_device(device)).sum2(
        wp.array(np.concatenate([other, a]), dtype=wp.vec2, device=device), o2, m2)
    assert o1.numpy()[0].tobytes() == o2.numpy()[1].tobytes() and m1.numpy()[0] == m2.numpy()[1]
