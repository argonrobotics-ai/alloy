# SPDX-License-Identifier: Apache-2.0
"""Per-world grouping (setup tables of digests, health and frames): same result as per-world masks, linear cost."""

from __future__ import annotations

import time

import numpy as np
import pytest

from alloy.precision.digest import world_items
from alloy.precision.world_groups import equal_size_blocks, group_by_world, world_reduce, world_sum


def _naive_items(item_world: np.ndarray, W: int, include_global: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    glob = np.flatnonzero(item_world < 0) if include_global else np.zeros(0, dtype=np.int64)
    lists = [np.concatenate([glob, np.flatnonzero(item_world == w)]) for w in range(W)]
    sizes = np.array([len(x) for x in lists], dtype=np.int64)
    end = np.cumsum(sizes)
    flat = np.concatenate(lists) if lists else np.zeros(0, dtype=np.int64)
    return end - sizes, end, flat


@pytest.mark.parametrize("include_global", [True, False])
def test_world_items_match_per_world_masks(include_global: bool) -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        W = int(rng.integers(1, 30))
        iw = rng.integers(-2, W + 2, size=int(rng.integers(0, 120)))
        got = world_items(iw, W, "cpu", include_global)
        want = _naive_items(iw, W, include_global)
        for arr, ref in zip((got.begin, got.end, got.items), want):
            assert np.array_equal(arr.numpy(), ref)


def test_reduce_sum_and_blocks_match_masks() -> None:
    rng = np.random.default_rng(2)
    W, n = 17, 400
    iw = rng.integers(-1, W + 1, size=n)
    x = rng.standard_normal((n, 3)).astype(np.float32)
    g = group_by_world(iw, W)
    lo = world_reduce(g, x, np.minimum, np.inf)
    for w in range(W):
        mine = x[iw == w]
        assert np.array_equal(x[g.order[g.begin[w]:g.end[w]]], mine)
        assert np.array_equal(lo[w], mine.min(axis=0) if len(mine) else np.full(3, np.inf, np.float32))
    assert np.allclose(world_sum(iw, x[:, 0], W), [x[iw == w, 0].sum(dtype=np.float64) for w in range(W)])
    size = int(np.bincount(iw[(iw >= 0) & (iw < W)], minlength=W).max())
    seen = [int(w) for ids, idx in equal_size_blocks(g, size, max_rows=size) for w in ids]
    assert seen == [w for w in range(W) if (iw == w).sum() == size]


def test_world_items_linear_at_many_worlds() -> None:
    """2^18 worlds x 4 items: per-world masks would cost ~3e11 element tests (WP8 xbench G4); grouping is ~0.1 s."""
    W = 1 << 18
    iw = np.concatenate([np.full(1, -1), np.repeat(np.arange(W), 4)])
    t0 = time.perf_counter()
    items = world_items(iw, W, "cpu")
    g = group_by_world(iw, W)
    world_reduce(g, np.ones((len(iw), 3), np.float32), np.maximum, -np.inf)
    elapsed = time.perf_counter() - t0
    assert items.end.numpy()[-1] == W * 5
    assert elapsed < 10.0, f"grouping {W} worlds took {elapsed:.1f} s"
