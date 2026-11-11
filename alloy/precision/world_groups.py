# SPDX-License-Identifier: Apache-2.0
"""Per-world grouping of host item arrays in O(n log n) (setup tables for digests, health and frames).

Items carry a world id; ids outside ``[0, world_count)`` (global items, -1) belong to no group. A stable sort keeps
each world's items in index order, so the grouping visits the same items in the same order as a per-world
``item_world == w`` mask, without the O(W·n) cost of one full-length mask per world.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorldGroups:
    """World ``w`` owns ``order[begin[w]:end[w]]`` (item indices, ascending)."""

    order: np.ndarray  # int64 item indices sorted by world, stable
    begin: np.ndarray  # [W] int64
    end: np.ndarray  # [W] int64

    @property
    def world_count(self) -> int:
        return len(self.begin)

    @property
    def counts(self) -> np.ndarray:
        return self.end - self.begin

    def world_of_sorted(self) -> np.ndarray:
        """World id of each entry of ``order``."""
        return np.repeat(np.arange(self.world_count, dtype=np.int64), self.counts)


def group_by_world(item_world: np.ndarray, world_count: int) -> WorldGroups:
    w = np.asarray(item_world, dtype=np.int64).reshape(-1)
    valid = np.flatnonzero((w >= 0) & (w < world_count))
    order = valid[np.argsort(w[valid], kind="stable")]
    counts = np.bincount(w[valid], minlength=world_count).astype(np.int64)
    end = np.cumsum(counts)
    return WorldGroups(order, end - counts, end)


def world_sum(item_world: np.ndarray, values: np.ndarray, world_count: int) -> np.ndarray:
    """[W] float64 sum of ``values`` per world (items outside ``[0, W)`` ignored)."""
    w = np.asarray(item_world, dtype=np.int64).reshape(-1)
    keep = (w >= 0) & (w < world_count)
    return np.bincount(w[keep], weights=np.asarray(values, dtype=np.float64)[keep], minlength=world_count)


def world_reduce(groups: WorldGroups, values: np.ndarray, ufunc: np.ufunc, empty: float) -> np.ndarray:
    """``ufunc.reduce`` of ``values[items of w]`` along axis 0 per world; ``empty`` for worlds with no items."""
    values = np.asarray(values)
    out = np.full((groups.world_count, *values.shape[1:]), empty, dtype=values.dtype)
    nonempty = groups.counts > 0
    if groups.order.size:
        out[nonempty] = ufunc.reduceat(values[groups.order], groups.begin[nonempty], axis=0)
    return out


def equal_size_blocks(groups: WorldGroups, size: int,
                      max_rows: int = 1 << 22) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(world ids, [k, size] item indices) for the worlds holding exactly ``size`` items, in chunks of at most
    ``max_rows`` items so a gathered [k, size, 3] block stays bounded."""
    worlds = np.flatnonzero(groups.counts == size)
    if size <= 0 or not len(worlds):
        return
    step = max(1, max_rows // size)
    lane = np.arange(size, dtype=np.int64)
    for k in range(0, len(worlds), step):
        ids = worlds[k:k + step]
        yield ids, groups.order[groups.begin[ids][:, None] + lane[None, :]]
