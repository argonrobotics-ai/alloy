# SPDX-License-Identifier: Apache-2.0
"""The dry-tile skip (design note §3.3, "Tiles"): a film step touches only the tiles near wet cells.

Each chart's grid is cut into tiles of about ``TILE x TILE`` cells (balanced: every tile of a split axis has at least
``TILE / 2`` cells). Per film substep:
1. ``tile_wet_kernel`` (world, tile): a tile is wet if any of its cells is (``cwet``, which the cell kernel writes
   for the cells it steps, the supply sets for the cells it sources and the host's writes refresh);
2. ``tile_on_kernel`` (world, tile): a tile is *active* if it or one of its 8 neighbours is wet (θ wraps on
   revolution charts), or a tile across a chart link (links.py) within reach: the tiles of the partners of its own
   and its neighbours' edge cells, and back. The film kernels step the tiles active now or at the previous substep.

The skip is exact: h, u, q and the ledger match the dense step bit for bit. A wet cell is one whose film can change:
deeper than ``H_DRY``, with a source (a volume, or a species until the substep that applies
its last share), or any film at all where evaporation or a drain can take it. A face whose two
cells are dry gets ``u = q = 0`` in the dense step; a tile leaving the active set is stepped once more, which writes
those zeros, and stays exact while skipped. Film moves at most one cell per substep, and every value a wet face reads
(depths, potentials, its neighbours' velocities and the transverse estimate from the previous substep) lies within
three cells of a cell that was wet at the previous substep, so inside the active tiles of both substeps.
One exception: a host ``set_depth`` into a dry region of a yield-stress film starts its transverse estimate from
the skipped faces' old values for one substep (the dense step uses the previous substep's).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.charts import RevolutionChart

TILE = 16                        # cells per tile side
NEAR = 15                        # a tile, its 8 neighbours and the tiles across chart links


@wp.struct
class Tiles:
    first: wp.array[wp.int32]                 # [T] global cell of the tile's first row and column
    rows: wp.array[wp.int32]                  # [T]
    cols: wp.array[wp.int32]                  # [T]
    pitch: wp.array[wp.int32]                 # [T] cells per chart row (ny)
    near: wp.array2d[wp.int32]                # [T, NEAR] the tile, its neighbours, across links; -1 = none


@dataclass
class TileTables:
    tiles: Tiles
    count: int
    ctile: wp.array                           # [C] int32: each cell's tile
    ftile: wp.array                           # [F] int32: each face's tile (its a cell's, else its b cell's)


def splits(n: int, size: int = TILE) -> np.ndarray:
    """Balanced tile edges along an axis of ``n`` cells: ``ceil(n / size)`` tiles."""
    k = max(1, -(-n // size))
    return (np.arange(k + 1) * n) // k


def _chart_tiles(nx: int, ny: int, wrap: bool, start: int, t0: int) -> tuple[list, np.ndarray]:
    ei, ej = splits(nx), splits(ny)
    ki, kj = len(ei) - 1, len(ej) - 1
    rows = []
    ctile = np.empty((nx, ny), np.int64)
    for ti in range(ki):
        for tj in range(kj):
            near = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = ti + di, tj + dj
                    if wrap:
                        nj %= kj
                    if 0 <= ni < ki and 0 <= nj < kj and t0 + ni * kj + nj not in near:
                        near.append(t0 + ni * kj + nj)
            first = start + int(ei[ti]) * ny + int(ej[tj])
            rows.append((first, int(ei[ti + 1] - ei[ti]), int(ej[tj + 1] - ej[tj]), ny, near))
            ctile[ei[ti]:ei[ti + 1], ej[tj]:ej[tj + 1]] = t0 + ti * kj + tj
    return rows, ctile.ravel()


def _across_links(near: list[list[int]], ctile: np.ndarray, pairs: np.ndarray) -> list[list[int]]:
    """Each tile's near set gains the tiles of the link partners of its own and its neighbours' cells, and each of
    those tiles gains it back (film within reach of a seam is within reach of the tiles on both sides)."""
    partner: dict[int, set[int]] = {}
    for a, b in pairs:
        partner.setdefault(int(ctile[a]), set()).add(int(ctile[b]))
        partner.setdefault(int(ctile[b]), set()).add(int(ctile[a]))
    extra = [set().union(*(partner.get(u, set()) for u in n)) for n in near]
    out = [list(n) for n in near]
    for t, more in enumerate(extra):
        for u in sorted(more):
            for x, y in ((t, u), (u, t)):
                if y not in out[x]:
                    out[x].append(y)
    worst = max(len(n) for n in out)
    if worst > NEAR:
        raise PluginError(f"surface_flow: a tile reaches {worst} tiles across chart links (at most {NEAR}); link "
                          "fewer edges into one tile, or use more cells")
    return out


def build_tiles(specs: Sequence[Any], tables: Any, device: Any) -> TileTables:
    """Tiles of every chart in ``tables`` (built from ``specs``, in order)."""
    rows, ctiles = [], []
    for spec, start, (nx, ny) in zip(specs, tables.starts, tables.shapes):
        r, ct = _chart_tiles(nx, ny, isinstance(spec, RevolutionChart), start, len(rows))
        rows += r
        ctiles.append(ct)
    ctile = np.concatenate(ctiles).astype(np.int32)
    near = _across_links([r[4] for r in rows], ctile, getattr(tables, "link_cells", np.zeros((0, 2), np.int64)))
    fa, fb = tables.fa.numpy(), tables.fb.numpy()
    ftile = ctile[np.where(fa >= 0, fa, fb)]

    def arr(a: Any, dtype: Any) -> wp.array:
        return wp.array(np.ascontiguousarray(a, np.int32), dtype=dtype, device=device)

    t = Tiles()
    t.first, t.rows, t.cols, t.pitch = (arr([r[k] for r in rows], wp.int32) for k in range(4))
    t.near = arr(np.asarray([n + [-1] * (NEAR - len(n)) for n in near]), wp.int32)
    return TileTables(tiles=t, count=len(rows), ctile=arr(ctile, wp.int32), ftile=arr(ftile, wp.int32))


def wet_cells(h: np.ndarray, source: np.ndarray, drain_of: np.ndarray, evaporation: float | np.ndarray,
              h_dry: float, flooded: np.ndarray | None = None) -> np.ndarray:
    """``cwet`` for a film set from the host (``[W, C]`` int32), as the cell kernel computes it. ``evaporation`` is
    the material's rate, or the per-cell sink rates ``[W, C]`` (evaporation + absorption)."""
    takes = (h > 0.0) & ((evaporation > 0.0) | (drain_of >= 0)[None, :])
    if flooded is not None:
        takes |= (h > 0.0) & (flooded >= 0)
    return ((h > h_dry) | (source > 0.0) | takes).astype(np.int32)


@wp.kernel
def tile_wet_kernel(tl: Tiles, tick: wp.array[wp.int32], stride: int, cwet: wp.array2d[wp.int32],
                    twet: wp.array2d[wp.int32]):
    w, t = wp.tid()
    if tick[w] % stride != 0:
        return
    first, pitch, cols = tl.first[t], tl.pitch[t], tl.cols[t]
    wet = int(0)
    r = int(0)
    while r < tl.rows[t] and wet == 0:
        row = first + r * pitch
        for j in range(cols):
            wet = wet | cwet[w, row + j]
        r += 1
    twet[w, t] = wet


@wp.kernel
def tile_on_kernel(tl: Tiles, tick: wp.array[wp.int32], stride: int, twet: wp.array2d[wp.int32],
                   act: wp.array2d[wp.int32], on: wp.array2d[wp.int32]):
    """``on``: active now or at the previous substep (``act`` keeps the previous substep's active set)."""
    w, t = wp.tid()
    if tick[w] % stride != 0:
        return
    a = int(0)
    for s in range(NEAR):
        n = tl.near[t, s]
        if n >= 0:
            a = a | twet[w, n]
    on[w, t] = a | act[w, t]
    act[w, t] = a


__all__ = ["NEAR", "TILE", "TileTables", "Tiles", "build_tiles", "splits", "tile_on_kernel", "tile_wet_kernel",
           "wet_cells"]
