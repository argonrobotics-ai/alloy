# SPDX-License-Identifier: Apache-2.0
"""Parallel per-world sums in a fixed order (mixed-speed lever #6, WP10's note 2026-11-07).

The D20 energy sums (health's ``world_energy``, the coupler's ``energy_kernel``) ran one thread per world over every
body and particle of that world: at 1 world, one GPU thread summed a whole scene (53 ms per call in table_drop). Now
each world's item sequence (its bodies, then its particles, in index order) is cut into fixed chunks of ``CHUNK``
items. One thread sums one chunk in float64 in item order, then one thread per world adds its chunks in chunk order.

- Deterministic: no atomics, so every run gives the same bits.
- Batch-invariant (D9): chunk boundaries count from each world's own first item, so a world's value does not depend
  on the other worlds.
- Unchanged where it fits: a world with at most ``CHUNK`` items is one chunk summed exactly as the serial loop was,
  bit for bit. Larger worlds differ from the serial sum by float64 rounding only (a different association).
- One scratch per owner: the partial sums live in the struct that carries the chunks (health's EnergyModel, the
  coupler's EnergyInputs / WorkInputs), so launches sharing one must run in order on one stream, as every caller
  does today (WP10's review). Concurrent streams would need a scratch each.
"""

from __future__ import annotations

import numpy as np
import warp as wp

CHUNK = 128   # items per chunk: one GPU thread each; the per-world pass then adds count / CHUNK partials


@wp.struct
class Chunks:
    """Chunk ``c`` covers positions ``lo[c]:hi[c]`` of world ``world[c]``'s item sequence; world ``w`` owns chunks
    ``start[w]:start[w + 1]`` ([W + 1])."""

    world: wp.array[wp.int32]
    lo: wp.array[wp.int32]
    hi: wp.array[wp.int32]
    start: wp.array[wp.int32]


def chunk_layout(counts: np.ndarray, size: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(world, lo, hi, start) of the chunks of worlds with ``counts[w]`` items each (host, once per model); ``size``
    defaults to ``CHUNK`` as it is at the call."""
    size = CHUNK if size is None else int(size)
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    counts = np.asarray(counts, dtype=np.int64)
    per = (counts + size - 1) // size
    start = np.concatenate([[0], np.cumsum(per)]).astype(np.int64)
    world = np.repeat(np.arange(len(counts), dtype=np.int64), per)
    lo = (np.arange(int(start[-1]), dtype=np.int64) - start[world]) * size
    hi = np.minimum(lo + size, counts[world])
    return world, lo, hi, start


def make_chunks(counts: np.ndarray, device: wp.context.Devicelike, size: int | None = None) -> Chunks:
    """The :class:`Chunks` of ``counts`` on ``device``."""
    ch = Chunks()
    for name, arr in zip(("world", "lo", "hi", "start"), chunk_layout(counts, size)):
        setattr(ch, name, wp.array(arr.astype(np.int32), dtype=wp.int32, device=device))
    return ch


def chunk_count(ch: Chunks) -> int:
    return int(ch.world.shape[0])


@wp.func
def chunk_sum(ch: Chunks, partial: wp.array[wp.float64], w: int) -> wp.float64:
    """World ``w``'s chunk partials added in chunk order (0 for a world without items)."""
    e = wp.float64(0.0)
    for c in range(ch.start[w], ch.start[w + 1]):
        e = e + partial[c]
    return e


__all__ = ["CHUNK", "Chunks", "chunk_count", "chunk_layout", "chunk_sum", "make_chunks"]
