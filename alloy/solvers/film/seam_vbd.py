# SPDX-License-Identifier: Apache-2.0
"""Seams inside VBD's own bookkeeping (seams.py holds them in the projection).

- ``recolor_twins``: Newton's colouring graph leaves springs out, so the two copies of a seam vertex could share a VBD
  colour (solved in parallel, against VBD's one-colour-per-element rule); each such twin end moves to a free colour.
- ``join_pieces`` / ``write_pieces``: VBD finds its pieces from triangles only (a pinned piece starts its iterations
  from x + h·v, the others from the inertial target), so the far side of a seam is a piece of its own. Unpinned, it
  started every step from the inertial target and fell h²·g per step that ten iterations of a stiff sheet did not
  undo: 8.5 um a step at the seam of a hanging kraft strip against 1.0 um unsplit, and the projection then dragged
  the pinned side to its strain cap (1.4 % everywhere against 0.05 % unsplit; 0.05 % joined). Each step the pieces
  joined by intact seams share one id; a seam torn through splits them again.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .seams import SeamArrays, Seams, _intact


def recolor_twins(model, s: Seams) -> int:
    """Give twin pairs different VBD colours (before the solver is built): a twin end sharing its partner's colour
    moves to the first colour none of its VBD neighbours (triangles, hinges, twins) uses, a new colour only if none is
    free. Returns how many particles moved (0 on the library sheets: the guard is for meshes where it is not).
    Deterministic (seam order)."""
    groups = [np.asarray(g.numpy(), np.int64) for g in model.particle_color_groups]
    if not groups:
        return 0
    n = int(model.particle_count)
    color = np.full(n, -1, np.int64)
    for g, ids in enumerate(groups):
        color[ids] = g
    seam_v = np.zeros(n, bool)
    seam_v[s.a] = True
    seam_v[s.b] = True
    nbr: dict[int, set[int]] = {}
    rows = [model.tri_indices.numpy().astype(np.int64)]
    if model.edge_count:
        rows.append(model.edge_indices.numpy().astype(np.int64))
    rows.append(np.stack([s.a, s.b], 1).astype(np.int64))
    for r in rows:
        r = r[seam_v[np.maximum(r, 0)].any(axis=1)]
        for row in r:
            row = [int(v) for v in row if v >= 0]
            for v in row:
                if seam_v[v]:
                    nbr.setdefault(v, set()).update(u for u in row if u != v)
    moved, n_colors = 0, len(groups)
    for a, b in zip(s.a.tolist(), s.b.tolist()):
        if color[a] != color[b]:
            continue
        used = {int(color[u]) for u in nbr.get(b, ())}
        free = next((c for c in range(n_colors) if c not in used), n_colors)
        n_colors = max(n_colors, free + 1)
        color[b] = free
        moved += 1
    if moved:
        dt = model.particle_color_groups[0].dtype
        model.particle_color_groups = [wp.array(np.flatnonzero(color == g).astype(np.int32), dtype=dt,
                                                device=model.device) for g in range(n_colors)]
    return moved


@wp.func
def _find(label: wp.array[wp.int32], p: int) -> int:
    r = p
    while label[r] != r:
        r = label[r]
    return r


@wp.kernel(enable_backward=False)
def init_labels(label: wp.array[wp.int32]):
    p = wp.tid()
    label[p] = p


@wp.kernel(enable_backward=False)
def join_pieces(sa: SeamArrays, intact: wp.array[wp.int8], base: wp.array[wp.int32], label: wp.array[wp.int32]):
    """One thread per world: union-find of VBD's triangle pieces over the world's intact seams (the root of a group
    is its smallest piece id, so the result does not depend on the order). Worlds share no piece."""
    w = wp.tid()
    for e in range(sa.world_start[w], sa.world_start[w + 1]):
        s = sa.by_world[e]
        if _intact(intact, sa.bond[s]):
            ra = _find(label, base[sa.a[s]])
            rb = _find(label, base[sa.b[s]])
            if ra != rb:
                label[wp.max(ra, rb)] = wp.min(ra, rb)


@wp.kernel(enable_backward=False)
def flatten_labels(label: wp.array[wp.int32]):
    """Every piece to its group's root (a concurrent read mid-walk sees an ancestor either way)."""
    p = wp.tid()
    label[p] = _find(label, p)


@wp.kernel(enable_backward=False)
def write_pieces(base: wp.array[wp.int32], label: wp.array[wp.int32], piece: wp.array[wp.int32]):
    i = wp.tid()
    piece[i] = label[base[i]]


__all__ = ["flatten_labels", "init_labels", "join_pieces", "recolor_twins", "write_pieces"]
