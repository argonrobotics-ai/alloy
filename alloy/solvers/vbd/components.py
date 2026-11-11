# SPDX-License-Identifier: Apache-2.0
"""Connected particle components of a VBD model and the fixed reduction layout of the coarse step (SCR-WP11-2).

A component is a set of particles joined by triangles, bending edges, springs or tetrahedra (one cloth, shell or soft
body). Its translation is an exact null space of every internal energy VBD evaluates, which is what the coarse step
uses. The layout is built once, on the host, at solver construction; nothing here runs per step:

- `members` lists the particles of every component in a fixed order (by component, then particle id), CSR-style;
- `chunk_start` cuts each component's run into chunks of at most `chunk` particles, never across components, so the
  per-step reduction is two fixed-order passes (chunks, then components): deterministic and capture-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from alloy._vendor import newton

CHUNK = 128          # particles summed by one thread in the first reduction pass
_ELEMENTS = (("tri_indices", 3), ("edge_indices", 4), ("spring_indices", 2), ("tet_indices", 4))


@dataclass(frozen=True)
class ComponentLayout:
    """Host-side layout of the coarse step (all int32)."""

    comp_of: np.ndarray            # [N] component of each particle, -1 for particles in no element
    members: np.ndarray            # [N'] particle ids grouped by component
    comp_start: np.ndarray         # [C + 1] into members
    chunk_start: np.ndarray        # [K + 1] into members; chunks never cross components
    comp_chunk_start: np.ndarray   # [C + 1] into chunks

    @property
    def count(self) -> int:
        return len(self.comp_start) - 1

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_start) - 1


def element_rows(model: newton.Model) -> list[np.ndarray]:
    """Particle index rows of every element type present (−1 entries, e.g. open bending-edge corners, kept)."""
    rows = []
    for name, width in _ELEMENTS:
        arr = getattr(model, name, None)
        if arr is not None and arr.shape[0] > 0:
            rows.append(np.asarray(arr.numpy(), dtype=np.int64).reshape(-1, width))
    return rows


def label_components(n: int, rows: list[np.ndarray]) -> np.ndarray:
    """Component label per particle in [0, C), −1 for particles in no element; labels ordered by lowest member."""
    src, dst = [], []
    in_elem = np.zeros(n, bool)
    for r in rows:
        valid = r >= 0
        keep = valid.any(axis=1)
        r, valid = r[keep], valid[keep]
        anchor = r[np.arange(len(r)), valid.argmax(axis=1)]           # first real vertex of each row
        for a in range(r.shape[1]):
            ok = valid[:, a]
            in_elem[r[ok, a]] = True
            src.append(anchor[ok])
            dst.append(r[ok, a])
    s = np.concatenate(src) if src else np.zeros(0, np.int64)
    d = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    labels = np.full(n, -1, np.int64)
    if not in_elem.any():
        return labels
    graph = coo_matrix((np.ones(len(s), np.int8), (s, d)), shape=(n, n))
    _, raw = connected_components(graph, directed=False)
    used = np.unique(raw[in_elem])
    order = np.argsort(_first_index(raw, in_elem, used), kind="stable")
    remap = np.full(raw.max() + 1, -1, np.int64)
    remap[used[order]] = np.arange(len(used))
    labels[in_elem] = remap[raw[in_elem]]
    return labels


def _first_index(raw: np.ndarray, in_elem: np.ndarray, used: np.ndarray) -> np.ndarray:
    """Lowest particle id of each raw label in `used` (vectorised)."""
    idx = np.flatnonzero(in_elem)
    first = np.full(raw.max() + 1, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, raw[idx], idx)
    return first[used]


def build_layout(model: newton.Model, chunk: int = CHUNK) -> ComponentLayout:
    """The coarse step's component and chunk layout for `model`."""
    n = int(model.particle_count)
    labels = label_components(n, element_rows(model))
    members = np.flatnonzero(labels >= 0)
    members = members[np.argsort(labels[members], kind="stable")]
    count = int(labels.max()) + 1 if members.size else 0
    sizes = np.bincount(labels[members], minlength=count)
    comp_start = np.concatenate([[0], np.cumsum(sizes)])
    per_comp = -(-sizes // chunk)                                   # chunks per component (ceil)
    comp_chunk_start = np.concatenate([[0], np.cumsum(per_comp)])
    chunk_start = [comp_start[c] + np.arange(per_comp[c]) * chunk for c in range(count)]
    chunk_start = np.concatenate(chunk_start + [[comp_start[-1]]]) if count else np.zeros(1, np.int64)
    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)       # noqa: E731
    return ComponentLayout(comp_of=i32(labels), members=i32(members), comp_start=i32(comp_start),
                           chunk_start=i32(chunk_start), comp_chunk_start=i32(comp_chunk_start))


__all__ = ["CHUNK", "ComponentLayout", "build_layout", "element_rows", "label_components"]
