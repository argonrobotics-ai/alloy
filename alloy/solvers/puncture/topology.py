# SPDX-License-Identifier: Apache-2.0
"""Host precompute of the puncturable tissue (tets with ``puncture:fp`` > 0) and its surface faces.

- Tissue tets are grouped per world (world of the tet's first particle), in index order.
- Surface faces are the faces of tissue tets that no other tissue tet shares, oriented outward (normal away from
  the tet's fourth vertex), grouped per world. Each keeps its tet and the tet-local slots of its three vertices, so
  a face barycentric point is also a tet barycentric point (the fourth weight is 0).
Everything is sized here once; the device arrays never change size (graph capture).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from .config import NAMESPACE

_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


@dataclass
class PunctureTopology:
    world_count: int
    tets: np.ndarray            # [T_t] tissue tet indices, grouped per world
    tet_start: np.ndarray       # [W+1]
    face_tri: np.ndarray        # [F, 3] particle indices, outward
    face_tet: np.ndarray        # [F]
    face_slot: np.ndarray       # [F, 3] tet-local slots of face_tri
    face_start: np.ndarray      # [W+1]

    @property
    def any(self) -> bool:
        return self.tets.size > 0


def tissue_values(model: newton.Model, name: str) -> np.ndarray:
    ns = getattr(model, NAMESPACE, None)
    arr = getattr(ns, name, None) if ns is not None else None
    if arr is None or model.tet_count == 0:
        return np.zeros(max(int(model.tet_count), 0), np.float32)
    return np.asarray(arr.numpy(), np.float32)


def build_topology(model: newton.Model) -> PunctureTopology:
    W = max(int(model.world_count), 1)
    T = int(model.tet_count)
    fp = tissue_values(model, "fp")
    tissue = np.flatnonzero(fp > 0.0) if T else np.zeros(0, np.int64)
    if tissue.size == 0:
        z = np.zeros(W + 1, np.int32)
        return PunctureTopology(W, np.zeros(0, np.int32), z, np.zeros((0, 3), np.int32), np.zeros(0, np.int32),
                                np.zeros((0, 3), np.int32), z.copy())
    tet_idx = model.tet_indices.numpy()
    pw = model.particle_world.numpy() if model.particle_count else np.zeros(1, np.int32)
    tw = np.maximum(pw[tet_idx[tissue, 0]], 0) if W > 1 else np.zeros(tissue.size, np.int64)
    order = np.lexsort((tissue, tw))
    tets, tw = tissue[order].astype(np.int32), tw[order]
    tet_start = np.searchsorted(tw, np.arange(W + 1)).astype(np.int32)
    q = model.particle_q.numpy()
    faces: dict[tuple[int, int, int], list[tuple[int, tuple[int, int, int]]]] = {}
    for t in tets:
        v = tet_idx[t]
        for slots in _FACES:
            key = tuple(sorted(int(v[s]) for s in slots))
            faces.setdefault(key, []).append((int(t), slots))
    tri, ftet, fslot, fw = [], [], [], []
    for owners in faces.values():
        if len(owners) != 1:
            continue
        t, slots = owners[0]
        v = tet_idx[t]
        other = ({0, 1, 2, 3} - set(slots)).pop()
        a, b, c = (int(v[s]) for s in slots)
        n = np.cross(q[b] - q[a], q[c] - q[a])
        s3 = list(slots)
        if np.dot(n, q[int(v[other])] - q[a]) > 0.0:
            a, b = b, a
            s3[0], s3[1] = s3[1], s3[0]
        tri.append((a, b, c))
        ftet.append(t)
        fslot.append(s3)
        fw.append(max(int(pw[a]), 0) if W > 1 else 0)
    fw = np.asarray(fw)
    order = np.lexsort((np.asarray(ftet), fw))
    return PunctureTopology(W, tets, tet_start, np.asarray(tri, np.int32)[order], np.asarray(ftet, np.int32)[order],
                            np.asarray(fslot, np.int32)[order], np.searchsorted(fw[order], np.arange(W + 1)).astype(
                                np.int32))


@wp.struct
class Surface:
    tets: wp.array[wp.int32]
    tet_start: wp.array[wp.int32]
    tri: wp.array2d[wp.int32]
    tet: wp.array[wp.int32]
    slot: wp.array2d[wp.int32]
    start: wp.array[wp.int32]


def upload(top: PunctureTopology, device) -> Surface:
    def a(x, dt):
        x = np.asarray(x)
        if x.shape[0] == 0:
            x = np.zeros((1,) + x.shape[1:], x.dtype)
        return wp.array(x, dtype=dt, device=device)

    s = Surface()
    s.tets, s.tet_start = a(top.tets, wp.int32), a(top.tet_start, wp.int32)
    s.tri, s.tet = a(top.face_tri, wp.int32), a(top.face_tet, wp.int32)
    s.slot, s.start = a(top.face_slot, wp.int32), a(top.face_start, wp.int32)
    return s
