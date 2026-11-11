# SPDX-License-Identifier: Apache-2.0
"""Full-surface proxy contacts: mute edge-edge records on flat rigid edges (SCR-SC4-6, WP4 part).

Newton's full-surface mesh pass takes every edge of a rigid mesh's triangulation. A box face's diagonal and the
near-coplanar edges of a convex hull (the 64-vertex view hulls of ``full_surface.contact_view``) are flat edges:
their two faces are (almost) coplanar, so a cloth edge near them is already held by the face's vertex-face records.
Their edge-edge records duplicate that contact several times over: 460 edge-edge against 100 vertex-face records on
two box pads. That makes a closed two-pad clamp bounce. The pad gap chatters 3.2-5.3 mm, no pad record is active at
the period ends, and the clamp holds ~0 N of a 12.6 N Coulomb limit.

``FlatEdgeMute`` flags the rigid edges whose dihedral angle is below ``FLAT_EDGE_DEG`` once, when the proxy pipeline
is built. After each collide it marks their edge-edge records sign-unknown: VBD applies no force for those, and the
pass-through check skips them. Sharp edges (a pad rim, a ridge) keep their records; they are what stops a feature
narrower than the cloth's vertex gap from threading it (SCR-SC4-4).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.coupling")

FLAT_EDGE_DEG = 10.0             # [deg] an edge whose two faces turn by less than this is flat
_EE = wp.constant(2)             # Newton mesh feature family: edge-edge
_FAMILY = wp.constant(7)
_SIGN_UNKNOWN = wp.constant(32)  # VBD skips the force of such records (Newton's _MESH_FEATURE_SIGN_UNKNOWN)


@wp.kernel(enable_backward=False)
def mute_flat_edge_records(count: wp.array[wp.int32], cap: int, features: wp.array[wp.vec3i],
                           flat: wp.array[wp.int32]):
    k = wp.tid()
    if k >= wp.min(count[0], cap):
        return
    f = features[k]
    if f[0] < 0 or (f[0] & _FAMILY) != _EE:
        return
    if flat[f[2]] != 0:
        features[k] = wp.vec3i(f[0] | _SIGN_UNKNOWN, f[1], f[2])


def _flat_edge_keys(mesh: Any, flat_deg: float) -> tuple[np.ndarray, np.ndarray, int]:
    """(canonical vertex id per index slot, sorted flat-edge keys a * n + b, n): edges shared by exactly two
    triangles whose normals differ by less than ``flat_deg``."""
    v = np.asarray(mesh.vertices, np.float64)
    idx = np.asarray(mesh.indices, np.int64).ravel()
    tri = idx.reshape(-1, 3)
    _, canon = np.unique(np.round(v, 9), axis=0, return_inverse=True)
    canon = canon.ravel()
    n = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    tc = canon[tri]
    edges = np.sort(tc[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    nv = int(canon.max()) + 1
    keys = edges[:, 0] * nv + edges[:, 1]
    face = np.repeat(np.arange(len(tri)), 3)
    order = np.argsort(keys, kind="stable")
    keys, face = keys[order], face[order]
    uniq, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    two = cnt == 2
    cos = np.einsum("ij,ij->i", n[face[start[two]]], n[face[start[two] + 1]])
    flat = uniq[two][cos > np.cos(np.radians(flat_deg))]
    return canon[idx], np.sort(flat), nv


def flat_edge_flags(edge_table: np.ndarray, sources: list[Any], flat_deg: float = FLAT_EDGE_DEG) -> np.ndarray:
    """[E] int32 flags over Newton's rigid edge table (rows: shape, a, b, where a and b are slots of the mesh's
    flattened triangle indices): 1 for a flat edge."""
    out = np.zeros(len(edge_table), np.int32)
    shapes = edge_table[:, 0]
    cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for s in np.unique(shapes):
        mesh = sources[int(s)]
        if mesh is None:
            continue
        if id(mesh) not in cache:
            cache[id(mesh)] = _flat_edge_keys(mesh, flat_deg)
        slot_canon, flat, nv = cache[id(mesh)]
        rows = np.flatnonzero(shapes == s)
        a, b = slot_canon[edge_table[rows, 1]], slot_canon[edge_table[rows, 2]]
        out[rows] = np.isin(np.minimum(a, b) * nv + np.maximum(a, b), flat)
    return out


class FlatEdgeMute:
    """Per built proxy pipeline: the flat-edge flags of its rigid edge table, and the post-collide mute."""

    def __init__(self, view: Any, pipeline: Any, flat_deg: float = FLAT_EDGE_DEG) -> None:
        data = getattr(pipeline, "_soft_mesh_contact_data", None)
        self.flags = None
        if data is None:
            return
        table = data.rigid_features[2].numpy()
        if not len(table):
            return
        flags = flat_edge_flags(table, list(view.shape_source), flat_deg)
        self.flags = wp.array(flags, dtype=wp.int32, device=view.device)
        log.info("proxy contacts %s: %d of %d rigid edges flat (< %.0f deg), their edge-edge records muted",
                 getattr(view, "name", "?"), int(flags.sum()), len(flags), flat_deg)

    def apply(self, contacts: Any) -> None:
        feats = getattr(contacts, "_soft_contact_mesh_features", None)
        if self.flags is None or feats is None:
            return
        cap = int(contacts.soft_contact_max)
        wp.launch(mute_flat_edge_records, dim=cap, device=contacts.soft_contact_count.device,
                  inputs=[contacts.soft_contact_count, cap, feats, self.flags])
