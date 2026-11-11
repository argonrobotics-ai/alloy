# SPDX-License-Identifier: Apache-2.0
"""Structured meshes for pre-split objects and twins: a triangle grid and a Kuhn tet grid (float64 numpy)."""

from __future__ import annotations

import numpy as np

# Kuhn (Freudenthal) split of the unit cube into 6 tets along the 0-7 diagonal; conforming across cubes.
_CUBE_CORNERS = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=np.int64)
_KUHN_TETS = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]], dtype=np.int64
)


def grid_tri_mesh(nx: int, ny: int, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [(nx+1)(ny+1), 3], tris [2·nx·ny, 3]) in the z = 0 plane, counter-clockwise, origin corner."""
    j, i = np.meshgrid(np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    verts = np.stack([i.ravel() * cell, j.ravel() * cell, np.zeros(i.size)], axis=1).astype(np.float64)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    v0 = (jj * (nx + 1) + ii).ravel()
    v1, v2, v3 = v0 + 1, v0 + nx + 1, v0 + nx + 2
    tris = np.stack([np.stack([v0, v1, v3], 1), np.stack([v0, v3, v2], 1)], axis=1).reshape(-1, 3)
    return verts, tris.astype(np.int64)


def grid_tet_mesh(nx: int, ny: int, nz: int, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, tets [6·nx·ny·nz, 4]) of a box [0, n·cell]^3 split into Kuhn tets; positive orientation."""
    k, j, i = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    verts = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.float64) * cell

    def vid(ci, cj, ck):
        return (ck * (ny + 1) + cj) * (nx + 1) + ci

    kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    base = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    corners = base[:, None, :] + _CUBE_CORNERS[None, :, :]                      # [C, 8, 3]
    ids = vid(corners[..., 0], corners[..., 1], corners[..., 2])               # [C, 8]
    tets = ids[:, _KUHN_TETS].reshape(-1, 4)
    return verts, _orient_positive(verts, tets)


def _orient_positive(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = verts[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    out = tets.copy()
    neg = vol < 0
    out[neg, 1], out[neg, 2] = tets[neg, 2], tets[neg, 1]
    return out


def tet_volumes(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = verts[tets]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def tri_areas(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = verts[tris]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Scenes for the topology tests (Newton models; CPU-sized).
