# SPDX-License-Identifier: Apache-2.0
"""Scoring of the measured-fabric drape rig (SCR-WP2-14; D03M and R6): drape coefficient, node count, strain and
settling of recorded Cusick frames.

DC = (A_s - A_d) / (A_D - A_d) (Cusick 1965; ISO 9073-9): A_s the silhouette of the draped specimen at 1 mm (the R6
twin's raster), A_d the support disc, A_D the flat specimen's own silhouette (the mesh, so its polygon edge cancels).
Nodes: the R6 twin's rim peaks about the support axis.
"""

from __future__ import annotations

import math

import numpy as np


def edges_of(tris: np.ndarray) -> np.ndarray:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def max_strain_pct(rest: np.ndarray, x: np.ndarray, edges: np.ndarray) -> float:
    l0 = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    l1 = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)
    return float(100.0 * np.abs(l1 / l0 - 1.0).max())


class Scorer:
    """Scores frames of one specimen mesh on a support of radius ``r_sup``."""

    def __init__(self, rest: np.ndarray, tris: np.ndarray, radius: float, r_sup: float) -> None:
        from alloy.scene.twins import drape as R6

        self.R6, self.rest, self.tris, self.r_sup = R6, rest, tris, r_sup
        self.edges = edges_of(tris)
        rr = np.hypot(rest[:, 0], rest[:, 1])
        self.rim = np.flatnonzero(np.isclose(rr, radius, atol=1e-6))
        self.rim_angle = np.arctan2(rest[self.rim, 1], rest[self.rim, 0])
        self.clamped = np.flatnonzero(rr < r_sup - 3e-3)      # held between the two discs
        self.a_d = math.pi * r_sup ** 2
        self.a_flat = R6.silhouette_area(rest[:, :2], tris)

    def dc(self, x: np.ndarray) -> float:
        return (self.R6.silhouette_area(x[:, :2], self.tris) - self.a_d) / (self.a_flat - self.a_d)

    def nodes(self, x: np.ndarray) -> int:
        return self.R6.node_count(self.rim_angle, x[self.rim, :2])

    def frames(self, t: np.ndarray, q: np.ndarray) -> dict:
        """t [F], q [F, W, P, 3] (the scored window) -> one row of window statistics."""
        dc = np.array([[self.dc(q[f, w]) for w in range(q.shape[1])] for f in range(q.shape[0])])   # [F, W]
        strain = max(max_strain_pct(self.rest, q[f, w], self.edges) for f in range(q.shape[0])
                     for w in range(q.shape[1]))
        nodes = [self.nodes(q[-1, w]) for w in range(q.shape[1])]
        dt = float(t[-1] - t[-2]) if len(t) > 1 else math.nan
        vmax = float(np.linalg.norm(q[-1] - q[-2], axis=-1).max() / dt) if len(t) > 1 else math.nan
        frame_mean = dc.mean(axis=1)
        vals, counts = np.unique(nodes, return_counts=True)
        return {"frames": int(len(t)), "worlds": int(q.shape[1]), "window_s": [float(t[0]), float(t[-1])],
                "dc_mean": float(dc.mean()), "dc_frame_min": float(frame_mean.min()),
                "dc_frame_max": float(frame_mean.max()), "dc_world_last": dc[-1].round(4).tolist(),
                "dc_world_mean": dc.mean(axis=0).tolist(),
                "dc_world_sd": float(dc[-1].std(ddof=1)) if q.shape[1] > 1 else 0.0,
                "nodes_world": nodes, "nodes_mode": int(vals[np.argmax(counts)]),
                "strain_max_pct": strain, "vmax_end_m_s": vmax, "finite": bool(np.isfinite(q).all()),
                "clamp_z_mm": [float(q[:, :, self.clamped, 2].min() * 1e3),
                               float(q[:, :, self.clamped, 2].max() * 1e3)]}


__all__ = ["Scorer", "edges_of", "max_strain_pct"]
