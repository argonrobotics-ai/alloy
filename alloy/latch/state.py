# SPDX-License-Identifier: Apache-2.0
"""Latch state read from geometry (never from a commanded flag), vectorised over worlds (numpy, float64).

A catch box (on the sprung member) is engaged with a keeper box (on the partner) when, in the keeper's frame:
- ``behind``: the catch box lies deeper than the keeper's deep face along the insertion axis, within contact
  penetration ``pen_tol`` (and, if ``window`` is given, no deeper than that past it);
- ``lift``: the two boxes overlap by ``>= min_overlap`` along the lift axis (the direction the member deflects
  to release);
- ``side``: they overlap along the third axis.
Box intervals along an axis a are ``c·a ± Σ_i h_i·|R_i·a|`` (exact for boxes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.latch.mass import quat_matrix


def quat_rows(q: np.ndarray) -> np.ndarray:
    """[N, 4] (x y z w) -> [N, 3, 3] rotation matrices."""
    q = np.atleast_2d(np.asarray(q, np.float64))
    return np.stack([quat_matrix(tuple(r)) for r in q])


def compose(body_q: np.ndarray, local_pos, local_quat) -> tuple[np.ndarray, np.ndarray]:
    """World centre [N, 3] and rotation [N, 3, 3] of a shape at (local_pos, local_quat) on bodies ``body_q``
    ([N, 7] x y z qx qy qz qw)."""
    bq = np.atleast_2d(np.asarray(body_q, np.float64))
    R = quat_rows(bq[:, 3:7])
    c = bq[:, :3] + np.einsum("nij,j->ni", R, np.asarray(local_pos, np.float64))
    return c, R @ quat_matrix(tuple(local_quat))


def interval(center: np.ndarray, R: np.ndarray, half, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N] lo/hi of boxes (centre [N,3], rotation [N,3,3], half [3]) projected on unit axes [N, 3]."""
    c = np.einsum("ni,ni->n", center, axis)
    r = np.abs(np.einsum("nij,ni->nj", R, axis)) @ np.asarray(half, np.float64)   # Σ_j h_j |R_:j · a|
    return c - r, c + r


@dataclass(frozen=True)
class CatchState:
    behind: np.ndarray        # [N] m, catch near face minus keeper deep face along the insertion axis
    lift: np.ndarray          # [N] m, overlap along the lift axis
    side: np.ndarray          # [N] m, overlap along the side axis
    engaged: np.ndarray       # [N] bool


def catch_state(catch_c: np.ndarray, catch_R: np.ndarray, catch_half, keeper_c: np.ndarray, keeper_R: np.ndarray,
                keeper_half, insert_local, lift_local, *, min_overlap: float = 1.0e-4, pen_tol: float = 5.0e-4,
                window: float | None = None) -> CatchState:
    """Engagement of a catch box with a keeper box; axes given in the keeper's frame (unit, orthogonal)."""
    a_in = np.einsum("nij,j->ni", keeper_R, np.asarray(insert_local, np.float64))
    a_up = np.einsum("nij,j->ni", keeper_R, np.asarray(lift_local, np.float64))
    a_side = np.cross(a_in, a_up)
    c_lo, _ = interval(catch_c, catch_R, catch_half, a_in)
    _, k_hi = interval(keeper_c, keeper_R, keeper_half, a_in)
    behind = c_lo - k_hi
    lift = _overlap(catch_c, catch_R, catch_half, keeper_c, keeper_R, keeper_half, a_up)
    side = _overlap(catch_c, catch_R, catch_half, keeper_c, keeper_R, keeper_half, a_side)
    ok = (behind >= -pen_tol) & (lift >= min_overlap) & (side > 0.0)
    if window is not None:
        ok &= behind <= window
    return CatchState(behind, lift, side, ok)


def _overlap(ca, Ra, ha, cb, Rb, hb, axis) -> np.ndarray:
    alo, ahi = interval(ca, Ra, ha, axis)
    blo, bhi = interval(cb, Rb, hb, axis)
    return np.minimum(ahi, bhi) - np.maximum(alo, blo)


def axial_depth(body_q: np.ndarray, nose_local, port_pos, port_axis) -> np.ndarray:
    """[N] signed depth of a body-frame point (the nose) past a port plane (point ``port_pos`` [N,3] or [3], unit
    ``port_axis`` into the port)."""
    c, _ = compose(body_q, nose_local, (0.0, 0.0, 0.0, 1.0))
    return np.einsum("ni,ni->n", c - np.broadcast_to(port_pos, c.shape), np.broadcast_to(port_axis, c.shape))


def box_penetration(ca: np.ndarray, Ra: np.ndarray, ha, cb: np.ndarray, Rb: np.ndarray, hb) -> np.ndarray:
    """[N] exact overlap depth of two boxes (separating-axis test over the 15 axes; 0 when separated); half
    extents [3] or per row [N, 3].
    Independent of any solver's contact reporting (HealthMonitor max_pen_m is blind on MuJoCo routes, SCR-SC0-1)."""
    ha = np.broadcast_to(np.asarray(ha, np.float64), (len(ca), 3))
    hb = np.broadcast_to(np.asarray(hb, np.float64), (len(cb), 3))
    axes = [Ra[:, :, i] for i in range(3)] + [Rb[:, :, j] for j in range(3)]
    axes += [np.cross(Ra[:, :, i], Rb[:, :, j]) for i in range(3) for j in range(3)]
    d = cb - ca
    depth = np.full(len(ca), np.inf)
    for a in axes:
        n = np.linalg.norm(a, axis=1)
        ok = n > 1e-9
        a = np.where(ok[:, None], a / np.where(ok, n, 1.0)[:, None], 0.0)
        ra = np.einsum("nj,nj->n", np.abs(np.einsum("nij,ni->nj", Ra, a)), ha)
        rb = np.einsum("nj,nj->n", np.abs(np.einsum("nij,ni->nj", Rb, a)), hb)
        ov = ra + rb - np.abs(np.einsum("ni,ni->n", d, a))
        depth = np.where(ok, np.minimum(depth, ov), depth)
    return np.maximum(depth, 0.0)
