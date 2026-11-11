# SPDX-License-Identifier: Apache-2.0
"""Success checks of the gift-bag tasks, from simulator state (numpy only; docs/notes/paper-pack.md).

The reference benchmark's (Gym-Anything, "GA") bag-candle task, "Put the boxed candle in the already-open bag, upright
on the bottom board, and leave the bag standing open on the mat", is judged by 11 criteria (its internal bag-purchase
checks, the tear check and the hands-off window). This module judges 10 of them from the bag's and carton's state and
lists the other one as not judged (``NOT_JUDGED``); ``success`` counts only the judged ones. Sized to this bag:

- ``candle_in_bag``: all eight carton corners inside the convex hull of the bag's vertices;
- ``candle_upright``: the carton's +z axis within 15 deg of vertical (signed: a carton on its lid reads 180 deg);
- ``candle_on_board``: the carton's lowest corner 0-5 mm above the board's top face (-1 mm allowed), its centre over
  the board;
- ``mouth_open``: the rim's area, projected on the plane normal to the bag's axis, at least 0.80·W·D; and the pinch
  guard: the least front-to-back gap across the middle 160 mm of the width, at every height, at least D/2;
- ``standing``: the bag's axis (bottom centroid to rim centroid) within 10 deg of vertical (upside down reads 180);
- ``on_mat``: the bottom centroid inside the mat;
- ``board_in_bag``: at least 90 % of the board's vertices within 4 mm of the bottom panel, above it and over it;
- ``bottom_squared``: at least 85 % of the bottom panel's area (triangle centroids) within 3 mm of the mat plane
  (the mat top plus 0.1 mm, GA's rest height of a 135 um sheet on a rigid surface) and the panel's area-summed normal
  within 10 deg of vertical (task.py:390-392); judged when the layout carries the bag's triangles;
- ``base_at_rest``: the bottom centroid's drift over the last 2 s of the window at most 2 · 2/3 mm (task.py:433-435);
  judged when ``bottom_mark`` (the centroid 2 s before the end) is given.

- ``hands_off_window``: no finger box within 5 mm of the bag, board, carton, mat or counter at any control period of
  the 3 s window (GA: no robot link with a reaction force over 1e-3 N; gift_hands_off.py). The caller measures it from
  the window's frames and passes it as ``hands_off``.

Not judged: ``bag_not_torn`` (GA's crack length of the bag since the start; the film route has no crack measure).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .gift_bag_mesh import BagMeshSpec
from .gift_gantry import ga_verdict
from .gift_hands_off import NOTE as HANDS_OFF_NOTE


@dataclass(frozen=True)
class Limits:
    upright_deg: float = 15.0
    on_board_mm: tuple[float, float] = (-1.0, 5.0)
    mouth_area: float = 0.80
    pinch_band: float = 0.160           # m of the width's middle where the front-to-back gap is guarded
    standing_deg: float = 10.0
    board_gap: float = 0.004            # m
    board_fraction: float = 0.90
    flat_fraction: float = 0.85         # bottom_squared: share of the panel's area ...
    flat_band: float = 0.003            # ... within this of the support plane [m]
    rest_drift: float = 2.0e-3 * 2.0 / 3.0   # base_at_rest: drift of the bottom over the last 2 s [m]


@dataclass(frozen=True)
class Layout:
    """Index sets of the bag mesh (alloy.tasks.gift_bag_mesh) and the scene's fixed geometry."""
    spec: BagMeshSpec
    rim: np.ndarray                     # rim vertex ids, around the perimeter
    bottom: np.ndarray                  # bottom panel vertex ids
    front: np.ndarray                   # [nz+1, nx+1] wall ids of the front face, left to right
    back: np.ndarray                    # [nz+1, nx+1] wall ids of the back face, left to right (facing the front)
    mat_center: np.ndarray              # (x, y) of the mat
    mat_half: np.ndarray                # half sizes (x, y) of the mat
    board_thickness: float
    candle_size: np.ndarray             # full edges (x, y, z) of the carton
    triangles: np.ndarray | None = None  # bag mesh triangles (vertex ids), for the bottom's area and normal
    support_z: float | None = None      # z of the plane the bag's mid-surface rests on [m]


def layout(spec: BagMeshSpec, mat_center, mat_half, board_thickness: float, candle_size, bottom: np.ndarray,
           rim: np.ndarray, triangles: np.ndarray | None = None, support_z: float | None = None) -> Layout:
    """Front and back faces from the wall's vertex numbering (gift_bag_mesh: wall[p, k] = k·P + p)."""
    P, nx, ny = 2 * (spec.nx + spec.ny), spec.nx, spec.ny
    rows = np.arange(spec.nz + 1)[:, None] * P
    front = rows + np.arange(nx + 1)[None, :]
    back = rows + ((2 * nx + ny - np.arange(nx + 1)) % P)[None, :]
    return Layout(spec, np.asarray(rim), np.asarray(bottom), front, back, np.asarray(mat_center, float),
                  np.asarray(mat_half, float), float(board_thickness), np.asarray(candle_size, float),
                  triangles, support_z)


def _tilt_deg(axis: np.ndarray) -> float:
    a = axis / max(np.linalg.norm(axis), 1e-12)
    return float(np.degrees(np.arccos(np.clip(a[2], -1.0, 1.0))))


def _quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def candle_corners(pos, quat_wxyz, size) -> np.ndarray:
    R, h = _quat_wxyz_matrix(quat_wxyz), 0.5 * np.asarray(size, float)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return np.asarray(pos, float) + (signs * h) @ R.T


def bag_frame(L: Layout, bag: np.ndarray) -> np.ndarray:
    """Rows: the bag's x (across the front), y (depth) and z (axis: bottom centroid to rim centroid) in world."""
    z = bag[L.rim].mean(0) - bag[L.bottom].mean(0)
    z = z / max(np.linalg.norm(z), 1e-12)
    x = bag[L.front[0, -1]] - bag[L.front[0, 0]]
    x = x - z * np.dot(x, z)
    x = x / max(np.linalg.norm(x), 1e-12)
    return np.stack([x, np.cross(z, x), z])


def mouth(L: Layout, bag: np.ndarray, lim: Limits) -> dict:
    """Rim area ratio over W·D (projected normal to the axis) and the least front-to-back gap in the middle band."""
    F = bag_frame(L, bag)
    rim2 = bag[L.rim] @ F[:2].T
    area = float(ConvexHull(rim2).volume) if len(rim2) >= 3 else 0.0
    ratio = area / (L.spec.width * L.spec.depth)
    origin = bag[L.bottom].mean(0)
    fx = (bag[L.front] - origin) @ F[0]
    gap = (bag[L.back] - bag[L.front]) @ F[1]
    band = np.abs(fx) <= 0.5 * lim.pinch_band
    least = float(gap[band].min()) if band.any() else 0.0
    ok = ratio >= lim.mouth_area and least >= 0.5 * L.spec.depth
    return {"mouth_open": bool(ok), "mouth_area_ratio": round(ratio, 3), "least_gap_mm": round(least * 1e3, 1)}


def board(L: Layout, bag: np.ndarray, board_pts: np.ndarray, lim: Limits) -> dict:
    """Board vertices within ``board_gap`` above the bottom panel's top face and over the panel (bag frame)."""
    F = bag_frame(L, bag)
    origin = bag[L.bottom].mean(0)
    b = (board_pts - origin) @ F.T
    panel = (bag[L.bottom] - origin) @ F.T
    floor = panel[:, 2].mean() + 0.5 * 135e-6
    under = b[:, 2] - 0.5 * L.board_thickness - floor
    over = Delaunay(panel[:, :2]).find_simplex(b[:, :2]) >= 0
    inside = (under >= -0.5e-3) & (under <= lim.board_gap) & over
    frac = float(inside.mean())
    return {"board_in_bag": frac >= lim.board_fraction, "board_fraction": round(frac, 3)}


def candle(L: Layout, bag: np.ndarray, board_pts: np.ndarray, pos, quat_wxyz, lim: Limits) -> dict:
    corners = candle_corners(pos, quat_wxyz, L.candle_size)
    inside = Delaunay(bag).find_simplex(corners) >= 0
    tilt = _tilt_deg(_quat_wxyz_matrix(quat_wxyz)[:, 2])
    over = Delaunay(board_pts[:, :2]).find_simplex(np.asarray(pos, float)[None, :2])[0] >= 0
    board_top = float(np.median(board_pts[:, 2])) + 0.5 * L.board_thickness
    lift = (corners[:, 2].min() - board_top) * 1e3
    on = over and lim.on_board_mm[0] <= lift <= lim.on_board_mm[1]
    return {"candle_in_bag": bool(inside.all()), "candle_upright": tilt <= lim.upright_deg,
            "candle_on_board": bool(on), "candle_tilt_deg": round(tilt, 1), "candle_lift_mm": round(lift, 1)}


def standing(L: Layout, bag: np.ndarray, lim: Limits) -> dict:
    tilt = _tilt_deg(bag[L.rim].mean(0) - bag[L.bottom].mean(0))
    c = bag[L.bottom].mean(0)[:2] - L.mat_center
    return {"standing": tilt <= lim.standing_deg, "bag_tilt_deg": round(tilt, 1),
            "on_mat": bool(np.all(np.abs(c) <= L.mat_half))}


def bottom_squared(L: Layout, bag: np.ndarray, lim: Limits) -> dict:
    """GA's bottom_squared: area share of the bottom panel within ``flat_band`` of the support plane, and its tilt.
    None when the layout has no triangles or support plane."""
    if L.triangles is None or L.support_z is None:
        return {"bottom_squared": None}
    inside = np.isin(L.triangles, L.bottom).all(axis=1)
    T = L.triangles[inside]
    n = np.cross(bag[T[:, 1]] - bag[T[:, 0]], bag[T[:, 2]] - bag[T[:, 0]])
    area = 0.5 * np.linalg.norm(n, axis=1)
    flat = np.abs(bag[T].mean(1)[:, 2] - L.support_z) <= lim.flat_band
    share = float(area[flat].sum() / area.sum())
    tilt = _tilt_deg(n.sum(0))
    tilt = min(tilt, 180.0 - tilt)                       # unsigned, as GA's abs(nm[2])
    return {"bottom_squared": bool(share >= lim.flat_fraction and tilt <= lim.standing_deg),
            "bottom_flat_fraction": round(share, 3), "bottom_tilt_deg": round(tilt, 1)}


def base_at_rest(L: Layout, bag: np.ndarray, bottom_mark: np.ndarray | None, lim: Limits) -> dict:
    """GA's base_at_rest (bag-purchase): drift of the bottom centroid since ``bottom_mark`` (2 s before the end)."""
    if bottom_mark is None:
        return {"base_at_rest": None}
    drift = float(np.linalg.norm(bag[L.bottom].mean(0) - bottom_mark))
    return {"base_at_rest": bool(drift <= lim.rest_drift), "bottom_drift_mm": round(drift * 1e3, 2)}


GA_CRITERIA = ("hands_off_window", "mouth_open", "standing", "bottom_squared", "on_mat", "base_at_rest",
               "board_in_bag", "candle_in_bag", "candle_upright", "candle_on_board", "bag_not_torn")
NOT_JUDGED = {
    "bag_not_torn": "GA measures the new crack length of the bag (task.py:622-637); the film route records none",
}
UNSET = {"hands_off_window": "no record of the hands-off window was given",
         "bottom_squared": "this layout has no bag triangles or support plane to measure the bottom with",
         "base_at_rest": "no bottom mark 2 s before the end was given"}


def evaluate(L: Layout, bag: np.ndarray, board_pts: np.ndarray, candle_pos, candle_quat_wxyz,
             bottom_mark: np.ndarray | None = None, lim: Limits = Limits(), port_checks: dict | None = None,
             hands_off: bool | None = None) -> dict:
    """Every check on one world's state. ``bottom_mark``: the bottom centroid 2 s before the end. ``success`` is
    every judged GA criterion (and every ``port_checks`` entry) true; ``ga_*`` keys say which criteria were judged.
    ``hands_off``: the window's verdict (gift_hands_off.py), None when no window was recorded."""
    out = {} if hands_off is None else {"hands_off_window": bool(hands_off)}
    out.update(candle(L, bag, board_pts, candle_pos, candle_quat_wxyz, lim))
    out.update(mouth(L, bag, lim))
    out.update(standing(L, bag, lim))
    out.update(board(L, bag, board_pts, lim))
    out.update(bottom_squared(L, bag, lim))
    out.update(base_at_rest(L, bag, bottom_mark, lim))
    criteria = {k: out.get(k) for k in GA_CRITERIA}
    out.update(ga_verdict(criteria, NOT_JUDGED | UNSET, port_checks or {},
                         (HANDS_OFF_NOTE,) if hands_off is not None else ()))
    return out
