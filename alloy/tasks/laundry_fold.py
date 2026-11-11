# SPDX-License-Identifier: Apache-2.0
"""task_laundry_fold: fold a bath towel in half or in thirds (Gym-Anything laundry gap, REPORT.md section 2 cloth row:
``laundry/towel/half_fold`` and ``laundry/towel/thirds_fold``; coordinator 2026-11-06 18:40Z).

The towel is the library's terry towel card (cloth route, ``features.self_contact`` on: WP2's VBD self contact) at
the size of their bath towel (450 x 700 mm, laundry report section 4.4) on a plywood table top. Two hands (corner
patches of the hem, ``laundry_crossings`` grasp "corners") carry each flap round its fold line, let go and the towel
settles; CT-13's crossing count runs throughout.

Gates, from their evaluator (laundry report p.29 Table 19, ``notes/sims/laundry.json`` in [internal note] xb4/gym-anything):
- half: the opposite long hems within a table-projected RMS <= 25 mm and max <= 60 mm (17 samples); convex projected
  area <= 62 % of the flat towel's;
- thirds: area <= 40 % of flat; both long hems inside the base ply's hull (15 mm margin); short-hem extent <= 0.20 m;
  length >= 0.63 m; normal extent <= 25 mm; three-ply fraction >= 0.80 over the base ply's interior (a mesh crease is
  a cell wide: the two crease rows cannot hold three layers, and with them a perfect fold read ~0.82);
- common: no cloth crossing at the end (their score is 0 with a confirmed crossing), stretch <= 0.30, finite.
Their towel is 3 mm loop-pile terry at 450 g/m2; this one keeps the library card's material and model thickness.

Run: ``python -m alloy.tasks.laundry_fold [--folds half|thirds] [--out F]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from alloy.tasks.laundry_crossings import FoldRig, FoldState, simulate

BATH_TOWEL = (0.70, 0.45)              # [m] their authored size, long x short
HEM_SAMPLES = 17
HULL_MARGIN = 0.015                    # [m]
MAX_STRETCH = 0.30


@dataclass(frozen=True)
class FoldVerdict:
    folds: str
    success: bool
    gates: dict[str, bool]
    metrics: dict[str, float]
    wall_s: float


def towel_rig(folds: str = "half", **over: Any) -> FoldRig:
    """The task's rig: bath towel size, two corner grasps on the hem, a low swing (half the fold radius)."""
    base = dict(folds=folds, size=BATH_TOWEL, grasp="corners", apex=0.5, fold_s=1.5, settle_s=0.75)
    return FoldRig(**{**base, **over})


def _rows(q0: np.ndarray, axis: int, side: str) -> np.ndarray:
    """Particle ids of the rest-shape boundary row at the low or high end of ``axis``."""
    v = q0[:, axis]
    tol = 0.25 * float(np.median(np.diff(np.unique(np.round(v, 6)))))
    edge = v.min() if side == "lo" else v.max()
    ids = np.flatnonzero(np.abs(v - edge) < tol)
    return ids[np.argsort(q0[ids, 1 - axis])]


def hem_alignment(q0: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """RMS and max table-plane distance between the two long hems at ``HEM_SAMPLES`` matched points."""
    a, b = _rows(q0, 1, "lo"), _rows(q0, 1, "hi")
    s = np.linspace(0.0, 1.0, HEM_SAMPLES)
    pa = np.stack([np.interp(s, np.linspace(0, 1, len(a)), q[a, k]) for k in (0, 1)], axis=1)
    pb = np.stack([np.interp(s, np.linspace(0, 1, len(b)), q[b, k]) for k in (0, 1)], axis=1)
    d = np.linalg.norm(pa - pb, axis=1)
    return float(np.sqrt(np.mean(d**2))), float(d.max())


def area(q: np.ndarray) -> float:
    return float(ConvexHull(q[:, :2]).volume)


def stretch(q0: np.ndarray, q: np.ndarray, tri: np.ndarray) -> float:
    """Largest relative edge elongation."""
    e = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    l0 = np.linalg.norm(q0[e[:, 0]] - q0[e[:, 1]], axis=1)
    return float((np.linalg.norm(q[e[:, 0]] - q[e[:, 1]], axis=1) / l0 - 1.0).max())


def ply_counts(q: np.ndarray, tri: np.ndarray, pts: np.ndarray, h: float) -> np.ndarray:
    """Cloth layers over each table point ``pts`` [P, 2]: heights where triangles cover it, merged within h/2."""
    a, b, c = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
    out = np.zeros(len(pts), dtype=int)
    v0, v1 = (b - a)[:, :2], (c - a)[:, :2]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(det) > 1e-14
    for i, p in enumerate(pts):
        w = p - a[:, :2]
        u = np.where(ok, (w[:, 0] * v1[:, 1] - w[:, 1] * v1[:, 0]) / np.where(ok, det, 1.0), -1.0)
        v = np.where(ok, (v0[:, 0] * w[:, 1] - v0[:, 1] * w[:, 0]) / np.where(ok, det, 1.0), -1.0)
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        z = np.sort(a[inside, 2] + u[inside] * (b - a)[inside, 2] + v[inside] * (c - a)[inside, 2])
        out[i] = 0 if not len(z) else 1 + int(np.sum(np.diff(z) > 0.5 * h))
    return out


def _inside_hull(pts: np.ndarray, hull_pts: np.ndarray, margin: float) -> np.ndarray:
    """Points within ``margin`` outside the convex hull of ``hull_pts`` (or inside it)."""
    eq = ConvexHull(hull_pts).equations                      # n.x + d <= 0 inside, n unit
    return np.all(pts @ eq[:, :2].T + eq[:, 2] <= margin, axis=1)


def _common(f: FoldState) -> tuple[dict[str, bool], dict[str, float]]:
    st = stretch(f.q0, f.q, f.tri)
    end = f.log[-1][1] if f.log else 0
    worst = max((c for _, c in f.log), default=0)
    gates = {"no_crossing_end": end == 0, "stretch": st <= MAX_STRETCH, "finite": bool(np.isfinite(f.q).all())}
    return gates, {"crossings_end": end, "crossings_max": worst, "stretch": st}


def half_gates(f: FoldState) -> tuple[dict[str, bool], dict[str, float]]:
    rms, worst = hem_alignment(f.q0, f.q)
    ratio = area(f.q) / area(f.q0)
    gates = {"hem_rms": rms <= 0.025, "hem_max": worst <= 0.060, "area": ratio <= 0.62}
    return gates, {"hem_rms": rms, "hem_max": worst, "area_ratio": ratio}


def thirds_gates(f: FoldState) -> tuple[dict[str, bool], dict[str, float]]:
    q0, q = f.q0, f.q
    y0, y1 = q0[:, 1].min(), q0[:, 1].max()
    w = (y1 - y0) / 3.0
    tol = 0.5 * float(np.median(np.diff(np.unique(np.round(q0[:, 1], 6)))))
    base = np.flatnonzero((q0[:, 1] >= y0 + w - tol) & (q0[:, 1] <= y1 - w + tol))   # middle third: the base ply
    hems = np.concatenate([_rows(q0, 1, "lo"), _rows(q0, 1, "hi")])
    short = [_rows(q0, 0, "lo"), _rows(q0, 0, "hi")]
    cell = 2.0 * tol
    inner = base[(q0[base, 1] > y0 + w + cell) & (q0[base, 1] < y1 - w - cell)]   # off the two crease rows
    plies = ply_counts(q, f.tri, q[inner[:: max(len(inner) // 300, 1)], :2], f.h)
    m = {"area_ratio": area(q) / area(q0), "hems_inside": float(_inside_hull(q[hems, :2], q[base, :2],
                                                                             HULL_MARGIN).mean()),
         "short_hem_extent": max(float(np.ptp(q[r, 1])) for r in short), "length": float(np.ptp(q[:, 0])),
         "normal_extent": float(np.ptp(q[:, 2])), "three_ply": float((plies >= 3).mean())}
    gates = {"area": m["area_ratio"] <= 0.40, "hems_inside": m["hems_inside"] == 1.0,
             "short_hem_extent": m["short_hem_extent"] <= 0.20, "length": m["length"] >= 0.63,
             "normal_extent": m["normal_extent"] <= 0.025, "three_ply": m["three_ply"] >= 0.80}
    return gates, m


def evaluate(f: FoldState, folds: str) -> tuple[dict[str, bool], dict[str, float]]:
    g, m = (half_gates if folds == "half" else thirds_gates)(f)
    gc, mc = _common(f)
    return {**g, **gc}, {**m, **mc}


def run(folds: str = "half", device: str = "cpu", **over: Any) -> FoldVerdict:
    t0 = time.time()
    f = simulate(towel_rig(folds, **over), device)
    gates, metrics = evaluate(f, folds)
    return FoldVerdict(folds, all(gates.values()), gates, metrics, round(time.time() - t0, 1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--folds", default="half", choices=("half", "thirds"))
    p.add_argument("--rig", default="{}", help="JSON FoldRig fields over the task's")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="append the verdict as a JSON line")
    a = p.parse_args(argv)
    over = json.loads(a.rig)
    row = json.dumps({"rig": over, **dataclasses.asdict(run(a.folds, a.device, **over))}, default=float)
    print(row)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(row + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
