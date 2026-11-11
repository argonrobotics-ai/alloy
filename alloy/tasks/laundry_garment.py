# SPDX-License-Identifier: Apache-2.0
"""A flat T-shirt for the laundry tasks (Gym-Anything laundry/tee: sleeve_tuck_fold, turn_face_down): front and back
panels from one outline (body, two short sleeves, a neck opening), lying flat, sewn along the side seams, the sleeve
seams and the shoulders; the hem, the neck and the two cuffs stay open.

One mesh: the back panel's mid-surface h/2 above the table, the front's ``gap`` x h above the back, sewn outline
vertices shared by both panels half way between. Points: a triangular lattice of spacing ``cell`` inside the outline
plus the outline sampled at ``cell``; Delaunay, triangles whose centroid is outside the outline dropped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class TeeShape:
    """Outline [m], front view, x across the chest, y from the hem up (a men's M T-shirt, roughly)."""
    width: float = 0.50              # body
    length: float = 0.68             # hem to shoulder line
    armhole: float = 0.20            # armpit below the shoulder line
    sleeve: float = 0.19             # sleeve length past the side seam (under-seam)
    sleeve_open: float = 0.16        # cuff width
    shoulder_drop: float = 0.03      # shoulder point in from the side, and down from the neck line
    neck: tuple[float, float] = (0.09, 0.05)       # half-width, depth (front and back alike)


@dataclass
class Tee:
    vertices: np.ndarray             # [N, 3]
    triangles: np.ndarray            # [T, 3]
    panel: np.ndarray                # [N]: 0 back, 1 front, 2 sewn (shared)
    region: np.ndarray               # [N]: 0 body, 1 left sleeve, 2 right sleeve (by the side seam lines)
    cuffs: list[np.ndarray] = field(default_factory=list)    # vertex ids within one cell of each cuff (left, right)
    shape: TeeShape = TeeShape()


OPEN = ("hem", "cuff_r", "neck", "cuff_l")        # outline edges left unsewn


def outline(s: TeeShape) -> list[tuple[np.ndarray, str]]:
    """Outline as (start point, name of the edge from it to the next point), counter-clockwise from the left hem
    corner: hem, right side seam, right sleeve under-seam, right cuff, right sleeve top seam, right shoulder, neck,
    left shoulder, left sleeve top seam, left cuff, left under-seam, left side seam."""
    w, L, a = 0.5 * s.width, s.length, s.armhole
    nx, nd = s.neck
    under = np.array([w + s.sleeve, L - a - 0.06])
    top = under + s.sleeve_open * np.array([0.42, 0.91])
    pts = [((-w, 0.0), "hem"), ((w, 0.0), "side_r"), ((w, L - a), "under_r"), (tuple(under), "cuff_r"),
           (tuple(top), "top_r"), ((w - s.shoulder_drop, L), "shoulder_r"), ((nx, L), "neck"),
           ((nx * 0.6, L - nd), "neck"), ((-nx * 0.6, L - nd), "neck"), ((-nx, L), "shoulder_l"),
           ((-w + s.shoulder_drop, L), "top_l"), ((-top[0], top[1]), "cuff_l"), ((-under[0], under[1]), "under_l"),
           ((-w, L - a), "side_l")]
    return [(np.asarray(p, dtype=float), name) for p, name in pts]


def _boundary(segs, cell: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Outline points every ~``cell``; a point is sewn when its edge is sewn and, at a corner, the edge before too."""
    pts, sewn, names = [], [], []
    for i, (p, name) in enumerate(segs):
        q = segs[(i + 1) % len(segs)][0]
        n = max(int(np.ceil(np.linalg.norm(q - p) / cell)), 1)
        here = name not in OPEN
        for k in range(n):
            pts.append(p + (q - p) * k / n)
            sewn.append(here and (k > 0 or segs[i - 1][1] not in OPEN))
            names.append(name)
    return np.asarray(pts), np.asarray(sewn), names


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd ray test of points [P, 2] against a closed polygon [V, 2]."""
    x, y = pts[:, :1], pts[:, 1:2]
    x0, y0, x1, y1 = poly[:, 0], poly[:, 1], np.roll(poly[:, 0], -1), np.roll(poly[:, 1], -1)
    dy = np.where(y1 != y0, y1 - y0, 1e-30)
    hit = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / dy + x0)
    return hit.sum(axis=1) % 2 == 1


def _outline_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30)[None], 0, 1)
    return np.linalg.norm(pts[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(axis=1)


def _interior(poly: np.ndarray, cell: float) -> np.ndarray:
    """A triangular lattice of spacing ``cell`` inside the outline, 0.4 cell clear of it."""
    lo, hi = poly.min(0), poly.max(0)
    dy = cell * np.sqrt(3.0) / 2.0
    pts = np.asarray([(x + (0.5 * cell if j % 2 else 0.0), y) for j, y in enumerate(np.arange(lo[1] + dy, hi[1], dy))
                      for x in np.arange(lo[0] + cell, hi[0], cell)])
    return pts[_inside(poly, pts) & (_outline_distance(poly, pts) > 0.4 * cell)]


def make_tee(h: float, cell: float = 0.02, gap: float = 2.0, s: TeeShape = TeeShape()) -> Tee:
    """The two-panel T-shirt mesh (module docstring); ``h`` sheet thickness [m]."""
    segs = outline(s)
    bpts, bsewn, bnames = _boundary(segs, cell)
    ipts = _interior(bpts, cell)
    xy = np.vstack([bpts, ipts])
    tri = Delaunay(xy).simplices
    cen = xy[tri].mean(axis=1)
    tri = tri[_inside(bpts, cen)]
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    area = 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    tri = tri[area > 0.05 * cell**2]          # outline slivers: a sewn edge would join 4 triangles
    sewn = np.concatenate([bsewn, np.zeros(len(ipts), bool)])
    n = len(xy)
    back_id = np.arange(n)
    front_id = np.where(sewn, back_id, n + np.cumsum(~sewn) - 1)    # sewn vertices are shared
    nf = int((~sewn).sum())
    z_back, z_front = 0.5 * h, 0.5 * h + gap * h
    verts = np.zeros((n + nf, 3))
    verts[:n, :2], verts[:n, 2] = xy, np.where(sewn, 0.5 * (z_back + z_front), z_back)
    verts[n:, :2], verts[n:, 2] = xy[~sewn], z_front
    tris = np.vstack([tri[:, ::-1], front_id[tri]])                  # back faces down, front faces up
    panel = np.concatenate([np.where(sewn, 2, 0), np.ones(nf, int)])
    region = _regions(verts[:, :2], s)
    names = np.asarray(bnames + [""] * len(ipts))
    cuffs = [_near_cuff(verts, back_id, front_id, names, side, cell) for side in ("cuff_l", "cuff_r")]
    return Tee(verts, tris, panel, region, cuffs, s)


def _regions(xy: np.ndarray, s: TeeShape) -> np.ndarray:
    w = 0.5 * s.width
    return np.where(xy[:, 0] < -w, 1, np.where(xy[:, 0] > w, 2, 0))


def _near_cuff(verts, back_id, front_id, names, side: str, cell: float) -> np.ndarray:
    """Both panels' vertices within one cell of the cuff (the hands' grasp)."""
    on = np.flatnonzero(names == side)
    ids = np.unique(np.concatenate([back_id[on], front_id[on]]))
    c = verts[ids, :2]
    d = np.min(np.linalg.norm(verts[:, None, :2] - c[None], axis=2), axis=1)
    return np.flatnonzero(d <= 1.01 * cell)


def write_obj(tee: Tee, path: Path) -> str:
    """Write the mesh as OBJ; returns its sha256."""
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in tee.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in tee.triangles]
    data = ("\n".join(lines) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
