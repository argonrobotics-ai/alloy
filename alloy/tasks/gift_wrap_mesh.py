# SPDX-License-Identifier: Apache-2.0
"""Mesh of a box wrapped in gift paper (docs/notes/paper-pack.md; the start of the reference task set's unwrap-gift).

The paper is cut as a flat sheet and folded onto the box analytically: a girth wrap round the box's length with the
seam overlapping under the box, and each end folded in the usual order (side flaps in, which tucks a triangle under
the top and bottom at each corner; the top flap down; the bottom flap up over it). The reference's present is a gift
box 184 x 134 x 55 mm in 81 g/m² wrap with a 40 mm seam; this one is 187 x 132 x 55 mm with a 44 mm seam and 44 mm end
flaps, so that every fold lies on the mesh's 11 mm grid.

- **Layers.** A face's paper stands one contact radius off the box (plus 0.1 mm), and every further layer one pitch
  (two contact radii) out. Order on an end face, from the box: side flaps, the top's tucks, the top flap, the bottom's
  tucks, the bottom flap (when the flaps overlap more than half the end, the bottom's tucks lie over the top flap).
- **Folds.** Each vertex is placed by the fold map of every region it belongs to; on a fold it sits outside every
  layer it joins (per face normal, the outermost layer's offset), so no two layers touch. Where the outer seam layer
  wraps the five layers of an end, its rest edges are up to 1.51 times the paper's flat length: the paper there starts
  slack, not stretched (the film takes its rest shape from this geometry).
- **OBJ.** ``v`` the wrapped positions (the box's bottom centre at the origin), ``vt`` the flat layout of each vertex
  (m; the film places tear lines on it, alloy/solvers/film/layout.py), ``f`` triangles, ``l`` fold lines (crease
  hinges: the girth corners, both end edges and the eight tuck diagonals).

Regenerate the library file from a worktree root: ``python -m alloy.tasks.gift_wrap_mesh``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBJ_PATH = Path("library/meshes/paper_gift_wrapped.obj")


@dataclass(frozen=True)
class WrapSpec:
    c: float = 0.011                    # cell [m]
    nW: int = 12                        # box width (y) in cells: 132 mm
    nH: int = 5                         # box height (z): 55 mm
    nL: int = 17                        # box length (x): 187 mm
    ne: int = 4                         # end flaps beyond the box: 44 mm (over half the 55 mm height: they overlap)
    nseam: int = 4                      # seam overlap under the box: 44 mm
    r: float = 6.0e-4                   # mid-surface off the box: the film's 0.5 mm contact radius + 0.1 mm
    pitch: float = 1.0e-3               # layer to layer: two contact radii

    @property
    def nu(self) -> int:
        return 2 * (self.nW + self.nH) + self.nseam

    @property
    def nv(self) -> int:
        return self.nL + 2 * self.ne

    @property
    def box(self) -> tuple[float, float, float]:
        return (self.nL * self.c, self.nW * self.c, self.nH * self.c)


SIDE, TOP_TRI, TOP_FLAP, BOT_TRI, BOT_FLAP = 1, 2, 3, 4, 5      # layers on an end face, from the box out


def _panels(s: WrapSpec) -> dict[str, tuple[int, int]]:
    """u-index spans round the girth from the seam's inner edge: bottom (inner), +y, top, -y, bottom (outer)."""
    b1 = s.nW // 2 + s.nseam // 2
    return {"bot_in": (0, b1), "side_p": (b1, b1 + s.nH), "top": (b1 + s.nH, b1 + s.nH + s.nW),
            "side_m": (b1 + s.nH + s.nW, b1 + 2 * s.nH + s.nW), "bot_out": (b1 + 2 * s.nH + s.nW, s.nu)}


def _in(x: float, lo: float, hi: float) -> bool:
    return lo - 1e-9 <= x <= hi + 1e-9


def _face(s: WrapSpec, name: str, t: float) -> tuple[float, float, tuple[int, int, int]]:
    """(y, z, outward normal) of girth position ``t`` cells along panel ``name``."""
    W2, H, h = s.nW / 2, s.nH, s.nseam // 2
    return {"bot_in": (-h + t, 0.0, (0, 0, -1)), "side_p": (W2, t, (0, 1, 0)), "top": (W2 - t, H, (0, 0, 1)),
            "side_m": (-W2, H - t, (0, -1, 0)), "bot_out": (-W2 + t, 0.0, (0, 0, -1))}[name]


def _end(s: WrapSpec, name: str, y: float, z: float, sd: float, ex: float, en: tuple, seam_outer: int) -> list:
    """Images on an end face (``sd`` cells beyond the end edge) of a girth point at (y, z)."""
    W2, H = s.nW / 2, s.nH
    if name in ("side_p", "side_m"):
        return [(np.array([ex, np.sign(y) * (W2 - sd), z]), en, SIDE)]
    top = name == "top"
    if top:                             # distance to the panel's own corner: the top has two, each bottom layer one
        a, ys = W2 - abs(y), np.sign(y)
    elif name == "bot_in":
        a, ys = W2 - y, 1.0
    else:
        a, ys = W2 + y, -1.0
    out = []
    if a >= sd - 1e-9:                  # flap: folds about the end edge
        out.append((np.array([ex, y, H - sd if top else sd]), en, TOP_FLAP if top else BOT_FLAP + seam_outer))
    if a <= sd + 1e-9:                  # tuck: (a, s) -> (W/2 - s, H - a) or (W/2 - s, a)
        out.append((np.array([ex, ys * (W2 - sd), H - a if top else a]), en, TOP_TRI if top else BOT_TRI))
    return out


def _maps(s: WrapSpec, i: int, j: int) -> list[tuple[np.ndarray, tuple, int]]:
    """(point in cells, outward normal, layer) for every region that holds grid vertex (i, j)."""
    v = j - s.nv / 2                    # cells from the box's middle along x
    sgn = 1.0 if v >= 0 else -1.0
    send = abs(v) - s.nL / 2            # cells beyond the end edge
    out = []
    for name, (lo, hi) in _panels(s).items():
        if not _in(i, lo, hi):
            continue
        y, z, nrm = _face(s, name, i - lo)
        seam_outer = int(name == "bot_out" and _in(y, -(s.nseam // 2), s.nseam // 2))
        if send <= 1e-9:
            out.append((np.array([v, y, z]), nrm, 1 + seam_outer))
        if send >= -1e-9:
            out += _end(s, name, y, z, max(send, 0.0), sgn * s.nL / 2, (int(sgn), 0, 0), seam_outer)
    return out


def _place(s: WrapSpec, maps) -> tuple[np.ndarray, float]:
    """The regions' common point, offset along each face normal by the outermost layer it joins; and the spread of the
    region images (0 for a consistent fold map)."""
    pts = np.array([p for p, _, _ in maps])
    lay: dict[tuple, int] = {}
    for _, n, layer in maps:
        lay[n] = max(lay.get(n, 0), layer)
    pos = pts.mean(0) * s.c + sum(np.array(n, float) * (s.r + (k - 1) * s.pitch) for n, k in lay.items())
    return pos, float(np.ptp(pts, axis=0).max()) * s.c


def _fold_diagonals(s: WrapSpec) -> dict[tuple[int, int], bool]:
    """Cells crossed by the 45 deg tuck folds: True = (i, j)-(i+1, j+1)."""
    P, out = _panels(s), {}
    for j0, dj in ((s.ne + s.nL, 1), (s.ne, -1)):
        for i0, di in ((P["bot_in"][1], -1), (P["top"][0], 1), (P["top"][1], -1), (P["bot_out"][0], 1)):
            for k in range(s.ne):
                out[(i0 + (k if di > 0 else -k - 1), j0 + (k if dj > 0 else -k - 1))] = (di > 0) == (dj > 0)
    return out


def _folds(s: WrapSpec, idx) -> list[list[int]]:
    P, lines = _panels(s), []
    for u in sorted({span[0] for span in P.values() if span[0] > 0}):            # girth corners, full length
        lines.append([idx(u, j) for j in range(s.nv + 1)])
    for j0 in (s.ne, s.ne + s.nL):                                                 # end edges, round the girth
        lines.append([idx(i, j0) for i in range(s.nu + 1)])
    for j0, dj in ((s.ne + s.nL, 1), (s.ne, -1)):                                  # tuck diagonals
        for i0, di in ((P["bot_in"][1], -1), (P["top"][0], 1), (P["top"][1], -1), (P["bot_out"][0], 1)):
            lines.append([idx(i0 + di * k, j0 + dj * k) for k in range(s.ne + 1)])
    return lines


@dataclass(frozen=True)
class WrapMesh:
    vertices: np.ndarray                # [N, 3] wrapped [m]
    layout: np.ndarray                  # [N, 2] flat sheet [m]
    triangles: np.ndarray               # [T, 3]
    folds: tuple[list[int], ...]


def wrap_mesh(s: WrapSpec = WrapSpec()) -> WrapMesh:
    idx = lambda i, j: i * (s.nv + 1) + j  # noqa: E731
    V, F = [], []
    for i in range(s.nu + 1):
        for j in range(s.nv + 1):
            p, spread = _place(s, _maps(s, i, j))
            if spread > 1e-9:
                raise ValueError(f"wrap fold map: vertex ({i}, {j}) has images {spread:.3g} m apart")
            V.append(p)
            F.append((i * s.c, j * s.c))
    diag, T = _fold_diagonals(s), []
    for i in range(s.nu):
        for j in range(s.nv):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [(a, b, c), (a, c, d)] if diag.get((i, j), (i + j) % 2 == 0) else [(a, b, d), (b, c, d)]
    return WrapMesh(np.asarray(V), np.asarray(F), np.asarray(T, np.int64), tuple(_folds(s, idx)))


def obj_text(m: WrapMesh, s: WrapSpec = WrapSpec()) -> str:
    L, W, H = (x * 1e3 for x in s.box)
    lines = [f"# a {L:.0f} x {W:.0f} x {H:.0f} mm box wrapped in gift paper: girth wrap, seam under the box, "
             "ends folded; generated by alloy/tasks/gift_wrap_mesh.py",
             "# v: wrapped positions (box bottom centre at the origin); vt: the flat sheet (m); l: fold lines",
             "# units: m"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.vertices]
    lines += [f"vt {u:.6f} {w:.6f}" for u, w in m.layout]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in m.triangles]
    lines += ["l " + " ".join(str(int(k) + 1) for k in line) for line in m.folds]
    return "\n".join(lines) + "\n"


def write(root: Path = Path(".")) -> tuple[Path, str]:
    out = root / OBJ_PATH
    out.write_text(obj_text(wrap_mesh()))
    return out, hashlib.sha256(out.read_bytes()).hexdigest()


if __name__ == "__main__":
    path, digest = write()
    print(path, digest)
