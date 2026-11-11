# SPDX-License-Identifier: Apache-2.0
"""Mesh of an open kraft gift bag (square block bottom), for the gift-bag tasks (docs/notes/paper-pack.md).

The bag is Uline's "Cub" kraft shopping bag, S-7098: 8 x 4 1/2 x 10 1/4 in (203 x 114 x 260 mm, width x depth x
height), 60 lb kraft (98 g/m²: the library's ``paper_kraft``, 100 g/m²), "square-bottom kraft paper bags stand
alone". The mesh is the bag opened and squared, as it stands on a counter: four walls and a bottom panel, one ply,
no handles and no turned-in top. Its fold lines are OBJ ``l`` polylines, which the film emitter reads as creases
(weaker hinges, ``alloy.solvers.film.emit.obj_creases``):

- the four vertical side folds, at 90 deg in the open bag;
- the gusset centre folds, flat in the open bag (the bag ships with them folded in);
- the bottom perimeter, at 90 deg;
- the four gusset feet: from each bottom corner up the gusset to its centre fold, the diagonals that let a block
  bottom fold flat.

Frame: the bottom panel's centre at the origin, x across the front, y the depth (gussets at x = ±W/2), z up. Cells are
about 12 mm (13 mm tall), inside the 10-15 mm range where paper on the film route bends as a plate. Triangles wind
counter-clockwise seen from outside.

Regenerate the library file from a worktree root: ``python -m alloy.tasks.gift_bag_mesh``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INCH = 0.0254
OBJ_PATH = Path("library/meshes/paper_gift_bag_kraft.obj")


@dataclass(frozen=True)
class BagMeshSpec:
    width: float = 8.0 * INCH           # 203.2 mm (Uline S-7098)
    depth: float = 4.5 * INCH           # 114.3 mm: the gusset
    height: float = 10.25 * INCH        # 260.35 mm
    nx: int = 16                        # cells across the width (12.7 mm)
    ny: int = 10                        # cells across the depth (11.4 mm; even: the gusset centre fold on vertices)
    nz: int = 20                        # cells up the height (13.0 mm)
    feet: int = 5                       # cells a gusset foot climbs (up to the gusset centre: ny / 2)


@dataclass(frozen=True)
class BagMesh:
    vertices: np.ndarray                # [N, 3] float64, m
    triangles: np.ndarray               # [T, 3] int64
    creases: tuple[np.ndarray, ...]     # polylines of vertex ids (0-based)
    rim: np.ndarray                     # vertex ids of the open top edge, around the perimeter
    bottom: np.ndarray                  # vertex ids of the bottom panel (perimeter included)


def _perimeter(s: BagMeshSpec) -> np.ndarray:
    """[P, 2] points around the bag at the bottom, counter-clockwise seen from above, from the front-left corner."""
    W, D = s.width, s.depth
    xs, ys = np.linspace(-W / 2, W / 2, s.nx + 1), np.linspace(-D / 2, D / 2, s.ny + 1)
    front = np.stack([xs[:-1], np.full(s.nx, -D / 2)], 1)
    right = np.stack([np.full(s.ny, W / 2), ys[:-1]], 1)
    back = np.stack([xs[::-1][:-1], np.full(s.nx, D / 2)], 1)
    left = np.stack([np.full(s.ny, -W / 2), ys[::-1][:-1]], 1)
    return np.concatenate([front, right, back, left])


def _corners(s: BagMeshSpec) -> dict[str, int]:
    """Perimeter indices of the four side folds and the two gusset centre folds."""
    return {"front_left": 0, "front_right": s.nx, "back_right": s.nx + s.ny, "back_left": 2 * s.nx + s.ny,
            "gusset_right": s.nx + s.ny // 2, "gusset_left": 2 * s.nx + s.ny + s.ny // 2}


def _foot_cells(s: BagMeshSpec) -> dict[tuple[int, int], bool]:
    """Wall cells (p, k) on a gusset foot -> True when the fold runs (p, k)-(p+1, k+1), False for (p+1, k)-(p, k+1)."""
    c, P, out = _corners(s), 2 * (s.nx + s.ny), {}
    for start in (c["front_right"], c["back_left"]):            # rising with p from a corner to the centre
        for k in range(s.feet):
            out[((start + k) % P, k)] = True
    for end in (c["back_right"], P):                            # rising against p
        for k in range(s.feet):
            out[((end - 1 - k) % P, k)] = False
    return out


def _bottom_ids(s: BagMeshSpec, wall: np.ndarray, n0: int) -> tuple[np.ndarray, int]:
    """[(nx+1), (ny+1)] ids of the bottom grid: its boundary is the walls' row 0, its interior is new (from n0)."""
    P = 2 * (s.nx + s.ny)
    grid = -np.ones((s.nx + 1, s.ny + 1), np.int64)
    for i in range(s.nx + 1):
        grid[i, 0] = wall[i % P, 0]                             # front, left to right
        grid[i, s.ny] = wall[(2 * s.nx + s.ny - i) % P, 0]      # back
    for j in range(s.ny + 1):
        grid[s.nx, j] = wall[s.nx + j, 0]                       # right gusset
        grid[0, j] = wall[(P - j) % P, 0]                       # left gusset
    n = n0
    for i in range(1, s.nx):
        for j in range(1, s.ny):
            grid[i, j], n = n, n + 1
    return grid, n


def _quad(a: int, b: int, c: int, d: int, ac: bool) -> list[tuple[int, int, int]]:
    """Two counter-clockwise triangles of the quad a b c d (counter-clockwise), split on a-c or b-d."""
    return [(a, b, c), (a, c, d)] if ac else [(a, b, d), (b, c, d)]


def bag_mesh(s: BagMeshSpec = BagMeshSpec()) -> BagMesh:
    if s.ny % 2 or s.feet > s.ny // 2:
        raise ValueError("BagMeshSpec: ny must be even and feet <= ny / 2")
    per, P = _perimeter(s), 2 * (s.nx + s.ny)
    zs = np.linspace(0.0, s.height, s.nz + 1)
    wall = np.arange(P * (s.nz + 1)).reshape(s.nz + 1, P).T          # wall[p, k]
    verts = [np.array([per[p, 0], per[p, 1], zs[k]]) for k in range(s.nz + 1) for p in range(P)]
    grid, n = _bottom_ids(s, wall, P * (s.nz + 1))
    xs, ys = np.linspace(-s.width / 2, s.width / 2, s.nx + 1), np.linspace(-s.depth / 2, s.depth / 2, s.ny + 1)
    verts += [np.array([xs[i], ys[j], 0.0]) for i in range(1, s.nx) for j in range(1, s.ny)]
    feet, tris = _foot_cells(s), []
    for p in range(P):
        for k in range(s.nz):
            a, b, c, d = wall[p, k], wall[(p + 1) % P, k], wall[(p + 1) % P, k + 1], wall[p, k + 1]
            tris += _quad(a, b, c, d, feet.get((p, k), (p + k) % 2 == 0))
    for i in range(s.nx):
        for j in range(s.ny):                                   # seen from below (outside): a d c b
            a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
            tris += _quad(a, d, c, b, (i + j) % 2 == 0)
    c = _corners(s)
    creases = [wall[c[name], :] for name in ("front_left", "front_right", "back_right", "back_left",
                                             "gusset_right", "gusset_left")]
    creases.append(np.append(wall[:, 0], wall[0, 0]))            # bottom perimeter, closed
    for start in (c["front_right"], c["back_left"]):
        creases.append(np.array([wall[(start + k) % P, k] for k in range(s.feet + 1)]))
    for end in (c["back_right"], P):
        creases.append(np.array([wall[(end - k) % P, k] for k in range(s.feet + 1)]))
    return BagMesh(np.asarray(verts), np.asarray(tris, np.int64), tuple(creases), wall[:, s.nz].copy(),
                   np.unique(grid))


def obj_text(m: BagMesh, s: BagMeshSpec = BagMeshSpec()) -> str:
    lines = [f"# open kraft gift bag, square block bottom: {s.width * 1e3:.1f} x {s.depth * 1e3:.1f} x "
             f"{s.height * 1e3:.2f} mm (Uline S-7098 'Cub', 60 lb kraft); generated by alloy/tasks/gift_bag_mesh.py",
             "# `l` polylines are fold lines (side folds, gusset centres, bottom perimeter, gusset feet), read as "
             "crease hinges by the film emitter (alloy.solvers.film.emit.obj_creases)", "# units: m"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in m.triangles]
    lines += ["l " + " ".join(str(int(i) + 1) for i in line) for line in m.creases]
    return "\n".join(lines) + "\n"


def write(root: Path = Path(".")) -> tuple[Path, str]:
    out = root / OBJ_PATH
    out.write_text(obj_text(bag_mesh()))
    return out, hashlib.sha256(out.read_bytes()).hexdigest()


if __name__ == "__main__":
    path, digest = write()
    print(path, digest)
