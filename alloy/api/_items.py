# SPDX-License-Identifier: Apache-2.0
"""Argument checks and derived values for ``alloy.Scene``'s ``add_*`` methods (kept out of scene.py for its size).

Example: ``_overrides(0.2, None, 0.8, None, static=False)`` -> ``{"mass": 0.2, "friction": 0.8}``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alloy.api import _args as A
from alloy.api._say import warn
from alloy.api.errors import SceneError
from alloy.contact.hull_limits import hull_vertices_problem


def _box_volume(half: Sequence[float]) -> float:
    """Volume of a box of half extents ``half``. Example: ``_box_volume([0.1, 0.1, 0.1])`` -> 0.008."""
    return 8.0 * half[0] * half[1] * half[2]


def _overrides(mass: Any, density: Any, friction: Any, restitution: Any, static: bool) -> dict[str, float]:
    """The item's checked value overrides. Example: ``_overrides(0.2, None, None, None, False)``."""
    if mass is not None and density is not None:
        raise SceneError("both mass and density given", "pass only one of mass= (kg) or density= (kg/m^3)",
                         field="mass")
    if static and mass is not None:
        raise SceneError("a static object has no mass", "drop mass= for static objects (or set static=False)",
                         field="mass")
    out: dict[str, float] = {}
    if mass is not None:
        out["mass"] = A.positive(mass, "mass", "kg")
    if density is not None:
        out["density"] = A.positive(density, "density", "kg/m^3")
    if friction is not None:
        out["friction"] = A.positive(friction, "friction", "-")
    if restitution is not None:
        r = float(restitution)
        if not 0.0 <= r <= 1.0:
            raise SceneError(f"restitution must be in [0, 1], got {restitution!r}", "pass restitution=0.5",
                             field="restitution")
        out["restitution"] = r
    return out


CLOTH_CELLS_MAX = 1600   # the most cells a sheet gets: alloy.scene.emitters.deformable.CLOTH_VERTS_TARGET (a law)


def _cloth_cells(name: str, resolution: Any, lx: float, ly: float) -> int:
    """The cells cloth ``name`` asks for, the card's budget: ``resolution`` along the longer side and the shorter
    side's share (at least 3), or nx * ny. The build lays them out as near-square cells (``cloth_grid``), so a sheet
    of n x m cells has (n + 1) x (m + 1) vertices; it warns when that grid is not the one asked for.

    Example: ``_cloth_cells("towel", 20, 0.5, 0.5)`` -> 400 (20 x 20 cells, 21 x 21 = 441 vertices)."""
    asked = _asked_grid(resolution, lx, ly)
    cells = asked[0] * asked[1]
    got = cloth_grid(cells, lx, ly)
    if got != asked:
        why = (f"a sheet gets at most {CLOTH_CELLS_MAX:,} cells" if cells > CLOTH_CELLS_MAX else
               f"its cells are as near square as whole counts allow, so they follow the sheet's {lx:g} x {ly:g} m")
        warn("add_cloth %r: resolution %r gives a %d x %d-cell sheet (%d vertices), not %d x %d: %s", name,
             resolution, got[0], got[1], (got[0] + 1) * (got[1] + 1), asked[0], asked[1], why)
    return cells


def _asked_grid(resolution: Any, lx: float, ly: float) -> tuple[int, int]:
    """Cells along x and y that ``resolution`` asks for (checked)."""
    if not isinstance(resolution, (int, float)):
        nx, ny = (int(v) for v in A.sizes(resolution, 2, "resolution"))
        return nx, ny
    n = int(resolution)
    if n < 3:
        raise SceneError(f"resolution must be >= 3, got {resolution!r}", "pass resolution=20 (cells along the longer "
                         "side: 21 vertices)", field="resolution")
    m = max(3, int(round(n * min(lx, ly) / max(lx, ly))))
    return (n, m) if lx >= ly else (m, n)


def cloth_grid(cells: int, lx: float, ly: float) -> tuple[int, int]:
    """Cells along x and y of the sheet the build makes for ``cells``: square cells of side sqrt(lx * ly / cells),
    at most ``CLOTH_CELLS_MAX`` in all, each side's count rounded (``alloy.scene.emitters.deformable.cloth_mesh``);
    it has (nx + 1) * (ny + 1) vertices. Example: ``cloth_grid(400, 0.5, 0.5)`` -> (20, 20)."""
    cell = math.sqrt(lx * ly / min(cells, CLOTH_CELLS_MAX))
    return max(1, round(lx / cell)), max(1, round(ly / cell))


COLLISIONS = ("convex", "decompose", "sdf", "mesh")


def _mesh_shape(scale: Any, collision: str, hull_vertices: int | None) -> dict[str, Any]:
    """``add_mesh``'s checked ``shape`` dict (``hull_vertices`` written only when set). Example:
    ``_mesh_shape(1.0, "convex", 64)``."""
    A.one_of(collision, COLLISIONS, "collision")
    s = [float(scale)] * 3 if isinstance(scale, (int, float)) else A.sizes(scale, 3, "scale")
    if problem := hull_vertices_problem(hull_vertices, collision):
        raise SceneError(problem, "leave hull_vertices=None (the hull keeps within 0.1 mm of the mesh's own), or give "
                         "an int from 4 to 2048 on a collision='convex' mesh (64: the earlier fast hull)",
                         field="hull_vertices")
    mesh: dict[str, Any] = {"path": "", "sha256": "", "scale": s, "collision": collision}
    if hull_vertices is not None:
        mesh["hull_vertices"] = int(hull_vertices)
    return {"mesh": mesh}


def _mesh_spec(path: Any, vertices: Any, faces: Any) -> dict[str, Any]:
    """A checked mesh source: a file or arrays. Example: ``_mesh_spec("part.obj", None, None)``."""
    if (path is None) == (vertices is None):
        raise SceneError("add_mesh needs exactly one of path / vertices", "pass path='part.obj', or vertices= and "
                         "faces= arrays", field="path")
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.is_file() or p.suffix.lower() not in (".obj", ".stl"):
            raise SceneError(f"mesh file {str(path)!r} not found or not .obj/.stl", "pass an existing .obj or .stl "
                             "file", field="path")
        return {"file": str(p)}
    import numpy as np  # noqa: PLC0415

    v, f = np.asarray(vertices, dtype=float), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise SceneError(f"vertices must be (V, 3) floats and faces (F, 3) ints, got {v.shape} / {f.shape}",
                         "pass vertices=np.array([[x, y, z], ...]), faces=np.array([[i, j, k], ...])", field="faces")
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise SceneError("faces index outside the vertex list", "use 0-based vertex indices", field="faces")
    bad = int((~np.isfinite(v)).sum())
    if bad:
        raise SceneError(f"vertices hold {bad} non-finite coordinate(s) (NaN or inf)",
                         "give every vertex finite x, y, z coordinates in metres", field="vertices")
    return {"vertices": v.tolist(), "faces": f.astype(int).tolist()}


def mixed_winding_edges(vertices: Any, faces: Any) -> int:
    """Edges that both of their faces run the same way, in a closed mesh (0 for a consistently wound or an open
    mesh): such a mesh's signed volume, so the mass a body gets from it, is wrong (half a cube's faces flipped: 0).
    Vertices at the same coordinates are merged first, as an OBJ repeats them along its texture seams.

    Example: ``mixed_winding_edges(v, f)`` -> 0 for ``trimesh.creation.box()``'s vertices and faces."""
    import numpy as np  # noqa: PLC0415

    from alloy.cards.geometry import MeshData, edge_report  # noqa: PLC0415

    v, f = np.asarray(vertices, dtype=float).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return 0
    _, same = np.unique(v, axis=0, return_inverse=True)
    open_edges, twice = edge_report(MeshData(v, same.reshape(-1)[f]))
    return twice if open_edges == 0 else 0


#: robot file suffix -> importer (a .usdz package opens as any USD stage does)
ROBOT_KINDS = {".xml": "mjcf", ".mjcf": "mjcf", ".urdf": "urdf", ".usd": "usd", ".usda": "usd", ".usdc": "usd",
               ".usdz": "usd"}


def _robot_kind(path: str) -> str:
    """mjcf / urdf / usd from a robot file's suffix. Example: ``_robot_kind("arm.xml")`` -> "mjcf"."""
    suffix = Path(path).suffix.lower()
    if suffix not in ROBOT_KINDS or not Path(path).is_file():
        raise SceneError(f"robot file {path!r} not found or not {'/'.join(ROBOT_KINDS)}",
                         "pass an existing MJCF (.xml), URDF (.urdf) or USD (.usd/.usda/.usdc/.usdz) file",
                         field="path")
    return ROBOT_KINDS[suffix]
