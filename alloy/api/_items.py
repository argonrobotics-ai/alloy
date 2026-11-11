# SPDX-License-Identifier: Apache-2.0
"""Argument checks and derived values for ``alloy.Scene``'s ``add_*`` methods (kept out of scene.py for its size).

Example: ``_overrides(0.2, None, 0.8, None, static=False)`` -> ``{"mass": 0.2, "friction": 0.8}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alloy.api import _args as A
from alloy.api.errors import SceneError


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


def _cloth_vertices(resolution: Any, lx: float, ly: float) -> int:
    """Vertex count of a cloth grid. Example: ``_cloth_vertices(21, 0.5, 0.5)`` -> 441."""
    if isinstance(resolution, (int, float)):
        n = int(resolution)
        if n < 3:
            raise SceneError(f"resolution must be >= 3, got {resolution!r}", "pass resolution=21 (vertices per side)",
                             field="resolution")
        long, short = max(lx, ly), min(lx, ly)
        return n * max(3, int(round(n * short / long)))
    nx, ny = (int(v) for v in A.sizes(resolution, 2, "resolution"))
    return nx * ny


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
    return {"vertices": v.tolist(), "faces": f.astype(int).tolist()}


def _robot_kind(path: str) -> str:
    """mjcf / urdf / usd from a robot file's suffix. Example: ``_robot_kind("arm.xml")`` -> "mjcf"."""
    suffix = Path(path).suffix.lower()
    kinds = {".xml": "mjcf", ".mjcf": "mjcf", ".urdf": "urdf", ".usd": "usd", ".usda": "usd", ".usdc": "usd"}
    if suffix not in kinds or not Path(path).is_file():
        raise SceneError(f"robot file {path!r} not found or not .xml/.urdf/.usd",
                         "pass an existing MJCF (.xml), URDF (.urdf) or USD file", field="path")
    return kinds[suffix]
