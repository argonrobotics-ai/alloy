# SPDX-License-Identifier: Apache-2.0
"""Test helpers for ``alloy.importers.gymanything``: tiny closed meshes and spec files, numpy only.

Example::

    >>> path = write_box_spec(tmp_path / "specs", key="crate", half=(0.1, 0.1, 0.05), mass=1.2)   # doctest: +SKIP
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from alloy.cards import geometry as geo


def box_mesh(half: tuple[float, float, float]) -> geo.MeshData:
    """Closed, outward-wound box centred at the origin."""
    hx, hy, hz = half
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    return geo.MeshData(v, f)


def ring_mesh(poly_rz: list[tuple[float, float]], sectors: int = 32) -> geo.MeshData:
    """Closed solid of revolution about z of an (r, z) polygon that does not touch the axis (e.g. a tube wall)."""
    poly = np.asarray(poly_rz, float)
    n = len(poly)
    th = 2.0 * math.pi * np.arange(sectors) / sectors
    v = np.array([[r * math.cos(t), r * math.sin(t), z] for t in th for r, z in poly])
    faces = []
    for s in range(sectors):
        a, b = s * n, ((s + 1) % sectors) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [[a + i, b + i, b + j], [a + i, b + j, a + j]]
    return geo.MeshData(v, np.asarray(faces, np.int64))


def tube_poly(r_in: float, r_out: float, height: float) -> list[tuple[float, float]]:
    """Wall cross-section of an open tube, counter-clockwise in (r, z)."""
    return [(r_in, 0.0), (r_out, 0.0), (r_out, height), (r_in, height)]


def cited(value: float, basis: str = "datasheet", cite: str = "params.py:1", quote: str = "") -> dict[str, Any]:
    return {"value": value, "basis": basis, "cite": cite, "quote": quote}


def write_spec(folder: Path, mesh: geo.MeshData, **fields: Any) -> Path:
    """Write ``<folder>/<sim>/<key>.json`` + ``.obj``; ``fields`` override the defaults of a minimal valid spec."""
    raw: dict[str, Any] = {"schema": "gar.spec/1", "key": "thing", "sim": "testsim", "name": "Thing",
                           "category": "other", "material": {"name": "test material"}}
    raw.update(fields)
    d = Path(folder) / raw["sim"]
    d.mkdir(parents=True, exist_ok=True)
    geo.write_obj(d / f"{raw['key']}.obj", mesh)
    raw["mesh"] = f"{raw['key']}.obj"
    path = d / f"{raw['key']}.json"
    path.write_text(json.dumps(raw))
    return path
