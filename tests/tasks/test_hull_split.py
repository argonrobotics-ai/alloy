# SPDX-License-Identifier: Apache-2.0
"""Error-bounded convex pieces for large hulls (SB-6, SCR-WP3-6; BUL-0006, BUL-0007): alloy.tasks.hull_split, and the
split T12 scene (every collision hull: at most 64 vertices per piece, within 0.2 mm of the recorded hull, never outside
it)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from alloy.tasks.hull_split import SplitConfig, clip, point_hull_distance, sampled_deviation, split_hull
from alloy.tasks.mjcf_tree import hull_vertex_count
from alloy.tasks.msh import read_msh
from alloy.testing.tasks import ROBOTS

TOL_M = 2e-4
T12 = ROBOTS / "task_t12"
MESHES = ROBOTS / "meshes"


def _sphere(n: int, radius: float, seed: int = 1) -> np.ndarray:
    p = np.random.default_rng(seed).normal(size=(n, 3))
    return radius * p / np.linalg.norm(p, axis=1, keepdims=True)


def _outside(original: np.ndarray, piece: np.ndarray) -> float:
    eq = ConvexHull(original).equations
    return float((piece @ eq[:, :3].T + eq[:, 3]).max())


def test_point_hull_distance_on_a_cube() -> None:
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    q = np.array([[0.5, 0.5, 0.5], [2, 0.5, 0.5], [2, 2, 0.5], [2, 2, 2]])
    assert point_hull_distance(q, cube) == pytest.approx([0.0, 1.0, np.sqrt(2), np.sqrt(3)], abs=1e-12)


def test_clip_splits_exactly() -> None:
    p = _sphere(500, 0.02)
    a, b = clip(p, np.array([1.0, 0.0, 0.0]), 0.004)
    assert ConvexHull(a).volume + ConvexHull(b).volume == pytest.approx(ConvexHull(p).volume, rel=1e-9)


def test_small_hull_is_kept_whole() -> None:
    p = _sphere(40, 0.01)
    res = split_hull(p)
    assert len(res.pieces) == 1 and len(res.pieces[0]) == 40 and res.deviation_m == 0.0


def test_sphere_split_respects_the_bounds() -> None:
    p = _sphere(2000, 0.015)  # a curved 30 mm hull needs several pieces at 64 vertices / 0.2 mm
    res = split_hull(p, SplitConfig(max_verts=64, tol_m=TOL_M))
    assert len(res.pieces) > 1 and max(len(x) for x in res.pieces) <= 64
    assert res.deviation_m <= TOL_M
    assert sampled_deviation(p, res.pieces) <= res.deviation_m + 1e-12
    assert max(_outside(p, x) for x in res.pieces) <= 1e-12


def _report() -> dict:
    return json.loads((T12 / "hull_split.json").read_text())


def test_t12_split_report() -> None:
    rep = _report()
    assert rep["bodies"] == "." and rep["config"]["max_verts"] == 64
    assert rep["max_piece_verts"] <= 64 and rep["max_bound_m"] <= TOL_M
    big = [m for m in rep["meshes"] if m["hull_verts"] > 64]
    assert len(rep["meshes"]) == 81 and len(big) == 74 and max(m["hull_verts"] for m in big) == 1423


def _piece_files() -> dict[str, list[Path]]:
    assets = ET.parse(T12 / "scene.xml").getroot().find("asset")
    out: dict[str, list[Path]] = {}
    for mesh in assets.findall("mesh"):
        base, _, idx = mesh.get("name", "").rpartition("_p")
        if base and idx.isdigit():
            out.setdefault(base, []).append(MESHES / Path(mesh.get("file", "")).name)
    return out


def test_t12_vendored_pieces_cover_the_recorded_hulls() -> None:
    """Independent of the construction: sample each recorded hull's surface and measure the distance to the union of
    the vendored pieces (float32 on disk); check that no piece reaches outside the recorded hull."""
    pieces = _piece_files()
    assert len(pieces) == 74
    for base, files in pieces.items():
        (original_file,) = MESHES.glob(f"{base}-*.msh")
        original = read_msh(original_file).vertices.astype(np.float64)
        verts = tuple(read_msh(f).vertices.astype(np.float64) for f in files)
        assert max(len(v) for v in verts) <= 64, base
        assert sampled_deviation(original, verts, per_face=2) <= TOL_M + 1e-7, base
        assert max(_outside(original, v) for v in verts) <= 1e-7, base


def test_t12_compiled_collision_hulls_at_most_64() -> None:
    m = mujoco.MjModel.from_xml_path(str(T12 / "scene.xml"))
    sizes = [hull_vertex_count(m, int(m.geom_dataid[g])) for g in range(m.ngeom)
             if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert len(sizes) == 200 and max(sizes) <= 64
