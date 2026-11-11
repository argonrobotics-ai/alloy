# SPDX-License-Identifier: Apache-2.0
"""Geometry facts and library meshes: extents, I/O, volume, watertightness (agrees with Newton), wall thickness,
deterministic regeneration of library/meshes."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from alloy.cards import geometry as geo
from alloy.cards import meshgen as mg
from alloy.cards import types as ct
from alloy.cards.library import default_root

MESH_DIR = default_root() / "meshes"


@pytest.mark.parametrize("kind, size, ext, member", [
    ("box", (0.01, 0.02, 0.03), (0.06, 0.04, 0.02), 0.02), ("sphere", (0.01,), (0.02, 0.02, 0.02), 0.02),
    ("capsule", (0.01, 0.02), (0.06, 0.02, 0.02), 0.02), ("cylinder", (0.01, 0.05), (0.1, 0.02, 0.02), 0.02),
    ("cone", (0.01, 0.002), (0.02, 0.02, 0.004), 0.004), ("ellipsoid", (0.01, 0.02, 0.03), (0.06, 0.04, 0.02), 0.02),
])
def test_primitive_extents_follow_newton_conventions(kind, size, ext, member) -> None:
    p = ct.Primitive(kind, size)
    assert geo.primitive_extents(p) == pytest.approx(ext) and geo.primitive_min_member(p) == pytest.approx(member)
    assert geo.primitive_extents(ct.Primitive("plane", (0.0, 2.0))) == (math.inf, 2.0, 0.0)


def test_obj_round_trip_and_groups(tmp_path: Path) -> None:
    m = mg.merge({"a": mg.extrude(mg.rect(0.01, 0.01), 0.0, 0.01), "b": mg.translate(mg.lathe([(0, 0), (0.005, 0), (0.005, 0.01), (0, 0.01)], 16), (0.05, 0, 0))})
    mg.write_obj(tmp_path / "m.obj", m, header="test")
    back = geo.read_obj(tmp_path / "m.obj")
    assert np.array_equal(back.vertices, m.vertices) and np.array_equal(back.faces, m.faces)
    assert {k: v.tolist() for k, v in back.groups.items()} == {k: v.tolist() for k, v in m.groups.items()}


def test_stl_binary_and_ascii(tmp_path: Path) -> None:
    m = mg.extrude(mg.rect(0.02, 0.02), 0.0, 0.02)
    tri = m.vertices[m.faces].astype(np.float32)
    data = b"\0" * 80 + struct.pack("<I", len(tri)) + b"".join(np.zeros(3, np.float32).tobytes() + t.tobytes() + b"\0\0" for t in tri)
    (tmp_path / "b.stl").write_bytes(data)
    ascii_ = "solid x\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {a} {b} {c}\n" for a, b, c in t) +
                                   "endloop\nendfacet\n" for t in tri.tolist()) + "endsolid x\n"
    (tmp_path / "a.stl").write_text(ascii_)
    for name in ("b.stl", "a.stl"):
        s = geo.read_mesh(tmp_path / name)
        assert geo.is_watertight(s) and geo.signed_volume(s) == pytest.approx(8e-6, rel=1e-5)


def test_watertight_rules() -> None:
    cube = mg.extrude(mg.rect(0.01, 0.01), 0.0, 0.01)
    assert geo.is_watertight(cube) and geo.signed_volume(cube) == pytest.approx(1e-6)
    open_ = geo.MeshData(cube.vertices, cube.faces[1:])
    flipped = geo.MeshData(cube.vertices, np.concatenate([cube.faces[:1, ::-1], cube.faces[1:]]))
    assert not geo.is_watertight(open_) and not geo.is_watertight(flipped)


def test_wall_thickness_and_obb() -> None:
    assert geo.wall_thickness(mg.box_pocket((0.05, 0.04, 0.03), (0.046, 0.036, 0.028))) == pytest.approx(0.002, rel=1e-6)
    pts = np.array([[x, y, z] for x in (0, 0.1) for y in (0, 0.02) for z in (0, 0.01)], float)
    c, s = math.cos(0.3), math.sin(0.3)
    rot = pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    assert geo.obb_extents(rot) == pytest.approx((0.1, 0.02, 0.01), rel=1e-6)


@pytest.mark.parametrize("name", sorted(mg.LIBRARY_MESHES))
def test_wall_thickness_within_a_cutoff_is_exact(name: str) -> None:
    """``within`` (the card validator's cutoff) skips far triangles: a thinner wall reads exactly as the full probe,
    a thicker one reads inf."""
    m = geo.read_obj(MESH_DIR / f"{name}.obj")
    full = geo._wall_thickness(m, 1024, 128)
    for within in (0.5 * full, 1.01 * full, 4.0 * full, 0.002):
        assert geo._wall_thickness(m, 1024, 128, within) == (full if full < within else math.inf), within


def test_wall_thickness_within_a_cutoff_on_a_pocket() -> None:
    pocket = mg.box_pocket((0.05, 0.04, 0.03), (0.046, 0.036, 0.028))                     # 2 mm walls
    assert geo.wall_thickness(pocket, within=0.003) == pytest.approx(0.002, rel=1e-6)
    assert geo.wall_thickness(pocket, within=0.0015) == math.inf


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    return mg.write_library_meshes(tmp_path_factory.mktemp("meshes"))


@pytest.mark.parametrize("name", sorted(mg.LIBRARY_MESHES))
def test_library_mesh_regenerates_bit_for_bit(name: str, regenerated: dict[str, str]) -> None:
    assert regenerated[name] == geo.sha256_file(MESH_DIR / f"{name}.obj"), "regenerate: python -m alloy.cards.meshgen library/meshes"
    m = geo.read_obj(MESH_DIR / f"{name}.obj")
    assert geo.is_watertight(m) and geo.signed_volume(m) > 0


def test_watertight_agrees_with_newton() -> None:
    from alloy._vendor import newton

    for name in sorted(mg.LIBRARY_MESHES):
        m = geo.read_obj(MESH_DIR / f"{name}.obj")
        nm = newton.Mesh(m.vertices.astype(np.float32), m.faces.reshape(-1).astype(np.int32), compute_inertia=False)
        assert bool(nm.is_watertight) == geo.is_watertight(m), name


def test_mesh_checks_are_computed_once_per_mesh_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """wall_thickness / is_closed / obb_extents are memoized by mesh content (SCR-GATE-1: rebuilt libraries re-validate
    the same meshes); a different mesh, or other arguments, is computed afresh."""
    monkeypatch.setattr(geo, "_DERIVED", {})
    calls: list[int] = []
    real = geo._wall_thickness
    monkeypatch.setattr(geo, "_wall_thickness", lambda m, r, c, *w: calls.append(r) or real(m, r, c, *w))
    a = mg.box_pocket((0.05, 0.04, 0.03), (0.046, 0.036, 0.028))
    same = geo.MeshData(a.vertices.copy(), a.faces.copy())
    other = mg.box_pocket((0.05, 0.04, 0.03), (0.044, 0.036, 0.028))
    assert geo.wall_thickness(a) == geo.wall_thickness(same) == pytest.approx(0.002, rel=1e-6)
    assert geo.wall_thickness(other) == pytest.approx(0.002, rel=1e-6) and len(calls) == 2
    geo.wall_thickness(a, max_rays=16)
    assert calls == [1024, 1024, 16]
    assert geo.is_closed(a) and geo.obb_extents(a.vertices) == geo.obb_extents(same.vertices)
