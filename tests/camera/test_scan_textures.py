# SPDX-License-Identifier: Apache-2.0
"""Scan textures, part A: the glb reader and the swap of a scanned object's display mesh for its textured glb mesh."""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera import scan_textures as st
from alloy.camera.from_newton import scene_from_model
from alloy.camera.glb import GlbError, read_glb, write_glb
from alloy.camera.scene import MeshGeom, SceneData
from tests.camera.scan_fixture import TILE, box_mesh, png_bytes, quadrants, tile_card, to_gltf

QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
QUAD_UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)


def _sim(root, item=TILE, worlds=1, look=None):
    scene = alloy.Scene(dt=0.01, worlds=worlds, roots=[root])
    scene.add_ground()
    scene.add_object("tile", item, pos=(0, 0, 0))
    if look is not None:
        scene.items["tile"]["look"] = look
    return alloy.build(scene, device="cpu")


def test_glb_round_trip_keeps_points_tris_uv_image_and_factor(tmp_path):
    path = write_glb(tmp_path / "q.glb", [{"points": QUAD, "tris": QUAD_TRIS, "uv": QUAD_UV, "image": b"img",
                                           "mime": "image/png", "factor": (0.5, 0.25, 1.0, 1.0)}])
    (part,) = read_glb(path)
    assert np.array_equal(part.points, QUAD) and np.array_equal(part.tris, QUAD_TRIS)
    assert np.array_equal(part.uv, QUAD_UV) and part.image.data == b"img" and part.image.mime == "image/png"
    assert part.factor == (0.5, 0.25, 1.0, 1.0)


def test_glb_without_triangles_or_not_a_glb_raises(tmp_path):
    (tmp_path / "x.glb").write_bytes(b"not a glb at all, just text")
    with pytest.raises(GlbError):
        read_glb(tmp_path / "x.glb")


def test_trimesh_default_factor_beside_an_image_stands_for_one():
    from alloy.camera.glb import GlbImage, GlbPart
    img = GlbImage(b"", "image/png")
    assert st.part_color(GlbPart(QUAD, QUAD_TRIS, QUAD_UV, (0.4, 0.4, 0.4, 1.0), img)) == (1.0, 1.0, 1.0)
    assert st.part_color(GlbPart(QUAD, QUAD_TRIS, QUAD_UV, (0.5, 0.4, 0.4, 1.0), img)) == (0.5, 0.4, 0.4)
    assert st.part_color(GlbPart(QUAD, QUAD_TRIS, None, (0.4, 0.4, 0.4, 1.0))) == (0.4, 0.4, 0.4)  # no image


def test_body_frame_turn_matches_the_importers():
    pts, _, _ = box_mesh()
    assert np.allclose(st.to_body(to_gltf(pts), (1, 1, 1)), pts)
    assert np.allclose(st.to_body(to_gltf(pts), (2, 1, 1))[:, 0], 2 * pts[:, 0])


def test_tile_scan_replaces_its_display_mesh_with_the_textured_one(tmp_path):
    sim = _sim(tile_card(tmp_path, image=quadrants(32)), worlds=2)
    plain = scene_from_model(sim.engine.model)
    sd = st.with_scan_textures(plain, sim)
    rows = [i for i, n in enumerate(sd.inst_name) if n == "tile/visual"]
    assert len(rows) == 2 and sd.num_instances == plain.num_instances
    mesh = sd.meshes[sd.inst_mesh[rows[0]]]
    assert mesh.texture == 0 and mesh.uv.shape == (24, 2) and len(sd.textures) == 1
    assert np.array_equal(sd.textures[0], quadrants(32))
    assert np.array_equal(sd.inst_color[rows], np.ones((2, 3), np.float32)) and sd.inst_textured.all()
    assert np.allclose(mesh.points.min(0), (-0.05, -0.05, 0.0), atol=1e-6)
    assert np.allclose(mesh.points.max(0), (0.05, 0.05, 0.01), atol=1e-6)
    assert np.array_equal(sd.inst_body, plain.inst_body) and np.array_equal(sd.inst_group, plain.inst_group)


def test_without_a_glb_or_with_a_look_the_object_keeps_its_flat_colour(tmp_path):
    sim = _sim(tile_card(tmp_path / "a", glb=False))
    plain = scene_from_model(sim.engine.model)
    assert st.with_scan_textures(plain, sim) is plain
    sim2 = _sim(tile_card(tmp_path / "b"), look={"color": [0.2, 0.5, 0.8]})
    assert st.scan_sources(sim2) == []


def test_a_glb_that_does_not_match_the_mesh_is_not_used(tmp_path, caplog):
    disp = MeshGeom("d", box_mesh()[0], box_mesh()[1])
    path = write_glb(tmp_path / "big.glb", [{"points": to_gltf(box_mesh()[0]) * 1.5, "tris": box_mesh()[1],
                                             "uv": box_mesh()[2], "image": png_bytes(quadrants(8))}])
    assert st.scan_parts(path, disp, np.array([0, 0, 0, 0, 0, 0, 1], np.float32)) is None
    assert "does not match" in caplog.text


def test_without_pillow_the_object_keeps_its_flat_colour(tmp_path, monkeypatch):
    sim = _sim(tile_card(tmp_path))
    plain = scene_from_model(sim.engine.model)
    monkeypatch.setattr(st, "decode_image", lambda data: None)
    sd = st.with_scan_textures(plain, sim)
    assert not sd.textured and np.array_equal(sd.inst_color, plain.inst_color)


def test_large_images_are_capped(monkeypatch):
    monkeypatch.setenv(st.TEXTURE_MAX_ENV, "16")
    monkeypatch.setattr(st, "_IMAGES", {})
    img, _ = st.decode_image(png_bytes(quadrants(64)))
    assert img.shape == (16, 16, 3) and tuple(img[0, 0]) == (230, 40, 40) and tuple(img[-1, -1]) == (240, 240, 240)


def test_scene_data_checks_texture_fields():
    m = MeshGeom("q", QUAD, QUAD_TRIS, QUAD_UV, texture=0)
    one = np.zeros((1, 7), np.float32)
    one[0, 6] = 1
    with pytest.raises(Exception, match="texture 0 of 0"):
        SceneData([m], [0], [0], [-1], one, [[1, 1, 1]])
    with pytest.raises(Exception, match="UVs for"):
        MeshGeom("q", QUAD, QUAD_TRIS, QUAD_UV[:3])
    sd = SceneData([m], [0], [0], [-1], one, [[1, 1, 1]], textures=[quadrants(4)])
    assert sd.textured and sd.inst_textured.tolist() == [True]


def test_textures_survive_beside_a_deformable(tmp_path):
    """The deformables' per-world meshes are added to the textured scene, not to the flat one (a soft or cloth object
    beside a scan once dropped every texture)."""
    from alloy.camera.camera import camera_scene
    scene = alloy.Scene(dt=0.01, roots=[tile_card(tmp_path)])
    scene.add_ground()
    scene.add_object("tile", TILE, pos=(0, 0, 0))
    scene.add_cloth("sheet", size=(0.1, 0.1), pos=(0.3, 0, 0.2), resolution=4, mass=0.01)
    sim = alloy.build(scene, device="cpu")
    sd = camera_scene(sim, device="cpu", quality="fast").sd
    tile = [sd.meshes[sd.inst_mesh[i]] for i, n in enumerate(sd.inst_name) if n == "tile/visual"]
    assert tile and tile[0].texture == 0 and len(sd.textures) == 1
    assert len(sd.inst_textured) == sd.num_instances


def test_soft_surface_uvs_follow_the_build_pose():
    """scan_soft: each particle-surface vertex takes the UV of the nearest glb vertex, the glb turned by the item's
    orientation and centred on the surface's box; a box of another size is refused."""
    from alloy.camera.scan_soft import _rotation, seam_fraction, surface_uvs
    pts, tris, _ = box_mesh()
    uv = (pts[:, :2] - pts[:, :2].min(0)) / np.ptp(pts[:, :2], axis=0)  # planar: one UV per position
    q = (0.0, 0.0, np.sin(0.35), np.cos(0.35))  # 40 degrees about z
    rot = _rotation(q)
    rest = pts @ rot.T + np.array([0.3, -0.2, 0.05], np.float32)  # the soft surface at its build pose
    got = surface_uvs(rest.astype(np.float32), tris, rot, pts, uv.astype(np.float32))
    assert got is not None and np.allclose(got, uv, atol=1e-6)
    assert surface_uvs(rest.astype(np.float32) * 1.5, tris, rot, pts, uv.astype(np.float32)) is None
    n = 10  # a fine grid: no triangle spans more than a tenth of the image unless a UV jumps (a seam)
    grid = np.stack(np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij"), -1).reshape(-1, 2) / n
    cells = [(i * (n + 1) + j, (i + 1) * (n + 1) + j, i * (n + 1) + j + 1) for i in range(n) for j in range(n)]
    gtris = np.array(cells, np.int32)
    assert seam_fraction(grid.astype(np.float32), gtris) == 0.0
    jump = grid.astype(np.float32).copy()
    jump[0] = (0.9, 0.9)
    assert 0.0 < seam_fraction(jump, gtris) < 0.05
