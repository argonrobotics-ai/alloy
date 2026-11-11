# SPDX-License-Identifier: Apache-2.0
"""Scene item looks on the fast tracer (``alloy.camera.item_fast``): colours, projected colour maps, camera_scene."""
from __future__ import annotations

import numpy as np
import pytest

from alloy.camera import CameraError, camera_scene
from alloy.camera.item_fast import projected, with_item_looks
from alloy.camera.scene import MeshGeom, SceneData
from tests.camera.scan_fixture import box_mesh, png_bytes, quadrants

WOOD = {"preset": "abs_black", "texture": "polyhaven:test_wood", "scale_m": 0.05, "color_gain": 0.8}
PAINT = {"preset": "abs_black", "texture": None, "base_color": [0.1, 0.2, 0.3], "color_gain": 2.0}
LOOK = {"materials": {"floor:wood": WOOD, "wall:paint": PAINT},
        "assign": [{"match": "^floor($|[/#])", "material": "floor:wood"},
                   {"match": "^wall($|[/#])", "material": "wall:paint"}]}


def _scene() -> SceneData:
    pts, tris, _ = box_mesh()
    names = ["floor", "wall", "other"]
    return SceneData([MeshGeom("box", pts, tris)], [0, 0, 0], [-1] * 3, [-1] * 3, [[0, 0, 0, 0, 0, 0, 1]] * 3,
                     [[0.5, 0.5, 0.5]] * 3, 1, names)


@pytest.fixture
def wood_png(tmp_path, monkeypatch):
    path = tmp_path / "wood_color.png"
    path.write_bytes(png_bytes(quadrants(16)))
    from alloy.camera.looks import textures
    monkeypatch.setattr(textures, "local_texture_files", lambda ref, res: {"color": path})
    return path


def test_projected_uvs_tile_each_face_along_its_axis():
    pts, tris, _ = box_mesh()
    m = projected(MeshGeom("box", pts, tris), 0.05, 0, "box:wood")
    assert len(m.points) == 3 * len(tris) and m.texture == 0
    p, uv = m.points.reshape(-1, 3, 3), m.uv.reshape(-1, 3, 2)
    for t in range(len(tris)):
        a = int(np.argmax(np.abs(np.cross(p[t, 1] - p[t, 0], p[t, 2] - p[t, 0]))))
        u, v = [(1, 2), (0, 2), (0, 1)][a]
        assert np.allclose(uv[t], np.stack([p[t, :, u], -p[t, :, v]], -1) / 0.05, atol=1e-5)


def test_item_looks_colour_and_texture_the_shapes_they_name(wood_png):
    sd = with_item_looks(_scene(), LOOK)
    floor, wall, other = range(3)
    mesh = sd.meshes[sd.inst_mesh[floor]]
    assert mesh.texture >= 0 and mesh.uv is not None and sd.inst_textured[floor]
    assert np.array_equal(sd.textures[mesh.texture], quadrants(16))
    assert np.allclose(sd.inst_color[floor], 0.8)                          # the colour map times the gain
    assert not sd.inst_textured[wall] and np.allclose(sd.inst_color[wall], [0.2, 0.4, 0.6])  # base colour x gain
    assert sd.inst_mesh[other] == 0 and np.allclose(sd.inst_color[other], 0.5)              # no look: unchanged
    assert with_item_looks(_scene(), None).meshes[0].texture == -1


def test_an_unreadable_colour_map_leaves_the_base_colour(monkeypatch):
    from alloy.camera.looks import textures

    def missing(ref, res):
        raise CameraError(f"texture {ref!r} is not cached", "fetch it")
    monkeypatch.setattr(textures, "local_texture_files", missing)
    sd = with_item_looks(_scene(), LOOK)
    assert sd.meshes[sd.inst_mesh[0]].texture == -1 and not sd.inst_textured[0] and not sd.textures


def test_the_fast_scene_of_a_sim_draws_its_item_looks_and_caches_by_them():
    import alloy
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.025), mass=0.2)
    s.items["block"]["look"] = {"materials": {"paint": dict(PAINT)}, "default": "paint"}
    sim = alloy.build(s, device="cpu")
    with_looks = camera_scene(sim, quality="fast", device="cpu").sd
    without = camera_scene(sim, quality="fast", device="cpu", items=None).sd
    rgb = {n: c for n, c in zip(with_looks.inst_name, with_looks.inst_color)}
    assert np.allclose(rgb["block"], [0.2, 0.4, 0.6])
    assert not np.allclose(dict(zip(without.inst_name, without.inst_color))["block"], [0.2, 0.4, 0.6])
