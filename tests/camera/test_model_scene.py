# SPDX-License-Identifier: Apache-2.0
"""model_scene: textured meshes (labels, decals) riding on a Newton model's bodies in every world."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera import Camera, CameraError, TexturedMesh, mesh_scene, model_scene
from alloy.camera.meshes import _body_row

LOOK = {"assign": [{"match": "box", "material": "alu"}],
        "materials": {"alu": {"preset": "aluminum_bare"}},
        "lights": [{"type": "area", "eye": [0.3, -0.3, 0.8], "target": [0, 0, 0], "size_m": 0.4, "watts": 60}]}
QUAD = np.array([[-0.03, -0.03, 0.0], [0.03, -0.03, 0.0], [0.03, 0.03, 0.0], [-0.03, 0.03, 0.0]], np.float32)
TRIS = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
RED = np.tile(np.array([230, 20, 20], np.uint8), (8, 8, 1))


def _model(worlds: int = 2):
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="box")
    b.add_shape_box(box, hx=0.08, hy=0.06, hz=0.04)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    return s.finalize(device="cpu")


def _label(**kw) -> TexturedMesh:
    # on the box's top face (z = +0.04 in the body frame), 1 mm proud
    return TexturedMesh("label", QUAD, TRIS, uv=UV, texture=RED, pose=(0, 0, 0.041, 0, 0, 0, 1), **kw)


def _red_columns(img: np.ndarray) -> float:
    """Mean column of the label's red pixels."""
    red = (img[..., 0] > 0.15) & (img[..., 0] > 3.0 * img[..., 1])
    assert red.sum() >= 4, "the label is not visible"
    return float(np.nonzero(red)[1].mean())


def test_a_label_rides_on_its_body_in_every_world():
    model = _model(2)
    state = model.state()
    q = state.body_q.numpy()
    q[1, 0] += 0.1                                   # world 1's box moves 0.1 m along +x (image right)
    state.body_q.assign(q)
    cam = Camera("top", look_at=((0.05, 0.0, 0.5), (0.05, 0.0, 0.0), (0.0, 1.0, 0.0)), width=48, height=48, spp=4)
    img = cam.render(model_scene(model, [_label(body="box")], look=LOOK), state, output="linear").numpy()
    c0, c1 = _red_columns(img[0]), _red_columns(img[1])
    assert c1 - c0 > 6.0                             # 0.1 m is ~11 px at this distance


def test_meshes_without_a_body_are_fixed_in_every_world():
    model = _model(2)
    state = model.state()
    q = state.body_q.numpy()
    q[1, 0] += 0.1
    state.body_q.assign(q)
    fixed = TexturedMesh("mat", QUAD, TRIS, uv=UV, texture=RED, pose=(0.0, 0.0, 0.2, 0, 0, 0, 1))
    cam = Camera("top", look_at=((0.05, 0.0, 0.5), (0.05, 0.0, 0.0), (0.0, 1.0, 0.0)), width=48, height=48, spp=4)
    img = cam.render(model_scene(model, [fixed], look=LOOK), state, output="linear").numpy()
    assert abs(_red_columns(img[0]) - _red_columns(img[1])) < 1.0


def test_body_lookup_by_label_or_last_part_and_errors_say_what_to_change():
    labels = ["a/base", "b/base", "item", "shelf"]
    worlds = np.array([0, 0, 0, -1])
    assert _body_row(labels, worlds, "a/base", 1).tolist() == [0]
    assert _body_row(labels, worlds, "item", 1).tolist() == [2]
    assert _body_row(labels, worlds, "shelf", 3).tolist() == [3, 3, 3]   # outside any world: every world
    with pytest.raises(CameraError, match="full label"):
        _body_row(labels, worlds, "base", 1)
    with pytest.raises(CameraError, match="no body 'lid'"):
        model_scene(_model(1), [_label(body="lid")])
    with pytest.raises(CameraError, match="model_scene"):
        mesh_scene([_label(body="box")])
