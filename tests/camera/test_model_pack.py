# SPDX-License-Identifier: Apache-2.0
"""Newton model -> full-material scene: meshes in body frames, smooth normals, look assignment, binding."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera import CameraError
from alloy.camera.model_pack import pack_from_model, studio_sky
from alloy.camera.smooth import corner_normals

LOOK = {"assign": [{"match": "box", "material": "alu"}],
        "materials": {"alu": {"preset": "aluminum_bare"}},
        "lights": [{"type": "area", "eye": [0.5, -0.4, 0.6], "target": [0, 0, 0], "size_m": 0.4, "watts": 40}]}


def _model(worlds: int = 2):
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="box")
    b.add_shape_box(box, hx=0.08, hy=0.06, hz=0.04)
    b.add_shape_sphere(box, xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), radius=0.03)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    return s.finalize(device="cpu")


def test_corner_normals_smooth_sphere_flat_box():
    from alloy._vendor import newton
    sph = newton.Mesh.create_sphere(1.0, compute_inertia=False)
    n = corner_normals(sph.vertices, np.asarray(sph.indices).reshape(-1, 3))
    pos = np.asarray(sph.vertices)[np.asarray(sph.indices).reshape(-1, 3)]
    radial = pos / np.linalg.norm(pos, axis=-1, keepdims=True)
    assert np.mean(np.sum(n * radial, axis=-1)) > 0.995  # smooth: corner normals follow the sphere
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-5)  # also on the zero-area pole triangles
    box = newton.Mesh.create_box(1.0, 1.0, 1.0, compute_inertia=False)
    nb = corner_normals(box.vertices, np.asarray(box.indices).reshape(-1, 3))
    assert np.allclose(np.abs(nb).max(axis=-1), 1.0, atol=1e-6)  # 90 degree edges stay sharp


def test_pack_from_model_binds_meshes_to_each_worlds_body():
    model = _model(2)
    pk, frame, (body, local, present) = pack_from_model(model, LOOK)
    names = [m.meta["materials"][0] for m in frame.meshes]
    assert len(frame.meshes) == 3 and names.count("alu") == 2  # "box" matches the box body's shapes
    assert body.shape == (3, 2) and (body[1:, 0] >= 0).all() and (body[1:, 1] >= 0).all()
    assert body[1, 0] != body[1, 1]  # each world's own copy of the box body
    assert np.allclose(local[1:, :, 6], 1.0)  # bodies: identity, the mesh is in the body frame
    assert present.all()  # rigid shapes are drawn in every world
    assert frame.lights[0]["type"] == "AREA" and set(pk.static["materials"]) == set(names)


def test_default_materials_follow_shape_colours_and_studio_sky():
    pk, frame, _ = pack_from_model(_model(1))
    assert all(n.startswith("shape_") for n in pk.static["materials_used_by_meshes"])
    assert frame.lights[0]["type"] == "SUN" and pk.static["world"]["use_mis"] is False
    sky = studio_sky()
    assert sky[0, 0, 0] == pytest.approx(1.0, abs=0.01) and sky[-1, 0, 0] == pytest.approx(0.03, abs=0.01)


def test_bad_look_rule_says_what_to_change():
    with pytest.raises(CameraError, match="valid regex"):
        pack_from_model(_model(1), {"assign": [{"match": "(", "material": "x"}]})


def _card_model():
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    ball = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="rubber_ball_40mm")
    b.add_shape_sphere(ball, radius=0.02, label="rubber_ball_40mm/hull_0")
    other = b.add_body(xform=wp.transform((0.1, 0.0, 0.1), wp.quat_identity()), label="my_box")
    b.add_shape_box(other, hx=0.02, hy=0.02, hz=0.02, label="my_box")
    return b.finalize(device="cpu")


def test_a_library_object_without_a_look_renders_with_its_card_look():
    from alloy.camera.card_looks import look_colors
    from alloy.camera.from_newton import scene_from_model
    model = _card_model()
    red = look_colors(["rubber_ball_40mm/hull_0"])[0]
    assert red[0] > 0.5 > red[1] + 0.3  # the library look is red, not grey
    pk, frame, _ = pack_from_model(model)
    mats = {m.meta["name"]: m.meta["materials"][0] for m in frame.meshes}
    assert mats["rubber_ball_40mm/hull_0"] == "rubber_ball_40mm.body" and mats["my_box"].startswith("shape_")
    sd = scene_from_model(model)
    colours = dict(zip(sd.inst_name, map(tuple, sd.inst_color)))
    assert np.allclose(colours["rubber_ball_40mm/hull_0"], red) and not np.allclose(colours["my_box"], red)


def test_the_scene_look_still_wins_over_the_card_look():
    look = {"assign": [{"match": "rubber_ball", "material": "steel"}],
            "materials": {"steel": {"preset": "steel_brushed"}}}
    _, frame, _ = pack_from_model(_card_model(), look)
    assert {m.meta["name"]: m.meta["materials"][0] for m in frame.meshes}["rubber_ball_40mm/hull_0"] == "steel"
