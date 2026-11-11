# SPDX-License-Identifier: Apache-2.0
"""Newton model -> camera scene: shapes, worlds, bodies, colours, filters."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera import CameraError
from alloy.camera.from_newton import scene_from_model


def _builder():
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), label="box")
    b.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
    b.add_shape_sphere(body, xform=wp.transform((0.0, 0.0, 0.4), wp.quat_identity()), radius=0.05)
    return b


def test_replicated_worlds_share_meshes():
    from alloy._vendor import newton
    s = newton.ModelBuilder()
    s.replicate(_builder(), 3)
    sd = scene_from_model(s.finalize(device="cpu"))
    assert sd.num_worlds == 3 and sd.num_instances == 9
    assert len(sd.meshes) == 3  # plane, box, sphere tessellated once
    assert sorted(set(sd.inst_group.tolist())) == [0, 1, 2]
    assert (sd.inst_body[sd.inst_group == 1] >= 0).sum() == 2
    assert np.isfinite(sd.inst_color).all() and (sd.inst_color <= 1.0).all()


def test_box_mesh_has_the_shape_size():
    sd = scene_from_model(_builder().finalize(device="cpu"))
    box = next(m for m in sd.meshes if len(m.points) == 24)
    assert np.allclose(box.points.max(0), [0.1, 0.2, 0.3], atol=1e-6)


def test_include_filter_and_empty_scene():
    model = _builder().finalize(device="cpu")
    sd = scene_from_model(model, include=lambda s, label: "ground" not in label)
    assert sd.num_instances == 2
    with pytest.raises(CameraError, match="no visible shapes"):
        scene_from_model(model, include=lambda s, label: False)


def test_a_scene_assembled_in_one_colour_gets_palette_colours_and_a_grey_ground():
    import alloy
    from alloy.camera.from_newton import GROUND_RGB
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.6), mass=0.2)
    model = alloy.build(s, device="cpu").engine.model
    assert len({tuple(c) for c in model.shape_color.numpy().round(6)}) == 1  # what the assembly leaves
    sd = scene_from_model(model)
    rgb = {n: tuple(c) for n, c in zip(sd.inst_name, sd.inst_color)}
    assert rgb["ground"] == GROUND_RGB and rgb["table"] != rgb["block"] and GROUND_RGB not in (rgb["table"], rgb["block"])


def test_a_palette_coloured_ground_draws_grey_beside_shapes_with_their_own_colour():
    """TA-2 F2: a robot's rgba makes the model's colours count as set, and the ground (added with no colour: a
    builder palette colour) then shared the objects' palette blue. An authored plane colour is kept."""
    from alloy._vendor import newton
    from alloy.camera.from_newton import GROUND_RGB
    b = newton.ModelBuilder()
    b.add_ground_plane(color=None)                                   # the builder's palette colour, as Alloy's
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), label="link")
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, color=(0.9, 0.1, 0.1))
    b.add_shape_plane(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), width=1.0, length=1.0,
                      color=(0.2, 0.6, 0.3))
    sd = scene_from_model(b.finalize(device="cpu"))
    colors = [tuple(np.round(c, 4)) for c in sd.inst_color]
    assert colors == [GROUND_RGB, (0.9, 0.1, 0.1), (0.2, 0.6, 0.3)], colors
