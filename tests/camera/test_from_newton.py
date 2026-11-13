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


def _three_boxes(worlds: int = 1, robot: bool = False):
    """Gemini audit S2-017's scene: three boxes of three materials on the ground, optionally beside a robot whose one
    geom has its own rgba."""
    import alloy
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    for k, (name, mat) in enumerate((("wood", "wood_pine"), ("cardboard", "cardboard_corrugated"),
                                     ("steel", "stainless_304"))):
        s.add_box(name, size=(0.15, 0.15, 0.15), pos=(0.22 * (k - 1), 0.0, 0.08), material=mat)
    if robot:
        s.add_robot("bot", mjcf="<mujoco><worldbody><body name='b' pos='0 0.35 0.05'><joint name='j' type='hinge' "
                                "axis='0 0 1'/><geom type='sphere' size='0.03' mass='0.2' rgba='0.9 0.8 0.1 1'/>"
                                "</body></worldbody></mujoco>")
    return alloy.build(s, device="cpu").engine.model


BOXES = ("wood", "cardboard", "steel")


def _apart(rgbs) -> float:
    """Smallest worst-channel difference between any two of ``rgbs``."""
    return min(float(np.abs(np.subtract(a, b)).max()) for i, a in enumerate(rgbs) for b in rgbs[i + 1:])


def test_shapes_with_no_colour_keep_palette_colours_beside_a_robots_own_colour():
    """S2-017: a robot geom's rgba made the model's colours count as set, and every box drew the palette's first
    blue (worst-channel spread 3 levels of 255). A builder palette colour is no colour of the shape's own."""
    from alloy.camera.from_newton import GROUND_RGB
    sd = scene_from_model(_three_boxes(robot=True))
    rgb = {n: tuple(float(v) for v in np.round(c, 4)) for n, c in zip(sd.inst_name, sd.inst_color)}
    assert _apart([rgb[b] for b in BOXES]) > 0.1, rgb
    assert rgb["ground"] == GROUND_RGB and np.allclose(rgb["worldbody/b/b_geom_0"], (0.9, 0.8, 0.1)), rgb  # its rgba


def test_the_full_tracer_draws_shapes_without_a_look_in_the_fast_tracers_colours():
    """S2-017: quality="full" read the raw shape colours: three boxes and the ground in one palette blue."""
    from alloy.camera.from_newton import GROUND_RGB
    from alloy.camera.model_pack import _materials, _template_shapes
    for robot in (False, True):
        model = _three_boxes(robot=robot)
        shapes = _template_shapes(model)
        names, mats, _ = _materials(model, shapes, {})
        fast = scene_from_model(model)
        want = dict(zip(fast.inst_name, fast.inst_color))
        labels = list(model.shape_label)
        for s, name in zip(shapes, names):                   # shape_rrggbb: the colour it is drawn in
            got = [int(name[-6:][i:i + 2], 16) / 255 for i in (0, 2, 4)]
            assert np.allclose(got, want[labels[s]], atol=0.003), (labels[s], got, want[labels[s]])
        ground = names[[labels[s] for s in shapes].index("ground")]
        assert ground == "shape_%02x%02x%02x" % tuple(round(255 * c) for c in GROUND_RGB)
        assert len({names[[labels[s] for s in shapes].index(b)] for b in BOXES}) == 3


def test_an_object_has_one_palette_colour_in_every_world():
    sd = scene_from_model(_three_boxes(worlds=3))
    for box in BOXES:
        rows = {tuple(np.round(c, 6)) for n, c in zip(sd.inst_name, sd.inst_color) if n == box}
        assert len(rows) == 1, (box, rows)
