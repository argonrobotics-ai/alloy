# SPDX-License-Identifier: Apache-2.0
"""Law (TA-7 finding 3): the Gym tasks' gantry gripper is drawn. Its palm and both fingers show in the fast camera's
segmentation, and the RGB of both tracers differs there from the same view without the gripper (CPU).

Newton's MJCF import gives no collider the VISIBLE flag once a file has one visual-only geom (contype = conaffinity
= 0). The gripper's three carriage markers are such geoms, so before ``from_newton.drawn_shapes`` the camera drew only
them (they sit inside the palm, out of sight) and the cube a policy carried seemed to float.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera import Camera, camera_scene
from alloy.camera.from_newton import drawn_shapes
from alloy.envs.robots import gantry_gripper

DEV = "cpu"
VIEW = ((0.25, -0.25, 0.22), (0.0, 0.0, 0.12))  # three-quarter view onto the palm and fingers
PARTS = ("palm", "finger_left", "finger_right")
MIN_PIXELS = 15  # each part covers far more at 96 x 72; a part that is not drawn covers none


def build(with_gripper: bool = True):
    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    if with_gripper:
        scene.add_robot("grip", mjcf=gantry_gripper(), pos=(0.0, 0.0, 0.2))  # as PickPlace-v0 places it
    else:  # the same view without the gripper; a scene needs one moving object, put far out of view
        scene.add_box("far", size=(0.02, 0.02, 0.02), pos=(5.0, 5.0, 0.01), mass=0.05)
    return alloy.build(scene, device=DEV)


def camera() -> Camera:
    return Camera("fixed", look_at=VIEW, width=96, height=72, fov=50, spp=4)


def part_ids(sim) -> dict[str, int]:
    """Segmentation id (2 + body index within its world; one world here) of each gripper part."""
    bodies = list(getattr(sim.engine.model, "body_label", None) or sim.engine.model.body_key)
    return {p: 2 + next(i for i, b in enumerate(bodies) if b.split("/")[-1] == p) for p in PARTS}


def test_every_gripper_part_is_a_drawn_shape():
    sim = build()
    model = sim.engine.model
    drawn, body = drawn_shapes(model), model.shape_body.numpy()
    bodies = list(getattr(model, "body_label", None) or model.body_key)
    for part in PARTS:
        b = next(i for i, name in enumerate(bodies) if name.split("/")[-1] == part)
        assert drawn[body == b].any(), part


def test_gripper_parts_show_in_segmentation_and_rgb():
    sim, empty = build(), build(with_gripper=False)
    out = camera().render(sim, output=("rgb", "segmentation"))
    seg, rgb = out.segmentation.numpy()[0], out.rgb.numpy()[0].astype(np.float64)
    bare = camera().render(empty).numpy()[0].astype(np.float64)
    for part, sid in part_ids(sim).items():
        mask = seg == sid
        assert mask.sum() >= MIN_PIXELS, (part, int(mask.sum()))
        assert np.abs(rgb[mask] - bare[mask]).mean() > 20.0, part  # 8-bit levels: the part, not the floor behind it


@pytest.mark.timeout(600)  # on a fresh GPU machine the first full-tracer render builds its kernels (minutes)
def test_full_tracer_draws_the_gripper_too():
    sim, empty = build(), build(with_gripper=False)
    seg = camera().render(sim, output="segmentation").numpy()[0]
    # the full-material scene is posed by the sim's state (without it, bodies sit at their build pose)
    full = camera().render(camera_scene(sim, quality="full"), sim.engine.state).numpy()[0].astype(np.float64)
    bare = camera().render(camera_scene(empty, quality="full"), empty.engine.state).numpy()[0].astype(np.float64)
    for part, sid in part_ids(sim).items():
        mask = seg == sid
        assert mask.sum() >= MIN_PIXELS, part
        assert np.abs(full[mask] - bare[mask]).mean() > 20.0, part


MIXED = """<mujoco><worldbody>
  <body name="table" pos="0 0 0.2"><geom type="box" size="0.2 0.2 0.01"/></body>
  <body name="cart" pos="0 0 0.5"><joint name="s" type="slide" axis="1 0 0"/>
    <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
    <body name="link" pos="0 0 0.1"><joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.1"/></body></body>
</worldbody><actuator><position joint="s" kp="10"/><position joint="h" kp="10"/></actuator></mujoco>"""


def test_moving_links_are_drawn_and_welded_bodies_keep_the_import_choice():
    """One camera-only marker in a file: the moving link's collider is drawn, the welded table's stays hidden."""
    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    scene.add_robot("m", mjcf=MIXED)
    model = alloy.build(scene, device=DEV).engine.model
    from alloy._vendor import newton
    vis = (model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)) != 0
    drawn, body = drawn_shapes(model), model.shape_body.numpy()
    names = [b.split("/")[-1] for b in (getattr(model, "body_label", None) or model.body_key)]
    link, table = names.index("link"), names.index("table")
    assert drawn[body == link].all() and not vis[body == link].any()
    assert not drawn[body == table].any()


def test_display_meshes_keep_their_hulls_hidden():
    """A body that has a visible shape keeps its hidden colliders hidden (the import's choice for display meshes)."""
    sim = build()
    model = sim.engine.model
    from alloy._vendor import newton
    flags, body = model.shape_flags.numpy(), model.shape_body.numpy()
    vis = (flags & int(newton.ShapeFlags.VISIBLE)) != 0
    drawn = drawn_shapes(model)
    with_visible = np.isin(body, np.unique(body[vis & (body >= 0)]))
    assert np.array_equal(drawn[with_visible], vis[with_visible])
    assert np.array_equal(drawn[body < 0], vis[body < 0])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
