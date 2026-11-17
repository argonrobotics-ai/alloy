# SPDX-License-Identifier: Apache-2.0
"""Law: a camera-only marker on a robot link no longer hides that link's collision geoms (ease-cpu item 8, BATCH-0117).

Newton's MJCF import hides every collider once a file has one visual-only geom (contype = conaffinity = 0). The camera
drew the colliders of moving bodies without a visible shape again, but a hand that carried both a marker and a jug of
box geoms kept the jug hidden, so the water in it floated. Now a moving body's colliders of MJCF groups 0 to 2 that
stick out of its display shapes are drawn, in both cameras, over as many pixels as without the marker. Collision
geoms that a display shape stands in for stay hidden: group-3 geoms (MuJoCo's viewer hides them too) and group-0
copies inside a display shape.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.camera import Camera, camera_scene
from alloy.camera.from_newton import drawn_shapes

VIEW = ((0.3, -0.35, 0.6), (0.0, 0.0, 0.42))  # from above the hand, into the jug
MIN_PIXELS = 40  # the hand covers far more at 96 x 72; a hand that is not drawn covers none
JUG = ("jug_floor", "jug_x0", "jug_x1", "jug_y0", "jug_y1")


def pourer(marker: bool) -> str:
    """A slide and a wrist tilt; the hand is a jug of five boxes, and with ``marker`` a 4 mm visual-only sphere at
    its centre."""
    tip = '<geom name="tip" type="sphere" size="0.004" contype="0" conaffinity="0"/>' if marker else ""
    return f"""<mujoco><compiler angle="radian"/><worldbody>
  <body name="carriage" pos="0 0 0.05"><joint name="slide" type="slide" axis="1 0 0" range="-0.2 0.2" damping="5"/>
    <geom name="carriage_box" type="box" size="0.03 0.03 0.02" mass="0.5"/>
    <body name="hand" pos="0 0 0.06"><joint name="tilt" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
      <geom name="jug_floor" type="box" size="0.03 0.03 0.003" mass="0.05"/>
      <geom name="jug_x0" type="box" size="0.003 0.03 0.03" pos="-0.03 0 0.03" mass="0.02"/>
      <geom name="jug_x1" type="box" size="0.003 0.03 0.03" pos="0.03 0 0.03" mass="0.02"/>
      <geom name="jug_y0" type="box" size="0.03 0.003 0.03" pos="0 -0.03 0.03" mass="0.02"/>
      <geom name="jug_y1" type="box" size="0.03 0.003 0.03" pos="0 0.03 0.03" mass="0.02"/>
      {tip}</body></body></worldbody>
  <actuator><position joint="slide" kp="400" kv="40"/><position joint="tilt" kp="50" kv="5"/></actuator></mujoco>"""


def build(mjcf: str | None):
    """The robot on a table; ``None``: the same view without it (a scene needs one moving object, put far away)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.4, 0.3), pos=(0.0, 0.0, 0.15), static=True)
    if mjcf is None:
        scene.add_box("far", size=(0.02, 0.02, 0.02), pos=(5.0, 5.0, 0.01), mass=0.05)
    else:
        scene.add_robot("pourer", mjcf=mjcf, pos=(0.0, 0.0, 0.3))
    return alloy.build(scene, device="cpu")


def shapes_named(model, names) -> np.ndarray:
    labels = list(getattr(model, "shape_label", None) or model.shape_key)
    return np.array([i for i, lb in enumerate(labels) if lb.split("/")[-1] in names])


@pytest.fixture(scope="module")
def sims():
    return build(pourer(False)), build(pourer(True))


def test_a_marker_on_the_hand_keeps_the_jug_drawn(sims):
    plain, marked = sims
    model = marked.engine.model
    jug = shapes_named(model, JUG)
    assert len(jug) == len(JUG)
    vis = (model.shape_flags.numpy()[jug] & int(newton.ShapeFlags.VISIBLE)) != 0
    assert not vis.any()                                        # the importer hid them: the case under test
    assert drawn_shapes(model)[jug].all()


def camera() -> Camera:
    return Camera("fixed", look_at=VIEW, width=96, height=72, fov=50, spp=4)


def hand_mask(sim) -> np.ndarray:
    """Pixels of the hand body in the fast camera's segmentation (2 + body index; one world)."""
    bodies = [b.split("/")[-1] for b in (getattr(sim.engine.model, "body_label", None) or sim.engine.model.body_key)]
    return camera().render(sim, output="segmentation").numpy()[0] == 2 + bodies.index("hand")


@pytest.mark.timeout(600)  # on a fresh machine the first full-tracer render builds its kernels (minutes)
def test_both_tracers_draw_the_jug_under_a_marker(sims):
    """The hand covers as many pixels with the marker as without it, and both tracers draw it there (their RGB
    differs from the same view without the robot)."""
    plain, marked = sims
    mask, ref = hand_mask(marked), hand_mask(plain)
    assert ref.sum() >= MIN_PIXELS and abs(int(mask.sum()) - int(ref.sum())) <= 2, (int(mask.sum()), int(ref.sum()))
    bare = build(None)
    fast = camera().render(marked).numpy()[0].astype(np.float64)
    # the full-material scene is posed by the sim's state (without it, bodies sit at their build pose)
    full = camera().render(camera_scene(marked, quality="full"), marked.engine.state).numpy()[0].astype(np.float64)
    bare_fast = camera().render(bare).numpy()[0].astype(np.float64)
    bare_full = camera().render(camera_scene(bare, quality="full"), bare.engine.state).numpy()[0].astype(np.float64)
    assert np.abs(fast[mask] - bare_fast[mask]).mean() > 20.0     # 8-bit levels: the jug, not the floor behind it
    assert np.abs(full[mask] - bare_full[mask]).mean() > 20.0


DISPLAY = """<mujoco><worldbody>
  <body name="link" pos="0 0 0.4"><joint name="h" type="hinge" axis="0 1 0"/>
    <geom name="shell" type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0" group="2"/>
    <geom name="copy" type="box" size="0.051 0.05 0.05" mass="0.2"/>
    <geom name="joint_cap" type="capsule" fromto="0 0 -0.12 0 0 0.05" size="0.03" group="3" mass="0.1"/>
  </body></worldbody><actuator><position joint="h" kp="10"/></actuator></mujoco>"""


def test_collision_geoms_a_display_shape_stands_for_stay_hidden():
    """A group-0 copy inside the display box and a group-3 capsule sticking out of it are not drawn; the shell is."""
    model = build(DISPLAY).engine.model
    drawn = drawn_shapes(model)
    assert drawn[shapes_named(model, ("shell",))].all()
    assert not drawn[shapes_named(model, ("copy", "joint_cap"))].any()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
