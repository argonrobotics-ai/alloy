# SPDX-License-Identifier: Apache-2.0
"""Law (TA-7 finding 3, frame export): a viewer fed by frame export shows the Gym tasks' gantry gripper. The topology
lists the shapes the camera draws (``alloy.camera.from_newton.drawn_shapes``): PickPlace-v0's palm and both fingers
get their boxes, while welded bodies and world shapes keep the import's VISIBLE choice (CPU).

Newton's MJCF import gives no collider the VISIBLE flag once a file has one visual-only geom (contype = conaffinity
= 0). The gripper's three carriage markers are such geoms, so the export listed only those 5 mm spheres and a viewer
showed the carried cube floating.
"""

from __future__ import annotations

import numpy as np

import alloy
from alloy._vendor import newton
from alloy.api._frame_export import frame_export
from alloy.camera.from_newton import drawn_shapes
from alloy.envs import make

PARTS = ("palm", "finger_left", "finger_right")
CARRIAGES = ("carriage_x", "carriage_y", "carriage_z")


def _names(model) -> list[str]:
    """Body names from the file (the export names a robot's root body after the robot), in the topology's order."""
    return [str(n).rsplit("/", 1)[-1] for n in model.body_label]


def _visible(model) -> np.ndarray:
    return (model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)) != 0


def _static_count(model, mask: np.ndarray, world: int = 0) -> int:
    body, shape_world = model.shape_body.numpy(), model.shape_world.numpy()
    return int((mask & (body < 0) & np.isin(shape_world, (world, -1))).sum())


def test_pick_place_exports_the_palm_and_both_fingers():
    env = make("PickPlace-v0", num_envs=1, device="cpu", backend="numpy", obj="cube_beech_30", vary=False)
    try:
        sim = env.sim
        top, model = frame_export(sim, 0)[0].topology, sim.engine.model
        names = _names(model)
        for part in PARTS:                                             # each link's one collider box
            shapes = top.body_shapes[names.index(part)]
            assert [s["kind"] for s in shapes] == ["box"], (part, shapes)
        for c in CARRIAGES:                                            # the markers, as before
            assert [s["kind"] for s in top.body_shapes[names.index(c)]] == ["sphere"], c
        drawn, vis, body = drawn_shapes(model), _visible(model), model.shape_body.numpy()
        assert [len(s) for s in top.body_shapes] == [int(drawn[body == b].sum()) for b in range(len(names))]
        assert len(top.statics) == _static_count(model, vis)          # world shapes: the import's choice
    finally:
        env.close()


MIXED = """<mujoco><worldbody>
  <body name="table" pos="0 0 0.2"><geom type="box" size="0.2 0.2 0.01"/></body>
  <body name="cart" pos="0 0 0.5"><joint name="s" type="slide" axis="1 0 0"/>
    <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
    <body name="link" pos="0 0 0.1"><joint name="h" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.1"/></body></body>
</worldbody><actuator><position joint="s" kp="10"/><position joint="h" kp="10"/></actuator></mujoco>"""


def test_moving_links_are_exported_and_welded_bodies_keep_the_import_choice():
    """One camera-only marker in a file: the moving link's collider is exported, the welded table's stays out."""
    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    scene.add_robot("m", mjcf=MIXED)
    sim = alloy.build(scene, device="cpu")
    top, model = frame_export(sim, 0)[0].topology, sim.engine.model
    names = _names(model)
    assert [s["kind"] for s in top.body_shapes[names.index("link")]] == ["capsule"]
    assert [s["kind"] for s in top.body_shapes[names.index("cart")]] == ["sphere"]
    assert top.body_shapes[names.index("table")] == []
    assert len(top.statics) == _static_count(model, _visible(model))
