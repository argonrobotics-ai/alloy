# SPDX-License-Identifier: Apache-2.0
"""A camera mounts on any body name ``sim.names()`` lists (audit CAMERA-1): robot parts are ``"<robot>/<body>"``."""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera import Camera, CameraError
from alloy.camera.pose import look_at

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""
POSE = look_at((0.0, -0.6, 0.2), (0.15, 0.0, 0.0))


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(dt=0.01, worlds=2)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0.2, 0.025), mass=0.2)
    return alloy.build(scene, device="cpu")


def _cam(attach_to) -> Camera:
    return Camera("wrist", pose=POSE, attach_to=attach_to, width=12, height=8, spp=1)


def test_every_listed_body_name_mounts_a_camera(sim):
    names = sim.names()["bodies"]
    assert "arm/link1" in names and "block" in names
    for name in names:
        cam = _cam(name)
        assert cam.render(sim).shape == (2, 8, 12, 3)
        rows = sim.index.per_world(sim.index.bodies[name], "body")
        assert np.array_equal(cam._cam_body.numpy(), rows)   # each world's own copy of that body


def test_api_name_and_bare_label_mount_the_same_body(sim):
    a, b = _cam("arm/link1"), _cam("link1")
    assert np.array_equal(a.render(sim, seed=3).numpy(), b.render(sim, seed=3).numpy())
    assert np.array_equal(a._cam_body.numpy(), b._cam_body.numpy())


def test_unknown_name_error_lists_the_sims_names(sim):
    with pytest.raises(CameraError) as err:
        _cam("arm/elbow").render(sim)
    assert "arm/link1" in str(err.value.fix) and "worldbody" not in str(err.value.fix)
