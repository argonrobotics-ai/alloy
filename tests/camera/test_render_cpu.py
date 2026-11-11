# SPDX-License-Identifier: Apache-2.0
"""The diffuse path tracer on the CPU device: energy against the analytic answer, attached cameras, world isolation,
determinism, the snapshot and the Sim entry point (tiny images; the kernel compiles once, about 10 s)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera import Camera, CameraError, Lighting, Sun, camera_scene, render, snapshot
from alloy.camera.scene import IDENTITY_XF, SceneData
from alloy.camera.view import standard_u8
from tests.camera.helpers import FakeState, body_scene, cube, plane_scene, quad

DEV = "cpu"
SKY = (0.6, 0.65, 0.7)


def test_lit_plane_matches_analytic_radiance():
    """A Lambert plane under a sun of irradiance E and a constant sky L: L_o = a (E cos(theta) / pi + L)."""
    sun = Sun.from_euler((0.6, 0.3, 0.8), 3.0)
    sc = camera_scene(plane_scene(0.8, Lighting([sun], SKY)), device=DEV)
    cam = Camera("down", look_at=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), width=6, height=6, fov=20,
                 spp=64)
    lin = cam.render(sc, output="linear").numpy().reshape(-1, 3)
    want = 0.8 * (3.0 * sun.direction_to_sun[2] / np.pi + np.asarray(SKY))
    assert np.allclose(lin.mean(0), want, rtol=2e-3), (lin.mean(0), want)
    assert lin.std(0).max() < 1e-2 * want.max()


def test_attached_camera_moves_with_its_body():
    sc = camera_scene(body_scene(1), device=DEV)
    cam = Camera("hand", look_at=((0.5, -0.5, 0.4), (0, 0, 0)), attach_to=0, width=12, height=12, spp=4)
    here = cam.render(sc, FakeState([IDENTITY_XF], DEV)).numpy().copy()
    moved = cam.render(sc, FakeState([[3.0, 1.0, 0.5, 0, 0, 0, 1]], DEV)).numpy().copy()
    assert np.abs(here.astype(int) - moved.astype(int)).max() <= 2
    assert here.std() > 5  # the cube is in view


def test_worlds_do_not_see_each_other():
    sc = camera_scene(body_scene(2), device=DEV)
    cam = Camera("fixed", look_at=((0.5, -0.5, 0.4), (0, 0, 0)), width=10, height=10, fov=25, spp=2)
    q = np.tile(IDENTITY_XF, (2, 1))
    q[1, :3] = (5.0, 0.0, 0.0)  # world 1's cube is out of view
    img = cam.render(sc, FakeState(q, DEV)).numpy()
    sky = standard_u8(np.asarray(SKY, np.float32))
    assert (img[1] == sky).all()
    assert (img[0] != sky).any(axis=-1).sum() > 10


def test_same_seed_same_image_new_seed_new_noise():
    sd = SceneData([quad(), cube()], [0, 1], [0, 0], [-1, -1], [IDENTITY_XF, [0, 0, 0.1, 0, 0, 0, 1]],
                   [(0.7, 0.7, 0.7), (0.8, 0.2, 0.2)], 1)
    sc = camera_scene(sd, device=DEV)
    cam = Camera("c", look_at=((0.6, -0.6, 0.5), (0, 0, 0.1)), width=16, height=16, spp=1)
    a = cam.render(sc, output="linear").numpy().copy()  # renders reuse the camera's buffers
    b = cam.render(sc, output="linear").numpy().copy()
    c = cam.render(sc, output="linear", seed=7).numpy().copy()
    d = cam.render(sc, output="linear").numpy().copy()
    assert np.array_equal(a, b) and not np.array_equal(a, c) and np.array_equal(a, d)


def test_all_outputs_and_errors():
    sc = camera_scene(plane_scene(), device=DEV)
    cam = Camera("c", look_at=((0, 0, 1), (0, 0, 0), (0, 1, 0)), width=4, height=3, spp=1)
    out = cam.render(sc, output="all")
    assert out.rgb.shape == (1, 3, 4, 3) and out.linear.shape == (1, 3, 4)
    n = out.normal.numpy().reshape(-1, 3)
    assert np.allclose(n, [0, 0, -1], atol=1e-5)  # camera space: the plane faces the camera
    with pytest.raises(CameraError, match="use one of"):
        cam.render(sc, output="png")
    with pytest.raises(CameraError, match="at least 1 sample"):
        Camera("c", spp=0)
    with pytest.raises(CameraError, match="one of them"):
        Camera("c", pose=IDENTITY_XF, look_at=((1, 1, 1), (0, 0, 0)))


def _model():
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()), label="box")
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return b.finalize(device=DEV)


def test_snapshot_and_sim_entry_point(tmp_path):
    model = _model()
    img = snapshot(model, model.state(), path=tmp_path / "s.png", width=16, height=12, spp=2, device=DEV)
    assert img.shape == (12, 16, 3) and img.dtype == np.uint8 and (tmp_path / "s.png").stat().st_size > 50

    class Engine:  # what alloy.Sim exposes
        def __init__(self, m):
            self.model, self.state = m, m.state()

    class FakeSim:
        engine = Engine(model)

    img2 = render(FakeSim(), width=16, height=12, spp=2)
    assert img2.shape == (12, 16, 3)


def test_attach_to_body_label_on_a_model():
    model = _model()
    cam = Camera("wrist", look_at=((0.4, -0.4, 0.3), (0, 0, 0)), attach_to="box", width=8, height=8, spp=1)
    img = cam.render(camera_scene(model, device=DEV), model.state())
    assert img.shape == (1, 8, 8, 3)
    with pytest.raises(CameraError, match="body labels|no body"):
        Camera("x", attach_to="nope", width=4, height=4, spp=1).render(model, model.state())


def test_camera_renders_an_alloy_sim():
    """cam.render(sim) reads sim.engine.model and sim.engine.state (alloy.Sim)."""
    model = _model()

    class Engine:
        def __init__(self, m):
            self.model, self.state = m, m.state()

    class FakeSim:
        engine = Engine(model)

    cam = Camera("c", look_at=((0.6, -0.6, 0.5), (0, 0, 0.1)), width=8, height=6, spp=1)
    camera_scene(FakeSim(), device=DEV)
    assert cam.render(FakeSim()).shape == (1, 6, 8, 3)
