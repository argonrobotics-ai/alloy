# SPDX-License-Identifier: Apache-2.0
"""The camera on CUDA: CPU/GPU agreement, many worlds with physics, torch output (skipped without a GPU)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera import Camera, Lighting, Sun, camera_scene
from tests.camera.helpers import plane_scene

pytestmark = pytest.mark.gpu


def test_cpu_and_gpu_agree_on_the_lit_plane():
    light = Lighting([Sun.from_euler((0.6, 0.3, 0.8), 3.0)], (0.6, 0.65, 0.7))
    cam_args = dict(look_at=((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), width=8, height=8, fov=20, spp=16)
    out = []
    for dev in ("cpu", "cuda:0"):
        sc = camera_scene(plane_scene(0.8, light), device=dev)
        out.append(Camera("down", **cam_args).render(sc, output="linear").numpy().copy())
    assert np.allclose(out[0], out[1], rtol=1e-4, atol=1e-5)


def test_many_worlds_with_physics_and_torch():
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_rpy(0.3, 0.2, 0.1)), label="box")
    b.add_shape_box(box, hx=0.08, hy=0.06, hz=0.04)
    s = newton.ModelBuilder()
    s.replicate(b, 64)
    model = s.finalize(device="cuda:0")
    solver = newton.solvers.SolverXPBD(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    cam = Camera("front", look_at=((0.7, -0.7, 0.5), (0.0, 0.0, 0.1)), width=32, height=32, fov=45, spp=4)
    first = cam.render(model, s0).numpy().copy()
    for _ in range(20):
        solver.step(s0, s1, ctrl, model.collide(s0), 1.0 / 60.0)
        s0, s1 = s1, s0
    later = cam.render(model, s0).numpy()
    assert first.shape == (64, 32, 32, 3) and not np.array_equal(first, later)  # the box fell
    # identical worlds look alike; each image has its own noise pattern (Cycles seeds every image differently)
    assert np.abs(later[0].astype(float) - later[63].astype(float)).mean() < 3.0
    torch = pytest.importorskip("torch")
    t = cam.render_torch(model, s0)
    assert isinstance(t, torch.Tensor) and t.is_cuda and t.shape == (64, 32, 32, 3)
