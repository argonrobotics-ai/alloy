# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: Newton imports and a falling box steps on a device."""

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton


def _drop_box(device: str, steps: int = 60) -> float:
    """Drop a box on a ground plane with XPBD; return its final z (m)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverXPBD(model)
        s0, s1 = model.state(), model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        dt = 1.0 / 600.0
        for _ in range(steps * 10):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        return float(s0.body_q.numpy()[0, 2])


def test_import_versions():
    assert newton.__version__.startswith("1.8")
    assert wp.config.version


def test_box_falls_and_rests_cpu():
    z = _drop_box("cpu")
    assert np.isfinite(z)
    assert 0.05 < z < 0.2, z  # rests on the plane, half-extent 0.1


@pytest.mark.gpu
def test_box_falls_and_rests_gpu():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    z = _drop_box("cuda:0")
    assert 0.05 < z < 0.2, z
