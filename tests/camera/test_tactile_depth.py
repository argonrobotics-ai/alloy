# SPDX-License-Identifier: Apache-2.0
"""Tactile press maps (OSS-21): exact ray-cast depth in the sensor frame against analytic shapes, batched worlds."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.camera.tactile.depth import MAX_CANDIDATES, PressMap

GEL_Z = 0.002  # gel surface on the pad body's +Z face


def pad_model(objects, worlds: int = 1, pad_rot=None):
    """A pad (gel at z = 2 mm, normal +Z) plus objects [(kind, size, pos)] per world; returns (model, pad label)."""
    from alloy._vendor import newton
    w = newton.ModelBuilder()
    q = pad_rot or wp.quat_identity()
    pad = w.add_body(xform=wp.transform((0.0, 0.0, 0.0), q), label="pad")
    w.add_shape_box(pad, hx=0.012, hy=0.01, hz=0.002)
    for k, (kind, size, pos) in enumerate(objects):
        body = w.add_body(xform=wp.transform(pos, wp.quat_identity()), label=f"obj{k}")
        if kind == "sphere":
            w.add_shape_sphere(body, radius=size)
        elif kind == "box":
            w.add_shape_box(body, hx=size[0], hy=size[1], hz=size[2])
        elif kind == "capsule":
            w.add_shape_capsule(body, radius=size[0], half_height=size[1],
                                xform=wp.transform((0, 0, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2)))
    if worlds == 1:
        return w.finalize(device="cpu")
    b = newton.ModelBuilder()
    b.replicate(w, copy_count=worlds)
    return b.finalize(device="cpu")


def press_of(model, bodies, h=60, w=80, mm=0.2, pose=(0, 0, GEL_Z, 0, 0, 0, 1.0), state=None):
    pm = PressMap(model, np.asarray(bodies), h, w, mm, max_depth_mm=4.0)
    pm.set_local_pose(np.asarray(pose, np.float64))
    st = state or model.state()
    return pm, pm.compute(st.body_q).numpy()


def test_sphere_press_is_the_spherical_cap():
    r, depth = 0.01, 0.0005
    model = pad_model([("sphere", r, (0.0, 0.0, GEL_Z + r - depth))])
    pm, p = press_of(model, [0])
    mm = 0.2
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float64)
    x, y = (xx + 0.5 - 40) * mm, (yy + 0.5 - 30) * mm
    want = np.clip(np.sqrt(np.clip(100.0 - x * x - y * y, 0, None)) - (10.0 - 0.5), 0, None)
    assert np.abs(p[0] - want).max() < 2e-4  # mm
    touched = pm.touched.numpy()[0]
    assert set(np.unique(touched[p[0] > 0])) == {1}  # shape 1 = the sphere (the pad's own box is skipped)


def test_box_press_is_flat_and_located_in_the_sensor_frame():
    model = pad_model([("box", (0.001, 0.001, 0.005), (0.006, 0.004, GEL_Z + 0.005 - 0.0003))])
    _, p = press_of(model, [0])
    ys, xs = np.nonzero(p[0])
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (45, 54, 65, 74)  # +x right, +y down (rows)
    assert np.allclose(p[0][p[0] > 0], 0.3, atol=1e-4)


def test_capsule_and_nothing_nearby():
    model = pad_model([("capsule", (0.002, 0.004), (0.0, 0.0, GEL_Z + 0.002 - 0.0004)),
                       ("sphere", 0.003, (0.0, 0.0, 0.05))])
    pm, p = press_of(model, [0])
    assert p[0].max() == pytest.approx(0.4 - (2.0 - math.sqrt(4.0 - 0.1 ** 2)), abs=1e-4)  # pixel centres 0.1 mm off axis
    assert pm.count.numpy()[0] == 1  # the far sphere is culled
    ys, xs = np.nonzero(p[0])
    assert ys.max() - ys.min() > 2 * (xs.max() - xs.min())  # capsule axis along y (rotated about x)


def test_tilted_pad_and_many_worlds():
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.7)
    rot = np.array(wp.quat_to_matrix(q)).reshape(3, 3)
    centre = rot @ np.array([0.0, 0.0, GEL_Z + 0.004 - 0.0006])
    model = pad_model([("sphere", 0.004, tuple(centre))], worlds=3, pad_rot=q)
    bodies = [0, 2, 4]  # pad of each world (2 bodies per world)
    _, p = press_of(model, bodies)
    assert p.shape == (3, 60, 80)
    peak = 0.6 - (4.0 - math.sqrt(16.0 - 2 * 0.1 ** 2))  # nearest pixel centres are 0.1 mm off in x and y
    assert np.allclose(p.max(axis=(1, 2)), peak, atol=2e-4) and np.allclose(p[0], p[2])


def test_ray_starting_inside_a_mesh_is_a_full_press():
    from alloy._vendor import newton
    w = newton.ModelBuilder()
    pad = w.add_body(label="pad")
    w.add_shape_box(pad, hx=0.012, hy=0.01, hz=0.002)
    body = w.add_body(xform=wp.transform((0.0, 0.0, GEL_Z), wp.quat_identity()), label="blob")
    mesh = newton.Mesh.create_box(0.008, 0.008, 0.008, compute_inertia=False)  # 16 mm cube, 8 mm deep into the gel
    w.add_shape_mesh(body, mesh=newton.Mesh(mesh.vertices, mesh.indices))
    model = w.finalize(device="cpu")
    _, p = press_of(model, [0])
    assert p[0, 30, 40] == pytest.approx(4.0)  # deeper than max_depth: clipped full press
    assert MAX_CANDIDATES >= 8


def test_primitive_deeper_than_max_depth_is_a_full_press():
    model = pad_model([("sphere", 0.004, (0.0, 0.0, GEL_Z - 0.001))])  # centre 1 mm inside the gel, 5 mm deep
    _, p = press_of(model, [0])
    assert p[0, 30, 40] == pytest.approx(4.0) and p[0, 0, 0] == 0.0
