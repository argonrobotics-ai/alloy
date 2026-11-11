# SPDX-License-Identifier: Apache-2.0
"""ColliderField and CROSS_PARTICLE are wave-3 stubs (SPEC §7 WP3): the behaviour tests are strict xfail."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.field import append_cross_particle
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.types import PrecisionPairSpec

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="wave 3 stub")


def _layer(device: str) -> AlloyContactLayer:
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -0.01), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.01,
                            cfg=cfg)
    ball = t.add_shape_sphere(t.add_body(xform=wp.transform((0, 0, 0.05), wp.quat_identity())), radius=0.01, cfg=cfg)
    t.add_shape_collision_filter_pair(plate, ball)
    b = newton.ModelBuilder()
    b.replicate(t, 1)
    model = b.finalize(device=device)
    return AlloyContactLayer(model, [PrecisionPairSpec(plate, ball, mode="sdf", pair_stiffness=1e5)], native="newton")


def test_collider_field_distance_to_plate(device):
    layer = _layer(device)
    pts = wp.array(np.array([[0.0, 0.0, 0.003]], np.float32), dtype=wp.vec3, device=device)
    world = wp.zeros(1, dtype=wp.int32, device=device)
    dist = wp.zeros(1, dtype=wp.float32, device=device)
    normal, vel = (wp.zeros(1, dtype=wp.vec3, device=device) for _ in range(2))
    shape = wp.zeros(1, dtype=wp.int32, device=device)
    layer.collider_field().query(pts, world, dist, normal, vel, shape)
    assert dist.numpy()[0] == pytest.approx(0.003, abs=1e-6)
    assert np.allclose(normal.numpy()[0], (0.0, 0.0, 1.0))


def test_cross_particle_rows(device):
    layer = _layer(device)
    q = wp.array(np.array([[0.0, 0.0, 0.0005]], np.float32), dtype=wp.vec3, device=device)
    r = wp.array(np.array([0.001], np.float32), dtype=wp.float32, device=device)
    append_cross_particle(layer, q, r, layer.contacts())
    assert int(layer.contacts().rigid_contact_count.numpy()[0]) > 0
