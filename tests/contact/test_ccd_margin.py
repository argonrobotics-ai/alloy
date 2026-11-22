# SPDX-License-Identifier: Apache-2.0
"""Regression test: sweep CCD must not double-count partner margin in emitted contact points."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.ccd import sweep_ccd
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.types import ContactFlags, PrecisionPairSpec


def test_sweep_ccd_does_not_double_count_partner_margin():
    """Emitted point0 must lie on the core plate face (z = 0) because margin0 carries the plate margin separately."""
    margin = 2.0e-3
    plate_t, coin_hh = 1.0e-3, 5.0e-4
    cfg_plate = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, gap=5e-4, margin=margin)
    cfg_coin = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -plate_t / 2.0), wp.quat_identity()),
                            hx=0.02, hy=0.02, hz=plate_t / 2.0, cfg=cfg_plate)
    coin = t.add_shape_cylinder(t.add_body(xform=wp.transform((0.0, 0.0, 0.01), wp.quat_identity())),
                                radius=5e-3, half_height=coin_hh, cfg=cfg_coin)
    t.add_shape_collision_filter_pair(plate, coin)
    b = newton.ModelBuilder()
    b.add_world(t)
    model = b.finalize(device="cpu")

    spec = PrecisionPairSpec(plate, coin, mode="sdf", ccd="sweep", speculative_gap_max=1e-4, wall_thickness=plate_t)
    layer = AlloyContactLayer(model, [spec], native="newton", precision_contacts_per_world=32)
    s = model.state()
    q, qd = s.body_q.numpy(), s.body_qd.numpy()
    q[0, 2], qd[0, 2] = coin_hh + 3.0e-3, -4.0
    s.body_q.assign(q)
    s.body_qd.assign(qd)

    layer.collide(s, layer.contacts(), dt=1.25e-3)
    sweep_ccd(layer, s, s, layer.contacts(), 1.25e-3)
    n = int(layer.contacts().rigid_contact_count.numpy()[0])
    swept = (layer.sidecar().flags.numpy()[:n] & int(ContactFlags.SWEPT)) > 0
    assert swept.any(), "expected at least one swept CCD contact row"
    p0_z = layer.contacts().rigid_contact_point0.numpy()[:n][swept, 2]
    m0 = layer.contacts().rigid_contact_margin0.numpy()[:n][swept]
    assert np.allclose(m0, margin, atol=1e-6), m0
    assert np.allclose(p0_z, 0.0, atol=2e-5), f"point0 shifted by partner margin: {p0_z}"
