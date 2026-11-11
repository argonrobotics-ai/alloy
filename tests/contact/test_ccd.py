# SPDX-License-Identifier: Apache-2.0
"""Sweep CCD on CPU: a 1 mm coin at 2 m/s onto a 0.25 mm plate at dt 1.25 ms (2.5 mm per step, L0 capped at 0.1 mm).
Without sweep rows the swept test counts tunnels; with ``ccd="sweep"`` every coin lands on the plate, 0 tunnels."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.ccd import sweep_ccd
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.shadow import set_flat_solimp
from alloy.contact.types import ContactFlags, PrecisionPairSpec
from alloy.core.errors import AssemblyError
from alloy.precision.types import StepHealth

PLATE_T, COIN_HH, DT, V = 2.5e-4, 5e-4, 1.25e-3, 2.0
WORLDS = 8


def _scene(device: str, ccd: str):
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -PLATE_T / 2), wp.quat_identity()), hx=0.02, hy=0.02,
                            hz=PLATE_T / 2, cfg=cfg)
    coin = t.add_shape_cylinder(t.add_body(xform=wp.transform((0, 0, 0.01), wp.quat_identity())), radius=5e-3,
                                half_height=COIN_HH, cfg=cfg)
    t.add_shape_collision_filter_pair(plate, coin)
    for shape in (plate, coin):
        set_flat_solimp(t, shape, 0.99)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, WORLDS)
    model = b.finalize(device=device)
    spec = PrecisionPairSpec(plate, coin, mode="sdf", ccd=ccd, speculative_gap_max=1e-4, wall_thickness=PLATE_T)
    return model, spec


def _drop(device: str, ccd: str, sweep: bool):
    model, spec = _scene(device, ccd)
    layer = AlloyContactLayer(model, [spec], native="mujoco", precision_contacts_per_world=32)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, integrator="euler", cone="elliptic",
                                         nconmax=64, njmax=256)
    layer.bind_mujoco(solver).install()
    mon = layer.monitor()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    q[:, 2] = COIN_HH + np.linspace(2.6e-3, 5.0e-3, WORLDS)          # bottom face 2.6-5 mm above the plate
    qd[:, 2] = -V
    s0.body_q.assign(q)
    s0.body_qd.assign(qd)
    newton.eval_ik(model, s0, s0.joint_q, s0.joint_qd)
    h = StepHealth()
    h.min_pair_gap_m = np.inf
    health = wp.array([h] * WORLDS, dtype=StepHealth, device=device)
    swept = 0
    for _ in range(16):
        layer.collide(s0, layer.contacts(), dt=DT)
        if sweep:
            sweep_ccd(layer, s0, s0, layer.contacts(), DT)
            n = int(layer.contacts().rigid_contact_count.numpy()[0])
            swept += int((layer.sidecar().flags.numpy()[:n] & int(ContactFlags.SWEPT) > 0).sum())
        solver.step(s0, s1, ctrl, layer.contacts(), DT)
        mon.update(s0, s1, health)
        s0, s1 = s1, s0
    return mon.report(), s0.body_q.numpy()[:, 2], swept


def test_without_sweep_rows_coins_tunnel(device):
    rep, _, _ = _drop(device, "speculative", sweep=False)
    assert rep.tunnel_events.sum() > 0


def test_sweep_rows_stop_every_coin(device):
    rep, z, swept = _drop(device, "sweep", sweep=True)
    assert swept > 0
    assert rep.tunnel_events.sum() == 0, rep.tunnel_events
    assert np.all(z > COIN_HH - 1e-4), z                     # resting on the plate top (z = 0), not below it


def test_sweep_rejects_foreign_buffer(device):
    model, spec = _scene(device, "sweep")
    layer = AlloyContactLayer(model, [spec], native="newton")
    s = model.state()
    with pytest.raises(AssemblyError):
        sweep_ccd(layer, s, s, newton.Contacts(8, 0, device=device), DT)
    with pytest.raises(ValueError):
        sweep_ccd(layer, s, s, layer.contacts(), 0.0)


def test_sweep_rows_use_each_worlds_own_plate(device):
    """Two worlds, plates centred at z = -0.5 mm: 0.25 mm thick in world 0 (top at -0.375 mm), 1 mm in world 1 (top
    at 0). A coin 2 mm above z = 0 at 2 m/s gets one sweep row per world, on that world's own top face (with world
    0's plate for both, world 1's row sat 0.375 mm inside its plate; SCR-XB4-10-1)."""
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, gap=5e-4, margin=0.0)
    b = newton.ModelBuilder()
    for t_w in (PLATE_T, 1e-3):
        t = newton.ModelBuilder()
        plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -5e-4), wp.quat_identity()), hx=0.02, hy=0.02,
                                hz=t_w / 2, cfg=cfg)
        coin = t.add_shape_cylinder(t.add_body(xform=wp.transform((0, 0, 0.01), wp.quat_identity())), radius=5e-3,
                                    half_height=COIN_HH, cfg=cfg)
        t.add_shape_collision_filter_pair(plate, coin)
        b.add_world(t)
    model = b.finalize(device=device)
    spec = PrecisionPairSpec(plate, coin, mode="sdf", ccd="sweep", speculative_gap_max=1e-4, wall_thickness=PLATE_T)
    layer = AlloyContactLayer(model, [spec], native="newton", precision_contacts_per_world=32)
    s = model.state()
    q, qd = s.body_q.numpy(), s.body_qd.numpy()
    q[:, 2], qd[:, 2] = COIN_HH + 2e-3, -V
    s.body_q.assign(q)
    s.body_qd.assign(qd)
    layer.collide(s, layer.contacts(), dt=DT)
    sweep_ccd(layer, s, s, layer.contacts(), DT)
    n = int(layer.contacts().rigid_contact_count.numpy()[0])
    side = layer.sidecar()
    swept = (side.flags.numpy()[:n] & int(ContactFlags.SWEPT)) > 0
    world, z = side.world.numpy()[:n][swept], layer.contacts().rigid_contact_point0.numpy()[:n][swept, 2]
    assert set(world.tolist()) == {0, 1}
    assert np.allclose(z[world == 0], -3.75e-4, atol=2e-6), z
    assert np.allclose(z[world == 1], 0.0, atol=2e-6), z
