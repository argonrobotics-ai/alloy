# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-1: ``TopologyController.request_break(bond_ids, world_mask)``, an external bond break ledgered like
``request_crumble``.

- A MuJoCo weld (RIGID_WELD) breaks only in the masked worlds, at the next ``apply``. The equality is disabled
  there, the box falls, and the ledger counts one event with exact mass and momentum balance.
- Requests share the per-world cap; an overflowed request stays pending, and a request on a broken bond is dropped.
- Device inputs need no host sync and can be captured in a CUDA graph; host inputs are validated.
- Without requests nothing changes: bonds with infinite thresholds stay intact.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.testing.topology import cloth_kwargs, grid_tri_mesh
from alloy.topology.bonds import add_weld_bond, bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_cloth
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget


def _welded_boxes(device: str, worlds: int) -> tuple[newton.Model, int]:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    body = t.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    t.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    row = add_weld_bond(t, body, -1, break_force=1.0e9)     # never breaks by force or strain on its own
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, worlds)
    return b.finalize(device=device), row


def _cloth(device: str, worlds: int = 2) -> newton.Model:
    """Three CLOTH_EDGE bonds per world, no failure card: thresholds are infinite."""
    t = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, tris = grid_tri_mesh(4, 2, 0.01)
    add_presplit_cloth(t, v, tris, SplitPlan(spacing=0.02, axes=(0,)), cloth_kwargs(),
                       budget=TopologyBudget(spare_vertices=10))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.replicate(t, worlds)
    b.color()
    return b.finalize(device=device)


def _run(ctl, solver, s0, s1, c, steps: int):
    for _ in range(steps):
        ctl.pre_step(s0, None, 1e-3)
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
    return s0, s1


def test_requested_weld_break_in_masked_worlds_is_ledgered(device):
    m, row = _welded_boxes(device, 3)
    bonds = bonds_from_model(m)
    solver = newton.solvers.SolverMuJoCo(m)
    ctl = TopologyController(m, bonds, TopologyBudget(), entries={"mujoco": solver})
    s0, s1, c = m.state(), m.state(), m.control()
    s0, s1 = _run(ctl, solver, s0, s1, c, 20)
    assert ctl.ledger().events.sum() == 0                              # no request, no break
    ctl.request_break([row], world_mask=[False, True, False])
    np.testing.assert_array_equal(bonds.intact.numpy(), [1, 1, 1])     # pending until the next apply
    s0, s1 = _run(ctl, solver, s0, s1, c, 1)
    np.testing.assert_array_equal(bonds.intact.numpy(), [1, 0, 1])
    np.testing.assert_array_equal(m.mujoco.equality_constraint_enabled.numpy(), [True, False, True])
    led = ctl.ledger()
    np.testing.assert_array_equal(led.events, [0, 1, 0])
    np.testing.assert_array_equal(led.reset_required, [False, True, False])
    assert led.mass_after[1] == led.mass_before[1] > 0.0               # a weld break moves no mass
    np.testing.assert_array_equal(led.momentum_after[1], led.momentum_before[1])
    ctl.request_break([row], world_mask=[False, True, False])          # already broken: dropped, no event
    s0, s1 = _run(ctl, solver, s0, s1, c, 150)
    assert ctl.ledger().events.sum() == 0
    assert ctl.state_arrays()["topology.break_request"].numpy().sum() == 0
    z = s0.body_q.numpy()[:, 2]
    assert z[0] > 0.99 and z[2] > 0.99 and z[1] < 0.95                 # only world 1's box fell


def test_requests_share_the_cap_and_overflow_stays_pending(device):
    m = _cloth(device)
    bonds = bonds_from_model(m)
    per_world = bonds.count // 2
    assert per_world == 3
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=1))
    state = m.state()
    ctl.apply(state)
    assert bonds.intact.numpy().sum() == bonds.count                   # infinite thresholds: nothing breaks
    ctl.request_break(np.arange(per_world))                            # None = every world
    ctl.apply(state)
    np.testing.assert_array_equal(ctl.step_events.numpy(), [1, 1])
    assert ctl.capacity_report()["overflow_this_period"] == 2 * (per_world - 1)
    pending = ctl.state_arrays()["topology.break_request"].numpy()
    assert pending.sum() == 2 * (per_world - 1)
    w = bonds.world.numpy()
    np.testing.assert_array_equal(ctl.state_worlds()["topology.break_request"].numpy(), w)
    for _ in range(per_world - 1):
        ctl.apply(state)
    assert bonds.intact.numpy().sum() == 0
    np.testing.assert_array_equal(ctl.ledger().events, [per_world, per_world])
    assert np.all(m.spring_stiffness.numpy()[m.topology.bond_spring.numpy()] == 0.0)


def test_device_inputs_and_world_local_ids(device):
    m = _cloth(device, worlds=3)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget())
    ids = wp.array(np.array([2, 0], np.int32), dtype=wp.int32, device=device)
    mask = wp.array(np.array([1, 0, 1], np.int8), dtype=wp.int8, device=device)
    ctl.request_break(ids, mask)
    ctl.apply(m.state())
    w, intact = bonds.world.numpy(), bonds.intact.numpy()
    for world in range(3):
        local = intact[w == world]
        expect = [0, 1, 0] if world != 1 else [1, 1, 1]                # world-local ids 0 and 2
        np.testing.assert_array_equal(local, expect)
    np.testing.assert_array_equal(ctl.ledger().events, [2, 0, 2])


def test_request_break_validates_host_inputs(device):
    m = _cloth(device)
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget())
    with pytest.raises(AssemblyError, match="outside"):
        ctl.request_break([3])
    with pytest.raises(AssemblyError, match="outside"):
        ctl.request_break([-1], world_mask=[1, 1])
    with pytest.raises(AssemblyError, match="world_mask"):
        ctl.request_break([0], world_mask=[1, 0, 1])
    ctl.request_break([7], world_mask=[0, 0])                          # no masked world: nothing to check or do
    b = newton.ModelBuilder()
    b.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    empty = b.finalize(device=device)
    with pytest.raises(AssemblyError, match="no bonds"):
        TopologyController(empty, bonds_from_model(empty), TopologyBudget()).request_break([0])


@pytest.mark.gpu
def test_request_break_captures_in_a_graph(device):
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")
    m = _cloth(device)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget())
    ids = wp.array(np.array([1], np.int32), dtype=wp.int32, device=device)
    mask = wp.zeros(2, dtype=wp.int8, device=device)
    state = m.state()
    with wp.ScopedCapture(device=device) as cap:
        ctl.request_break(ids, mask)
        ctl.apply(state)
    wp.capture_launch(cap.graph)
    assert bonds.intact.numpy().sum() == bonds.count                   # mask all zero: nothing requested
    mask.assign(np.array([0, 1], np.int8))                              # a device-side criterion flips world 1
    wp.capture_launch(cap.graph)
    w, intact = bonds.world.numpy(), bonds.intact.numpy()
    np.testing.assert_array_equal(intact[w == 1], [1, 0, 1])
    assert intact[w == 0].all()
