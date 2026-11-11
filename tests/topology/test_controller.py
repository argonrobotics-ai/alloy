# SPDX-License-Identifier: Apache-2.0
"""TopologyController contract: TopologyLike, consuming ledger, reset_required, per-world snapshot, empty models."""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.protocols import TopologyLike
from alloy.core.snapshot import WorldSnapshot
from alloy.testing.topology import cloth_kwargs, grid_tri_mesh
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_cloth
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget


def _two_world_cloth(device: str) -> newton.Model:
    t = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, tris = grid_tri_mesh(4, 2, 0.01)
    add_presplit_cloth(t, v, tris, SplitPlan(spacing=0.02, axes=(0,)), cloth_kwargs(),
                       budget=TopologyBudget(spare_vertices=10))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b.replicate(t, 2)
    b.color()
    return b.finalize(device=device)


def test_is_topology_like_and_empty_model_is_a_no_op(device):
    b = newton.ModelBuilder()
    b.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    m = b.finalize(device=device)
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget())
    assert isinstance(ctl, TopologyLike)
    state = m.state()
    ctl.pre_step(state, None, 1e-3)
    ctl.apply(state)
    led = ctl.ledger()
    assert led.events.shape == (1,) and led.events.sum() == 0 and not led.reset_required.any()


def test_ledger_is_a_consuming_read_and_flags_reset_required(device):
    m = _two_world_cloth(device)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget())
    state = m.state()
    world1 = np.flatnonzero(bonds.world.numpy() == 1)
    bs = bonds.break_strain.numpy()
    bs[world1] = -1.0                                   # world 1 bonds break on the first apply
    bonds.break_strain.assign(bs)
    ctl.apply(state)
    led = ctl.ledger()
    np.testing.assert_array_equal(led.events, [0, len(world1)])
    np.testing.assert_array_equal(led.reset_required, [False, True])
    np.testing.assert_array_equal(ctl.ledger().events, [0, 0])            # a fresh period
    assert ctl.last_ledger.events.sum() == 0
    springs = m.topology.bond_spring.numpy()
    assert np.all(m.spring_stiffness.numpy()[springs[world1]] == 0.0)
    assert np.all(m.spring_stiffness.numpy()[springs[bonds.world.numpy() == 0]] > 0.0)


def test_per_world_snapshot_restores_bonds_and_model_arrays(device):
    m = _two_world_cloth(device)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget())
    state = m.state()
    snap = WorldSnapshot(m, state, extra=ctl.state_arrays(), extra_world=ctl.state_worlds())
    assert snap.whole_arrays == ()
    snap.save(state)
    ke0 = m.spring_stiffness.numpy().copy()
    ctl.preset_broken(np.arange(bonds.count))
    assert np.all(m.spring_stiffness.numpy() == 0.0)
    mask = wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device=device)
    snap.restore(state, mask)
    w = bonds.world.numpy()
    np.testing.assert_array_equal(bonds.intact.numpy(), (w == 1).astype(np.int8))
    springs = m.topology.bond_spring.numpy()
    np.testing.assert_array_equal(m.spring_stiffness.numpy()[springs[w == 1]], ke0[springs[w == 1]])
    assert np.all(m.spring_stiffness.numpy()[springs[w == 0]] == 0.0)


def test_capacity_report_counts_overflow(device):
    m = _two_world_cloth(device)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=1))
    bonds.break_strain.assign(np.full(bonds.count, -1.0, np.float32))
    state = m.state()
    ctl.apply(state)
    per_world = bonds.count // 2
    np.testing.assert_array_equal(ctl.step_events.numpy(), [1, 1])
    assert ctl.capacity_report()["overflow_this_period"] == 2 * (per_world - 1)
    for _ in range(per_world - 1):                      # deferred events are applied on later steps
        ctl.apply(state)
    assert bonds.intact.numpy().sum() == 0
    led = ctl.ledger()
    np.testing.assert_array_equal(led.events, [per_world, per_world])
