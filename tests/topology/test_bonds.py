# SPDX-License-Identifier: Apache-2.0
"""Bond rows as custom attributes, replication offsets, MuJoCo weld/connect breaks by force and strain, springs."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import AssemblyError
from alloy.topology.bonds import add_connect_bond, add_spring_bond, add_weld_bond, bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.types import BondKind, TopologyBudget


def _welded_boxes(device: str, worlds: int, *, connect: bool = False) -> tuple[newton.Model, int]:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    body = t.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    t.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)                 # + 1 kg from density 1000 -> 2 kg
    if connect:
        add_connect_bond(t, -1, body, (0.0, 0.0, 1.0), break_force=1.0e9)
    else:
        add_weld_bond(t, body, -1, break_force=1.0e9)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, worlds)
    return b.finalize(device=device), body


def test_rows_replicate_with_offsets(device):
    m, _ = _welded_boxes(device, 3)
    bonds = bonds_from_model(m)
    assert bonds.count == 3
    np.testing.assert_array_equal(bonds.world.numpy(), [0, 1, 2])
    np.testing.assert_array_equal(bonds.a.numpy(), [0, 1, 2])                       # body of each world
    np.testing.assert_array_equal(m.topology.bond_eq.numpy(), [0, 1, 2])
    assert set(bonds.kind.numpy().tolist()) == {int(BondKind.RIGID_WELD)}


@pytest.mark.parametrize("connect", [False, True])
def test_weld_breaks_by_force_in_one_world_only(device, connect):
    m, _ = _welded_boxes(device, 2, connect=connect)
    bonds = bonds_from_model(m)
    bf = bonds.break_force.numpy()
    bf[:] = (30.0, 15.0)                                   # hold 2 kg x 9.81 m/s² = 19.6 N in world 0 only
    bonds.break_force.assign(bf)
    solver = newton.solvers.SolverMuJoCo(m)
    ctl = TopologyController(m, bonds, TopologyBudget(), entries={"mujoco": solver})
    s0, s1, c = m.state(), m.state(), m.control()
    for _ in range(150):
        ctl.pre_step(s0, None, 1e-3)
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
    np.testing.assert_array_equal(bonds.intact.numpy(), [1, 0])
    f = ctl.force.numpy()
    assert 15.0 < f[0] < 30.0                              # the weld force read back from MJWarp efc rows
    z = s0.body_q.numpy()[:, 2]
    assert z[0] > 0.99 and z[1] < 0.95
    led = ctl.ledger()
    np.testing.assert_array_equal(led.events, [0, 1])
    np.testing.assert_array_equal(led.reset_required, [False, True])


def test_weld_breaks_by_strain(device):
    """Strain = drift of b in a's frame / rest distance; the MuJoCo weld sags slightly under gravity."""
    m, _ = _welded_boxes(device, 1)
    bonds = bonds_from_model(m)
    bonds.break_strain.assign(np.array([1.0e-7], np.float32))
    solver = newton.solvers.SolverMuJoCo(m)
    ctl = TopologyController(m, bonds, TopologyBudget(), entries={"mujoco": solver})
    s0, s1, c = m.state(), m.state(), m.control()
    for _ in range(30):
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
    assert bonds.intact.numpy()[0] == 0
    assert not bool(m.mujoco.equality_constraint_enabled.numpy()[0])


def test_spring_bond_breaks_on_stretch(device):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    i = b.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.0)
    j = b.add_particle(wp.vec3(0.1, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.01)
    add_spring_bond(b, i, j, 100.0, 0.0, break_strain=0.2)
    m = b.finalize(device=device)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget())
    state = m.state()
    for x, expect in ((0.115, 1), (0.125, 0)):
        q = state.particle_q.numpy()
        q[j, 0] = x
        state.particle_q.assign(q)
        ctl.apply(state)
        assert bonds.intact.numpy()[0] == expect
    assert m.spring_stiffness.numpy()[0] == 0.0
    assert ctl.strain.numpy()[0] == pytest.approx(0.25, rel=1e-5)


def test_bond_without_native_element_is_rejected(device):
    from alloy.topology.types import BondSet

    b = newton.ModelBuilder()
    b.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    b.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    m = b.finalize(device=device)
    arr = lambda x, dt: wp.array(np.asarray(x), dtype=dt, device=device)  # noqa: E731
    bonds = BondSet(arr([0], wp.int32), arr([1], wp.int32), arr([int(BondKind.SPRING)], wp.int8), arr([0], wp.int32),
                    arr([np.inf], wp.float32), arr([np.inf], wp.float32), arr([0.0], wp.float32), arr([1], wp.int8))
    with pytest.raises(AssemblyError):
        TopologyController(m, bonds, TopologyBudget())
