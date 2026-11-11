# SPDX-License-Identifier: Apache-2.0
"""Scene attachments become WP10 bonds on the template: weld/joint as MuJoCo equality rows, spring as a particle
spring; replicate gives one bond per world; endpoint resolution and the unsupported combinations raise."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.cards.types import Attachment
from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.topology.types import BondKind

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def _assemble(objs, places, atts, worlds=2):
    card, lib = S.scene(objs, places, worlds=worlds)
    card = dataclasses.replace(card, attachments=tuple(atts))
    return SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS))).assemble(card, "cpu")


def _cloth(name: str):
    from alloy.cards.types import Budget

    return T.box(name, T.material("cotton", **T.COTTON, beta=1e-3), 0.05, 0.05, 2.5e-4, budget=Budget(particles=36))


def test_weld_to_static_object_holds_the_box_in_every_world():
    built = _assemble([S.table(), S.box()], [S.place("table"), S.place("box", z=0.2)],
                      [Attachment("table", "box", "weld", break_force=T.q(5.0))], worlds=3)
    bonds = built.bonds
    assert bonds.count == 3
    assert set(bonds.kind.numpy()) == {int(BondKind.RIGID_WELD)}
    assert list(bonds.a.numpy()) == [-1, -1, -1]
    assert len(set(bonds.b.numpy())) == 3 and list(bonds.world.numpy()) == [0, 1, 2]
    assert np.allclose(bonds.break_force.numpy(), 5.0)
    sim = Simulation(built)
    z0 = sim.state.body_q.numpy()[:, 2].copy()
    for _ in range(5):
        sim.step_period()
    assert np.max(np.abs(sim.state.body_q.numpy()[:, 2] - z0)) < 1e-3   # free fall would drop 6 mm in 0.05 s


def test_joint_attachment_is_a_connect_bond_between_two_bodies():
    built = _assemble([S.table(), S.box("a"), S.box("b")],
                      [S.place("table"), S.place("a", z=0.2), S.place("b", x=0.07, z=0.2)],
                      [Attachment("a", "b", "joint", anchor=(0.035, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))], worlds=2)
    assert built.bonds.count == 2 and set(built.bonds.kind.numpy()) == {int(BondKind.JOINT)}
    a, b = built.bonds.a.numpy(), built.bonds.b.numpy()
    assert np.all(a >= 0) and np.all(b >= 0) and np.all(a != b)


def test_spring_joins_the_nearest_particles_of_two_cloths():
    built = _assemble([S.table(), _cloth("c1"), _cloth("c2")],
                      [S.place("table"), S.place("c1", z=0.05), S.place("c2", x=0.11, z=0.05)],
                      [Attachment("c1", "c2", "spring", break_strain=T.q(0.5))], worlds=2)
    bonds = built.bonds
    assert bonds.count == 2 and set(bonds.kind.numpy()) == {int(BondKind.SPRING)}
    q = built.model.particle_q.numpy()
    i, j = int(bonds.a.numpy()[0]), int(bonds.b.numpy()[0])
    assert np.linalg.norm(q[i] - q[j]) == pytest.approx(0.01, abs=1e-6)   # facing edges, 1 cm apart
    assert np.allclose(bonds.break_strain.numpy(), 0.5)


def test_chain_endpoint_is_the_link_nearest_the_anchor_and_asks_for_cg():
    built = _assemble([S.table(), S.cable(), S.box()],
                      [S.place("table"), S.place("cable", z=0.03), S.place("box", x=0.29, z=0.05)],
                      [Attachment("box", "cable", "weld")], worlds=1)
    cable = built.handles["cable"]
    assert cable.decision.route == Route.CHAIN
    assert int(built.bonds.b.numpy()[0]) == cable.template.body.stop - 1    # the +x end link
    assert built.entries[0].solver_kwargs["solver"] == "cg"
    assert "mujoco:solver=cg" in built.report.scene_effects


@pytest.mark.parametrize("att,match", [
    (Attachment("towel", "box", "spring"), "no particles"),
    (Attachment("table", "table2", "weld"), "both ends are static"),
    (Attachment("ghost", "box", "weld"), "no placed object"),
    (Attachment("box", "towel:3", "weld"), "no such link"),
])
def test_unsupported_attachments_raise(att, match):
    table2 = T.box("table2", T.material("wood", **T.WOOD), 0.1, 0.1, 0.02, motion="static")
    with pytest.raises(AssemblyError, match=match):
        _assemble([S.table(), table2, S.box(), S.cloth()],
                  [S.place("table"), S.place("table2", x=0.6), S.place("box", z=0.2), S.place("towel", x=0.3, z=0.05)],
                  [att], worlds=1)
