# SPDX-License-Identifier: Apache-2.0
"""SOFT_VBD particle contact radius (SCR-XB4-5-1, `alloy.scene.emitters.deformable.soft_particle_radius`): 1/5 of the
mesh's shortest tet edge, on the plain and the pre-split paths, read back at assembly (`readback_checks.check_soft`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.cards.types import Budget, Failure
from alloy.core.errors import ReadbackError
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.scene.emitters.deformable import SOFT_RADIUS_FRACTION, box_mesh, box_tets, soft_particle_radius
from alloy.scene.readback import readback
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.topology.types import TopologyBudget


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def _block(presplit: bool):
    mat, topo = T.material("foam", **T.PU_FOAM, beta=1e-3), None
    if presplit:
        mat = dataclasses.replace(mat, failure=Failure(fracture_strain=T.q(0.3), tear_energy=T.q(500.0)))
        topo = TopologyBudget(spare_vertices=400)
    return T.box("sponge", mat, 0.04, 0.03, 0.02, budget=Budget(tets=200), topology=topo)


def _built(block):
    card, lib = S.scene([S.table(), block], [S.place("table"), S.place("sponge", z=0.045)], worlds=2)
    return SceneAssembler(lib, Router()).assemble(card, "cpu"), lib


def test_the_radius_is_a_fifth_of_the_shortest_cell_edge_of_a_kuhn_grid():
    verts, tets = box_tets(np.array([0.04, 0.03, 0.01]), np.array([8, 6, 4]))      # cells 5 x 5 x 2.5 mm
    assert soft_particle_radius(verts, tets) == pytest.approx(SOFT_RADIUS_FRACTION * 0.0025)


@pytest.mark.parametrize("presplit", [False, True], ids=["plain", "presplit"])
def test_every_soft_particle_gets_the_rule_radius_in_every_world(presplit):
    block = _block(presplit)
    built, _ = _built(block)                                     # assembly reads the radius back (finish_scene)
    assert built.handles["sponge"].decision.route == Route.SOFT_VBD
    assert (built.bonds is not None and built.bonds.count > 0) == presplit
    m = built.model
    r = m.particle_radius.numpy()[m.particle_mass.numpy() > 0]
    expected = soft_particle_radius(*box_mesh(block.card.shape, block.card.budget.tets))
    assert r.size == m.particle_count and 1.0e-4 < expected < 0.01      # not Newton's 0.1 m default
    np.testing.assert_allclose(r, expected, rtol=1e-6)


@pytest.mark.parametrize("presplit", [False, True], ids=["plain", "presplit"])
def test_readback_flags_a_soft_radius_off_the_rule(presplit):
    built, lib = _built(_block(presplit))
    built.model.particle_radius.fill_(0.1)                       # Newton's default, as before SCR-XB4-5-1
    with pytest.raises(ReadbackError, match="sponge.card.budget.tets"):
        readback(built, lib)
