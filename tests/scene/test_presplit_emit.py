# SPDX-License-Identifier: Apache-2.0
"""Objects with a TopologyBudget and failure values are emitted pre-split through WP10 (D19): seam bonds exist in
every world, the BondSet is read back after finalize, and splitting keeps the mass."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.cards.types import Budget, Failure
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.topology.types import BondKind, TopologyBudget


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def _tearable(mat, topology=True):
    failure = Failure(fracture_strain=T.q(0.3), tear_energy=T.q(500.0))
    return dataclasses.replace(mat, failure=failure), TopologyBudget(spare_vertices=400) if topology else None


def _built(obj, z):
    card, lib = S.scene([S.table(), obj], [S.place("table"), S.place(obj.card.name, z=z)], worlds=2)
    return SceneAssembler(lib, Router()).assemble(card, "cpu")


def test_tearable_cloth_is_presplit_with_cloth_edge_bonds():
    mat, topo = _tearable(T.material("cotton", **T.COTTON, beta=1e-3))
    towel = T.box("towel", mat, 0.1, 0.1, 2.5e-4, budget=Budget(particles=121), topology=topo)
    plain = T.box("towel", T.material("cotton", **T.COTTON, beta=1e-3), 0.1, 0.1, 2.5e-4, budget=Budget(particles=121))
    built, ref = _built(towel, 0.03), _built(plain, 0.03)
    assert built.handles["towel"].decision.route == Route.CLOTH_VBD
    assert built.bonds is not None and built.bonds.count > 0
    assert set(built.bonds.kind.numpy()) == {int(BondKind.CLOTH_EDGE)}
    assert set(built.bonds.world.numpy()) == {0, 1}
    assert built.model.particle_count > ref.model.particle_count            # seam vertices duplicated
    m, m0 = built.model.particle_mass.numpy().sum(), ref.model.particle_mass.numpy().sum()
    assert m == pytest.approx(m0, rel=1e-5)


def test_cuttable_soft_block_is_presplit_with_tet_cut_bonds():
    mat, topo = _tearable(T.material("foam", **T.PU_FOAM, beta=1e-3))
    block = T.box("sponge", mat, 0.04, 0.03, 0.02, budget=Budget(tets=200), topology=topo)
    built = _built(block, 0.045)
    assert built.handles["sponge"].decision.route == Route.SOFT_VBD
    assert built.bonds is not None and built.bonds.count > 0
    assert set(built.bonds.kind.numpy()) == {int(BondKind.TET_CUT)}
    tet_vol = built.model.particle_mass.numpy().sum() / 2 / mat.density.value
    assert tet_vol == pytest.approx(0.08 * 0.06 * 0.04, rel=1e-4)


def test_no_failure_values_keeps_the_plain_mesh():
    towel = T.box("towel", T.material("cotton", **T.COTTON, beta=1e-3), 0.1, 0.1, 2.5e-4, budget=Budget(particles=121),
                  topology=TopologyBudget(spare_vertices=400))
    assert _built(towel, 0.03).bonds is None
