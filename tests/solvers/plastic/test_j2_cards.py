# SPDX-License-Identifier: Apache-2.0
"""The j2 card fields and router rule (SCR-WP7-15, P0): no existing card changes route, a j2 solid keeps its solid
route instead of becoming a paste, the new fields are validated and leave existing card digests alone."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alloy.cards.library import CardLibrary
from alloy.cards.types import Plastic
from alloy.cards.validate import validate_material
from alloy.core.ids import canonical_json
from alloy.core.registry import isolated_registry, registered_routes
from alloy.route.phase import route_phase
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.testing import route as T

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}
LIB = CardLibrary(assets=None)
# the plastic cards from before j2 (pastes, soils, film papers); j2 cards (WP1's metals) are pinned in their own test
PLASTIC_CARDS = tuple(n for n in LIB.names("materials")
                      if LIB.material(n).plastic is not None and LIB.material(n).plastic.model is None)


class _NoJ2Router(Router):
    """The router as it was before the j2 rule: every card goes through the phase step."""

    def _phase(self, obj, ctx, p):
        return route_phase(self.cfg, obj, ctx, p)


@pytest.fixture(scope="module")
def routes():
    with isolated_registry(empty=True):
        register_builtin_solvers()
        yield registered_routes()


def _route(router: Router, mat, routes, size=(0.05, 0.05, 0.05)):
    return router.route(T.box("probe", mat, *size), T.context(routes)).route


def test_the_library_has_pre_j2_plastic_cards():
    assert len(PLASTIC_CARDS) >= 10                         # pastes, soils, film papers


@pytest.mark.parametrize("name", PLASTIC_CARDS)
def test_every_plastic_library_card_keeps_its_route(name, routes):
    mat = LIB.material(name)
    new = _route(Router(rod_self_test=lambda rod, h: dict(PASS)), mat, routes)
    old = _route(_NoJ2Router(rod_self_test=lambda rod, h: dict(PASS)), mat, routes)
    assert new == old, (name, new, old)


def test_a_j2_solid_keeps_its_solid_route_where_a_yield_stress_alone_makes_a_paste(routes):
    router = Router(rod_self_test=lambda rod, h: dict(PASS))
    steel = T.material("steel", **T.STEEL)
    pasty = T.material("steel_y", **T.STEEL, yield_stress=235e6)
    j2 = replace(pasty, plastic=replace(pasty.plastic, model="j2", hardening_modulus=T.q(1.06e9)))
    assert _route(router, pasty, routes) == Route.PASTE_MPM               # today's rule, kept for non-j2 cards
    for size in ((0.05, 0.05, 0.05), (0.1, 0.1, 0.001)):
        assert _route(router, j2, routes, size) == _route(router, steel, routes, size) == Route.RIGID
    soft = T.material("foam", **T.PU_FOAM)
    soft_j2 = replace(soft, plastic=Plastic(yield_stress=T.q(750.0), model="j2"))
    assert _route(router, soft_j2, routes) == _route(router, soft, routes)
    assert validate_material(j2) == [] and validate_material(soft_j2) == []


def test_j2_fields_are_validated():
    steel = T.material("steel", **T.STEEL)

    def errors(plastic):
        return {i.field for i in validate_material(replace(steel, plastic=plastic)) if i.level == "error"}

    assert errors(Plastic(model="j2")) == {"plastic.yield_stress"}
    assert errors(Plastic(yield_stress=T.q(2e8), model="j2", cohesion=T.q(1e3))) == {"plastic.cohesion"}
    assert errors(Plastic(yield_stress=T.q(2e8), hardening_modulus=T.q(1e9))) == {"plastic.hardening_modulus"}
    assert errors(Plastic(yield_stress=T.q(2e8), model="j2", hardening_modulus=T.q(-1.0))) == {
        "plastic.hardening_modulus"}
    no_e = replace(steel, elastic=None, plastic=Plastic(yield_stress=T.q(2e8), model="j2"))
    assert "plastic.model" in {i.field for i in validate_material(no_e) if i.level == "error"}


def test_the_new_fields_leave_existing_digests_alone():
    for name in PLASTIC_CARDS:
        plastic = json.loads(canonical_json(LIB.material(name)))["plastic"]
        assert "model" not in plastic and "hardening_modulus" not in plastic, name
