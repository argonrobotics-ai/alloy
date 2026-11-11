# SPDX-License-Identifier: Apache-2.0
"""The metal cards' J2 plasticity values (SCR-WP7-15 item 5; sources in each card note and in
[internal note]): which cards carry ``plastic.model: j2``, their sourced yield and derived
hardening, and that taking the values changes no library object's route and no material ``alloy.create`` picks."""

from __future__ import annotations

import dataclasses
import math

import pytest

from alloy.cards.library import CardLibrary
from alloy.cards.validate import validate_material

MPA = 1e6
# card: (yield_stress σ_y0 [MPa], hardening_modulus H [MPa] or None (no sourced uniform elongation: perfectly plastic))
J2_METALS: dict[str, tuple[float, float | None]] = {
    "aluminium_6061_t6": (276.0, 998.0), "copper_c110": (69.0, 603.0), "brass_c360": (310.0, None),
    "stainless_303": (240.0, 1590.0), "stainless_304": (290.0, 1648.0), "steel_carbon": (235.0, 1057.0),
    "steel_zinc_plated": (640.0, None), "steel_tinplate": (580.0, None),
}
UNSOURCED = ("steel_nickel_plated", "steel_crv", "steel_silver")      # no proof stress found: no plastic block
LIB = CardLibrary(assets=None)


def _secant(sy: float, su: float, eu: float) -> float:
    """WP7's H: the secant from σ_y0 to the true UTS, (σu(1+eu) − σy0) / ln(1+eu)."""
    return (su * (1.0 + eu) - sy) / math.log1p(eu)


@pytest.mark.parametrize("name", sorted(J2_METALS))
def test_each_metal_card_is_j2_with_its_sourced_values(name: str) -> None:
    mat = LIB.material(name)
    sy, h = J2_METALS[name]
    assert mat.plastic.model == "j2" and mat.plastic.yield_stress.value == pytest.approx(sy * MPA, rel=0, abs=1.0)
    assert mat.plastic.yield_stress.source == "catalog" and "http" in mat.plastic.yield_stress.note
    if h is None:
        assert mat.plastic.hardening_modulus is None
    else:
        assert mat.plastic.hardening_modulus.value == pytest.approx(h * MPA, rel=0, abs=1.0)
        assert mat.plastic.hardening_modulus.source == "model" and "derived" in mat.plastic.hardening_modulus.note
    assert [i for i in validate_material(mat) if i.level == "error"] == []


def test_the_metals_without_a_sourced_yield_keep_no_plastic_block() -> None:
    assert all(LIB.material(n).plastic is None for n in UNSOURCED)


def test_the_derived_hardening_follows_its_stated_inputs() -> None:
    """The notes' arithmetic, to the MPa: EN 1993-1-4 Annex C e_u = 1 − σy/σu (stainless), Yun & Gardner's S235
    e_u 20.83 %, and a Hollomon n (Considère: ε_u,true = n) for 6061-T6 and annealed copper."""
    assert _secant(290, 655, 1 - 290 / 655) == pytest.approx(1648, abs=0.5)
    assert _secant(240, 620, 1 - 240 / 620) == pytest.approx(1590, abs=0.5)
    assert _secant(235, 360, 0.2083) == pytest.approx(1057, abs=0.5)
    assert _secant(276, 310, math.expm1(0.05)) == pytest.approx(998, abs=0.5)
    assert _secant(69, 221, math.expm1(0.474)) == pytest.approx(603, abs=0.5)


def test_no_library_object_of_these_metals_changes_route() -> None:
    """Every library object made of a j2 metal routes as it did before the plastic block (WP7's rule: a j2 solid
    keeps its solid route); today all 18 are rigid."""
    from alloy.core.registry import isolated_registry, registered_routes
    from alloy.route.router import Router
    from alloy.route.types import SceneContext
    from alloy.scene.builtin_plugins import register_builtin_solvers

    names = [n for n in LIB.names("objects") if LIB.object(n).material in J2_METALS]
    assert len(names) >= 18
    with isolated_registry(empty=True):
        register_builtin_solvers()
        for name in names:
            obj = LIB.resolve(LIB.object(name))
            bare = dataclasses.replace(obj, material=dataclasses.replace(obj.material, plastic=None))
            ctx = SceneContext(rigid_partners={name: ()}, free_bodies_per_world=1,
                               registered_routes=registered_routes())
            routes = {str(Router().route(o, ctx).route) for o in (obj, bare)}
            assert routes == {"rigid"}, (name, routes)


def test_create_still_picks_the_metal_cards_for_metal_words() -> None:
    """alloy.create chooses a rigid library material for a word ("steel"); a j2 metal is still one."""
    from alloy.create.physics import choose_material

    lib = CardLibrary()
    for word, card in (("steel", "steel_carbon"), ("aluminium", "aluminium_6061_t6"), ("brass", "brass_c360"),
                       ("copper", "copper_c110")):
        assert choose_material(lib, None, None, f"a {word} cup")[0] == card, word
