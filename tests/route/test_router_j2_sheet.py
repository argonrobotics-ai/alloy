# SPDX-License-Identifier: Apache-2.0
"""A deformable sheet of a ``plastic.model: j2`` material routes to the film (SCR-WP7-15 P3): its hinges carry the fibre
section and keep a dent. Nothing else moves: the same sheet without j2 stays elastic cloth, a thick j2 plate stays
rigid, and without the film plug-in the sheet falls back to cloth_vbd, said in its reasons, while a card that asks for
cloth_film itself still fails, naming the fix. (Every library object of a j2 metal keeps its route:
tests/cards/test_metal_j2_cards.py.)"""

from __future__ import annotations

import dataclasses

import pytest

from alloy.cards.types import Crease, RouteOverride
from alloy.core.registry import isolated_registry, registered_routes
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.testing import route as T


@pytest.fixture(scope="module")
def routes():
    with isolated_registry(empty=True):
        register_builtin_solvers()
        yield registered_routes()


def _j2(mat):
    return dataclasses.replace(mat, plastic=dataclasses.replace(mat.plastic, model="j2"))


def _aluminium(j2: bool):
    mat = T.material("al_6061", **T.ALUMINIUM, yield_stress=276e6)
    return _j2(mat) if j2 else mat


def test_a_thin_j2_sheet_goes_to_the_film(routes):
    d = Router().route(T.box("sheet", _aluminium(True), 0.1, 0.1, 2.5e-4), T.context(routes))
    assert d.route == Route.CLOTH_FILM
    assert any("j2 sheet" in r for r in d.reasons), d.reasons


def test_the_same_sheet_without_j2_keeps_its_route(routes):
    plain = Router().route(T.box("sheet", T.material("al", **T.ALUMINIUM), 0.1, 0.1, 2.5e-4), T.context(routes))
    assert plain.route == Route.CLOTH_VBD


def test_a_creased_j2_sheet_keeps_its_panels(routes):
    """The card's fold lines are its author's choice: creases route PANELS before the j2 rule (WP11's review)."""
    crease = Crease(a=(0.0, -0.05, 0.0), b=(0.0, 0.05, 0.0), stiffness=T.q(0.05))
    lid = T.box("lid", _aluminium(True), 0.1, 0.1, 2.5e-4, features=T.features(creases=(crease,)))
    assert Router().route(lid, T.context(routes)).route == Route.PANELS


def test_a_thick_j2_plate_stays_rigid(routes):
    assert Router().route(T.box("tray", _aluminium(True), 0.1, 0.1, 0.001), T.context(routes)).route == Route.RIGID


def test_without_the_film_plugin_a_j2_sheet_is_elastic_cloth_and_says_so(routes, caplog, monkeypatch):
    from alloy.route import router

    monkeypatch.setattr(router, "_J2_FALLBACK_WARNED", set())              # warned once per card name per process
    with caplog.at_level("WARNING", logger="alloy"):
        d = Router().route(T.box("sheet", _aluminium(True), 0.1, 0.1, 2.5e-4),
                           T.context(frozenset(r for r in routes if r != str(Route.CLOTH_FILM))))
    assert d.route == Route.CLOTH_VBD
    assert any("film plug-in is not registered" in r for r in d.reasons), d.reasons
    assert any("keeps no dent" in m and "rigid=True" in m for m in caplog.messages), caplog.messages
    caplog.clear()
    with caplog.at_level("WARNING", logger="alloy"):
        Router().route(T.box("sheet", _aluminium(True), 0.1, 0.1, 2.5e-4),
                       T.context(frozenset(r for r in routes if r != str(Route.CLOTH_FILM))))
    assert not any("keeps no dent" in m for m in caplog.messages), caplog.messages        # the second time: no repeat


def test_add_cloths_film_override_without_the_film_plugin_keeps_the_route_and_warns(routes, caplog, monkeypatch):
    """Scene.add_cloth gives a j2 sheet a cloth_film override (alloy/api/_cloth_route.py). Without the film plug-in it
    does not fail on the missing entry: the router keeps its own route (a thick plate: rigid) and warns, naming the
    fix (WP2's review)."""
    from alloy.route import router

    monkeypatch.setattr(router, "_J2_FALLBACK_WARNED", set())
    plate = T.box("plate", _aluminium(True), 0.03, 0.03, 2.5e-4,
                  route=RouteOverride("cloth_film", "set through alloy.Scene.add_cloth: test"))
    with caplog.at_level("WARNING", logger="alloy"):
        d = Router().route(plate, T.context(frozenset(r for r in routes if r != str(Route.CLOTH_FILM))))
    assert d.route != Route.CLOTH_FILM and not d.overridden, d
    assert any("film plug-in is not registered" in r for r in d.reasons), d.reasons
    assert any("register the film plug-in" in m for m in caplog.messages), caplog.messages
    film = Router().route(plate, T.context(routes))
    assert film.route == Route.CLOTH_FILM and film.overridden


def test_add_cloth_route_follows_a_scene_copy_to_its_card():
    from alloy.api._cloth_route import add_cloth_route

    assert add_cloth_route("aluminium_6061_t6", {}, [])["route"]["route"] == "cloth_film"
    assert add_cloth_route("my_al", {"my_al": {"like": "aluminium_6061_t6", "density": 2700.0}}, []) != {}
    assert add_cloth_route("cotton_woven", {}, []) == {} and add_cloth_route("my_rubber", {"my_rubber": {}}, []) == {}


def test_add_cloth_sends_paper_and_polymer_films_to_the_film_and_refuses_a_paste():
    """add_cloth(material="paper_80gsm" | "film_ldpe") fell to the router's paste rule and the build failed (5.1
    million MPM particles against a budget of 81). A card with a film: block, or a plastic yield_stress with an
    elastic youngs of at least 1 MPa, is a sheet the film caps at yield / E. A paste (dough: 52 kPa, mayonnaise: no
    modulus) is refused with a SceneError naming add_box; granular cards are left to the router."""
    from alloy.api._cloth_route import add_cloth_route, film_reason
    from alloy.api.errors import SceneError

    for name in ("paper_80gsm", "paper_board", "paper_kraft", "film_ldpe", "film_hdpe", "film_lldpe"):
        assert add_cloth_route(name, {}, [])["route"]["route"] == "cloth_film", name
    assert "film: block" in film_reason({"film": {}})
    assert "caps its stretch" in film_reason({"plastic": {"yield_stress": 1.6e7}, "elastic": {"youngs": "340 MPa"}})
    assert film_reason({"plastic": {"yield_stress": 1994.0}, "elastic": {"youngs": 5.2e4}}) is None
    for name in ("dough_wheat", "mayonnaise"):
        with pytest.raises(SceneError, match="add_box") as err:
            add_cloth_route(name, {}, [])
        assert err.value.field == "material"
    for name in ("soil_sand_dry", "cotton_woven"):
        assert add_cloth_route(name, {}, []) == {}, name


def test_a_library_film_card_without_the_film_plugin_fails_and_names_register(routes):
    """Only add_cloth's j2 override falls back. A card that asks for cloth_film itself (the library's polymer films)
    still fails without the plug-in, and the error names the fix (WP2's review)."""
    from alloy.cards.library import CardLibrary
    from alloy.core.errors import RouteError

    lib = CardLibrary()
    obj = lib.resolve(lib.object("film_ldpe_sheet"))
    assert obj.card.route is not None and obj.card.route.route == "cloth_film"
    with pytest.raises(RouteError, match=r"alloy\.solvers\.film\.plugin\.register\(\)"):
        Router().route(obj, T.context(frozenset(r for r in routes if r != str(Route.CLOTH_FILM))))


def test_the_library_materials_add_cloth_sends_to_the_film_are_pinned():
    """Every library material film_reason moves (a non-j2 sheet) is listed here, so a new card with a yield stress and
    a modulus shows up in review instead of silently becoming a film sheet."""
    from pathlib import Path

    import yaml

    from alloy.api._cloth_route import film_reason
    from alloy.cards.library import default_root

    moved = set()
    for path in sorted((Path(default_root()) / "materials").glob("*.yaml")):
        card = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(card, dict) and (card.get("plastic") or {}).get("model") != "j2" and film_reason(card):
            moved.add(path.stem)
    assert moved == {"paper_80gsm", "paper_board", "paper_kraft", "film_hdpe", "film_ldpe", "film_lldpe"}, sorted(moved)


def test_an_override_reason_comes_first():
    """A card's route override is the first reason Sim.routes() shows; the natural pick's reasons follow (a paper
    sheet's object card listed "paste: yield stress ..." first while it ran cloth_film, WP2 2026-11-08)."""
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_object("p", "paper_a4", pos=(0.0, 0.0, 0.05))
    route = alloy.estimate(scene, 1, 0.1, device="cpu")["routes"]["p"]
    assert route["route"] == "cloth_film" and route["reasons"][0].startswith("override ")
