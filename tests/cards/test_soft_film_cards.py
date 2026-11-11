# SPDX-License-Identifier: Apache-2.0
"""The soft-asset pack's film and cloth cards: every authored value is tagged, each object takes its route, and the
one-line example drops a film sheet and a towel on CPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from alloy.cards.library import CardLibrary
from alloy.cards.paths import iter_q_leaves
from alloy.core.registry import isolated_registry, registered_routes
from alloy.route.router import Router
from alloy.route.types import SceneContext
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.sourcing.tags import has_link, tag_of
from alloy.testing.soft_film import OBJECTS

LIB = CardLibrary(assets=None)
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "soft" / "soft_film_drop.py"


def _leaves(name: str):
    obj = LIB.peek_object(name)
    yield from ((f"{name}.{p}", q) for p, q in iter_q_leaves(obj))
    yield from ((f"{obj.material}.{p}", q) for p, q in iter_q_leaves(LIB.material(obj.material)))


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_every_value_is_tagged(name: str) -> None:
    bad = []
    for path, q in _leaves(name):
        if q.source == "default":      # not authored: the load envelope stays at the engine defaults (task specific)
            continue
        tag = tag_of(q)
        if tag == "MISSING":
            bad.append((path, "no source"))
        elif tag == "ESTIMATE" and not (q.unc is not None and q.unc[0] < q.unc[1]):
            bad.append((path, "an ESTIMATE needs a range lo < hi"))
        elif tag == "DOC" and not has_link(q.note):
            bad.append((path, "a DOC value needs its URL"))
        if q.unc is not None and not q.unc[0] <= q.value <= q.unc[1]:
            bad.append((path, f"value {q.value} outside its range {q.unc}"))
    assert bad == []


def test_each_object_takes_its_route() -> None:
    router = Router()
    with isolated_registry(empty=True):
        register_builtin_solvers()
        routes = registered_routes()
        got = {}
        for name in OBJECTS:
            obj = LIB.resolve(LIB.object(name))
            ctx = SceneContext(rigid_partners={name: ()}, free_bodies_per_world=1, registered_routes=routes)
            got[name] = str(router.route(obj, ctx).route)
    assert got == OBJECTS


def test_films_keep_their_folds() -> None:
    """Every film keeps a fold: a yield angle from its card's film block, else from the plug-in's FILM_DEFAULTS."""
    from alloy.solvers.film.emit import film_extras

    for name, route in OBJECTS.items():
        if route == "cloth_film":
            assert film_extras(LIB.material(LIB.peek_object(name).material)).yield_angle > 0.0, name


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize("card", ["film_hdpe_sheet", "paper_kraft_sheet", "towel_cotton"])
def test_example_drops_the_card_onto_the_ground(card: str) -> None:
    spec = importlib.util.spec_from_file_location("soft_film_drop", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.drop(card, seconds=0.3, device="cpu")
    assert out["route"] == OBJECTS[card] and out["health_ok"]
    assert 0.0 < out["z_min_m"] < 0.05                   # falling, above the ground


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_bag_rests_flat_on_the_ground() -> None:
    """The flat-folded two-layer bag (seam hinges at rest near ±pi) stays flat: no vertex moves more than the layer
    gap closing plus the 0.5 mm drop, and nothing lifts (a bending wrap at pi made its seams fold in by 9-40 mm)."""
    import numpy as np

    import alloy

    scene = alloy.Scene()
    scene.add_ground()
    scene.add_object("bag", "bag_hdpe_tshirt", pos=(0, 0, 0.00155))     # bottom layer 0.5 mm above the ground
    sim = alloy.build(scene, device="cpu")
    p0 = np.asarray(sim.state("bag")["bag.points"])[0]
    sim.run(0.2)
    p = np.asarray(sim.state("bag")["bag.points"])[0]
    assert sim.health()["ok"]
    assert float(np.linalg.norm(p - p0, axis=1).max()) < 3.0e-3
    assert float(p[:, 2].max()) < 2.0e-3
