# SPDX-License-Identifier: Apache-2.0
"""Soft solids in a composed scene (audit TOOLS-11): a library object whose material makes it a soft body (a sponge,
a fruit) is placed as a rigid copy unless the sentence asks for a soft one, the log says which and what a soft body
costs, and the saved folder loads and routes that way on its own."""

from __future__ import annotations

import pytest

import alloy
from alloy.cards.library import CardLibrary, default_root
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.load import load_scene


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def test_a_soft_solid_is_placed_rigid_and_the_log_says_so(cat, tmp_path):
    assert cat.info("sponge_kitchen").route == "soft_vbd"                    # the library card alone is a soft body
    r = compose("a sponge and a mug on a table", variations=1, check="load", device="cpu", catalog=cat)
    assert r.ok and len(r.scenes) == 2
    for s in r.scenes:
        assert {o["name"]: o["card"] for o in s.objects} == {"sponge_kitchen_rigid": "sponge_kitchen",
                                                             "mug_stoneware": "mug_stoneware"}
        note = next(n for n in s.log if n.startswith("sponge_kitchen: placed as a rigid body"))
        assert "far slower" in note and "'a soft sponge'" in note
    assert "sponge_kitchen_rigid" in r.generated
    r.save(tmp_path)
    assert (tmp_path / "objects" / "sponge_kitchen_rigid.yaml").is_file()
    sim = alloy.build(load_scene(tmp_path / "scenes" / "scene.yaml"), device="cpu")
    assert {n: v["route"] for n, v in sim.routes().items()} == {
        "table_top_plywood": "static", "mug_stoneware": "rigid", "sponge_kitchen_rigid": "rigid"}
    sim.step(5)
    assert sim.check()["ok"]


def test_a_sentence_that_asks_for_a_soft_object_keeps_it_and_states_the_cost(cat, caplog):
    with caplog.at_level("WARNING", logger="alloy.compose"):
        r = compose("a soft sponge on a table", check="none", device="cpu", catalog=cat)
    s = r.scenes[0]
    assert [o.object for o in s.card.objects][1:] == ["sponge_kitchen"] and not r.generated
    assert any(n.startswith("sponge_kitchen: a soft body, as asked") and "far slower" in n for n in s.log)
    assert any("a soft body, as asked" in rec.getMessage() for rec in caplog.records)
    assert not any("rigid body" in n for n in s.log)


def test_rigid_objects_get_no_copy_and_no_note(cat):
    r = compose("a mug and two cubes on a table", check="none", device="cpu", catalog=cat)
    assert not r.generated and not any("soft body" in n for n in r.scenes[0].log)


def test_a_saved_copy_names_its_library_object_so_another_machine_fetches_it(cat, tmp_path, monkeypatch):
    """A copy keeps the object's mesh path and material. A build fetches the objects a scene names, not what a copy
    was made from, so loading the saved folder asks for that object (on a machine without it, an apple's rigid copy
    failed with "no material 'ycb_013_apple_eff'")."""
    import alloy.api._build as build_mod
    from alloy.compose.load import fetch_sources
    from alloy.compose.soft.cards import copied_from, liquid_column, static_copy

    r = compose("a sponge on a table", check="none", device="cpu", catalog=cat)
    assert copied_from(r.generated["sponge_kitchen_rigid"]) == "sponge_kitchen"
    assert copied_from(static_copy("mug_still", cat.library.object("mug_stoneware")).text) == "mug_stoneware"
    assert copied_from(liquid_column("w", "water", radius=0.03, height=0.05, passage=0.06, where="a mug").text) == ""
    r.save(tmp_path)
    assert fetch_sources(tmp_path) == ["sponge_kitchen"]            # on disk here: nothing to fetch
    asked = []
    monkeypatch.setattr(build_mod, "fetch_imported",
                        lambda lib, names: asked.append((lib.has_object("sponge_kitchen_rigid"), list(names))))
    load_scene(tmp_path / "scenes" / "scene.yaml")
    assert asked == [(True, ["sponge_kitchen"])]                    # asked of a library that sees the saved folder
