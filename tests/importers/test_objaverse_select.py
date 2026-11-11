# SPDX-License-Identifier: Apache-2.0
"""Objaverse importer, metadata stage: licence filter, category table, ranking and caps (no network, no meshes)."""

from __future__ import annotations

import doctest

import pytest

from alloy.cards import CardLibrary
from alloy.importers.objaverse import categories, licence, materials, physical, quality, select
from alloy.importers.objaverse.categories import SCHEMA, load_categories, lvis_index, parse_categories
from alloy.importers.objaverse.errors import ImporterConfigError, ImportRejected
from alloy.importers.objaverse.materials import food_materials


def _row(uid="u1", **kw):
    row = {"uid": uid, "license": "by", "lvis": ["mug"], "name": "Ceramic mug", "tags": [], "faceCount": 20000,
           "glb_textures": 2, "glb_size": 4_000_000, "animationCount": 0, "likeCount": 3, "viewCount": 200,
           "user_display": "Ann", "viewerUrl": "https://sketchfab.com/3d-models/u1"}
    row.update(kw)
    return row


@pytest.mark.parametrize("mod", [licence, select, categories, physical, quality, materials])
def test_docstring_examples_run(mod):
    assert doctest.testmod(mod, optionflags=doctest.ELLIPSIS).failed == 0


@pytest.mark.parametrize("slug,ok", [("by", True), ("cc0", True), ("CC-BY-4.0", True), ("by-sa", False),
                                     ("by-nd", False), ("by-nc", False), ("by-nc-sa", False), ("st", False),
                                     (None, False), ("gpl", False)])
def test_only_cc0_and_cc_by_are_accepted(slug, ok):
    assert licence.check_licence(slug).accepted is ok


def test_attribution_needs_an_author_and_names_the_changes():
    att = licence.attribution(_row())
    assert att["spdx"] == "CC-BY-4.0" and "Ann" in att["text"] and licence.CHANGES in att["text"]
    with pytest.raises(ImportRejected, match="author"):
        licence.attribution(_row(user_display="", user=""))


@pytest.mark.parametrize("kw,stage,reason", [
    ({"license": "by-nc"}, "licence", "non-commercial"),
    ({"lvis": ["giraffe"]}, "select", "no robot-relevant category"),
    ({"glb_textures": 0}, "select", "no textures"),
    ({"faceCount": 100}, "select", "faces outside"),
    ({"animationCount": 2}, "select", "animated"),
    ({"name": "Coca Cola can"}, "select", "blocked term"),
    ({"tags": ["lowpoly"]}, "select", "blocked term"),
])
def test_screen_rejects_with_a_stage_and_reason(kw, stage, reason):
    with pytest.raises(ImportRejected) as e:
        select.screen(_row(**kw))
    assert e.value.stage == stage and reason in e.value.reason


def test_scan_and_staff_pick_rank_higher_and_caps_hold():
    cats = load_categories()
    plain = [_row(f"p{i}") for i in range(80)]
    scan = _row("s", tags=["photogrammetry"], staffpickedAt="2020")
    kept, rejects = select.select(plain + [scan], cats)
    assert kept[0].uid == "s" and "scan" in kept[0].flags
    assert len(kept) == cats["mug"].cap * 2 and sum(r["stage"] == "cap" for r in rejects) == 81 - len(kept)


def test_category_table_is_consistent_with_the_material_library():
    cats = load_categories()
    lib = CardLibrary()
    food = {m.name for m in food_materials()}
    missing = sorted({c.material for c in cats.values()} - food - {n for n in lib.names("materials")})
    assert not missing, f"category materials not in library/ or food_materials(): {missing}"
    assert set(c.group for c in cats.values()) == set(categories.GROUPS)
    assert len(lvis_index(cats)) >= len(cats)


@pytest.mark.parametrize("row,msg", [
    (["kitchen", "abs", [0.1, 0.05, 0.2], [1, 1, 1], "", "x"], "lo <= typical <= hi"),
    (["garden", "abs", [0.1, 0.1, 0.1], [1, 1, 1], "", "x"], "is not one of"),
    (["kitchen", "abs", [0.1, 0.1, 0.1], [1, 1, 1], "squishy", "x"], "unknown"),
    (["kitchen", "abs", [0.1, 0.1]], "expected"),
])
def test_bad_category_rows_name_the_field(row, msg):
    with pytest.raises(ImporterConfigError, match=msg):
        parse_categories({"schema": SCHEMA, "categories": {"thing": row}})


def test_mass_rule_keeps_plausible_solid_and_rejects_absurd_volume():
    mug = load_categories()["mug"]
    assert not physical.choose_mass(mug, 2300.0, 1.4e-4).effective
    eff = physical.choose_mass(mug, 2300.0, 6e-4)
    assert eff.effective and eff.mass == mug.mass.value and eff.note.startswith("ESTIMATE")
    with pytest.raises(ImportRejected, match="effective density"):
        physical.choose_mass(mug, 2300.0, 1e-9)


def test_cli_select_and_review_print_json(tmp_path, capsys):
    import gzip
    import json

    from alloy.importers.objaverse.__main__ import main

    meta = tmp_path / "meta.jsonl.gz"
    with gzip.open(meta, "wt") as fh:
        for row in (_row("a"), _row("b", license="by-nc")):
            fh.write(json.dumps(row) + "\n")
    assert main(["select", "--meta", str(meta), "--out", str(tmp_path / "c.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["candidates"] == 1 and out["rejected"] == {"licence": 1}
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"uid": "a", "category": "mug", "verdict": "ok"}) + "\n")
    (tmp_path / "v.json").write_text(json.dumps({"a": {"keep": True, "category": "cup"}}))
    assert main(["review", "--rows", str(rows), "--verdicts", str(tmp_path / "v.json"),
                 "--out", str(tmp_path / "k.json")]) == 0
    assert json.loads(capsys.readouterr().out)["kept"] == 1
    assert json.loads((tmp_path / "k.json").read_text())[0]["category"] == "cup"
