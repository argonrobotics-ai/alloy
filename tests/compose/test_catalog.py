# SPDX-License-Identifier: Apache-2.0
"""OSS-19 catalog: search over the library by noun, synonym and hint; placeable or why not; and a new import library
(``imports/<source>/`` with a manifest) is searched by its titles with no code change."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.compose.catalog import Catalog, import_roots
from alloy.compose.spec import ComposeError


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def test_search_noun_synonym_and_hints(cat):
    assert cat.search("cube")[0].name == "cube_beech_30"
    assert cat.search("cubes")[0].name == "cube_beech_30"
    assert cat.search("spoon", ["wooden"])[0].name == "spoon_wood_oxo"
    assert cat.search("ball", ["steel"])[0].name == "steel_ball_25mm"
    assert "mug_stoneware" in [m.name for m in cat.search("cup", k=50)]           # synonym
    assert cat.search("mug_stoneware")[0].score == 100.0                          # exact card name
    assert cat.search("unicorn") == []
    assert cat.search("wooden spoon")[0].name == "spoon_wood_oxo"                 # leading words are hints


def test_head_noun_skips_a_trailing_colour():
    from alloy.compose.catalog import _head

    assert _head("gso_android_figure_orange", "ycb_gso") == {"figure"}
    assert _head("ycb_017_orange", "ycb_gso") == {"orange"}
    assert _head("spoon_wood_oxo", "library") == {"spoon", "oxo"}
    assert _head("robocasa_mug_tree_001", "robocasa") == {"tree"}


def test_info_placeable_and_reasons(cat):
    mug = cat.info("mug_stoneware")
    assert mug.placeable and mug.route == "rigid" and mug.size[2] == pytest.approx(0.096, abs=2e-3)
    assert cat.info("sponge_kitchen").placeable and cat.info("sponge_kitchen").route == "soft_vbd"   # soft solid
    towel = cat.info("towel_cotton")
    assert not towel.placeable and "cloth_vbd" in towel.reason
    water = cat.info("water_cup")
    assert not water.placeable and "ParticleSource" in water.reason
    assert cat.body("pencil_wood").reoriented                                    # authored upright, laid down
    with pytest.raises(ComposeError) as e:
        cat.body("towel_cotton")
    assert "cannot be placed" in e.value.message
    with pytest.raises(ComposeError):
        cat.info("no_such_card")


def test_shell_objects_place_and_material_words_match_whole_words(cat):
    from alloy.compose.catalog import _family
    from alloy.compose.choose import choose_one
    from alloy.compose.spec import ItemSpec

    bag = cat.info("gift_bag_kraft")                         # an open kraft bag: a film shell placed as an object
    assert bag.placeable and bag.route == "cloth_film" and not bag.sheet
    assert bag.size == pytest.approx((0.2032, 0.1143, 0.2604), abs=1e-3)
    assert cat.body("gift_bag_kraft").com[2] == pytest.approx(0.114, abs=0.01)     # area centroid of the sheet
    board = cat.info("gift_bag_board")
    assert board.placeable and board.sheet                    # a flat sheet: placed only when named
    for words, want in ((("gift",), "gift_bag_kraft"), (("kraft",), "gift_bag_kraft"),
                        (("shopping",), "gift_bag_kraft"), (("plastic",), "bag_hdpe_tshirt")):
        assert choose_one(ItemSpec("bag", words=words), cat, None, []) == want, words
    assert choose_one(ItemSpec("board"), cat, None, []) != "gift_bag_board"
    assert choose_one(ItemSpec("board", words=("bottom",)), cat, None, []) == "gift_bag_board"
    assert choose_one(ItemSpec("insert", words=("bag",)), cat, None, []) == "gift_bag_board"
    assert not _family("pe", "paper") and not _family("pp", "shopping")       # "plastic" is not paper
    assert _family("pe", "pe") and _family("alumin", "aluminium")


def _import_root(tmp_path: Path) -> Path:
    root = tmp_path / "imports" / "demo_source"
    (root / "objects" / "demo_source").mkdir(parents=True)
    (root / "objects" / "demo_source" / "demo_00042.yaml").write_text(
        "schema: alloy.object/1\nname: demo_00042\nshape:\n  primitive:\n    kind: cylinder\n"
        "    size: [0.04, 0.05]\nmaterial: stoneware_glazed\n", encoding="utf-8")
    for name in ("demo_mug_tree_001", "demo_tall_coffee_mug_002"):
        (root / "objects" / "demo_source" / f"{name}.yaml").write_text(
            f"schema: alloy.object/1\nname: {name}\nshape:\n  primitive:\n    kind: box\n"
            "    size: [0.05, 0.05, 0.1]\nmaterial: abs\n", encoding="utf-8")
    manifest = {"schema": "alloy.import_manifest/1", "lib": "demo_source",
                "objects": {"demo_00042": {"title": "Blue Coffee Mug", "category": "mug"}}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_a_new_import_library_is_searched_by_title(tmp_path):
    root = _import_root(tmp_path)
    assert import_roots(tmp_path) == [root]
    cat = Catalog(library=CardLibrary([default_root(), root], assets=None))
    hits = {m.name: m for m in cat.search("mug", k=50)}
    assert "demo_00042" in hits and hits["demo_00042"].entry.source == "demo_source"
    assert hits["demo_00042"].entry.title == "Blue Coffee Mug"
    assert cat.info("demo_00042").placeable
    assert cat.search("coffee")[0].name == "demo_00042"
    ranked = [m.name for m in cat.search("mug", k=50)]
    assert ranked.index("demo_tall_coffee_mug_002") < ranked.index("demo_mug_tree_001")   # a mug, not a mug tree
