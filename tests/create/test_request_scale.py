# SPDX-License-Identifier: Apache-2.0
"""create: request validation, size inference (units from the category, generated meshes from the typical
size, absurd sizes refused) and the provenance of mass and material. Pure Python plus the category table."""

from __future__ import annotations

import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.create.physics import Category, choose_material, find_category, mass_value
from alloy.create.request import CreateError, CreateRequest
from alloy.create.scale import resolve_scale

MUG = Category("mug", (0.08, 0.11, 0.14), (0.2, 0.33, 0.5), "stoneware_glazed", "typical mug", "ESTIMATE", (),
               "typical mug")


def test_request_needs_one_input_and_says_what_to_pass():
    with pytest.raises(CreateError) as e:
        CreateRequest()
    assert "mesh=" in e.value.fix
    with pytest.raises(CreateError):
        CreateRequest(mesh="a.obj", text="a mug")
    with pytest.raises(CreateError) as e:
        CreateRequest(mesh="a.obj", size=-1)
    assert e.value.field == "size"
    assert CreateRequest(text="a red ceramic coffee mug").card_name == "a_red_ceramic_coffee"
    assert CreateRequest(mesh="dir/My Part-2.STL").card_name == "my_part_2"
    with pytest.raises(CreateError):
        CreateRequest.from_dict({"mesh": "a.obj", "colour": "red"})


def test_units_inferred_from_the_category_and_from_plain_ranges():
    s = resolve_scale(110.0, None, None, False, MUG)                 # 110 units, a mug: millimetres
    assert s.factor == pytest.approx(0.001) and s.value.tag == "ESTIMATE" and "mm" in s.value.source
    assert resolve_scale(0.11, None, None, False, MUG).factor == 1.0
    assert resolve_scale(0.3, None, None, False, None).factor == 1.0
    assert resolve_scale(250.0, None, None, False, None).factor == pytest.approx(0.001)
    assert resolve_scale(2.0, 0.11, None, False, MUG).value.tag == "USER"
    assert resolve_scale(4.33, None, "in", False, None).value.value == pytest.approx(0.11, abs=1e-3)


def test_generated_meshes_take_the_typical_size_and_absurd_sizes_are_refused():
    s = resolve_scale(1.7, None, None, True, MUG)
    assert s.value.value == pytest.approx(0.11) and s.value.lo == 0.08 and "measure" in s.value.verify
    with pytest.raises(CreateError) as e:
        resolve_scale(1.7, None, None, True, None)
    assert "size=" in e.value.fix
    with pytest.raises(CreateError):
        resolve_scale(1e6, None, None, False, None)                  # units unclear
    with pytest.raises(CreateError):
        resolve_scale(1.0, 12.0, None, False, None)                  # a 12 m object
    assert "far from a typical mug" in resolve_scale(1.0, 1.0, None, False, MUG).value.verify


def test_mass_provenance_user_cited_category_density():
    assert mass_value(CreateRequest(mesh="a.obj", mass=0.3), MUG, 2000, 1e-4).tag == "USER"
    cited = mass_value(CreateRequest(mesh="a.obj", mass=0.3, mass_source="kitchen scale"), MUG, 2000, 1e-4)
    assert cited.tag == "CITED" and cited.verify == ""
    est = mass_value(CreateRequest(mesh="a.obj"), MUG, 2000, 1e-4)
    assert (est.tag, est.value, est.lo, est.hi) == ("ESTIMATE", 0.33, 0.2, 0.5) and "weigh" in est.verify
    solid = mass_value(CreateRequest(mesh="a.obj"), None, 2000, 1e-4)
    plastic = mass_value(CreateRequest(mesh="a.obj", material="plastic"), MUG, 1050, 1e-4, "abs")
    assert plastic.value == pytest.approx(0.105) and "not the typical" in plastic.source  # not a ceramic mug's mass
    assert mass_value(CreateRequest(text="a ceramic mug"), MUG, 2000, 1e-4, MUG.material).value == MUG.mass[1]
    board = Category("cutting_board_wood", (0.25, 0.38, 0.5), (0.8, 1.2, 2.0), "wood_hardwood")
    same = mass_value(CreateRequest(text="a wooden cutting board"), board, 720, 9.6e-3, "wood_beech")
    assert same.value == 1.2 and same.lo == 0.8                     # beech is a hardwood: the category's mass
    assert solid.value == pytest.approx(0.2) and solid.lo == pytest.approx(0.14)


def test_material_from_a_word_the_category_or_the_default():
    lib = CardLibrary([default_root()], assets=None)
    name, row = choose_material(lib, "wood", None, None)
    assert "wood" in name or "beech" in name or "plywood" in name
    assert choose_material(lib, "steel_crv", None, None)[1].tag == "USER"
    assert choose_material(lib, None, MUG, None)[0] == "stoneware_glazed"
    assert choose_material(lib, None, None, None)[0] == "abs"


def test_category_lookup_from_text():
    cat = find_category("a red ceramic coffee mug with a handle", None)
    assert cat is not None and "mug" in cat.name and cat.size is not None and cat.mass is not None
    assert find_category(None, None, "zz_qq") is None


def test_soft_words_do_not_pick_soft_materials_but_a_named_one_is_kept():
    lib = CardLibrary([default_root()], assets=None)
    name, row = choose_material(lib, None, None, "a small yellow rubber duck")
    assert name == "abs" and "soft" in row.source                     # rubber would make a soft body
    assert choose_material(lib, "foam", None, None)[0] == "abs"
    assert choose_material(lib, "silicone_40a", None, None)[1].tag == "USER"



class _WithMaterial:
    """The library with one material replaced in memory."""

    def __init__(self, lib, card):
        self._lib, self._card = lib, card

    def material(self, name):
        return self._card if name == self._card.name else self._lib.material(name)

    def __getattr__(self, attr):
        return getattr(self._lib, attr)


def test_a_j2_metal_is_still_rigid_but_a_paste_is_not():
    import dataclasses

    from alloy.cards.types import Plastic

    lib = CardLibrary([default_root()], assets=None)
    steel = lib.material("steel_carbon")
    yield_stress = dataclasses.replace(steel.elastic.youngs, value=250e6, unc=None, note="in-memory test card")
    j2 = dataclasses.replace(steel, plastic=Plastic(model="j2", yield_stress=yield_stress))
    assert choose_material(_WithMaterial(lib, j2), None, None, "a steel cup")[0] == "steel_carbon"
    paste = dataclasses.replace(steel, plastic=Plastic(yield_stress=yield_stress))      # no model: a paste
    assert choose_material(_WithMaterial(lib, paste), None, None, "a steel cup")[0] != "steel_carbon"

def test_a_drop_test_that_raises_is_a_finding_not_a_crash(monkeypatch, tmp_path):
    from alloy.create import build
    from alloy.create.result import CreateResult

    def boom(*a, **k):
        raise ValueError("cannot reshape array of size 7 into shape (4)")

    monkeypatch.setattr("alloy.check.check_object", boom)
    r = CreateResult(name="duck", card=str(tmp_path / "objects/duck.yaml"))
    report = build._drop_test(r, tmp_path, "cpu")
    assert report["ok"] is False and report["findings"][0]["check"] == "drop_test"
    assert "material=" in report["findings"][0]["fix"]
    r.check = report
    assert not r.ok and r.verify[0].startswith("drop test failed: the drop test could not run")


def test_material_words_in_any_case_and_several_words_pick_the_material_meant():
    """Gemini audit S2-035: 'Steel', 'stainless steel' and 'titanium' all became abs with a false 'soft' note."""
    lib = CardLibrary([default_root()], assets=None)
    assert choose_material(lib, "steel", None, None)[0] == "steel_carbon"
    name, row = choose_material(lib, "Steel", None, None)
    assert name == "steel_carbon" and row.tag == "ESTIMATE" and "soft" not in row.source
    assert choose_material(lib, "stainless steel", None, None)[0].startswith("stainless_")
    assert choose_material(lib, "Stainless 304", None, None) == choose_material(lib, "stainless_304", None, None)
    assert choose_material(lib, "stainless_304", None, None)[1].tag == "USER"
    assert choose_material(lib, "paper", None, None)[0] == "paper_80gsm"          # the preferred pick, not cardboard
    assert choose_material(lib, None, None, "a paper cup")[0] == "paper_80gsm"


def test_an_unknown_material_or_a_liquid_is_refused_with_the_fix():
    lib = CardLibrary([default_root()], assets=None)
    with pytest.raises(CreateError) as e:
        choose_material(lib, "titanium", None, None)
    assert e.value.field == "material" and "'steel'" in e.value.fix and "alloy list materials" in e.value.fix
    with pytest.raises(CreateError) as e:
        choose_material(lib, "water", None, None)
    assert "a liquid" in e.value.message and e.value.field == "material"
    assert choose_material(lib, None, None, "a titanium widget")[0] == "abs"     # a description is only a hint


def test_a_material_word_prefers_alloys_own_library_over_imported_materials(tmp_path):
    (tmp_path / "materials").mkdir()
    text = (default_root() / "materials" / "abs.yaml").read_text(encoding="utf-8")
    (tmp_path / "materials" / "aa_book_paperback.yaml").write_text(text.replace("name: abs", "name: aa_book_paperback"),
                                                                    encoding="utf-8")
    lib = CardLibrary([tmp_path, default_root()], assets=None)
    assert "aa_book_paperback" in lib.names("materials")
    assert choose_material(lib, "kraft", None, None)[0].startswith("paper_kraft")
