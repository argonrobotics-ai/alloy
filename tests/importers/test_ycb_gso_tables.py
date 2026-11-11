# SPDX-License-Identifier: Apache-2.0
"""alloy.importers.ycb_gso: YCB table, GSO metadata, estimates, specs and job lists (no network, no meshes)."""

from __future__ import annotations

import pytest

from alloy.cards.library import CardLibrary
from alloy.importers.ycb_gso import estimates as est
from alloy.importers.ycb_gso import gso, jobs, specs, ycb

META = '''model {
  file: "model.sdf"
}
name: "Pepsi_Cola_Wild_Cherry_Diet_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt"
description: "Pepsi Cola, Wild Cherry, Diet - 12 - 12 fl oz (355 ml) cans [144 fl oz (4.26 lt)]"
legal {
  copyright: "Copyright 2020 Google LLC."
  license: "Creative Commons Attribution 4.0 International"
}
annotations {
  key: "brand"
  value: "Pepsi"
}
annotations {
  key: "gtin"
  value: "012000018244"
}
categories {
  first: "Consumer Goods"
}
'''


def test_ycb_table_is_complete_and_consistent():
    names = [o.card_name for o in ycb.OBJECTS]
    assert len(names) == len(set(names)) == 78
    lib = CardLibrary()
    measured = {o.card_name for o in ycb.OBJECTS if o.mass_g is not None}
    for o in ycb.OBJECTS:
        lib.material(o.material)                       # every surface material exists and validates
        assert (o.mass_g is None) == (o.ref == "")
        assert o.mass_g is None or o.mass_g > 0
        if o.estimate.startswith("like:"):
            assert o.estimate.split(":")[1] in measured
        assert o.url.endswith(f"{o.dataset_id}_google_16k.tgz")
    assert ycb.by_card_name()["ycb_025_mug"].mass_g == 118
    assert ycb.by_card_name()["ycb_055_baseball"].ref == ycb.REF_IJRR


def test_gso_metadata_octal_escapes_match_listing_names():
    meta = gso.parse_metadata('name: "Pok\\303\\251mon_Yellow"\ndescription: "say \\"hi\\"\\nbye"\n')
    assert meta.name == "Pokémon_Yellow" and meta.description == 'say "hi"\nbye'
    assert gso.card_name(meta.name) == jobs.gso_job("Pokémon_Yellow")["name"] == "gso_pok_mon_yellow"


def test_gso_metadata_and_names():
    meta = gso.parse_metadata(META)
    assert meta.brand == "Pepsi" and meta.gtin == "012000018244"
    assert meta.categories == ("Consumer Goods",)
    assert meta.license == gso.LICENCE_NAME
    assert gso.card_name("Vtech_Cruise_Learn_Car_25_Years") == "gso_vtech_cruise_learn_car_25_years"
    assert gso.archive_url("Pokémon_X").endswith("/Pok%C3%A9mon_X/1/Pok%C3%A9mon_X.zip")


@pytest.mark.parametrize("text, kg", [
    ("Pepsi Cola, Wild Cherry, Diet - 12 - 12 fl oz (355 ml) cans [144 fl oz (4.26 lt)]", 4.26),
    ("Nestle Raisinets, Milk Chocolate - 3.5 oz (99.2 g)", 0.0992233),
    ("Glycerin 11 ... Weight: 11.8 oz", 0.3345244),
    ("no quantity here, 2 pack", None),
    ("Crayola Washable Fingerpaint, Red, Blue, Yellow - 3 count, 8 fl oz bottes each", 3 * 0.2365882365),
    ("Crayola Model Magic Modeling Material, Single Packs - 6 pack, 0.5 oz packs", 6 * 0.5 * 0.028349523125),
    ("Skinny Cow Heavenly Crisp Candy Bar ,Chocolate Raspberry - 6 pack, 4.62 oz total", 4.62 * 0.028349523125),
    ("Little Debbie Donut, Sticks - 6 cake donuts, 10 oz total", 10 * 0.028349523125),
    ("PureFlow 2 ... to last approximately 250-300 miles. Weight: 7.2 oz", 7.2 * 0.028349523125),
])
def test_net_contents(text, kg):
    got = est.net_contents(text)
    if kg is None:
        assert got is None
    else:
        assert got[0] == pytest.approx(kg, rel=1e-4)


def test_category_rules_use_library_materials():
    lib = CardLibrary()
    for rule in [*est.CATEGORY.values(), est.DEFAULT_RULE]:
        lib.material(rule.material)
        assert (rule.density is not None) + (rule.mass is not None) + bool(rule.analogue) == 1
    assert set(est.YCB_ANALOGUES["carton"]) <= set(ycb.by_card_name())


def _fits() -> dict[str, float]:
    return {n: 500.0 + 10 * i for i, n in enumerate(n for v in est.YCB_ANALOGUES.values() for n in v)}


def test_ycb_specs_measured_and_estimated():
    fits = _fits() | {"ycb_072_a_toy_airplane": 365.0}
    mug = specs.ycb_spec(ycb.by_card_name()["ycb_025_mug"], 905.0, fits)
    assert mug.mass.kind == "mass" and mug.mass.value == pytest.approx(0.118) and not mug.queue
    cube = specs.ycb_spec(ycb.by_card_name()["ycb_077_rubiks_cube"], 1040.0, fits)
    assert cube.mass.kind == "density" and cube.mass.value == 365.0
    assert cube.queue[0].field == "material.density"


def test_gso_spec_prefers_net_contents_and_queues_gtin():
    meta = gso.parse_metadata(META)
    s = specs.gso_spec(meta, 5e-3, _fits())                  # 5 L closed scan of a 12-can pack
    assert s.mass.kind == "density"
    assert s.mass.value * 5e-3 == pytest.approx(4.26 * (1 + est.PACKAGING))
    tiny = gso.parse_metadata(META.replace("Pepsi Cola, Wild Cherry, Diet - 12 - 12 fl oz (355 ml) cans [144 fl oz "
                                           "(4.26 lt)]", "Firming Eye Cream 0.5 oz"))
    jar = specs.gso_spec(tiny, 1e-4, _fits())
    assert jar.mass.value * 1e-4 == pytest.approx(0.5 * 0.028349523125 + est.MIN_PACKAGING)
    assert s.queue[0].extra["gtin"] == "012000018244"
    assert "ESTIMATE" in s.queue[0].note
    tiny = specs.gso_spec(meta, 1e-5, _fits())               # implausible density -> category rule
    assert "net contents" not in tiny.mass.note


def test_job_lists():
    yj = jobs.ycb_jobs()
    assert len(yj) == 78 and all(j["mesh"].endswith("google_16k/textured.obj") for j in yj)
    listing = [{"name": "A_b", "license_name": gso.LICENCE_NAME}, {"name": "Nc", "license_name": "Other"}]
    gj = jobs.gso_jobs(listing)
    assert [j["name"] for j in gj] == ["gso_a_b"]
    assert gj[0]["copy"] == [["materials/textures/texture.png", "meshes/texture.png"]]


def test_build_applies_only_the_mass_of_a_contradictory_record():
    sourcing = pytest.importorskip("alloy.sourcing")
    from alloy.importers.common import cards as cw  # noqa: PLC0415
    from alloy.importers.ycb_gso.build import _usable  # noqa: PLC0415

    base = CardLibrary().material("cardboard_corrugated")            # mu_dynamic 0.4
    spec = cw.CardSpec("imp_box", "cardboard_corrugated", "test", cw.MassSpec("density", 300.0, (150.0, 500.0), "t"),
                       1e-3, "meshes/imp_box/physics.obj", "0" * 64, 2, "test")
    sv = sourcing.SourcedValue
    rec = sourcing.ObjectRecord(library="imp", object="imp_box", values={
        "mu_static": sv(field="mu_static", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE", reason="t"),
        "mass": sv(field="mass", value=0.5, lo=0.4, hi=0.6, tag="ESTIMATE", reason="t")})
    used = _usable(spec, base, rec)
    assert set(used.values) == {"mass"}
    assert cw.material_card(spec, base, used).density.value == pytest.approx(500.0)


def test_relabelled_gso_models_get_their_real_names_and_estimates():
    from alloy.importers.ycb_gso.relabels import RELABELS  # noqa: PLC0415

    assert gso.card_name("Ortho_Forward_Facing") == "gso_plush_giraffe"
    assert gso.plain_card_name("Ortho_Forward_Facing") == "gso_ortho_forward_facing"
    assert len({r.card for r in RELABELS.values()}) == len(RELABELS) == 11
    lib = CardLibrary()
    for r in RELABELS.values():
        lib.material(r.rule.material)
    meta = gso.parse_metadata('name: "Granimals_20_Wooden_ABC_Blocks_Wagon"\ndescription: "20 blocks, 2 lb"\n')
    s = specs.gso_spec(meta, 1.66e-4, _fits())
    assert s.name == "gso_can_opener_black" and "mislabel" in s.title
    assert s.mass.value * 1.66e-4 == pytest.approx(0.18)          # the can opener, not the 2 lb block set
    assert "gtin" not in s.queue[0].extra


def test_thumbs_json_overrides_thumbnail_hashes(tmp_path):
    import json  # noqa: PLC0415

    from alloy.importers.ycb_gso.build import apply_thumbs  # noqa: PLC0415

    man = {"objects": {"a": {"files": {"thumb.png": {"sha256": "old", "bytes": 1}}}, "b": {"files": {}}}}
    (tmp_path / "thumbs.json").write_text(json.dumps({"objects": {"a": {"sha256": "new", "bytes": 9, "notes": "x"},
                                                                  "zz": {"sha256": "q", "bytes": 2}}}))
    assert apply_thumbs(tmp_path, man) == 1
    assert man["objects"]["a"]["files"]["thumb.png"] == {"sha256": "new", "bytes": 9}


def test_manifest_files_fall_back_to_the_batch_result(tmp_path):
    from alloy.importers.ycb_gso.build import _files  # noqa: PLC0415

    (tmp_path / "physics.obj").write_text("v 0 0 0\n")
    result = {"files": {"physics.obj": "x", "visual.glb": "abc"}, "sizes": {"physics.obj": 1, "visual.glb": 42}}
    files = _files(tmp_path, result)
    assert files["visual.glb"] == {"sha256": "abc", "bytes": 42}          # deleted locally, kept from result.json
    assert files["physics.obj"]["bytes"] == 8                             # on disk: hashed from the file
