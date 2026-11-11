# SPDX-License-Identifier: Apache-2.0
"""Cache, queue, fill, records and coverage over a throwaway alloy-assets tree."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from alloy.cards.library import CardLibrary
from alloy.sourcing.cache import Cache, CacheEntry, load_entry
from alloy.sourcing.cli import run
from alloy.sourcing.coverage import summarize, to_markdown
from alloy.sourcing.fields import SourcingError
from alloy.sourcing.match import fill_from_cache
from alloy.sourcing.paths import sources_root
from alloy.sourcing.queue import scan_library
from alloy.sourcing.record import Evidence, SourcedValue
from alloy.sourcing.store import ObjectRecord, apply_to_material, card_overrides, load_records, write_record

EV = Evidence(url="https://example.org/mug", quote="Weight 340 g", tag="DOC")
MASS = SourcedValue(field="mass", value=0.34, lo=0.28, hi=0.45, tag="DOC", evidence=(EV,))
REST = SourcedValue(field="restitution", value=0.5, lo=0.3, hi=0.7, tag="ESTIMATE", reason="ceramic impacts")


def _mug_entry() -> CacheEntry:
    return CacheEntry(key="ceramic_mug", kind="category", title="Ceramic mug", aliases=("mug",),
                      material="stoneware", values={"mass": MASS})


def _cache(root: Path) -> Cache:
    cache = Cache(sources_root(root))
    cache.put(_mug_entry())
    cache.put(CacheEntry(key="stoneware", kind="material", title="Glazed stoneware", values={"restitution": REST}))
    return cache


def test_cache_round_trip_and_whole_word_match(assets: Path) -> None:
    cache = _cache(assets)
    path = sources_root(assets) / "cache" / "categories" / "ceramic_mug.yaml"
    assert load_entry(path) == _mug_entry()
    fresh = Cache(sources_root(assets))
    assert fresh.find("Porcelain_Coffee_Mug_12oz").key == "ceramic_mug"
    assert fresh.find("smuggler_box") is None
    assert cache.get("stoneware", "material").values["restitution"] == REST


def test_cache_key_must_be_snake_case() -> None:
    with pytest.raises(SourcingError, match="snake_case"):
        CacheEntry(key="Ceramic Mug", kind="category", title="x")


def test_queue_reads_card_tags(assets: Path) -> None:
    scan = scan_library("demo", assets)
    tags = {(i.object, i.field): i.tag for i in scan.items}
    assert tags[("mug_white", "density")] == "ESTIMATE"
    assert tags[("mug_white", "mu_static")] == "DOC"
    assert tags[("mug_white", "mu_dynamic")] == "MISSING"
    assert tags[("mug_white", "restitution")] == "UNVERIFIED"
    assert {i.field for i in scan.open()} == {"density", "mu_dynamic"}
    assert not scan.skipped


def test_hints_add_mass_items(assets: Path) -> None:
    hint = '{"object": "mug_white", "category": "ceramic_mug", "fields": ["mass"]}\n'
    (assets / "imports" / "demo" / "sourcing_queue.jsonl").write_text(hint)
    items = [i for i in scan_library("demo", assets).items if i.object == "mug_white"]
    mass = next(i for i in items if i.field == "mass")
    assert (mass.tag, mass.card_path, mass.category) == ("ESTIMATE", "material.density", "ceramic_mug")


def test_fill_writes_priors_and_never_downgrades(assets: Path) -> None:
    (assets / "imports" / "demo" / "sourcing_queue.jsonl").write_text('{"object": "mug_white", "fields": ["mass"]}\n')
    cache = _cache(assets)
    strict = fill_from_cache(scan_library("demo", assets), cache, sources_root(assets), dry_run=True)
    assert strict.written == [] and strict.unmatched == ["demo/bowl_red", "demo/mug_white"]
    report = fill_from_cache(scan_library("demo", assets), cache, sources_root(assets), fuzzy=True)
    assert report.written == ["demo/mug_white"] and report.unmatched == ["demo/bowl_red"]
    rec = load_records("demo", sources_root(assets))["mug_white"]
    assert rec.category == "ceramic_mug"
    assert rec.values["mass"].tag == "ESTIMATE" and rec.values["mass"].evidence == (EV,)
    assert "restitution" not in rec.values  # the card's UNVERIFIED value is not open, so not replaced
    after = {(i.object, i.field): (i.tag, i.recorded) for i in scan_library("demo", assets).items}
    assert after[("mug_white", "mass")] == ("ESTIMATE", True)


def test_specific_card_estimates_are_kept(assets: Path) -> None:
    lib = assets / "imports" / "demo"
    (lib / "sourcing_queue.jsonl").write_text(
        '{"object": "mug_white", "category": "ceramic_mug", "fields": ["mass"], '
        '"note": "ESTIMATE: net contents 12 oz in the product description"}\n')
    scan = scan_library("demo", assets)
    assert not scan.skipped
    mass = next(i for i in scan.items if (i.object, i.field) == ("mug_white", "mass"))
    assert mass.tag == "ESTIMATE" and "product description" in mass.note
    assert ("mug_white", "density") in {(i.object, i.field) for i in scan.items}  # shared material keeps density
    report = fill_from_cache(scan, _cache(assets), sources_root(assets), dry_run=True)
    assert report.written == [] and report.unmatched == ["demo/bowl_red"]


def test_record_overrides_and_apply(assets: Path) -> None:
    rec = ObjectRecord(library="demo", object="mug_white", values={"mass": MASS, "restitution": REST})
    write_record(rec, sources_root(assets))
    assert load_records("demo", sources_root(assets))["mug_white"] == rec
    over = card_overrides(rec, volume=1.7e-4)
    assert over["material.density"].value == pytest.approx(2000.0)
    assert over["material.density"].unc == pytest.approx((0.28 / 1.7e-4, 0.45 / 1.7e-4))
    mat = apply_to_material(CardLibrary().material("abs"), rec, volume=1.7e-4)
    assert mat.surface.restitution.value == 0.5 and mat.density.source == "fit"
    with pytest.raises(SourcingError, match="volume"):
        card_overrides(rec, volume=0.0)


def test_coverage_shares(assets: Path) -> None:
    cov = summarize({"demo": scan_library("demo", assets)})
    total = cov["total"]
    assert total["values"] == 8 and cov["libraries"]["demo"]["objects"] == 2
    assert (total["sourced"], total["weak"], total["estimated"], total["missing"]) == (2, 2, 2, 2)
    assert total["estimated_documented"] == 0
    assert total["sourced_share"] == 0.25
    assert "| demo | 2 | 8 |" in to_markdown(cov)


def test_cli_json(assets: Path) -> None:
    _cache(assets)
    out = run({"cmd": "queue", "assets": str(assets)})
    assert out["ok"] and out["open"] == 4 and out["by_field"] == {"density": 2, "mu_dynamic": 2}
    assert run({"cmd": "lookup", "name": "white mug", "assets": str(assets)})["entry"]["key"] == "ceramic_mug"
    assert not run({"cmd": "lookup", "name": "kettle", "assets": str(assets)})["ok"]
    good = sources_root(assets) / "cache" / "categories" / "ceramic_mug.yaml"
    bad = assets / "bad.yaml"
    bad.write_text("schema: alloy.sourcing.category/1\nkey: x\nkind: category\ntitle: t\nvalues: {mass: {value: 1}}\n")
    out = run({"cmd": "validate", "paths": [str(good), str(bad)], "assets": str(assets)})
    assert not out["ok"] and out["checked"] == 2 and "missing keys" in out["errors"][0]["error"]
    assert run({"cmd": "nope"})["ok"] is False


def test_category_mass_kept_inside_or_clamped_and_flagged(assets: Path) -> None:
    import json

    row = {"card": "mug_white", "field": "material.density", "value": 2300, "unc": [2000, 2500],
           "note": "ESTIMATE: category rule", "category": "ceramic_mug", "query": "", "extra": {"mass_kg": 0.30}}
    lib = assets / "imports" / "demo"
    (lib / "queue.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "card": "bowl_red",
                                                                           "extra": {"mass_kg": 2.0}}) + "\n")
    report = fill_from_cache(scan_library("demo", assets), _cache(assets), sources_root(assets))
    recs = load_records("demo", sources_root(assets))
    assert recs["mug_white"].values["mass"].value == pytest.approx(0.30)  # inside 0.28-0.45: kept, range documented
    assert "inside the documented range" in recs["mug_white"].values["mass"].reason
    bowl = recs["bowl_red"].values["mass"]  # 2.0 kg contradicts the mug range: clamped, reasoned, listed for review
    assert bowl.value == pytest.approx(0.45) and "clamped" in bowl.reason
    assert [f["object"] for f in report.flags] == ["bowl_red"]


def test_effective_density_times_scan_volume_is_used_first(assets: Path) -> None:
    import json

    cache = _cache(assets)
    rho = SourcedValue(field="effective_density", value=600.0, lo=400.0, hi=700.0, tag="ESTIMATE", evidence=(EV,),
                       reason="rubberwood")
    cache.put(CacheEntry(key="wooden_toy", kind="category", title="Wooden toy", values={"effective_density": rho}))
    row = {"card": "mug_white", "field": "material.density", "value": 300, "unc": [100, 600], "note": "ESTIMATE: rule",
           "category": "wooden_toy", "query": "", "extra": {"mass_kg": 0.05, "volume_m3": 1e-4}}
    (assets / "imports" / "demo" / "queue.jsonl").write_text(json.dumps(row) + "\n")
    fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets))
    mass = load_records("demo", sources_root(assets))["mug_white"].values["mass"]
    assert (mass.value, mass.lo, mass.hi) == pytest.approx((0.05, 0.04, 0.07))


def test_mass_implying_impossible_density_is_not_recorded(assets: Path) -> None:
    import json

    row = {"card": "mug_white", "field": "material.density", "value": 300, "unc": [100, 600], "note": "ESTIMATE: x",
           "category": "ceramic_mug", "query": "", "extra": {"volume_m3": 2.6e-5}}
    big = {**row, "extra": {"volume_m3": 1e-5}}  # 0.34 kg in 10 cm^3 = 34,000 kg/m^3
    (assets / "imports" / "demo" / "queue.jsonl").write_text(json.dumps(big) + "\n")
    report = fill_from_cache(scan_library("demo", assets), _cache(assets), sources_root(assets))
    assert any(f.get("implied_density", 0) > 25_000 for f in report.flags)
    assert "mug_white" not in load_records("demo", sources_root(assets))


def test_mass_denser_than_the_category_material_is_not_recorded(assets: Path) -> None:
    import json

    cache = _cache(assets)
    rho = SourcedValue(field="density", value=700.0, lo=500.0, hi=900.0, tag="ESTIMATE", evidence=(EV,), reason="wood")
    cache.put(CacheEntry(key="stoneware", kind="material", title="Glazed stoneware", values={"density": rho}))
    row = {"card": "mug_white", "field": "material.density", "value": 300, "unc": [100, 600], "note": "ESTIMATE: x",
           "category": "ceramic_mug", "query": "", "extra": {"volume_m3": 2.0e-4}}  # 0.34 kg / 0.2 L = 1,700 kg/m^3
    thin = {**row, "card": "bowl_red", "extra": {"volume_m3": 0.9e-4}}  # 3,778 kg/m^3
    (assets / "imports" / "demo" / "queue.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(thin) + "\n")
    report = fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets))
    # an open category (no effective density) allows thinned scan walls: 1.5 x 2.5 x 900 = 3,375 kg/m^3
    assert [(f["object"], f["max_density"]) for f in report.flags if "max_density" in f] == [("bowl_red", 3375.0)]
    assert "mug_white" in load_records("demo", sources_root(assets))
    closed = SourcedValue(field="effective_density", value=1500.0, lo=1400.0, hi=1600.0, tag="ESTIMATE",
                          evidence=(EV,), reason="filled mug")
    cache.put(dataclasses.replace(_mug_entry(), values={"mass": MASS, "effective_density": closed}))
    report = fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets))
    assert {f["max_density"] for f in report.flags if "max_density" in f} == {1350.0}  # closed: 1.5 x 900


def test_friction_prior_contradicting_the_card_is_left_out(assets: Path) -> None:
    cache = _cache(assets)  # the card: mu_static 0.5 (cited), mu_dynamic 0.4 with no source (open)
    for value, kept in ((0.3, True), (0.6, False)):
        mu = SourcedValue(field="mu_dynamic", value=value, lo=value - 0.1, hi=value + 0.1, tag="ESTIMATE",
                          evidence=(EV,), reason="glaze on wood")
        cache.put(CacheEntry(key="stoneware", kind="material", title="Glazed stoneware",
                             values={"restitution": REST, "mu_dynamic": mu}))
        report = fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets),
                                 fuzzy=True)
        rec = load_records("demo", sources_root(assets)).get("mug_white")
        assert (rec is not None and rec.values["mu_dynamic"].value == value) is kept
        assert report.friction_dropped == (0 if kept else 1)


def test_density_from_measured_mass_is_fit(tmp_path: Path) -> None:
    v = SourcedValue(field="mass", value=0.1, lo=0.1, hi=0.1, tag="DOC", evidence=(EV,), q_source="measured")
    q = card_overrides(ObjectRecord(library="demo", object="x", values={"mass": v}), volume=1e-4)["material.density"]
    assert (q.source, q.value) == ("fit", pytest.approx(1000.0))


def test_refill_is_idempotent_and_keeps_object_findings(assets: Path) -> None:
    cache = _cache(assets)  # the card's mu_dynamic has no source (MISSING); the prior below has no evidence either
    bare = SourcedValue(field="mu_dynamic", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE", reason="glaze, no source")
    cache.put(CacheEntry(key="stoneware", kind="material", title="Glazed stoneware", values={"mu_dynamic": bare}))
    (assets / "imports" / "demo" / "sourcing_queue.jsonl").write_text('{"object": "mug_white", "fields": ["mass"]}\n')
    own = SourcedValue(field="mass", value=0.31, lo=0.3, hi=0.32, tag="ESTIMATE", evidence=(EV,), reason="2 listings")
    write_record(ObjectRecord(library="demo", object="mug_white", values={"mass": own}), sources_root(assets))
    for _ in range(2):
        fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets), fuzzy=True)
        rec = load_records("demo", sources_root(assets))["mug_white"]
        assert rec.values["mu_dynamic"].value == 0.3  # written for a MISSING card value, and kept on the refill
        assert rec.values["mass"] == own  # an object finding is never replaced by the category's mass range


def test_prior_the_card_already_carries_keeps_its_record(assets: Path) -> None:
    cache = _cache(assets)
    bare = SourcedValue(field="mu_dynamic", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE", reason="glaze, no source")
    cache.put(CacheEntry(key="stoneware", kind="material", title="Glazed stoneware", values={"mu_dynamic": bare}))
    fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets), fuzzy=True)
    prior = load_records("demo", sources_root(assets))["mug_white"].values["mu_dynamic"]
    mat = assets / "imports" / "demo" / "materials" / "demo_ceramic.yaml"  # an importer applies the record ...
    mat.write_text(mat.read_text().replace("  mu_dynamic: 0.4\n", "  mu_dynamic: {value: 0.3, source: model, note: "
                                           f"'ESTIMATE: {prior.reason}'}}\n"))
    fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets), fuzzy=True)
    assert load_records("demo", sources_root(assets))["mug_white"].values["mu_dynamic"] == prior  # ... record stays


def test_no_density_prior_beside_a_recorded_mass(assets: Path) -> None:
    cache = _cache(assets)  # the card's density is ESTIMATE (open); the material has a documented density
    rho = SourcedValue(field="density", value=2400.0, lo=2300.0, hi=2500.0, tag="ESTIMATE", evidence=(EV,), reason="r")
    cache.put(CacheEntry(key="stoneware", kind="material", title="Glazed stoneware", values={"density": rho}))
    own = SourcedValue(field="mass", value=0.31, lo=0.3, hi=0.32, tag="ESTIMATE", evidence=(EV,), reason="2 listings")
    write_record(ObjectRecord(library="demo", object="mug_white", values={"mass": own}), sources_root(assets))
    cache.put_map("demo", {"bowl_red": "ceramic_mug"})
    fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets), fuzzy=True)
    rec = load_records("demo", sources_root(assets))["mug_white"]
    assert "density" not in rec.values and rec.values["mass"] == own  # mass / volume gives the card's density
    assert "density" in load_records("demo", sources_root(assets))["bowl_red"].values  # no mass: the prior applies
