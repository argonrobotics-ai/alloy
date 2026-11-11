# SPDX-License-Identifier: Apache-2.0
"""Quote checking against fetched pages (no network: the fetcher is injected) and dataset name maps."""

from __future__ import annotations

from pathlib import Path

from alloy.sourcing.cache import Cache, CacheEntry, dump_entry
from alloy.sourcing.match import fill_from_cache
from alloy.sourcing.paths import sources_root
from alloy.sourcing.queue import scan_library
from alloy.sourcing.record import Evidence, SourcedValue
from alloy.sourcing.store import load_records
from alloy.sourcing.verify import check_quote, html_text, quote_found, to_json, verify_files

PAGE = "<html><script>var x=9.99</script><table><tr><td>Weight</td><td>13.4 oz</td></tr></table></html>"


def _entry(url: str, quote: str) -> CacheEntry:
    ev = Evidence(url=url, quote=quote, tag="DOC")
    v = SourcedValue(field="mass", value=0.38, lo=0.3, hi=0.45, tag="ESTIMATE", evidence=(ev,), reason="listings")
    return CacheEntry(key="ceramic_mug", kind="category", title="Mug", values={"mass": v})


def test_quote_found_needs_every_number_and_most_words() -> None:
    text = html_text(PAGE)
    assert "9.99" not in text
    assert quote_found("Weight 13.4 oz", text)
    assert quote_found("Weight: 13.4 oz", text)
    assert not quote_found("Weight 14.3 oz", text)
    assert not quote_found("Shipping weight 13.4 oz per carton", text)
    assert not quote_found("", text)
    assert check_quote("Item weight is 13.4 ounces total", text) == "reworded"
    assert check_quote("Item weight is 14.3 ounces", text) == "missing"


def test_verify_files_counts_found_missing_unchecked(tmp_path: Path) -> None:
    pages = {"https://a.example/mug": html_text(PAGE), "https://b.example/mug.pdf": None}
    files = []
    for i, (url, quote) in enumerate([("https://a.example/mug", "Weight 13.4 oz"),
                                      ("https://a.example/mug", "Weight 15 oz"),
                                      ("https://b.example/mug.pdf", "Weight 13.4 oz")]):
        path = tmp_path / f"f{i}" / "ceramic_mug.yaml"
        path.parent.mkdir()
        path.write_text(dump_entry(_entry(url, quote)))
        files.append(path)
    calls: list[str] = []

    def fetch(url: str) -> str | None:
        calls.append(url)
        return pages[url]

    result = to_json(verify_files(files, fetch, retry_delay_s=0.0))
    assert result["counts"] == {"found": 1, "reworded": 0, "missing": 1, "unchecked": 1}
    assert result["missing"][0]["quote"] == "Weight 15 oz"
    assert sorted(calls) == sorted([*pages, "https://a.example/mug"])  # once each, plus one re-fetch on a miss


def test_name_map_assigns_category_through_importer_prefix(assets: Path) -> None:
    cache = Cache(sources_root(assets))
    cache.put(_entry("https://a.example/mug", "Weight 13.4 oz"))
    cache.put_map("demo_dataset", {"Bowl_Red": "ceramic_mug"})
    assert Cache(sources_root(assets)).mapped("demo_bowl_red") == "ceramic_mug"
    assert Cache(sources_root(assets)).mapped("bowl_blue") is None
    lib = assets / "imports" / "demo"
    (lib / "sourcing_queue.jsonl").write_text('{"object": "bowl_red", "fields": ["mass"]}\n')
    report = fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets))
    assert "demo/bowl_red" in report.written


def test_pipeline_queue_rows_turn_density_into_mass(assets: Path) -> None:
    from alloy.sourcing.handoff import read_hints

    row = ('{"card": "mug_white", "field": "material.density", "value": 2300, "unc": [2000, 2500], '
           '"note": "ESTIMATE: category", "category": "Bottles and Cans and Cups", "query": "white mug weight", '
           '"extra": {"brand": "Acme"}}\n')
    lib = assets / "imports" / "demo"
    (lib / "queue.jsonl").write_text(row)
    hints = read_hints(lib)
    assert hints["mug_white"]["fields"] == ["mass"] and hints["mug_white"]["extra"] == {"brand": "Acme"}
    items = {(i.object, i.field): i for i in scan_library("demo", assets).items}
    assert ("mug_white", "density") not in items
    mass = items[("mug_white", "mass")]
    assert (mass.tag, mass.card_path, mass.product) == ("ESTIMATE", "material.density", "white mug weight")
    assert ("bowl_red", "density") in items


def test_pdf_text_without_pypdf_or_bad_bytes_is_unchecked() -> None:
    from alloy.sourcing.verify import pdf_text

    assert pdf_text(b"%PDF-1.4 not really a pdf") is None


def test_fetch_text_never_raises_on_odd_urls() -> None:
    from alloy.sourcing.verify import fetch_text

    assert fetch_text("ftp://example.org/x") is None
    assert fetch_text("http://127.0.0.1:9/a b.pdf", timeout=0.5) is None  # space in path, closed port


def test_ingest_applies_only_stronger_findings(tmp_path: Path) -> None:
    import json

    from alloy.sourcing.ingest import ingest_file

    ev = {"url": "https://example.org/x", "quote": "Weight 50 g", "tag": "DOC"}
    est = {"card": "gso_x", "field": "mass", "value": 0.06, "lo": 0.04, "hi": 0.08, "tag": "ESTIMATE", "reason": "r"}
    doc = {"card": "gso_x", "field": "mass", "value": 0.05, "lo": 0.05, "hi": 0.05, "tag": "DOC", "evidence": [ev]}
    bad = {"card": "gso_y", "field": "mass", "value": 0.05, "lo": 0.05, "hi": 0.05, "tag": "DOC"}
    f = tmp_path / "f.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in (est, doc, est, bad)) + "\nnot json\n")
    report = ingest_file(f, "demo", tmp_path / "src").to_json()
    assert report["applied"] == 2 and report["weaker_or_equal"] == 1
    assert [r["line"] for r in report["rejected"]] == [4, 5]
    assert load_records("demo", tmp_path / "src")["gso_x"].values["mass"].tag == "DOC"


def test_queue_rows_in_card_and_field_form(assets: Path) -> None:
    from alloy.sourcing.handoff import read_hints

    lib = assets / "imports" / "demo"
    (lib / "sourcing_queue.jsonl").write_text(
        '{"card": "mug_white", "category": "coffee mug", "field": "mass", "value": 0.31, "note": "ESTIMATE: x"}\n'
        '{"card": "mug_white", "category": "coffee mug", "field": "flag", "why": "mass outside plausible"}\n')
    hint = read_hints(lib)["mug_white"]
    assert hint["fields"] == ["mass"] and hint["extra"] == {"mass_kg": 0.31, "flag": "mass outside plausible"}
    cache = Cache(sources_root(assets))
    cache.put(_entry("https://a.example/mug", "Weight 13.4 oz"))
    assert Cache(sources_root(assets)).lookup("Coffee Mug") is None  # not an alias of the test entry
    assert Cache(sources_root(assets)).lookup("ceramic mug").key == "ceramic_mug"


def test_ingest_object_estimate_beats_category_prior(tmp_path: Path) -> None:
    import json

    from alloy.sourcing.ingest import ingest_file
    from alloy.sourcing.store import ObjectRecord, load_records, write_record

    prior = SourcedValue(field="mass", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE", reason="category shoe (Shoe): x")
    write_record(ObjectRecord(library="demo", object="gso_x", values={"mass": prior}), tmp_path / "src")
    est = {"card": "gso_x", "field": "mass", "value": 0.35, "lo": 0.3, "hi": 0.4, "tag": "ESTIMATE",
           "reason": "sibling model weight"}
    f = tmp_path / "f.jsonl"
    f.write_text(json.dumps(est) + "\n" + json.dumps({**est, "value": 0.36}) + "\n")
    report = ingest_file(f, "demo", tmp_path / "src").to_json()
    assert (report["applied"], report["weaker_or_equal"]) == (1, 1)
    assert load_records("demo", tmp_path / "src")["gso_x"].values["mass"].value == 0.35


def test_scan_volume_from_a_mesh_density_note(assets: Path) -> None:
    import json

    import pytest

    from alloy.sourcing.handoff import read_hints

    lib = assets / "imports" / "demo"
    note = "ESTIMATE: density x visual-mesh volume; Mug: demo_ceramic 2000 kg/m3, solid"
    rows = [{"card": "mug_white", "category": "mug", "field": "mass", "value": 0.2, "note": note},
            {"card": "bowl_red", "category": "mug", "field": "mass", "value": 2000.0, "note": note},
            {"card": "bowl_red", "category": "mug", "field": "flag", "why": "mass 2000 kg outside plausible"},
            {"card": "bowl_red", "category": "mug", "field": "flag", "why": "size 5.2 m outside plausible"}]
    (lib / "sourcing_queue.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    hints = read_hints(lib)
    assert hints["mug_white"]["extra"]["volume_m3"] == pytest.approx(1e-4)
    assert "volume_m3" not in hints["bowl_red"]["extra"]  # the importer flagged the size: a mis-scaled mesh
    assert hints["bowl_red"]["extra"]["flag"].endswith("; size 5.2 m outside plausible")


def test_name_map_to_a_material_gives_its_values_and_no_mass(assets: Path) -> None:
    ev = Evidence(url="https://a.example/glaze", quote="density 2.3 g/cm3, kinetic friction 0.3", tag="DOC")
    rho = SourcedValue(field="density", value=2300.0, lo=2200.0, hi=2400.0, tag="ESTIMATE", evidence=(ev,), reason="r")
    mu = SourcedValue(field="mu_dynamic", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE", evidence=(ev,), reason="r")
    cache = Cache(sources_root(assets))
    cache.put(CacheEntry(key="glaze", kind="material", title="Glazed ceramic",
                         values={"density": rho, "mu_dynamic": mu}))
    cache.put_map("demo_dataset", {"bowl_red": "material:glaze"})
    report = fill_from_cache(scan_library("demo", assets), Cache(sources_root(assets)), sources_root(assets))
    assert report.written == ["demo/bowl_red"]  # mug_white has no map entry and no category
    rec = load_records("demo", sources_root(assets))["bowl_red"]
    assert rec.category == "material_glaze" and set(rec.values) == {"density", "mu_dynamic"}
    assert rec.values["mu_dynamic"].reason.startswith("category material_glaze (single-material object of Glazed")
