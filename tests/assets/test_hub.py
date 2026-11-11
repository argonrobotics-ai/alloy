# SPDX-License-Identifier: Apache-2.0
"""alloy.assets.hub on CPU: stage a tiny library, write the top files, fetch it from a fake Hub into a temp cache."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from alloy.assets.hub import build, card, fetch, source
from alloy.assets.hub.layout import object_dir, sha256_file, tree_digest

CARD = """schema: alloy.object/1
name: {name}
shape:
  mesh:
    path: meshes/{name}/physics.obj
    collision: convex
material: {name}_eff
"""
MATERIAL = """schema: alloy.material/1
name: {name}_eff
density: {{value: 500, source: fit, note: "effective: 50 g over 100 cm3"}}
surface:
  mu_static: {{value: 0.5, source: model, note: "typical"}}
"""
OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def _library(tmp: Path, names: list[str]) -> tuple[Path, Path]:
    """An import root (cards, materials, manifest, thumbs.json) and a store copy holding the binaries."""
    root, store = tmp / "imports" / "toylib", tmp / "store" / "toylib"
    objects = {}
    for n in names:
        for rel, text in ((f"objects/toylib/{n}.yaml", CARD.format(name=n)),
                          (f"materials/toylib/{n}_eff.yaml", MATERIAL.format(name=n))):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        files = {}
        for f, data in (("physics.obj", OBJ), ("thumb.png", "png-" + n)):
            p = store / "meshes" / n / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)
            files[f] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        objects[n] = {"title": n, "licence": "CC-BY-4.0", "attribution": f"{n} by someone, CC BY 4.0",
                      "source": f"https://example.org/{n}", "category": "cups", "card": f"objects/toylib/{n}.yaml",
                      "material": f"materials/toylib/{n}_eff.yaml", "files": files,
                      "physics": {"mass_status": "sourced"}}
    if "brand_box" in objects:                        # imported with a plausibility flag: kept, with the reason
        objects["brand_box"]["status"] = "flagged"
        card = root / "objects/toylib/brand_box.yaml"
        card.write_text("# Flags: mass 9 kg outside plausible 0.1-1 kg.\n" + card.read_text())
    (root / "manifest.json").write_text(json.dumps({"schema": "x", "lib": "toylib", "objects": objects}, indent=1))
    thumbs = {n: {"thumb": f"meshes/{n}/thumb.png", "showcase_ok": n != "brand_box",
                  **({"showcase_note": "brand"} if n == "brand_box" else {})} for n in names}
    (root / "thumbs.json").write_text(json.dumps({"objects": thumbs}))
    return root, tmp / "store"


@pytest.fixture
def dataset(tmp_path):
    root, store = _library(tmp_path, ["cup_a", "brand_box"])
    out, rows = tmp_path / "ds", tmp_path / "rows" / "toylib.jsonl"
    res = build.stage("toylib", root, store, out, rows)
    assert res["objects"] == 2 and not res["failed"]
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "CC-BY-4.0.txt").write_text("Attribution 4.0 International")
    top = card.write_top(rows.parent, texts, out)
    assert top["objects"] == 2 and top["licenses"] == ["CC-BY-4.0"]
    return out


def test_stage_writes_object_folders_and_index(dataset):
    rows = [json.loads(x) for x in (dataset / "index.jsonl").read_text().splitlines()]
    cup = next(r for r in rows if r["name"] == "cup_a")
    assert sorted(f["file"] for f in cup["files"]) == ["card.yaml", "material.yaml", "physics.obj", "thumb.png"]
    assert cup["sha256"] == tree_digest(cup["files"]) and cup["category"] == "cups"
    assert cup["mass_basis"] == "sourced" and not cup["estimate"] and cup["sources"]["density"] == "fit"
    box = next(r for r in rows if r["name"] == "brand_box")
    assert box["showcase_ok"] is False and box["flags"].startswith("mass 9 kg") and "flags" not in cup
    for f in cup["files"]:
        assert sha256_file(dataset / object_dir("toylib", "cup_a") / f["file"]) == f["sha256"]
    assert "cup_a by someone" in (dataset / "ATTRIBUTION.md").read_text()
    readme = (dataset / "README.md").read_text()
    assert readme.startswith("---\npretty_name: Alloy object library") and "alloy fetch" in readme
    assert "The dataset is public" in readme and "request access" not in readme       # no login step
    assert "(CC BY 4.0: 2 objects)" in readme and "ATTRIBUTION.md" in readme
    assert "2 object cards" in readme and "| `toylib` | toylib | 2 | 0 | 1 |" in readme      # cards, estimates, flagged
    assert "flagged 1 of the cards" in readme and "0 of the 2 masses are estimates" in readme


def test_the_dataset_card_says_what_each_library_is_and_claims_no_more():
    """The audit's do-not-claim items for the card: no 'physics-ready' or 'sourced' headline, estimates and flagged
    cards counted, each source named (the kitchen pack is Lightwheel's, from NVIDIA's dataset, as packaged by
    RoboCasa; Gym-Anything cards are parts), licences as the source declares them, trademarks not covered."""
    from alloy.assets.hub.readme import readme

    rows = ([{"library": "robocasa", "size": 10, "files": [], "licence": "CC-BY-4.0", "estimate": True,
              "mass_basis": "estimate", "showcase_ok": i > 0, **({"flags": "mass 9 kg"} if i == 0 else {})}
             for i in range(3)]
            + [{"library": "gymanything", "size": 5, "files": [], "licence": "MIT", "estimate": False,
                "mass_basis": b, "showcase_ok": True} for b in ("datasheet", "material density (estimate)", "none")]
            + [{"library": "ycb_gso", "size": 5, "files": [], "licence": "CC-BY-4.0", "estimate": True,
                "mass_basis": "sourced", "showcase_ok": False}])
    libs = card.summary(rows)
    assert (libs["robocasa"]["flagged"], libs["robocasa"]["mass_estimate"]) == (1, 3)
    assert libs["gymanything"]["mass_estimate"] == 2 and libs["ycb_gso"]["mass_estimate"] == 0
    text = readme(libs, ["CC-BY-4.0", "MIT"])
    low = text.lower()
    for claim in ("physics-ready", "sourced material", "permissive", "allow commercial use", "allows commercial use",
                  "every object is under", "every object carries", "every object passed"):
        assert claim not in low, claim
    assert "7 object cards" in text and "5 of the 7 masses are estimates" in text and "flagged 1 of the cards" in text
    assert "- `robocasa` (3): kitchen objects and fixtures made by Lightwheel, from NVIDIA's" in text
    assert "no Gym-Anything task, robot or score is here" in text and "28 simulations" not in text
    assert "denoiser/ " in text
    assert "as packaged by RoboCasa" in text and "RoboCasa's kitchens and tasks are not here" in text
    assert "- `gymanything` (3): object parts" in text and "recorded as declared" in text
    assert "the licence its source declares" in text and "before commercial use" in text
    assert "covers the scan, not the\ntrademark" in text and "nvidia_kitchen_mjcf" in text
    assert "| `robocasa` | Lightwheel kitchen assets (NVIDIA" in text and "| 3 | 3 | 1 |" in text


def test_a_pack_credit_line_fills_credit_and_source_and_no_credit_is_refused(tmp_path):
    """Objects credited per pack (the kitchen assets): the pack's line of the library's ATTRIBUTION.md is the credit,
    its ``Source:`` the upstream URL (the licence id ``CC-BY-4.0.`` has dots of its own). An object whose licence
    asks for credit and has none stops the top files."""
    root = tmp_path / "imports" / "packlib"
    (root / "objects" / "cups").mkdir(parents=True)
    (root / "objects" / "cups" / "cup_p.yaml").write_text("schema: alloy.object/1\nname: cup_p\n")
    credit = "Maker kitchen assets from Some Dataset (CC BY 4.0), as packaged by Packer; converted for Alloy."
    (root / "ATTRIBUTION.md").write_text(f"# Attribution\n\n- **pack_a** (1 assets): CC-BY-4.0. {credit} "
                                         "Source: https://example.org/datasets/some\n")
    entry = {"card": "objects/cups/cup_p.yaml", "model": "mjcf/cups/cup_p/model.xml", "licence": "CC-BY-4.0",
             "pack": "pack_a", "status": "ok", "files": {}}
    (root / "manifest.json").write_text(json.dumps({"assets": {"cup_p": entry}}))
    meta = build.import_specs(root, "packlib")["cup_p"].meta
    assert meta["attribution"] == f"{credit} Source: https://example.org/datasets/some"
    assert meta["source"] == "https://example.org/datasets/some"

    row = {"name": "cup_p", "library": "packlib", "licence": "CC-BY-4.0", "attribution": "", "size": 1, "files": []}
    free = {**row, "name": "cup_free", "licence": "CC0-1.0"}
    assert card.uncredited([row, free]) == ["cup_p"]
    (tmp_path / "rows").mkdir()
    (tmp_path / "rows" / "packlib.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="1 objects have a licence that asks for credit and no credit line"):
        card.write_top(tmp_path / "rows", tmp_path / "texts", tmp_path / "ds")


def test_stage_reports_a_file_that_differs_from_the_manifest(tmp_path):
    root, store = _library(tmp_path, ["cup_a"])
    (store / "toylib" / "meshes" / "cup_a" / "physics.obj").write_text("changed")
    res = build.stage("toylib", root, store, tmp_path / "ds", tmp_path / "rows.jsonl")
    assert "cup_a" in res["failed"] and "store copy" in res["failed"]["cup_a"]


@pytest.fixture
def hub(dataset, tmp_path, monkeypatch):
    """A fake Hub serving ``dataset``; the cache is a temp folder. Returns the list of downloaded paths."""
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    calls: list[str] = []

    def fake_download(path, local_dir, repo=None):
        calls.append(path)
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(dataset / path, dst)
        return dst

    monkeypatch.setattr(source, "download", fake_download)
    monkeypatch.setattr(fetch, "download", fake_download)
    return calls


def test_fetch_places_objects_where_their_cards_expect_and_is_lazy(hub, tmp_path):
    res = fetch.fetch(names=["cup_a"])
    base = tmp_path / "cache" / "imports" / "toylib"
    assert res["objects"] == 1 and res["roots"] == [str(base)]
    assert (base / "objects/toylib/cup_a.yaml").is_file() and (base / "meshes/cup_a/physics.obj").is_file()
    assert (base / "materials/toylib/cup_a_eff.yaml").is_file() and (base / "manifest.json").is_file()
    assert not (base / "meshes/brand_box").exists()
    again = fetch.fetch(names=["cup_a"])
    assert again["downloaded"] == 0 and again["kept"] == 4
    (base / "meshes/cup_a/physics.obj").write_text("damaged")
    assert fetch.fetch(names=["cup_a"])["downloaded"] == 1


def test_two_fetches_at_once_do_not_remove_each_others_downloads(hub, tmp_path, monkeypatch):
    """A second fetch that starts and ends while a file of the first is downloaded and not yet in place (another
    process filling the same cache) leaves that file alone: each fetch has a download folder of its own."""
    plain, state = fetch.download, {"nested": None}

    def slow_download(path, local_dir, repo=None):
        got = plain(path, local_dir, repo)
        if state["nested"] is None and path.endswith("cup_a/physics.obj"):
            state["nested"] = {}                                         # set first: the inner fetch must not recurse
            state["nested"] = fetch.fetch(names=["brand_box"], workers=1)
        return got

    monkeypatch.setattr(fetch, "download", slow_download)
    res = fetch.fetch(names=["cup_a"], workers=1)
    base, hub_dir = tmp_path / "cache" / "imports" / "toylib", fetch.hub_dir()
    assert res["objects"] == 1 and state["nested"]["objects"] == 1
    assert (base / "meshes/cup_a/physics.obj").is_file() and (base / "meshes/brand_box/physics.obj").is_file()
    assert not list(hub_dir.glob("staging*"))                            # both removed their own folder
    left = hub_dir / "staging-killed"                                    # a killed fetch's folder: removed once old
    (left / "objects").mkdir(parents=True)
    (base / "meshes/cup_a/physics.obj").unlink()
    fetch.fetch(names=["cup_a"])
    assert left.is_dir()
    monkeypatch.setattr(fetch, "STALE_S", -1.0)
    (base / "meshes/cup_a/physics.obj").unlink()
    fetch.fetch(names=["cup_a"])
    assert not left.exists()


def test_fetch_by_category_and_listing(hub):
    assert fetch.fetch(library="toylib", category="cups")["objects"] == 2
    listing = fetch.list_objects(library="toylib")
    assert listing["categories"] == {"toylib/cups": 2} and listing["objects"] == 2


def test_unknown_names_and_filters_say_what_to_do(hub):
    with pytest.raises(source.HubError) as e:
        fetch.fetch(names=["cup_b"])
    assert "cup_a" in e.value.fix and "alloy fetch --list" in e.value.fix
    with pytest.raises(source.HubError) as e:
        fetch.fetch(library="nope")
    assert "toylib" in e.value.fix
    with pytest.raises(source.HubError):
        fetch.fetch()


def test_a_missing_or_private_repo_error_says_check_the_id_then_log_in():
    errors = pytest.importorskip("huggingface_hub.errors")

    class Resp:
        status_code = 401
        headers: dict = {}
        request = None

    exc = errors.RepositoryNotFoundError("401 Client Error", response=Resp())
    err = source.explain(exc, "owner/alloy-library", "index.jsonl")
    assert "private" in err.message and "hf auth login" in err.fix and "request access" in err.fix
    assert "is public and needs no login" in err.fix and "hf auth logout" in err.fix     # the default library
    assert err.to_dict()["error"] == "HubError"


def test_repo_id_from_env(monkeypatch):
    monkeypatch.delenv("ALLOY_ASSETS_REPO", raising=False)
    assert source.repo_id() == source.DEFAULT_REPO
    monkeypatch.setenv("ALLOY_ASSETS_REPO", "someone/copy")
    assert source.repo_id() == "someone/copy"


def test_fetched_card_resolves_its_mesh_in_a_card_library(hub, tmp_path):
    from alloy.cards.library import CardLibrary
    from alloy.compose.fetch import cache_root

    fetch.fetch(names=["cup_a"])
    lib = CardLibrary(roots=[cache_root("toylib")])
    assert lib.asset_path("meshes/cup_a/physics.obj").is_file()


def _authored_root(tmp: Path) -> Path:
    """A card root with one hand-authored object: card -> MJCF (meshdir) -> mesh, own material, looks, a pair."""
    root = tmp / "cards"
    files = {
        "objects/kitchen/pot.yaml": "# pot: own model (INTERNAL-7 notes)\nschema: alloy.object/1\nname: pot\nshape:\n"
                                    "  mjcf:\n    path: mjcf/kitchen/pot.xml\n    body: pot\nmaterial: pot_steel\n",
        "mjcf/kitchen/pot.xml": '<mujoco model="pot"><compiler meshdir="../../meshes/kitchen"/><asset>'
                                '<mesh name="m" file="pot.obj"/></asset><worldbody><body name="pot">'
                                '<geom type="mesh" mesh="m"/></body></worldbody></mujoco>',
        "meshes/kitchen/pot.obj": OBJ,
        "materials/kitchen/pot_steel.yaml": MATERIAL.format(name="pot").replace("pot_eff", "pot_steel"),
        "looks/objects/kitchen/pot.json": '{"object": "pot", "note": "steel (see INTERNAL-7)"}',
        "looks/materials/pot_steel.json": '{"material": "pot_steel"}',
        "pairs/kitchen/pot_steel__stainless_304.yaml": "schema: alloy.pair/1\na: pot_steel\nb: stainless_304\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_authored_objects_follow_their_references_and_scrub_on_export(tmp_path):
    from alloy.assets.hub.authored import authored_specs

    root = _authored_root(tmp_path)
    specs, shared = authored_specs(root, ["objects/kitchen/pot.yaml"], {"pot": {"licence": "CC0-1.0", "category": "pots"}})
    assert sorted(specs["pot"].files) == ["looks/materials/pot_steel.json", "looks/objects/kitchen/pot.json",
                                          "materials/kitchen/pot_steel.yaml", "meshes/kitchen/pot.obj",
                                          "mjcf/kitchen/pot.xml", "objects/kitchen/pot.yaml"]
    assert shared == ["pairs/kitchen/pot_steel__stainless_304.yaml"]
    rewrites = {"objects/kitchen/pot.yaml": [[" (INTERNAL-7 notes)", ""]], "*": [[" (see INTERNAL-7)", ""]]}
    out, rows = tmp_path / "ds", tmp_path / "rows" / "alloy.jsonl"
    res = build.stage("alloy", root, tmp_path / "store", out, rows, None, (specs, shared), rewrites)
    assert res["objects"] == 1 and not res["failed"] and res["library_files"] == 1
    row = json.loads(rows.read_text())
    folder = out / object_dir("alloy", "pot")
    assert sorted(f["file"] for f in row["files"]) == ["card.yaml", "look.json", "material.look.json", "material.yaml",
                                                      "pot.obj", "pot.xml"]
    assert "INTERNAL" not in (folder / "card.yaml").read_text() + (folder / "look.json").read_text()
    assert all(sha256_file(folder / f["file"]) == f["sha256"] for f in row["files"])     # hashes of the export
    assert row["licence"] == "CC0-1.0" and row["category"] == "pots" and row["mass_basis"] == "authored"


def test_a_rewrite_whose_text_is_gone_fails_the_object(tmp_path):
    from alloy.assets.hub.authored import authored_specs

    root = _authored_root(tmp_path)
    specs, shared = authored_specs(root, ["objects/kitchen/pot.yaml"], {})
    res = build.stage("alloy", root, tmp_path / "store", tmp_path / "ds", tmp_path / "rows.jsonl", None,
                      (specs, shared), {"objects/kitchen/pot.yaml": [["text that is not there", ""]]})
    assert "pot" in res["failed"] and "rewrite text not found" in res["failed"]["pot"]
