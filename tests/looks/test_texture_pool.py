# SPDX-License-Identifier: Apache-2.0
"""Texture pool (alloy.looks.pool): CC0 texture looks in families, for scene variants. CPU only, no network."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from alloy.cards import CardLibrary
from alloy.looks import LookFileError, pool, validate_look
from alloy.looks.schema import FAMILIES
from alloy.render.blender.presets import PRESETS

GOOD = {"schema": "alloy.look/1", "look": "tex_ph_oak", "family": "wood", "preset": "wood_oak",
        "texture": "polyhaven:oak_veneer_01", "res": "1k", "scale_m": 1.0, "source": "https://polyhaven.com/a/oak_veneer_01",
        "author": "someone", "licence": "CC0-1.0", "albedo": 0.24, "note": "oak"}


@pytest.fixture
def lib() -> CardLibrary:
    return CardLibrary()


@pytest.fixture
def small_root(tmp_path: Path) -> CardLibrary:
    """A card root whose pool holds two wood textures and one metal texture."""
    d = tmp_path / "looks" / "textures"
    d.mkdir(parents=True)
    for name, fam, tex in (("w1", "wood", "polyhaven:a"), ("w2", "wood", "polyhaven:b"), ("m1", "metal", "ambientcg:c")):
        (d / f"{name}.json").write_text(json.dumps({**GOOD, "look": name, "family": fam, "texture": tex}))
    return CardLibrary(roots=[tmp_path], assets=None)


def test_library_pool_has_every_family_with_6_to_10_cc0_textures(lib):
    sizes = pool.families(lib)
    assert set(sizes) == set(FAMILIES) and all(6 <= n <= 10 for n in sizes.values()), sizes
    es = pool.entries(lib)
    assert 55 <= len(es) <= 70
    assert len({e.name for e in es}) == len(es) and len({e.texture for e in es}) == len(es)
    for e in es:
        assert e.licence == "CC0-1.0" and e.source.startswith("https://") and e.author.strip(), e.name
        assert e.spec["res"] in ("1k", "2k") and e.spec["scale_m"] > 0 and 0.0 < e.albedo < 1.0, e.name
        assert "albedo" not in e.spec


def test_every_texture_a_preset_names_is_in_the_pool_under_the_presets_family(lib):
    """A preset's texture is a pool set, so ``--fetch-textures --family <f>`` brings it and variants can swap it
    (american_walnut_veneer of wood_walnut_oiled was outside the pool until 0.1.6)."""
    from alloy.render.blender.presets import PRESETS
    by_tex = {e.texture: e for e in pool.entries(lib)}
    named = {k: v["texture"] for k, v in PRESETS.items() if v.get("texture")}
    assert "polyhaven:american_walnut_veneer" in named.values()
    for preset, tex in named.items():
        assert tex in by_tex, (preset, tex)
        assert by_tex[tex].family == pool.PRESET_FAMILY.get(preset, by_tex[tex].family), (preset, tex)
    assert not set(named.values()) & set(pool.look_textures(lib))


def test_library_pool_looks_render_in_the_path_tracer(lib):
    """Every pool look builds a camera material the path tracer accepts (no subsurface, decal, tint, ...)."""
    from alloy.camera.looks import material_tree
    from alloy.camera.looks.spec import check_supported, resolve_spec
    from alloy.camera.pbr.dispatch import _check_material
    for e in pool.entries(lib):
        check_supported(e.name, resolve_spec(e.name, e.spec))
        maps = {"color": "/x/c.jpg", "roughness": "/x/r.jpg", "height": "/x/h.png"}
        _check_material(e.name, material_tree(e.name, e.spec, textures={e.texture: maps}))


def test_family_of_reads_the_pool_texture_then_the_preset(small_root):
    assert pool.family_of({"preset": "plastic_matte", "texture": "ambientcg:c"}, small_root) == "metal"
    assert pool.family_of({"preset": "wood_walnut_oiled"}, small_root) == "wood"
    assert pool.family_of({"preset": "glass_clear"}, small_root) is None
    assert set(pool.PRESET_FAMILY) <= set(PRESETS) and set(pool.PRESET_FAMILY.values()) <= set(FAMILIES)


def test_swap_stays_in_the_family_and_never_returns_the_same_texture(small_root):
    for seed in range(20):
        name, spec = pool.swap({"preset": "wood_oak", "texture": "polyhaven:a"}, random.Random(seed), small_root)
        assert name == "w2" and spec["texture"] == "polyhaven:b" and "family" not in spec and "source" not in spec
    picks = {pool.swap({"preset": "wood_oak"}, random.Random(s), small_root)[0] for s in range(20)}
    assert picks == {"w1", "w2"}
    assert pool.swap({"preset": "wood_oak"}, random.Random(3), small_root) == \
        pool.swap({"preset": "wood_oak"}, random.Random(3), small_root)
    assert pool.swap({"preset": "steel_polished", "texture": "ambientcg:c"}, random.Random(0), small_root) is None
    assert pool.swap({"preset": "glass_clear"}, random.Random(0), small_root) is None
    with pytest.raises(ValueError, match="unknown texture family"):
        pool.entries(small_root, "lava")


@pytest.mark.parametrize("change, fragment", [
    ({"family": "lava"}, "'family' must be one of"),
    ({"licence": "CC-BY-4.0"}, "'licence' must be 'CC0-1.0'"),
    ({"source": "polyhaven.com/a/x"}, "'source' must be the set's https page"),
    ({"author": " "}, "'author' must name who made the set"),
    ({"texture": "file:wood.jpg"}, "'texture' must be a CC0 set"),
    ({"res": "4k"}, "'res' must be one of"),
    ({"albedo": 1.5}, "'albedo' must be the colour map's mean linear luminance"),
    ({"albedo": None}, "'albedo' must be"),
    ({"roughnes": 0.3}, "unknown key 'roughnes'"),
])
def test_pool_file_validation_names_the_problem(change, fragment):
    assert validate_look(GOOD) == []
    assert any(fragment in i for i in validate_look({**GOOD, **change})), validate_look({**GOOD, **change})


def test_a_bad_pool_file_raises_with_its_path(small_root):
    bad = small_root.search_roots[0] / "looks" / "textures" / "w9.json"
    bad.write_text(json.dumps({**GOOD, "look": "w9", "licence": "unknown"}))
    pool._POOL.clear()
    with pytest.raises(LookFileError, match="w9.json"):
        pool.entries(small_root)
    pool._POOL.clear()


def _fake_hub(monkeypatch) -> dict[str, bytes]:
    """Serve a two-set texture library (wood set polyhaven:a, metal set ambientcg:c) instead of the Hub; returns the
    blobs, which a test may change after the listing was made."""
    import hashlib

    from alloy.assets.hub import source
    blobs = {"polyhaven/a/a_diff_1k.jpg": b"wood-a", "ambientcg/c_1K/c_Color.jpg": b"metal-c"}
    listing = {"files": [{"file": p, "path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                          "texture": "polyhaven:a" if p.startswith("poly") else "ambientcg:c"} for p, b in blobs.items()]}

    def fake_download(path, local_dir, repo=None):
        rel = path.split("libraries/textures/", 1)[1]
        out = Path(local_dir) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(json.dumps(listing).encode() if rel == "files.json" else blobs[rel])
        return out
    monkeypatch.setattr(source, "download", fake_download)
    return blobs


def test_fetch_from_the_hub_takes_only_the_selected_sets_and_checks_sha256(small_root, tmp_path, monkeypatch):
    blobs = _fake_hub(monkeypatch)
    root = tmp_path / "cache"
    report = pool.fetch("wood", lib=small_root, root=root)
    assert report["downloaded"] == 1 and report["missing"] == ["polyhaven:b"]
    assert (root / "polyhaven/a/a_diff_1k.jpg").read_bytes() == b"wood-a"
    assert not (root / "ambientcg").exists() and not (root / ".hub_staging").exists()
    assert pool.fetch("wood", lib=small_root, root=root)["kept"] == 1
    assert pool.fetch("metal", lib=small_root, root=root)["downloaded"] == 1    # ambientCG: the cache marker is written
    assert (root / "ambientcg/c_1K/.done").read_text() == "ok" and (root / "ambientcg/c_1K/c_Color.jpg").is_file()
    blobs["polyhaven/a/a_diff_1k.jpg"] = b"tampered"
    (root / "polyhaven/a/a_diff_1k.jpg").unlink()
    with pytest.raises(Exception, match="different sha256"):
        pool.fetch("wood", lib=small_root, root=root)


def test_alloy_cache_moves_the_texture_cache_for_fetch_and_the_camera(small_root, tmp_path, monkeypatch):
    """With ALLOY_CACHE set (as `alloy fetch` honours it), fetched textures land in $ALLOY_CACHE/render and the
    camera reads them there; ALLOY_RENDER_ASSETS still overrides; neither set: ~/.cache/alloy-render."""
    from alloy.camera.looks.textures import local_texture_files
    from alloy.render.blender.assets import default_cache
    _fake_hub(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ALLOY_RENDER_ASSETS", raising=False)
    monkeypatch.delenv("ALLOY_CACHE", raising=False)
    assert default_cache() == tmp_path / ".cache" / "alloy-render"
    monkeypatch.setenv("ALLOY_CACHE", "~/ac")
    assert default_cache() == tmp_path / "ac" / "render"
    report = pool.fetch("metal", lib=small_root)
    assert report["root"] == str(tmp_path / "ac" / "render") and report["downloaded"] == 1
    assert local_texture_files("ambientcg:c", "1k")["color"] == tmp_path / "ac/render/ambientcg/c_1K/c_Color.jpg"
    assert not (tmp_path / ".cache").exists()
    monkeypatch.setenv("ALLOY_RENDER_ASSETS", str(tmp_path / "ra"))
    assert default_cache() == tmp_path / "ra"


def test_a_set_cached_at_another_resolution_serves_a_default_look_with_one_notice(tmp_path, monkeypatch, caplog):
    """The public texture library holds 1k sets and looks ask for 2k by default: the camera's cache serves the nearest
    resolution on disk, the larger on a tie (one notice per set, naming the maps it used), so a default textured look
    renders; a set not there at all still raises."""
    from alloy.camera.errors import CameraError
    from alloy.camera.looks import material_tree, textures
    monkeypatch.setenv("ALLOY_RENDER_ASSETS", str(tmp_path))
    monkeypatch.setattr(textures, "_NOTICED", set())
    ph = tmp_path / "polyhaven" / "a"
    ph.mkdir(parents=True)
    listing = {key: {r: {"jpg": {"url": f"https://x/{r}/a_{tag}_{r}.jpg"}} for r in ("1k", "2k", "4k")}
               for key, tag in (("Diffuse", "diff"), ("Rough", "rough"))}
    (ph / "files.json").write_text(json.dumps(listing))      # the API listing names every resolution
    for tag in ("diff", "rough"):
        (ph / f"a_{tag}_1k.jpg").write_bytes(b"jpg")        # only 1k is on disk, as a library fetch leaves it
    for res in ("1K", "4K"):
        (tmp_path / "ambientcg" / f"c_{res}").mkdir(parents=True)
        (tmp_path / "ambientcg" / f"c_{res}" / "c_Color.jpg").write_bytes(b"jpg")
    (tmp_path / "ambientcg" / "c_1K" / ".done").write_text("ok")
    with caplog.at_level("WARNING", logger=textures.__name__):
        assert textures.local_texture_files("polyhaven:a", "2k")["color"] == ph / "a_diff_1k.jpg"
        assert textures.local_texture_files("polyhaven:a", "2k")["roughness"] == ph / "a_rough_1k.jpg"
        tree = material_tree("board", {"preset": "wood_oak", "texture": "polyhaven:a"})   # res: the 2k default
        assert "a_diff_1k.jpg" in tree["textures"]
        notices = [r.getMessage() for r in caplog.records]
        assert notices == ["texture polyhaven:a: no 2k maps in the local asset cache; using the 1k maps there"]
        assert textures.local_texture_files("ambientcg:c", "1k")["color"].parent.name == "c_1K"   # exact: no notice
        assert len(caplog.records) == 1
        (tmp_path / "ambientcg" / "c_4K" / ".done").write_text("ok")
        assert textures.local_texture_files("ambientcg:c", "2k")["color"].parent.name == "c_4K"   # 4k ties 1k: larger
        assert "using the 4k maps" in caplog.records[-1].getMessage()
        for res in ("1K", "8K"):                             # nearest, not largest: 1k serves a 2k ask before 8k
            (tmp_path / "ambientcg" / f"e_{res}").mkdir(parents=True)
            (tmp_path / "ambientcg" / f"e_{res}" / "c_Color.jpg").write_bytes(b"jpg")
            (tmp_path / "ambientcg" / f"e_{res}" / ".done").write_text("ok")
        assert textures.local_texture_files("ambientcg:e", "2k")["color"].parent.name == "e_1K"
        d = tmp_path / "polyhaven" / "d"                     # listed up to 4k, only 4k on disk: the notice says 4k
        d.mkdir()
        (d / "files.json").write_text(json.dumps({"Diffuse": {r: {"jpg": {"url": f"https://x/d_diff_{r}.jpg"}}
                                                              for r in ("1k", "2k", "4k")}}))
        (d / "d_diff_4k.jpg").write_bytes(b"jpg")
        assert textures.local_texture_files("polyhaven:d", "2k")["color"] == d / "d_diff_4k.jpg"
        assert caplog.records[-1].getMessage() == ("texture polyhaven:d: no 2k maps in the local asset cache; using "
                                                   "the 4k maps there")
        f = tmp_path / "polyhaven" / "f"                     # listed up to 2k: a 4k ask (8k tried first) says 2k
        f.mkdir()
        (f / "files.json").write_text(json.dumps({"Diffuse": {r: {"jpg": {"url": f"https://x/f_diff_{r}.jpg"}}
                                                              for r in ("1k", "2k")}}))
        (f / "f_diff_2k.jpg").write_bytes(b"jpg")
        assert textures.local_texture_files("polyhaven:f", "4k")["color"] == f / "f_diff_2k.jpg"
        assert caplog.records[-1].getMessage().endswith("no 4k maps in the local asset cache; using the 2k maps there")
    with pytest.raises(CameraError, match="not in the local asset cache"):
        textures.local_texture_files("polyhaven:missing", "2k")


def test_a_full_fetch_also_takes_the_textures_that_presets_and_library_looks_name(small_root, tmp_path, monkeypatch):
    mats = small_root.search_roots[0] / "looks" / "materials"
    mats.mkdir()
    (mats / "board.json").write_text(json.dumps({"schema": "alloy.look/1", "material": "board", "preset": "wood_oak",
                                                 "texture": "polyhaven:extra"}))
    extra = pool.look_textures(small_root)
    assert extra["polyhaven:extra"] == "2k" and extra["polyhaven:oak_veneer_01"] == "2k"   # a library look's, a preset's
    assert not {"polyhaven:a", "polyhaven:b", "ambientcg:c"} & set(extra)                  # pool sets are not repeated
    _fake_hub(monkeypatch)
    report = pool.fetch(lib=small_root, root=tmp_path / "cache")
    assert report["downloaded"] == 2 and {"polyhaven:b", "polyhaven:extra", "polyhaven:oak_veneer_01"} <= set(
        report["missing"])
    assert pool.fetch("wood", lib=small_root, root=tmp_path / "c2")["missing"] == ["polyhaven:b"]   # a family: pool only


def test_a_1k_only_cache_renders_a_full_quality_look_with_no_res(tmp_path, monkeypatch, caplog):
    """sim.render(quality="full", look=...) with a look naming polyhaven:plywood and no res (so 2k) on a cache that has
    the 1k set, as the public fetch leaves it: the frame renders from the 1k maps with one notice naming 1k. With no
    size cached the error's fix names the public command (with the set's family for a pool set), never an internal
    module, and for a set no fetch brings, what the fetch takes."""
    import numpy as np

    import alloy
    from alloy.camera.errors import CameraError
    from alloy.camera.looks import textures
    from tests.camera.helpers import fake_texture
    monkeypatch.setenv("ALLOY_RENDER_ASSETS", str(tmp_path))
    monkeypatch.setattr(textures, "_NOTICED", set())
    fake_texture(tmp_path, "polyhaven:plywood", "1k", rgb=(200, 150, 90))
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("board", size=(0.3, 0.3, 0.02), pos=(0, 0, 0.01), static=True)
    s.add_box("cube", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.04), mass=0.1)
    sim = alloy.build(s, device="cpu")
    look = {"materials": {"ply": {"preset": "wood_plywood", "texture": "polyhaven:plywood"}},
            "assign": [{"match": "board", "material": "ply"}]}
    with caplog.at_level("WARNING", logger=textures.__name__):
        img = sim.render(width=32, height=24, quality="full", look=look, spp=1)
    assert img.shape == (24, 32, 3) and np.isfinite(img.astype(float)).all()
    assert [r.getMessage() for r in caplog.records if r.name == textures.__name__] == [
        "texture polyhaven:plywood: no 2k maps in the local asset cache; using the 1k maps there"]
    with pytest.raises(CameraError) as err:
        textures.local_texture_files("polyhaven:oak_veneer_01", "2k")
    assert "python -m alloy.looks --fetch-textures --family wood" in err.value.fix and "AssetCache" not in err.value.fix
    with pytest.raises(CameraError) as err:                 # a library look's set outside the pool: the full fetch has it
        textures.local_texture_files("polyhaven:coated_pine", "2k")
    assert err.value.fix.startswith("fetch it with python -m alloy.looks --fetch-textures, or")
    with pytest.raises(CameraError) as err:                 # no fetch brings it: say what the fetch takes
        textures.local_texture_files("polyhaven:not_in_the_pool", "2k")
    assert err.value.fix == ("python -m alloy.looks --fetch-textures takes only the texture pool's sets and those "
                             "presets and library looks name: use one of those (python -m alloy.looks --pool lists "
                             "the pool), or render with quality='fast'")
