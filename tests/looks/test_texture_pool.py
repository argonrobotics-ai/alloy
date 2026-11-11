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
