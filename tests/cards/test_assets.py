# SPDX-License-Identifier: Apache-2.0
"""SCR-DC-1: CardLibrary also reads the alloy-assets checkout ($ALLOY_ASSETS, else the sibling ../alloy-assets).

A temp root in the alloy-assets layout (objects/<area>/, meshes/<area>/, scenes/<area>/, materials/, pairs/) covers
recursive loading, mesh paths relative to the assets root, clashes with library/ (raise, both paths named), put and
path-independent identity (card_hash, model_digest). The last test checks the real checkout and skips without it.
"""

from __future__ import annotations

import dataclasses as dc
import json
import shutil
from pathlib import Path

import pytest

from alloy.cards import roots as R
from alloy.cards.library import CardLibrary, default_root
from alloy.core.errors import CardError
from alloy.core.quantity import Q

LIB = default_root()
TRAY_MESH = "meshes/eth/dc_eth_tray.obj"


def _copy_card(src: Path, dst: Path, *subs: tuple[str, str]) -> Path:
    text = src.read_text(encoding="utf-8")
    for old, new in subs:
        assert old in text, old
        text = text.replace(old, new)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return dst


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    """An assets root: a material, a pair, a mesh object (eth area) and a scene that also places a library object."""
    root = tmp_path / "alloy-assets"
    _copy_card(LIB / "materials" / "aluminium_6061_t6.yaml", root / "materials" / "eth" / "dc_eth_alu.yaml",
               ("name: aluminium_6061_t6", "name: dc_eth_alu"))
    _copy_card(LIB / "pairs" / "abs__abs.yaml", root / "pairs" / "eth" / "abs__dc_eth_alu.yaml",
               ("b: abs", "b: dc_eth_alu"))
    (root / TRAY_MESH).parent.mkdir(parents=True)
    shutil.copy(LIB / "meshes" / "tray_aluminium.obj", root / TRAY_MESH)
    _copy_card(LIB / "objects" / "tray_aluminium.yaml", root / "objects" / "eth" / "dc_eth_tray.yaml",
               ("name: tray_aluminium", "name: dc_eth_tray"), ("meshes/tray_aluminium.obj", TRAY_MESH),
               ("material: aluminium_6061_t6", "material: dc_eth_alu"))
    scene = ("schema: alloy.scene/1\nrobots: []\nobjects:\n"
             "  - {object: table_top_plywood, pose: [0, 0, -0.009, 0, 0, 0, 1]}\n"
             "  - {object: dc_eth_tray, pose: [0, 0, 0.05, 0, 0, 0, 1]}\nworlds: 2\n")
    (root / "scenes" / "eth").mkdir(parents=True)
    (root / "scenes" / "eth" / "task_dc_eth_demo.yaml").write_text(scene, encoding="utf-8")
    (root / "objects" / ".scratch").mkdir()
    (root / "objects" / ".scratch" / "ignored.yaml").write_text("not: a card\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------------------------------------ root


def test_assets_root_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(R.ASSETS_ENV, str(tmp_path))
    assert R.assets_root() == tmp_path.resolve()
    monkeypatch.setenv(R.ASSETS_ENV, "")
    assert R.assets_root() is None
    monkeypatch.setenv(R.ASSETS_ENV, str(tmp_path / "missing"))
    with pytest.raises(CardError) as exc:
        R.assets_root()
    assert exc.value.field == R.ASSETS_ENV


def test_assets_root_defaults_to_sibling_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(R.ASSETS_ENV, raising=False)
    monkeypatch.setattr(R, "repo_root", lambda: tmp_path / "alloy-dev")      # a worktree next to the checkout
    assert R.assets_root() is None
    (tmp_path / "alloy-assets").mkdir()
    assert R.assets_root() == (tmp_path / "alloy-assets").resolve()


def test_default_library_takes_env_root_explicit_roots_do_not(monkeypatch: pytest.MonkeyPatch, assets: Path) -> None:
    monkeypatch.setenv(R.ASSETS_ENV, str(assets))
    assert CardLibrary().assets == assets.resolve() and CardLibrary().has_object("dc_eth_tray")
    assert CardLibrary([LIB]).assets is None and not CardLibrary([LIB]).has_object("dc_eth_tray")
    monkeypatch.setenv(R.ASSETS_ENV, "")
    assert CardLibrary().assets is None


# ------------------------------------------------------------------------------------------------ loading


def test_loads_cards_recursively_with_asset_relative_meshes(assets: Path) -> None:
    lib = CardLibrary(assets=assets)
    assert "dc_eth_tray" in lib.names("objects") and "task_dc_eth_demo" in lib.names("scenes")
    assert lib.has_object("peg_d12") and lib.material("abs").name == "abs"           # library/ still there
    assert lib.path_of("objects", "dc_eth_tray") == assets.resolve() / "objects" / "eth" / "dc_eth_tray.yaml"
    assert lib.in_assets("objects", "dc_eth_tray") and not lib.in_assets("objects", "peg_d12")
    assert lib.asset_path(TRAY_MESH) == assets.resolve() / TRAY_MESH
    assert lib.object("dc_eth_tray").material == "dc_eth_alu"                         # validated: mesh, sha256
    assert lib.pair("dc_eth_alu", "abs") is not None
    assert [p.object for p in lib.scene("task_dc_eth_demo").objects] == ["table_top_plywood", "dc_eth_tray"]
    r = lib.resolve(lib.object("dc_eth_tray"))
    assert r.extents == lib.resolve(lib.object("tray_aluminium")).extents
    assert "ignored" not in lib.names("objects")


def test_missing_asset_mesh_names_the_field(assets: Path) -> None:
    (assets / TRAY_MESH).unlink()
    with pytest.raises(CardError) as exc:
        CardLibrary(assets=assets).object("dc_eth_tray")
    assert exc.value.field == "shape.path" and str(assets.resolve()) in str(exc.value)


# ------------------------------------------------------------------------------------------------ clashes


def test_name_in_library_and_assets_raises_with_both_paths(assets: Path) -> None:
    dup = _copy_card(LIB / "objects" / "peg_d12.yaml", assets / "objects" / "eth" / "peg_d12.yaml")
    lib = CardLibrary(assets=assets)
    assert "peg_d12" in lib.names("objects") and lib.has_object("peg_d12")
    with pytest.raises(CardError) as exc:
        lib.object("peg_d12")
    assert exc.value.field == "name"
    assert str(LIB / "objects" / "peg_d12.yaml") in exc.value.message and str(dup.resolve()) in exc.value.message
    with pytest.raises(CardError) as exc:
        lib.scene("demo_peg_d12")                                          # a scene placing it is invalid too
    assert exc.value.field.startswith("objects.")


def test_name_twice_in_assets_raises(assets: Path) -> None:
    (assets / "objects" / "conn").mkdir()
    other = shutil.copy(assets / "objects" / "eth" / "dc_eth_tray.yaml", assets / "objects" / "conn")
    with pytest.raises(CardError) as exc:
        CardLibrary(assets=assets).object("dc_eth_tray")
    assert "objects/eth/dc_eth_tray.yaml" in exc.value.message and str(Path(other).resolve()) in exc.value.message


def test_material_clash_is_an_object_error(assets: Path) -> None:
    _copy_card(LIB / "materials" / "aluminium_6061_t6.yaml", assets / "materials" / "eth" / "aluminium_6061_t6.yaml")
    lib = CardLibrary(assets=assets)
    with pytest.raises(CardError) as exc:
        lib.material("aluminium_6061_t6")
    assert exc.value.field == "name"
    with pytest.raises(CardError) as exc:
        lib.object("tray_aluminium")
    assert exc.value.field == "material"


def test_asset_path_in_both_roots_raises(assets: Path) -> None:
    shutil.copytree(LIB / "meshes", assets / "meshes", dirs_exist_ok=True)     # meshes/tray_aluminium.obj twice
    lib = CardLibrary(assets=assets)
    with pytest.raises(CardError):
        lib.asset_path("meshes/tray_aluminium.obj")
    with pytest.raises(CardError) as exc:
        lib.object("tray_aluminium")
    assert exc.value.field == "shape.path"
    assert lib.asset_path(TRAY_MESH) is not None                         # only under the assets root: fine


# ------------------------------------------------------------------------------------------------ put


def test_put_replaces_an_asset_card_in_place(tmp_library: Path, assets: Path) -> None:
    lib = CardLibrary([tmp_library], assets=assets)
    m = lib.material("dc_eth_alu")
    fitted = dc.replace(m, surface=dc.replace(m.surface, mu_static=Q(0.6, "fit", (0.55, 0.65), "rig run 3")))
    path = lib.put(fitted, reason="fit")
    assert path == assets.resolve() / "materials" / "eth" / "dc_eth_alu.yaml"
    assert lib.material("dc_eth_alu") == fitted and not (tmp_library / "materials" / "dc_eth_alu.yaml").exists()
    row = json.loads((path.parent / "history.jsonl").read_text().splitlines()[-1])
    assert row["name"] == "dc_eth_alu" and row["path"] == "materials/eth/dc_eth_alu.yaml"
    new = dc.replace(lib.object("dc_eth_tray"), name="dc_eth_tray_b")
    assert lib.put(new, reason="variant") == tmp_library / "objects" / "dc_eth_tray_b.yaml"   # new: first root


# ------------------------------------------------------------------------------------------------ identity


def test_card_hash_does_not_depend_on_where_the_root_is(tmp_path: Path, assets: Path) -> None:
    moved = shutil.copytree(assets, tmp_path / "elsewhere" / "assets-copy")
    lib_a, lib_b = CardLibrary(assets=assets), CardLibrary(assets=moved)
    h = lib_a.resolve(lib_a.object("dc_eth_tray")).card_hash
    assert lib_b.resolve(lib_b.object("dc_eth_tray")).card_hash == h
    as_library = tmp_path / "as-library"                                   # same card under a library root
    shutil.copytree(assets, as_library)
    for kind in ("materials", "objects"):
        for p in list((as_library / kind).rglob("*.yaml")):
            p.rename(as_library / kind / p.name)
    lib_c = CardLibrary([as_library, LIB])
    assert lib_c.resolve(lib_c.object("dc_eth_tray")).card_hash == h


@pytest.mark.compile_heavy
def test_model_digest_does_not_depend_on_where_the_root_is(tmp_path: Path, assets: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    assembler = pytest.importorskip("alloy.scene.assembler")
    from alloy.core.registry import isolated_registry
    from alloy.route.router import Router

    moved = shutil.copytree(assets, tmp_path / "elsewhere" / "assets-copy")
    monkeypatch.chdir(tmp_path)
    digests = []
    for root in (assets, moved):
        lib = CardLibrary(assets=root)
        with isolated_registry(empty=True):
            built = assembler.SceneAssembler(lib, Router()).assemble(lib.scene("task_dc_eth_demo"), "cpu")
        digests.append(built.model_digest)
    assert digests[0] == digests[1] and digests[0]


# ------------------------------------------------------------------------------------------------ real checkout


# ~53 s alone on host-1 (2026-11-06): alloy-assets' JOB2 bottle meshes (job_bottle_jar 25 s, lid 13 s, beaker 10 s)
# spend it in the thin-wall ray check (alloy/cards/geometry.py wall_thickness); the 60 s default failed under load.
@pytest.mark.timeout(300)
def test_real_assets_checkout_cards_load_and_validate() -> None:
    root = R.assets_root()
    if root is None:
        pytest.skip("no alloy-assets checkout (set ALLOY_ASSETS or clone it next to the alloy checkout)")
    lib = CardLibrary(imports=False)  # imported libraries (SCR-OSS3-4) are their importers' to validate
    check = {"materials": lib.material, "objects": lib.object, "scenes": lib.scene,
             "pairs": lambda key: lib.pair(*key.split("__", 1))}
    bad: list[str] = []
    for kind, get in check.items():
        for name in lib.names(kind):
            try:
                if not lib.path_of(kind, name).is_relative_to(LIB):  # library/ cards: test_library_cards.py
                    get(name)
            except CardError as exc:                                 # clash, parse or validation error
                bad.append(f"{kind}/{name}: {exc}")
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------------------------------------ imported libraries


def _imports(base: Path, *rels: str) -> None:
    for rel in rels:
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text("schema: alloy.object/1\n")


def test_default_library_includes_every_imported_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCR-OSS3-4: ``<assets>/imports/<source>/`` folders with objects/ join the default roots after library/ and each
    one's fetch cache, so ``CardLibrary()`` sees every imported object; a library/ card keeps its name; a fetched copy
    wins over the import root's; ``imports=False``, ``assets=None`` and explicit roots leave them out."""
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    assets = tmp_path / "assets"
    _imports(assets, "imports/src_b/objects/imp_b.yaml", "imports/src_a/objects/kitchen/imp_a.yaml",
             "imports/src_a/objects/peg_d12.yaml")
    _imports(tmp_path / "cache", "imports/src_a/objects/imp_a.yaml")                 # a fetched copy
    (assets / "imports/no_objects/meshes").mkdir(parents=True)
    found = R.import_roots(assets)
    assert found == [assets / "imports/src_a", assets / "imports/src_b"]
    lib = CardLibrary(assets=assets)
    caches = (tmp_path / "cache/imports/src_a", tmp_path / "cache/imports/src_b")
    assert lib.roots == tuple(p.resolve() for p in (LIB, *caches, *found))
    assert lib.has_object("imp_a") and lib.has_object("imp_b")
    assert lib.path_of("objects", "imp_a").is_relative_to(tmp_path / "cache")          # the fetched copy wins
    assert lib.path_of("objects", "peg_d12").is_relative_to(LIB)                       # library/ wins
    assert not CardLibrary(assets=assets, imports=False).has_object("imp_a")
    assert not CardLibrary(assets=None).has_object("imp_a") and not CardLibrary([LIB]).has_object("imp_a")


def test_a_wheel_install_finds_its_packaged_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no alloy-assets checkout, the imports an installed wheel carries in alloy/_data/imports/ are used (OSS-2's
    layout, the same rule as default_root's wheel fallback)."""
    monkeypatch.setenv("ALLOY_ASSETS", "")
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(R, "package_data", lambda: tmp_path / "_data")
    assert R.import_roots() == []
    _imports(tmp_path / "_data", "imports/ycb_gso/objects/mug.yaml")
    assert R.import_roots() == [tmp_path / "_data/imports/ycb_gso"]
    assert R.default_roots()[1:] == [tmp_path / "cache/imports/ycb_gso", tmp_path / "_data/imports/ycb_gso"]
    assert CardLibrary().has_object("mug") and CardLibrary().assets is None


def test_a_fetched_library_with_no_import_root_is_a_default_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCR-OSS11-8: a clean wheel install has no alloy-assets and no packaged imports; a library ``alloy fetch`` put
    in the cache (``<cache>/imports/<library>/``) is still a default root, after the import roots' caches and before
    the import roots, once; its cards load with their meshes. ``imports=False`` and ``assets=None`` leave it out."""
    monkeypatch.setenv("ALLOY_ASSETS", "")
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(R, "package_data", lambda: tmp_path / "_data")
    toylib = tmp_path / "cache/imports/toylib"
    (toylib / TRAY_MESH).parent.mkdir(parents=True)
    shutil.copy(LIB / "meshes" / "tray_aluminium.obj", toylib / TRAY_MESH)
    _copy_card(LIB / "objects" / "tray_aluminium.yaml", toylib / "objects" / "x.yaml",
               ("name: tray_aluminium", "name: x"), ("meshes/tray_aluminium.obj", TRAY_MESH))
    assert R.import_roots() == [] and R.default_roots() == [LIB, toylib]
    lib = CardLibrary()
    assert lib.has_object("x") and lib.object("x").material == "aluminium_6061_t6"      # validated: mesh, sha256
    assert lib.resolve(lib.object("x")).extents == lib.resolve(lib.object("tray_aluminium")).extents
    assert not CardLibrary(imports=False).has_object("x") and not CardLibrary(assets=None).has_object("x")

    _imports(tmp_path / "_data", "imports/ycb_gso/objects/mug.yaml")                  # with an import root too
    (tmp_path / "cache/imports/ycb_gso").mkdir()
    assert R.default_roots() == [LIB, tmp_path / "cache/imports/ycb_gso", toylib, tmp_path / "_data/imports/ycb_gso"]
