# SPDX-License-Identifier: Apache-2.0
"""Scene variants (alloy.scene.variants, CPU): jittered free objects, same-family texture swaps, deterministic files.

Laws: fixed things stay (ground, static table, kinematic pusher, robot); free objects move within +-pos / +-yaw, z
unchanged, touching ones together; a tight row is redrawn until nothing overlaps, and two variants built through
alloy.check have no start overlap and settle healthy; a look swaps only within its family; same seed, same files.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

import alloy
from alloy.api.errors import SceneError
from alloy.scene.variants import looks as L
from alloy.scene.variants import scene_look, vary

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.2"/>
  </body></worldbody><actuator><position joint="shoulder" kp="20" kv="1"/></actuator></mujoco>"""
FREE = ("a", "b", "c", "ball", "cap")


def _table(tight: bool = False) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")
    gap = 0.042 if tight else 0.08                                    # tight: 2 mm between 40 mm boxes
    s.add_box("a", size=(0.04, 0.04, 0.04), pos=(-gap, 0.0, 0.42), material="wood_beech")
    s.add_box("b", size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.42), material="bamboo")
    s.add_box("c", size=(0.03, 0.03, 0.03), pos=(0.0, 0.0, 0.455), material="steel_carbon")    # rests on b
    s.add_sphere("ball", 0.015, pos=(gap, 0.0, 0.415), material="abs")
    if not tight:
        s.add_sphere("cap", 0.015, pos=(0.0, 0.08, 0.415), material="nitrile_70a")   # rubber: no texture family
        s.add_box("pusher", size=(0.02, 0.02, 0.02), pos=(0.0, -0.1, 0.41), kinematic=True)
        s.add_robot("arm", mjcf=ARM, pos=(0.5, 0.0, 0.0))
    return s


def _yaw(q) -> float:
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@pytest.fixture(scope="module")
def table_variants(tmp_path_factory):
    out = tmp_path_factory.mktemp("variants")
    return _table(), vary(_table(), 3, pos=0.01, yaw_deg=8.0, seed=4, out=out, contact_sheet=True,
                          sheet={"quality": "fast", "width": 48, "height": 32, "spp": 1}), out


def test_fixed_things_stay_and_free_objects_move_within_the_bounds(table_variants):
    base, variants, _ = table_variants
    for v in variants:
        for name in ("ground", "table", "pusher", "arm"):
            assert v.items[name]["pose"] == base.items[name]["pose"], name
        for name in FREE:
            p0, p1 = np.asarray(base.items[name]["pose"]), np.asarray(v.items[name]["pose"])
            assert np.all(np.abs(p1[:2] - p0[:2]) <= 0.01 + 1e-9) and p1[2] == pytest.approx(p0[2], abs=1e-12)
            assert abs(_yaw(p1[3:]) - _yaw(p0[3:])) <= 8.0 + 1e-6
    moved = [np.asarray(v.items["a"]["pose"][:2]) - np.asarray(base.items["a"]["pose"][:2]) for v in variants]
    assert min(float(np.linalg.norm(d)) for d in moved) > 0.0


def test_touching_objects_move_together(table_variants):
    base, variants, out = table_variants
    manifest = json.loads((out / "variants.json").read_text())
    assert ["b", "c"] in manifest["groups"]
    for row in manifest["variants"]:
        assert row["moves"]["b"] == row["moves"]["c"]


def test_files_round_trip_and_the_same_seed_gives_the_same_variants(table_variants, tmp_path):
    base, variants, out = table_variants
    manifest = json.loads((out / "variants.json").read_text())
    assert manifest["schema"] == "alloy.variants/1" and [r["file"] for r in manifest["variants"]] == \
        ["variant_000.json", "variant_001.json", "variant_002.json"]
    for k, v in enumerate(variants):
        assert alloy.Scene.from_dict(json.loads((out / f"variant_{k:03d}.json").read_text())).to_dict() == v.to_dict()
    again = vary(_table(), 3, pos=0.01, yaw_deg=8.0, seed=4, out=tmp_path)
    assert [v.to_dict() for v in again] == [v.to_dict() for v in variants]
    other = vary(_table(), 1, pos=0.01, yaw_deg=8.0, seed=5)
    assert other[0].items["a"]["pose"] != variants[0].items["a"]["pose"]
    sheet = manifest["contact_sheet"]
    assert sheet["tiles"] == ["fast"] * 3 and (out / "contact_sheet.png").stat().st_size > 0


def test_looks_swap_within_their_family(table_variants):
    _, variants, _ = table_variants
    pool = L.texture_pool(_lib())
    by_ref = {t.ref: t for ts in pool.values() for t in ts}
    assert {"wood", "metal", "plastic", "fabric"} <= set(pool)                   # alloy.looks.pool's families
    for v in variants:
        for name, family in (("table", "wood"), ("a", "wood"), ("b", "wood"), ("c", "metal"), ("ball", "plastic")):
            (spec,) = v.items[name]["look"]["materials"].values()
            tex = by_ref[spec["texture"]]                                    # untextured steel / ABS take one too
            assert tex.family == family and spec["res"] == tex.res and spec["scale_m"] == tex.scale_m, name
        assert v.items["b"]["look"]["materials"]["bamboo"]["preset"] == "wood_oak"     # the preset stays
        (rubber,) = v.items["cap"]["look"]["materials"].values()
        assert "texture" not in rubber and 0.9 <= rubber["color_gain"] <= 1.1  # no family: tint only
        assert "look" not in v.items["arm"]
    swapped = [v.items["b"]["look"]["materials"]["bamboo"]["texture"] for v in variants]
    assert "polyhaven:bamboo_veneer" not in swapped                   # never its own texture while others exist


def test_textured_mode_leaves_untextured_looks_untextured():
    (v,) = vary(_table(), 1, seed=4, textures="textured")
    assert "texture" not in v.items["c"]["look"]["materials"]["steel_carbon"]
    assert v.items["b"]["look"]["materials"]["bamboo"]["texture"] != "polyhaven:bamboo_veneer"


def test_the_pool_family_wins_over_the_preset(tmp_path):
    (tmp_path / "looks" / "textures").mkdir(parents=True)
    for name, family in (("tex_a", "stone"), ("tex_b", "stone"), ("tex_c", "wood")):
        (tmp_path / "looks" / "textures" / f"{name}.json").write_text(json.dumps(
            {"schema": "alloy.look/1", "look": name, "family": family, "preset": "wood_oak",
             "texture": f"polyhaven:{name}", "res": "1k", "scale_m": 0.5, "source": "https://example.org",
             "author": "test", "licence": "CC0-1.0", "albedo": 0.3}))
    lib = type("Lib", (), {"search_roots": [tmp_path]})()
    pool = L.texture_pool(lib)
    assert [t.ref for t in pool["stone"]] == ["polyhaven:tex_a", "polyhaven:tex_b"]
    assert [(t.ref, t.res, t.albedo) for t in pool["wood"]] == [("polyhaven:tex_c", "1k", 0.3)]
    look = type("Look", (), {"materials": {"m": {"preset": "wood_oak", "texture": "polyhaven:tex_a"}}, "assign": (),
                             "default": "m"})()
    got = {L.vary_look(look, pool, np.random.default_rng(k), "same-family", 0.0)[1]["m"] for k in range(8)}
    assert got == {"polyhaven:tex_b"}                         # stone by the pool, though its preset is wood


def test_every_pool_texture_carries_res_and_albedo():
    pool = L.texture_pool(_lib())
    assert sum(len(ts) for ts in pool.values()) >= 60
    assert all(t.res in ("1k", "2k") and t.albedo is not None and 0.0 <= t.albedo <= 1.0
               for ts in pool.values() for t in ts)


def test_a_tight_row_is_redrawn_and_its_variants_build_clean(tmp_path):
    variants = vary(_table(tight=True), 2, pos=0.01, yaw_deg=10.0, seed=7, out=tmp_path, textures="keep")
    draws = [r["draws"] for r in json.loads((tmp_path / "variants.json").read_text())["variants"]]
    assert sum(draws) > len(draws)        # some draws were refused (7 and 4 at seed 7); unchecked, both overlap
    for k, v in enumerate(variants):
        report = alloy.check(str(v.save(tmp_path / f"cards{k}")), worlds=1, seconds=0.2, device="cpu")
        assert report.ok and report.checks["overlap"] == "pass", [f.message for f in report.findings]


def test_no_room_raises_naming_the_overlap():
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("left", size=(0.02, 0.1, 0.05), pos=(-0.03, 0, 0.025), static=True)
    s.add_box("right", size=(0.02, 0.1, 0.05), pos=(0.03, 0, 0.025), static=True)
    s.add_box("slot", size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.02))           # touches both walls
    with pytest.raises(SceneError, match="overlaps"):
        vary(s, 1, pos=0.01, max_tries=5)


def test_scene_look_rules_match_the_item_and_its_parts_only(table_variants):
    _, variants, _ = table_variants
    look = scene_look(variants[0])
    rule = next(r for r in look["assign"] if r["material"] == "a:wood_beech")
    assert all(re.search(rule["match"], s) for s in ("a", "a/hull_0", "a#1"))
    assert not any(re.search(rule["match"], s) for s in ("ab", "ball", "table"))
    assert "a:wood_beech" in look["materials"]


def test_bad_arguments_name_the_field():
    with pytest.raises(SceneError, match="pos"):
        vary(_table(), 1, pos=-1.0)
    with pytest.raises(SceneError, match="textures"):
        vary(_table(), 1, textures="random")


def _lib():
    from alloy.cards.library import CardLibrary  # noqa: PLC0415

    return CardLibrary()
