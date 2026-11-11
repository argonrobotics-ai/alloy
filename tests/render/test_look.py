# SPDX-License-Identifier: Apache-2.0
"""Look loading, validation, frame list and cost estimate (no GPU, fake bundle)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alloy.render import look as lk


def fake_bundle(frames=100):
    scene = {"bodies": [{"id": 0, "label": "nut_body", "world": 0}],
             "shapes": [{"id": 0, "label": "table_top", "body": -1}, {"id": 1, "label": "geom1", "body": 0}]}
    labels = {"shapes": ["table_top", "geom1"], "bodies": ["nut_body"], "surfaces": ["towel"],
              "points": ["honey"], "curves": [], "instancers": []}
    return SimpleNamespace(scene=scene, frame_count=frames, labels=lambda: labels)


def good_look():
    cam = {"lens_mm": 85, "keys": [{"t": 0, "eye": [1, 0, 1], "target": [0, 0, 0]}]}
    return {"look": 1, "render": {"resolution": [1080, 1920], "samples": 512}, "cameras": {"v": cam},
            "materials": {"table": {"preset": "brass"}}, "assign": [{"match": "^table", "material": "table"}],
            "liquids": {"honey": {"material": "table"}}, "frames": {"start": 0, "end": -1, "step": 4}}


def test_load_checks_version(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"look": 2}))
    with pytest.raises(lk.LookError, match="supports 1"):
        lk.load_look(p)
    p.write_text("{bad")
    with pytest.raises(lk.LookError, match="not valid JSON"):
        lk.load_look(p)


def test_validate_reports_unmatched_as_warnings():
    out = lk.validate_look(good_look(), fake_bundle())
    errs = [o for o in out if not o.startswith(lk.WARN)]
    assert errs == []
    warns = [o for o in out if o.startswith(lk.WARN)]
    assert any("geom1" in w for w in warns) and any("towel" in w for w in warns)
    assert not any("honey" in w or "table_top" in w for w in warns)


def test_body_label_matches_shape():
    look = good_look()
    look["assign"].append({"match": "nut", "material": "table"})
    warns = [o for o in lk.validate_look(look, fake_bundle()) if "geom1" in o]
    assert warns == []


def test_bad_regex_and_camera_are_errors():
    look = good_look()
    look["assign"][0]["match"] = "(["
    look["cameras"]["v"]["focus"] = {"label": "(("}
    errs = lk.validate_look(look, fake_bundle())
    assert any("assign[0].match" in e for e in errs) and any("focus.label" in e for e in errs)


def test_unknown_preset_checked_when_presets_file_exists(tmp_path):
    f = tmp_path / "presets.py"
    f.write_text("PRESETS = {'honey': {}, 'steel': {}}\n")
    assert lk.preset_names(f) == {"honey", "steel"}
    assert lk.preset_names(tmp_path / "missing.py") is None


def test_frame_list():
    look = good_look()
    assert lk.frame_list(look, fake_bundle(10)) == [0, 4, 8]
    assert lk.frame_list(look, fake_bundle(100), "10:20") == [12, 16]
    assert lk.frame_list(look, fake_bundle(100), "90:") == [92, 96]
    with pytest.raises(lk.LookError):
        lk.frame_list(look, fake_bundle(), "a:b")


def test_estimate_cost_scales():
    ref = lk.estimate_cost(80, (1080, 1920), 512)
    assert ref.seconds_per_frame == 45 and ref.gpu_hours == pytest.approx(1.0)
    assert ref.usd == pytest.approx(0.28)
    half = lk.estimate_cost(80, (1080, 1920), 256)
    assert half.gpu_hours == pytest.approx(0.5)


def test_unknown_preset_is_error():
    look = good_look()
    look["materials"]["table"]["preset"] = "unobtainium"
    assert any("unknown preset" in e for e in lk.validate_look(look, fake_bundle()))


def test_bevels_are_checked():
    look = good_look()
    look["bevels"] = [{"match": "^table", "bevel_m": 0.0003}, {"match": "(", "bevel_m": 0.001}, {"match": "x", "bevel_m": 0.5}]
    errs = [o for o in lk.validate_look(look, fake_bundle()) if not o.startswith(lk.WARN)]
    assert any("bevels[1].match" in e for e in errs) and any("bevels[2].bevel_m" in e for e in errs)
    assert not any("bevels[0]" in e for e in errs)


def test_skin_covers_its_bodies_and_checks_files(tmp_path):
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "lid.obj").write_text("v 0 0 0\n")
    (tmp_path / "extra" / "w.json").write_text(json.dumps({"bodies": ["nut_body"], "weights": [[[0, 1.0]]]}))
    b = fake_bundle()
    b.root = tmp_path
    look = good_look()
    look["skins"] = {"lid": {"mesh": "extra/lid.obj", "weights": "extra/w.json", "material": "table"}}
    out = lk.validate_look(look, b)
    assert [o for o in out if not o.startswith(lk.WARN)] == []
    assert not any("geom1" in o for o in out)          # the skinned body's shape is covered by the skin
    look["skins"]["lid"]["weights"] = "extra/missing.json"
    assert any("skins.lid.weights" in o for o in lk.validate_look(look, b))
    (tmp_path / "extra" / "w.json").write_text(json.dumps({"bodies": ["ghost"], "weights": [[[0, 1.0]]]}))
    look["skins"]["lid"]["weights"] = "extra/w.json"
    assert any("ghost" in o for o in lk.validate_look(look, b))


def test_curve_shapes_need_no_material():
    b = fake_bundle()
    b.scene["shapes"][1]["curve"] = "harness"
    assert not any("geom1" in o for o in lk.validate_look(good_look(), b))


def test_props_and_white_balance(tmp_path):
    b = fake_bundle()
    b.root = tmp_path
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "jar.glb").write_bytes(b"glb")
    look = good_look()
    look["render"]["white_balance"] = {"kelvin": 5600, "tint": 10}
    look["props"] = [{"model": "polyhaven:wooden_bowl_01", "pose": [0, 0, 0, 0, 0, 0, 1]},
                     {"model": "file:extra/jar.glb", "pose": [0.3, 0, 0, 0, 0, 0, 1], "scale": 0.5}]
    assert [o for o in lk.validate_look(look, b) if not o.startswith(lk.WARN)] == []
    look["render"]["white_balance"] = "sunny"
    look["props"] = [{"model": "bowl", "pose": [0, 0, 0]}, {"model": "file:extra/none.glb", "pose": [0] * 7}]
    assert any("white_balance" in e for e in lk.validate_look(look, b))
    look["render"]["white_balance"] = "auto"
    errs = lk.validate_look(look, b)
    assert any("props[0].model" in e for e in errs) and any("props[0].pose" in e for e in errs)
    assert any("props[1].model" in e for e in errs)


def test_rigid_glb_skin_file_texture_and_glare(tmp_path):
    b = fake_bundle()
    b.root = tmp_path
    (tmp_path / "extra").mkdir()
    for f in ("plug.glb", "print.png"):
        (tmp_path / "extra" / f).write_bytes(b"x")
    look = good_look()
    look["materials"]["jacket"] = {"preset": "pvc_jacket_matte", "texture": "file:extra/print.png"}
    look["skins"] = {"plug": {"mesh": "extra/plug.glb", "body": "nut_body", "material_map": {"Shell": "table"}}}
    look["render"]["glare"] = {"type": "fog_glow", "threshold": 1.0}
    assert [o for o in lk.validate_look(look, b) if not o.startswith(lk.WARN)] == []
    assert not any("geom1" in o for o in lk.validate_look(look, b))     # the rigid skin hides its body's proxy
    look["skins"]["plug"].update(body="ghost", material_map={"Shell": "nope"})
    look["materials"]["jacket"]["texture"] = "file:extra/missing.png"
    look["render"]["glare"] = {"type": "starburst"}
    errs = lk.validate_look(look, b)
    assert any("glare" in e for e in errs)
    look["render"].pop("glare")
    errs = lk.validate_look(look, b)
    assert any("skins.plug.body" in e for e in errs) and any("'nope'" in e for e in errs)
    assert any("materials.jacket.texture" in e for e in errs)


def test_liquid_contact_rules():
    look = good_look()
    look["liquids"]["honey"]["contact"] = [{"match": "glass", "distance_m": 0.0015, "overlap_m": 0.0002}]
    assert [o for o in lk.validate_look(look, fake_bundle()) if not o.startswith(lk.WARN)] == []
    look["liquids"]["honey"]["contact"] = [{"distance_m": 0.001}, {"match": "(", "overlap_m": 0.5}]
    errs = lk.validate_look(look, fake_bundle())
    assert any("contact[0]" in e for e in errs) and any("contact[1].match" in e for e in errs)
    assert any("contact[1].overlap_m" in e for e in errs)


def test_ground_is_checked():
    look = good_look()
    look["ground"] = {"z": -0.75, "material": "table", "size_m": 4000}
    assert not [o for o in lk.validate_look(look, fake_bundle()) if o.startswith("ground")]
    look["ground"] = {"material": "table"}
    assert any(o.startswith("ground") for o in lk.validate_look(look, fake_bundle()))


def test_skin_replicas_are_checked(tmp_path):
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "lid.obj").write_text("v 0 0 0\n")
    (tmp_path / "extra" / "lid_lod.obj").write_text("v 0 0 0\n")
    b = fake_bundle()
    b.root = tmp_path
    look = good_look()
    look["skins"] = {"lid": {"mesh": "extra/lid.obj", "body": "nut_body", "material": "table",
                             "replicas": {"mesh": "extra/lid_lod.obj"}}}
    errors = lambda: [o for o in lk.validate_look(look, b) if not o.startswith(lk.WARN)]  # noqa: E731
    assert errors() == []
    for ok in ("proxy", "skin", "none"):
        look["skins"]["lid"]["replicas"] = ok
        assert errors() == []
    look["skins"]["lid"]["replicas"] = {"mesh": "extra/missing.obj"}
    assert any("replicas.mesh" in e for e in errors())
    look["skins"]["lid"]["replicas"] = "lod"
    assert any("skins.lid.replicas" in e for e in errors())


def test_planar_mapping_checks():
    look = good_look()
    mat = look["materials"]["table"]
    mat.update(mapping="planar", planar_axis=[1, 0, 0], planar_origin=[0.4, 0, 0.8], planar_up=[0, 0, 1])
    assert [o for o in lk.validate_look(look, fake_bundle()) if not o.startswith(lk.WARN)] == []
    mat["planar_up"] = [2, 0, 0]
    assert any("parallel" in o for o in lk.validate_look(look, fake_bundle()))
    mat.update(planar_up=None, planar_axis=[0, 0, 0])
    assert any("must not be zero" in o for o in lk.validate_look(look, fake_bundle()))
    mat.update(planar_axis=[1, 0], mapping="planar")
    assert any("not 3 numbers" in o for o in lk.validate_look(look, fake_bundle()))
    mat.update(planar_axis=[1, 0, 0], mapping="box")
    assert any("not one of" in o for o in lk.validate_look(look, fake_bundle()))
