# SPDX-License-Identifier: Apache-2.0
"""Render parts -> ``visual.glb`` (base colour, roughness, metallic, UV texture; glass flagged by alpha)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alloy.cards import geometry as geo
from alloy.importers.gymanything.pipeline import ImportOptions, import_specs
from alloy.importers.gymanything.spec import SpecError, load_spec
from alloy.importers.gymanything.visualglb import read_obj_uv, srgb_to_linear, write_visual_glb
from alloy.testing.imp_gymanything import box_mesh, cited, write_spec

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("PIL")


def _part_obj(path, mesh: geo.MeshData, uv: bool = False):
    lines = [f"v {x} {y} {z}" for x, y, z in mesh.vertices]
    if uv:
        lines += [f"vt {(x + 1) / 2} {(y + 1) / 2}" for x, y, _ in mesh.vertices]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in mesh.faces]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    path.write_text("\n".join(lines) + "\n")


def _spec_with_parts(tmp_path, **extra):
    box = box_mesh((0.1, 0.05, 0.02))
    p = write_spec(tmp_path / "specs", box, key="crate", category="container",
                   mass=cited(0.8, "datasheet", "params.py:7"), **extra)
    d = p.parent
    _part_obj(d / "crate__vis0.obj", box_mesh((0.1, 0.05, 0.015)))
    _part_obj(d / "crate__vis1.obj", box_mesh((0.02, 0.02, 0.02)), uv=True)
    from PIL import Image
    Image.fromarray(np.full((4, 4, 3), 200, np.uint8)).save(d / "crate__vis1.png")
    raw = json.loads(p.read_text())
    raw["visual"] = [
        {"mesh": "crate__vis0.obj", "base_color": [0.8, 0.2, 0.1], "cite": "scene.py:12", "roughness": 0.6,
         "metallic": 0.0, "name": "body"},
        {"mesh": "crate__vis1.obj", "base_color": [1, 1, 1], "cite": "scene.py:14", "texture": "crate__vis1.png",
         "name": "label"}]
    p.write_text(json.dumps(raw))
    return p


def test_srgb_to_linear_ends_and_mid():
    assert srgb_to_linear([0, 1, 0.5]) == pytest.approx([0.0, 1.0, 0.214], abs=1e-3)


def test_parts_parse_and_uv_round_trip(tmp_path):
    spec = load_spec(_spec_with_parts(tmp_path))
    assert [p.name for p in spec.visual] == ["body", "label"]
    assert spec.visual[0].roughness == 0.6 and spec.visual[1].texture.name == "crate__vis1.png"
    v, f, uv = read_obj_uv(spec.visual[1].mesh)
    assert uv is not None and uv.shape == (len(v), 2) and f.shape[1] == 3


def test_bad_colour_is_a_spec_error(tmp_path):
    p = _spec_with_parts(tmp_path)
    raw = json.loads(p.read_text())
    raw["visual"][0]["base_color"] = [2.0, 0, 0]
    p.write_text(json.dumps(raw))
    with pytest.raises(SpecError, match="base_color"):
        load_spec(p)


def test_glb_carries_pbr_values_and_texture(tmp_path):
    spec = load_spec(_spec_with_parts(tmp_path))
    path = write_visual_glb(spec, tmp_path / "out")
    scene = trimesh.load(path)
    mats = {g.visual.material.name: g.visual.material for g in scene.geometry.values()}
    body, label = mats["body"], mats["label"]
    assert np.asarray(body.baseColorFactor)[:3] / 255 == pytest.approx(srgb_to_linear([0.8, 0.2, 0.1]), abs=0.01)
    assert body.roughnessFactor == pytest.approx(0.6)
    assert label.baseColorTexture is not None
    ext = scene.bounds[1] - scene.bounds[0]           # glTF +Y up: the body's z extent (0.04) is glTF y
    assert ext == pytest.approx([0.2, 0.04, 0.1], abs=1e-6)


def test_import_writes_visual_glb_into_the_manifest(tmp_path):
    _spec_with_parts(tmp_path)
    res = import_specs(tmp_path / "specs", tmp_path / "out", opt=ImportOptions(hulls=False))
    assert res["visual"] == {"objects": 1, "failures": {}, "bbox_off": []}
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert "visual.glb" in man["objects"]["gar_testsim_crate"]["files"]
