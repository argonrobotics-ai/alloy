# SPDX-License-Identifier: Apache-2.0
"""Blender-free look materials against the oracle: the C3 export made by Alloy's Blender build (no GPU, no Blender)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alloy.camera import CameraError
from alloy.camera.looks import material_tree
from alloy.camera.pbr.codegen import gen_material

GOLDEN = json.loads((Path(__file__).parent / "data" / "looks_golden.json").read_text())
WALNUT = "polyhaven:american_walnut_veneer"
ROLE_BY_TAG = {"_diff_": "color", "_disp_": "height"}


def _texture_files() -> dict:
    """The walnut set the export used, as local files {ref: {role: path}} (the files themselves are not read)."""
    files = {role: Path("/assets") / key for key in GOLDEN["textures"] for tag, role in ROLE_BY_TAG.items() if tag in key}
    return {WALNUT: files}


def _tree_hash(tree: dict) -> str:
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(GOLDEN["materials"]))
def test_material_matches_blender_export(name):
    want = GOLDEN["materials"][name]
    mat = material_tree(name, want["spec"], _texture_files())
    src, meta = gen_material(name, mat, mat["textures"])
    for im in meta["images"]:  # the export names files "textures/NN_<key>"; compare by key
        im["file"] = im["key"]
    assert src == want["source"]
    assert meta == want["meta"]
    assert _tree_hash(mat["tree"]) == want["tree_sha256"]


def test_textured_material_reports_its_images():
    mat = material_tree("walnut_worktop", GOLDEN["materials"]["walnut_worktop"]["spec"], _texture_files())
    assert {k: v["colorspace"] for k, v in mat["textures"].items()} == {k: v["colorspace"] for k, v in GOLDEN["textures"].items()}


def test_untextured_preset_defaults():
    mat = material_tree("pad", "rubber_black")
    types = [n["type"] for n in mat["tree"]["nodes"]]
    assert mat["textures"] == {} and "ShaderNodeBevel" in types and "ShaderNodeBump" in types
    ins = mat["principled"]["inputs"]
    assert ins["Roughness"]["linked"] and ins["Sheen Weight"]["value"] == pytest.approx(0.15)
    assert ins["Metallic"] == {"value": 0.0, "linked": False}


def test_override_changes_values_and_float32_rounding():
    base = material_tree("m", {"preset": "pvc_cable_black"})
    over = material_tree("m", {"preset": "pvc_cable_black", "coat": 0.4, "ior": 1.3})
    assert base["principled"]["inputs"]["Coat Weight"]["value"] == pytest.approx(0.2)
    assert over["principled"]["inputs"]["Coat Weight"]["value"] == 0.4000000059604645  # float32 of 0.4
    assert over["principled"]["inputs"]["IOR"]["value"] == pytest.approx(1.3)


def test_unknown_preset_names_the_fix():
    with pytest.raises(CameraError) as err:
        material_tree("m", {"preset": "unobtainium"})
    assert "unobtainium" in err.value.problem and "abs_black" in err.value.fix


def test_unsupported_detail_and_missing_texture_raise(tmp_path, monkeypatch):
    with pytest.raises(CameraError, match="crust"):
        material_tree("loaf", "bread_crust")
    monkeypatch.setenv("ALLOY_RENDER_ASSETS", str(tmp_path))
    with pytest.raises(CameraError, match="not in the local asset cache"):
        material_tree("top", {"preset": "wood_oak"})
