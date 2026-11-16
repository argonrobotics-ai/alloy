# SPDX-License-Identifier: Apache-2.0
"""The Blender renderer gives a static ``<item>/visual`` shape its item's look rules, as the full-material camera does.

Law, on a decomposed concave mesh (an L prism split into two box hulls), static, so its shapes have no body label: a
bundle recorded from the scene holds the drawn ``shelf__glass/visual`` shape and none of its hulls, and the look rule
``"__glass$"`` picks that shape's material in alloy/render/blender (``RigidSet.labels`` then ``Look.assigned``, which
is what ``MaterialLibrary.for_labels`` does) and its ``hide`` rule hides it. Blender is not needed: its own modules
(bpy, mathutils, bmesh) are stubbed so that rigid.py imports.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest

import alloy
from alloy.render import RenderRecorder, load_bundle, validate_bundle
from tests.camera.test_look_visual_shape import HULLS, L_POLY, _box, _obj, _prism

BLENDER_DIR = Path(alloy.__file__).resolve().parent / "render" / "blender"
LOOK = {"look": 1, "cameras": {"front": {"keys": [{"t": 0.0, "eye": [0.5, -0.5, 0.4], "target": [0, 0, 0]}]}},
        "materials": {"glass": {"preset": "glass_clear"}}, "assign": [{"match": "__glass$", "material": "glass"}]}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """One frame of a static and a free ``__glass`` item, each the decomposed L card ``ell_test``."""
    root = tmp_path_factory.mktemp("blender_visual_labels")
    (root / "meshes").mkdir()
    (root / "objects").mkdir()
    mesh = root / "meshes" / "ell_test.obj"
    mesh.write_text(_obj([_prism(L_POLY)]))
    sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
    (root / "meshes" / "ell_test.hulls.obj").write_text(_obj([_box(*h) for h in HULLS], f"# source_sha256: {sha}"))
    (root / "objects" / "ell_test.yaml").write_text(
        "schema: alloy.object/1\nname: ell_test\nshape:\n  mesh:\n    path: meshes/ell_test.obj\n"
        f"    sha256: {sha}\n    collision: decompose\n    max_parts: 2\nmaterial: abs\n")
    scene = alloy.Scene(dt=0.01, roots=[str(root)])
    scene.add_ground()
    scene.add_object("shelf__glass", "ell_test", pos=(0.0, 0.0, 0.0), static=True)
    scene.add_object("cup__glass", "ell_test", pos=(-0.3, 0.0, 0.0))
    model = alloy.build(scene, device="cpu").engine.model
    rec = RenderRecorder(model, root / "bundle", fps=30.0)
    rec.capture(model.state(), 0.0)
    rec.close()
    out = load_bundle(root / "bundle")
    assert validate_bundle(out) == []
    return out


@pytest.fixture
def blender(monkeypatch):
    """(rigid, look) of alloy/render/blender, imported with Blender's own modules stubbed; the Blender-side modules
    this import added are dropped again afterwards."""
    monkeypatch.syspath_prepend(str(BLENDER_DIR))
    for name in ("bpy", "mathutils", "bmesh"):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock(name=name))
    before = set(sys.modules)
    yield importlib.import_module("rigid"), importlib.import_module("look")
    for name in set(sys.modules) - before:
        if str(getattr(sys.modules[name], "__file__", "") or "").startswith(str(BLENDER_DIR)):
            del sys.modules[name]


def test_a_rule_on_the_item_name_reaches_the_static_visual_shape_in_blender(bundle, blender, monkeypatch):
    rigid, blook = blender
    shapes = {s["label"]: s for s in bundle.scene["shapes"]}
    assert {"shelf__glass/visual", "cup__glass/visual"} <= set(shapes), sorted(shapes)
    assert not [lb for lb in shapes if "/hull_" in lb]                       # the hulls are not exported
    assert shapes["shelf__glass/visual"]["body"] == -1                       # static: no body label to match
    look = blook.parse_look({**LOOK, "hide": ["^shelf__glass$"]})
    monkeypatch.setattr(rigid.RigidSet, "_build", lambda self: None)         # labels only: no Blender objects
    rs = rigid.RigidSet(bundle, look, mats=None, coll=None)
    assert look.assigned(rs.labels(shapes["shelf__glass/visual"])) == "glass"
    assert look.assigned(rs.labels(shapes["cup__glass/visual"])) == "glass"
    assert look.assigned(rs.labels(shapes["ground"])) is None
    assert look.is_hidden(rs.labels(shapes["shelf__glass/visual"]))
    assert not look.is_hidden(rs.labels(shapes["cup__glass/visual"]))
