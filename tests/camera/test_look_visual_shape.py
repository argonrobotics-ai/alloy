# SPDX-License-Identifier: Apache-2.0
"""A look rule on an item colours the ``<item>/visual`` mesh the camera draws in place of its collision hulls.

Law, on a decomposed concave mesh (an L prism split into two box hulls): a rule anchored on the item name
(``"__glass$"``) gives the drawn shape its material whether the item is static (no body label) or free, and a rule
that matches only a hull, which the camera does not draw, warns once, naming the item to match instead.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import alloy
from alloy.camera import model_pack
from alloy.camera.model_pack import pack_from_model

S = 0.05                                                   # [m] per unit
L_POLY = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]  # concave: reflex vertex at (1, 1)
HULLS = [((0, 0, 0), (2, 1, 1)), ((0, 1, 0), (1, 2, 1))]   # an exact partition of the L
RED = {"preset": "plastic_matte", "base_color": [0.8, 0.05, 0.05]}


def _prism(poly) -> tuple[list, list]:
    n = len(poly)
    v = [(x * S, y * S, 0.0) for x, y in poly] + [(x * S, y * S, S) for x, y in poly]
    fan = [(3, (3 + k) % n, (3 + k + 1) % n) for k in range(1, n - 1)]                   # from the reflex vertex
    f = [(a, c, b) for a, b, c in fan] + [(a + n, b + n, c + n) for a, b, c in fan]
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, j + n), (i, j + n, i + n)]
    return v, f


def _box(lo, hi) -> tuple[list, list]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x * S, y * S, z * S) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v, f


def _obj(parts, header: str = "") -> str:
    lines, base = ([header] if header else []), 0
    for k, (v, f) in enumerate(parts):
        lines += [f"o hull_{k}"] if header else []
        lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in v]
        lines += [f"f {a + base + 1} {b + base + 1} {c + base + 1}" for a, b, c in f]
        base += len(v)
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    """A static and a free ``__glass`` item and a static ``plain`` one, each the decomposed L card ``ell_test``."""
    root = tmp_path_factory.mktemp("look_visual_shape")
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
    scene.add_object("plain", "ell_test", pos=(0.3, 0.0, 0.0), static=True)
    scene.add_object("cup__glass", "ell_test", pos=(-0.3, 0.0, 0.0))
    return alloy.build(scene, device="cpu")


@pytest.fixture(scope="module")
def model(sim):
    return sim.engine.model


def _materials(model, look) -> dict[str, str]:
    _, frame, _ = pack_from_model(model, look)
    return {m.meta["name"]: m.meta["materials"][0] for m in frame.meshes}


def test_a_rule_on_the_item_name_colours_the_visual_shape_of_a_decomposed_mesh(model, caplog, monkeypatch):
    monkeypatch.setattr(model_pack, "_WARNED", set())
    labels = [str(lb) for lb in model.shape_label]
    assert {"shelf__glass/hull_0", "shelf__glass/hull_1", "shelf__glass/visual"} <= set(labels)
    look = {"materials": {"glass": RED}, "assign": [{"match": "__glass$", "material": "glass"}]}
    with caplog.at_level("WARNING", logger=model_pack.__name__):
        got = _materials(model, look)
    assert not [n for n in got if "/hull_" in n], got                       # the camera draws no hull
    assert got["shelf__glass/visual"] == "glass", got                       # static: no body label to match
    assert got["cup__glass/visual"] == "glass", got
    assert got["plain/visual"] != "glass", got
    assert not [r for r in caplog.records if r.name == model_pack.__name__]


def test_a_rule_that_matches_only_a_hull_warns_once_naming_the_item(model, caplog, monkeypatch):
    monkeypatch.setattr(model_pack, "_WARNED", set())
    look = {"materials": {"glass": RED}, "assign": [{"match": r"shelf__glass/hull_\d+$", "material": "glass"}]}
    with caplog.at_level("WARNING", logger=model_pack.__name__):
        got = _materials(model, look)
        _materials(model, look)
    assert got["shelf__glass/visual"] != "glass", got
    assert [r.getMessage() for r in caplog.records if r.name == model_pack.__name__] == [
        "look: 'shelf__glass/visual' gets no look material, but the rule for 'glass' matches its collision hull "
        "'shelf__glass/hull_0', which the camera does not draw; match the item name 'shelf__glass' instead (the "
        "drawn shape takes the item's rules)"]


def test_a_full_render_draws_the_static_decomposed_item_in_the_rules_material(sim):
    """Through sim.render(quality="full"), aimed at the static L: it takes the rule's colour (red, then green) where
    it drew the default white in both before."""

    def tinted(rgb) -> tuple[int, int]:
        look = {"materials": {"glass": {**RED, "base_color": rgb}},
                "assign": [{"match": "^shelf__glass$", "material": "glass"}]}
        img = sim.render(width=32, height=24, quality="full", look=look, spp=2,
                         look_at=((0.2, -0.15, 0.15), (0.05, 0.05, 0.025)))
        img = np.asarray(img, float).reshape(-1, 3)
        return int((img[:, 0] > img[:, 1] + 60).sum()), int((img[:, 1] > img[:, 0] + 60).sum())

    red, green = tinted([0.8, 0.05, 0.05]), tinted([0.05, 0.8, 0.05])
    assert red[0] > 30 and red[1] == 0 and green[1] > 30 and green[0] == 0, (red, green)   # 72 of 768 measured
