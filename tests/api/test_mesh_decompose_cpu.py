# SPDX-License-Identifier: Apache-2.0
"""``Scene.add_mesh(..., collision="decompose")``: the API writes the convex-parts file the engine reads (CPU)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api._materialize import materialize
from alloy.importers.common import hulls as importer_hulls
from alloy.importers.common.errors import MissingExtraError

H = 0.04
#: An L in plan, 10 cm arms 3 cm wide, extruded 4 cm: its convex hull fills the inner corner, its two parts do not.
POLY = [(0, 0), (0.1, 0), (0.1, 0.03), (0.03, 0.03), (0.03, 0.1), (0, 0.1)]
CAP = [(3, 4, 5), (3, 5, 0), (3, 0, 1), (3, 1, 2)]      # fan from the inner corner, counter-clockwise from above
NOTCH = (0.055, 0.055)                                  # inside the L's hull, outside the L


def l_mesh() -> tuple[np.ndarray, np.ndarray]:
    n = len(POLY)
    v = [(x, y, 0.0) for x, y in POLY] + [(x, y, H) for x, y in POLY]
    f = [(a + n, b + n, c + n) for a, b, c in CAP] + [(a, c, b) for a, b, c in CAP]
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, j + n), (i, j + n, i + n)]
    return np.array(v, float), np.array(f, int)


def l_scene(**mesh_kw) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    if "path" not in mesh_kw and "vertices" not in mesh_kw:
        mesh_kw["vertices"], mesh_kw["faces"] = l_mesh()
    scene.add_mesh("ell", collision="decompose", static=True, **mesh_kw)
    return scene


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_CACHE_DIR", str(tmp_path / "cache"))


@pytest.mark.compile_heavy
def test_decompose_mesh_builds_and_keeps_its_inner_corner(tmp_path):
    scene = l_scene()
    scene.add_sphere("ball", 0.01, pos=(*NOTCH, 0.08), mass=0.01)
    sim = alloy.build(scene, device="cpu")
    z = sim.run(0.6, record=["ball"]).world(0)["ball.pos"][-1, 2]
    assert z == pytest.approx(0.01, abs=1e-3)         # fell past the L's top (z 0.04) to the ground: two parts
    assert sim.check()["ok"]
    sim.close()
    assert len(list((tmp_path / "cache" / "hulls").glob("*.hulls.obj"))) == 1


def test_shipped_hull_file_is_used_without_coacd(tmp_path, monkeypatch):
    v, f = l_mesh()
    obj = tmp_path / "ell.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in v) + "".join(f"f {a + 1} {b + 1} {c + 1}\n"
                                                                          for a, b, c in f))
    boxes = [((0, 0, 0), (0.1, 0.03, H)), ((0, 0.03, 0), (0.03, 0.1, H))]
    hulls = [importer_hulls.convex_hull(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                                  for z in (lo[2], hi[2])], float)) for lo, hi in boxes]
    from alloy.importers.common.objio import hulls_obj_text

    side = tmp_path / "ell.hulls.obj"
    side.write_text(hulls_obj_text(hulls, hashlib.sha256(obj.read_bytes()).hexdigest()))
    monkeypatch.setattr(importer_hulls, "decompose", lambda *a, **k: pytest.fail("CoACD ran"))
    materialize(l_scene(path=obj), tmp_path / "cards")
    assert (tmp_path / "cards" / "meshes" / "ell.hulls.obj").read_text() == side.read_text()


def test_decompose_without_coacd_says_how_to_install(tmp_path, monkeypatch):
    def missing(module):
        raise MissingExtraError(module)

    monkeypatch.setattr(importer_hulls, "need", missing)
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        materialize(l_scene(), tmp_path / "cards")
    assert "alloy-engine[import]" in err.value.fix and 'collision="convex"' in err.value.fix


def test_decompose_inside_out_mesh_is_a_scene_error(tmp_path):
    v, f = l_mesh()
    with pytest.raises(alloy.SceneError) as err:
        materialize(l_scene(vertices=v, faces=f[:, ::-1].copy()), tmp_path / "cards")
    assert err.value.field == "collision" and "outward" in err.value.message
