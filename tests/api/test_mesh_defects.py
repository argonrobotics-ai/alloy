# SPDX-License-Identifier: Apache-2.0
"""Broken meshes raise an API error with a fix, never a raw numpy or Python error (CPU).

A closed mesh with some faces wound inside out has a wrong signed volume (half of a cube's faces flipped: exactly 0),
and a dynamic body gets its mass from that volume: the rigid step divided by the zero mass (ZeroDivisionError). A NaN
vertex made the card check's oriented-box fit raise numpy's LinAlgError. ``add_mesh`` now refuses both with a
SceneError, and the engine's own checks catch the same meshes on paths that skip ``add_mesh`` (an object card from a
card root): the card check names the non-finite vertices, the rigid emitter a dynamic mesh body of no mass.

Only a dynamic rigid body whose mass comes out 0 or less is refused: the same meshes static or kinematic, with an
explicit mass=, or as a cloth sheet (no volume by nature) behave as before.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api import meshes
from alloy.api._items import mixed_winding_edges

V, F = meshes.box((0.1, 0.1, 0.1))                      # closed, wound outward
MIXED = F.copy()
MIXED[::2] = MIXED[::2, ::-1]                           # half the faces inside out: signed volume 0
ONE = F.copy()
ONE[0] = ONE[0, ::-1]                                   # one face inside out: the volume stays positive
NAN_V = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [np.nan, 0.0, 0.1]])
TET_F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
THICKNESS = "features:\n  thickness:\n    value: 0.002\n    source: measured\n    note: a test sheet\n"


def _obj(v: np.ndarray, f: np.ndarray) -> str:
    return "".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in v.tolist()) + "".join(
        f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f.tolist())


def _card_root(root, name: str, v: np.ndarray, f: np.ndarray, material: str = "abs", extra: str = ""):
    """A card root holding one object card ``name`` made from the mesh (v, f), as a YAML card would be."""
    text = _obj(v, f)
    (root / "meshes").mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(parents=True, exist_ok=True)
    (root / "meshes" / f"{name}.obj").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (root / "objects" / f"{name}.yaml").write_text(
        f"schema: alloy.object/1\nname: {name}\nshape:\n  mesh:\n    path: meshes/{name}.obj\n    sha256: {sha}\n"
        f"    collision: convex\nmaterial: {material}\n{extra}")
    return root


def _sheet(n: int = 6, size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """A flat n x n vertex grid: an open sheet with no volume."""
    xs = np.linspace(-size / 2, size / 2, n)
    v = np.array([[x, y, 0.0] for y in xs for x in xs])
    f = [[a, a + 1, a + n + 1] for j in range(n - 1) for a in range(j * n, j * n + n - 1)]
    f += [[a, a + n + 1, a + n] for j in range(n - 1) for a in range(j * n, j * n + n - 1)]
    return v, np.asarray(f)


def _scene(**kw) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, **kw)
    scene.add_ground()
    return scene


def test_mixed_winding_edges_counts_only_closed_meshes_wound_both_ways():
    assert mixed_winding_edges(V, F) == 0
    assert mixed_winding_edges(V, MIXED) > 0
    assert mixed_winding_edges(V, ONE) == 3                # one face flipped: its three edges
    assert mixed_winding_edges(V, F[1:]) == 0              # open (a face missing): not judged here
    seam = np.vstack([V, V[:1]])                           # an OBJ repeats a vertex along a texture seam
    first = np.nonzero((F == 0).any(axis=1))[0][0]
    for faces, want in ((F, 0), (ONE, 3)):
        f_seam = faces.copy()
        f_seam[first] = np.where(f_seam[first] == 0, len(V), f_seam[first])
        assert mixed_winding_edges(seam, f_seam) == want  # the copies are merged before the edges are counted


def test_nan_vertices_are_refused_by_add_mesh():
    with pytest.raises(alloy.SceneError) as err:
        _scene().add_mesh("m", vertices=NAN_V, faces=TET_F, pos=(0, 0, 0.2))
    assert err.value.field == "vertices" and "non-finite" in str(err.value) and err.value.fix


@pytest.mark.parametrize("source", ["arrays", "file"])
def test_a_dynamic_mesh_of_no_volume_wound_both_ways_is_refused_with_a_fix(tmp_path, source):
    scene = _scene()
    if source == "arrays":
        scene.add_mesh("m", vertices=V, faces=MIXED, pos=(0, 0, 0.2))
    else:
        (tmp_path / "mixed.obj").write_text(_obj(V, MIXED))
        scene.add_mesh("m", tmp_path / "mixed.obj", pos=(0, 0, 0.2))
    with pytest.raises(alloy.SceneError) as err:
        alloy.build(scene, device="cpu")
    assert "wound inside out" in str(err.value) and "counter-clockwise" in err.value.fix
    assert err.value.field == ("faces" if source == "arrays" else "path")


def test_mass_given_over_a_mesh_of_no_volume_raises_as_before():
    scene = _scene()
    scene.add_mesh("m", vertices=V, faces=MIXED, pos=(0, 0, 0.2), mass=0.2)
    with pytest.raises(alloy.SceneError, match="cannot derive a density from mass"):
        alloy.build(scene, device="cpu")


@pytest.mark.compile_heavy
@pytest.mark.parametrize("faces, kw", [(MIXED, {"static": True}), (MIXED, {"kinematic": True}),
                                       (ONE, {"mass": 0.2}), (ONE, {})],
                         ids=["no-volume static", "no-volume kinematic", "mixed with mass=", "mixed, volume > 0"])
def test_meshes_that_built_before_still_build_and_step(faces, kw):
    scene = _scene()
    scene.add_mesh("m", vertices=V, faces=faces, pos=(0, 0, 0.3), **kw)
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.5, 0, 0.1), mass=0.1)
    sim = alloy.build(scene, device="cpu")
    sim.step(10)
    assert sim.check()["ok"]
    if "mass" in kw:
        assert float(sim.engine.model.body_mass.numpy()[sim.index.body("m")]) == pytest.approx(0.2, rel=1e-5)


def test_nan_vertices_in_a_mesh_file_fail_the_card_check_with_a_fix(tmp_path):
    (tmp_path / "nan.obj").write_text(_obj(NAN_V, TET_F))
    scene = _scene()
    scene.add_mesh("m", tmp_path / "nan.obj", pos=(0, 0, 0.2))
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(scene, device="cpu")
    assert "non-finite vertex" in str(err.value) and "finite" in err.value.fix


@pytest.mark.compile_heavy
def test_an_object_card_whose_mesh_gives_a_dynamic_body_no_mass_fails_the_build_with_a_fix(tmp_path):
    scene = _scene(roots=[str(_card_root(tmp_path / "mine", "mixed_box", V, MIXED))])
    scene.add_object("m", "mixed_box", pos=(0, 0, 0.2))
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(scene, device="cpu")
    assert "encloses no volume" in str(err.value) and "counter-clockwise" in err.value.fix


@pytest.mark.compile_heavy
def test_a_cloth_sheet_mesh_of_no_volume_builds_as_before(tmp_path):
    v, f = _sheet()
    scene = _scene(roots=[str(_card_root(tmp_path / "mine", "sheet", v, f, "cotton_woven", THICKNESS))])
    scene.add_object("sheet", "sheet", pos=(0, 0, 0.2))
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["sheet"]["route"] == "cloth_vbd"
    sim.step(5)
    assert sim.check()["ok"]
