# SPDX-License-Identifier: Apache-2.0
"""A TetSource card's relative ``from_mesh`` resolves like every other card asset path (CPU).

WP4's report: ``SceneAssembler._with_asset_paths`` made only ``shape.path`` absolute, so ``from_mesh`` stayed relative
to the working directory and a card whose mesh sat in its own library root failed to build ("mesh meshes/ball.obj not
found"). It now resolves against the library roots, then the assets root, as mesh, SDF, MJCF and tet ``.npz`` paths
do. A path found in none of them is refused by card validation, naming the field, the path and the roots searched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pytetwild")
trimesh = pytest.importorskip("trimesh")

import alloy  # noqa: E402
from alloy.cards.library import CardLibrary, default_root  # noqa: E402
from alloy.scene.assembler import SceneAssembler  # noqa: E402
from alloy.scene.emitters import deformable as D  # noqa: E402

MESH = "meshes/ball.obj"


def _root(base: Path, write_mesh: bool = True) -> Path:
    """A card root holding ``tet_ball``: a silicone TetSource whose from_mesh is root-relative."""
    root = base / "cards"
    (root / "objects").mkdir(parents=True)
    if write_mesh:
        (root / MESH).parent.mkdir(parents=True)
        trimesh.creation.icosphere(subdivisions=2, radius=0.03).export(root / MESH)
    (root / "objects" / "tet_ball.yaml").write_text(
        "schema: alloy.object/1\nname: tet_ball\nshape:\n  tet:\n"
        f"    from_mesh: {MESH}\n    edge_length: 0.01\nmaterial: silicone_40a\n")
    return root


def _scene(root: Path) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, roots=[str(root)])
    scene.add_ground()
    scene.add_object("ball", "tet_ball", pos=(0.0, 0.0, 0.04))
    return scene


@pytest.fixture
def elsewhere(tmp_path, monkeypatch) -> Path:
    """Work from a folder that does not hold the mesh, with a fresh fTetWild cache."""
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("ALLOY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "_TETS", {})
    return tmp_path


def test_the_assembler_resolves_from_mesh_against_the_card_root(elsewhere):
    root = _root(elsewhere)
    lib = CardLibrary(roots=[root, default_root()])
    obj = SceneAssembler(lib)._with_asset_paths(lib.resolve(lib.object("tet_ball")))
    assert Path(obj.card.shape.from_mesh) == (root / MESH).resolve()
    assert obj.card.shape.path is None


def test_a_root_relative_from_mesh_builds_from_another_working_directory(elsewhere):
    sim = alloy.build(_scene(_root(elsewhere)), device="cpu")
    assert sim.routes()["ball"]["route"] == "soft_vbd"
    assert sim.engine.model.tet_count > 0


def test_a_from_mesh_found_nowhere_names_the_field_the_path_and_the_roots(elsewhere):
    root = _root(elsewhere, write_mesh=False)
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(_scene(root), device="cpu")
    msg = str(err.value)
    assert "shape.from_mesh" in msg and f"file not found: {MESH!r}" in msg and str(root.resolve()) in msg
