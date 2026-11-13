# SPDX-License-Identifier: Apache-2.0
"""The ``import`` extra carries what the mesh pipeline needs (Gemini audit S2-005): fast-simplification is in it,
a mesh over the face limit is simplified, and a missing module is reported with the real package name instead of
being swallowed as "kept"."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from alloy.importers.common import meshops
from alloy.importers.common.errors import MissingExtraError, pip_name

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parents[2]


def test_the_import_extra_lists_every_module_the_mesh_pipeline_needs():
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["import"]
    names = {r.split("==")[0].split(">")[0].strip().lower() for r in extra}
    assert {"trimesh", "manifold3d", "scikit-image", "fast-simplification"} <= names


def test_a_mesh_over_the_limit_is_simplified():
    pytest.importorskip("fast_simplification")
    for subdivisions, faces in ((4, 5120), (5, 20480)):
        ball = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.03)
        small, info = meshops.physics_mesh(ball, max_faces=5000)
        assert info.source_faces == faces and info.simplify.startswith("qem:") and len(small.faces) <= 5200
        assert small.is_watertight and abs(small.volume / ball.volume - 1.0) < 0.02


def test_without_the_module_a_mid_size_mesh_is_kept_with_one_warning_and_a_big_one_raises(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "fast_simplification", None)           # import raises ImportError
    monkeypatch.setattr(meshops, "_WARNED", [])
    ball = trimesh.creation.icosphere(subdivisions=4, radius=0.03)           # 5,120 faces: usable as it is
    with caplog.at_level("WARNING", logger="alloy.importers"):
        _, info = meshops.physics_mesh(ball, max_faces=5000)
        meshops.physics_mesh(ball, max_faces=5000)
    said = [r.getMessage() for r in caplog.records if "kept unsimplified" in r.getMessage()]
    assert info.simplify == "kept" and info.faces == 5120 and len(said) == 1
    assert "5120 faces" in said[0] and "pip install 'alloy-engine[import]'" in said[0]
    tiny = trimesh.creation.icosphere(subdivisions=5, radius=0.004)          # 20,480 faces, 8 mm: the voxel remesh
    _, info = meshops.physics_mesh(tiny, max_faces=5000)                     # is small enough without simplifying
    assert info.method == "voxel" and info.simplify == "none" and info.faces <= 5000
    big = trimesh.creation.icosphere(subdivisions=5, radius=0.03)            # 20,480 faces: over KEEP_FACES
    with pytest.raises(MissingExtraError) as e:
        meshops.physics_mesh(big, max_faces=5000)
    assert "pip install 'alloy-engine[import]'" in e.value.message and "'alloy[import]'" not in e.value.message
    assert "pip install fast-simplification" in e.value.message
    assert pip_name("skimage") == "scikit-image" and pip_name("trimesh") == "trimesh"


def test_create_turns_the_missing_module_into_an_error_with_the_install_command(tmp_path, monkeypatch):
    from alloy.create import create
    from alloy.create.request import CreateError

    monkeypatch.setitem(sys.modules, "fast_simplification", None)
    src = tmp_path / "ball_20k.obj"
    trimesh.creation.icosphere(subdivisions=5, radius=30.0).export(src)
    with pytest.raises(CreateError) as e:
        create(mesh=str(src), name="ball_20k", units="mm", material="abs", check=False, out=str(tmp_path / "out"))
    assert "fast_simplification" in e.value.message and "alloy-engine[import]" in e.value.fix
