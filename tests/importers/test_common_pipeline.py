# SPDX-License-Identifier: Apache-2.0
"""alloy.importers.common mesh pipeline (needs the ``import`` extra) and the drop test (Newton, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alloy.importers.common.manifold_fix import is_clean
from alloy.importers.common.objio import read_hulls_obj, sha256_path

CUBE_V = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (0.0, 0.1)])
CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                   [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def test_process_torus_end_to_end(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("fast_simplification")
    pytest.importorskip("coacd")
    pytest.importorskip("PIL")
    from alloy.importers.common import hulls as hl  # noqa: PLC0415
    from alloy.importers.common.process import ProcessOptions, process  # noqa: PLC0415

    ring = trimesh.creation.torus(major_radius=40.0, minor_radius=10.0, major_sections=48, minor_sections=16)
    ring.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # axis Z -> Y: a Y-up file
    src = tmp_path / "ring.obj"
    ring.export(src)                                             # millimetres
    opts = ProcessOptions(scale=1e-3, up_axis="y", max_faces=1000, hulls=hl.HullParams(threshold=0.1, max_parts=16))
    r = process(src, tmp_path / "out", "imp_ring", opts)
    out = tmp_path / "out"
    assert r["ok"] and r["watertight"] and r["faces"] <= 1500
    assert r["volume"] == pytest.approx(ring.volume * 1e-9, rel=0.03)
    assert 2 <= r["n_hulls"] <= 16 and r["hulls"]["max_verts"] <= 64
    sha, hulls = read_hulls_obj(out / "physics.hulls.obj")
    assert sha == sha256_path(out / "physics.obj") == r["files"]["physics.obj"]
    phys = trimesh.load(out / "physics.obj", process=False)
    assert is_clean(np.asarray(phys.faces))
    assert np.ptp(np.asarray(phys.vertices)[:, 2]) < 0.025      # the ring's axis is now +Z (thickness 20 mm)
    assert json.loads((out / "result.json").read_text())["files"] == r["files"]
    assert (out / "visual.glb").stat().st_size > 0 and (out / "thumb.png").stat().st_size > 0


def test_drop_test_cube_rests_on_the_plane():
    pytest.importorskip("alloy._vendor.newton")
    from alloy.importers.common.droptest import REST_PEN_TOL, drop_test  # noqa: PLC0415

    r = drop_test([(CUBE_V, CUBE_F)], mass=0.2, mu=0.5, restitution=0.3, duration=0.8)
    assert r.ok and r.settled, r.reason
    assert r.mass == pytest.approx(0.2, rel=1e-5)
    assert r.min_clearance > -1e-3                               # stiff plane: no deep sink at impact
    assert r.rest_clearance > -REST_PEN_TOL


def _cube_library(root):
    """A one-object import library: a 10 cm cube card, its effective material, mesh, hull sidecar and manifest."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.importers.common import cards as cw  # noqa: PLC0415
    from alloy.importers.common import manifest as mf  # noqa: PLC0415
    from alloy.importers.common.objio import hulls_obj_text, mesh_obj_text, sha256_bytes  # noqa: PLC0415

    mesh = mesh_obj_text(CUBE_V, CUBE_F)
    d = root / "meshes" / "imp_cube"
    d.mkdir(parents=True)
    (d / "physics.obj").write_text(mesh)
    (d / "physics.hulls.obj").write_text(hulls_obj_text([(CUBE_V, CUBE_F)], sha256_bytes(mesh.encode())))
    spec = cw.CardSpec("imp_cube", "abs", "test", cw.MassSpec("mass", 0.2, (0.19, 0.21), "test scale"), 1e-3,
                       "meshes/imp_cube/physics.obj", sha256_bytes(mesh.encode()), 1, "test card")
    obj, mat = cw.card_texts(spec, CardLibrary().material("abs"))
    (root / "objects").mkdir()
    (root / "materials").mkdir()
    (root / "objects" / "imp_cube.yaml").write_text(obj)
    (root / "materials" / "imp_cube_eff.yaml").write_text(mat)
    man = mf.empty("imp")
    man["objects"]["imp_cube"] = {"physics": {"volume": 1e-3}, "files": {}}
    mf.save(root / "manifest.json", man)


def test_checks_validate_and_drop_a_library(tmp_path):
    pytest.importorskip("alloy._vendor.newton")
    from alloy.importers.common.checks import THIN, check_one, run_checks  # noqa: PLC0415

    _cube_library(tmp_path)
    name, res = check_one((str(tmp_path), "imp_cube", 1e-3))
    assert name == "imp_cube" and res["cards"]["errors"] == []
    assert res["drop"]["ok"], res["drop"]["reason"]
    assert res["drop"]["mass"] == pytest.approx(0.2, rel=1e-5)
    assert run_checks(tmp_path, "imp", workers=1)["drop_failures"] == []
    assert (tmp_path / "checks.json").is_file()
    assert 1e-4 < THIN < 1e-2


def test_batch_processes_a_bare_mesh_url(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    for mod in ("fast_simplification", "coacd", "PIL"):
        pytest.importorskip(mod)
    from alloy.importers.common.batch import run_one  # noqa: PLC0415

    src = tmp_path / "box.glb"
    trimesh.creation.box((0.1, 0.05, 0.02)).export(src)
    r = run_one({"name": "imp_box", "url": src.as_uri(), "mesh": "", "up_axis": "y"}, tmp_path / "out")
    assert r["ok"] and r["n_hulls"] >= 1
    assert r["volume"] == pytest.approx(1e-4, rel=1e-3)


def test_decompose_drops_flat_hulls(monkeypatch):
    pytest.importorskip("trimesh")
    from alloy.importers.common import hulls as hl  # noqa: PLC0415

    import trimesh  # noqa: PLC0415

    box = trimesh.creation.box((0.1, 0.1, 0.1))
    box.apply_translation((0.05, 0.05, 0.05))
    cube = np.array([[x, y, z] for x in (0, 0.1) for y in (0, 0.1) for z in (0, 0.1)])
    flat = np.array([[0, 0, 0.1], [0.1, 0, 0.1], [0, 0.1, 0.1], [0.1, 0.1, 0.1 + 1e-9]])   # 1 nm sliver on top
    mesh, coacd = (np.asarray(box.vertices), np.asarray(box.faces)), hl.HullParams(solid_min=2.0)   # never one hull
    monkeypatch.setattr(hl, "_run_coacd", lambda v, f, p: [(cube, None), (flat, None)])
    hulls, stats = hl.decompose(*mesh, 1e-3, coacd)
    assert len(hulls) == 1 and stats.dropped == 1 and stats.volume_ratio == pytest.approx(1.0)
    assert hl.is_thin(flat) and not hl.is_thin(*hulls[0])
    assert not hl.is_thin(cube * [1, 1, 0.01])                     # a 1 mm plate is thin, not flat: kept
    monkeypatch.setattr(hl, "_run_coacd", lambda v, f, p: [(flat, None)])
    with pytest.raises(hl.ImporterError, match="flat"):
        hl.decompose(*mesh, 1e-3, coacd)
