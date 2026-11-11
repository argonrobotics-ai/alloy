# SPDX-License-Identifier: Apache-2.0
"""Objaverse importer, re-centring: a mesh-pipeline folder whose object lies far from its origin (as some uploads do)
is moved so the physics mesh's bounding-box centre is the origin, with the hull sidecar, the visual and result.json
moved and re-hashed to match, and a second run changes nothing."""

from __future__ import annotations

import doctest
import hashlib
import json

import numpy as np
import pytest

from alloy.importers.common import objio
from alloy.importers.common.errors import ImporterError
from alloy.importers.objaverse import __main__ as cli
from alloy.importers.objaverse import recentre as R

FAR = np.array([-434.4, -616.5, 0.03])          # the worst upload's offset (m), cassette-sized box around it
HALF = np.array([0.05, 0.0025, 0.033])


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * HALF + FAR
    f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                  (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return v, f


def _folder(tmp_path, with_visual: bool = True):
    trimesh = pytest.importorskip("trimesh")
    from alloy.importers.common.visual import export_glb

    d = tmp_path / "out" / "objv_cassette_e8498e35"
    d.mkdir(parents=True)
    v, f = _box()
    (d / R.PHYSICS).write_text(objio.mesh_obj_text(v, f, header="objv_cassette_e8498e35: physics mesh"))
    (d / R.HULLS).write_text(objio.hulls_obj_text([(v[:4], np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])),
                                                   (v[4:], np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]))],
                                                  _sha(d / R.PHYSICS), header="2 CoACD hulls"))
    files = {R.PHYSICS: _sha(d / R.PHYSICS), R.HULLS: _sha(d / R.HULLS)}
    if with_visual:
        export_glb(trimesh.Trimesh(v, f), d / R.VISUAL)                   # body frame -> glTF +Y up
        files[R.VISUAL] = _sha(d / R.VISUAL)
    result = {"name": d.name, "files": files, "volume": float(np.prod(2 * HALF)),
              "mass_props": {"center_mass": FAR.tolist(), "bounds": [(FAR - HALF).tolist(), (FAR + HALF).tolist()],
                             "extents": (2 * HALF).tolist()}}
    (d / R.RESULT).write_text(json.dumps(result))
    return d


def test_docstring_examples_run():
    assert doctest.testmod(R).failed == 0


def test_far_object_is_moved_to_its_bounding_box_centre(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    d = _folder(tmp_path)
    assert not R.is_centred(json.loads((d / R.RESULT).read_text()))
    out = R.recentre_dir(d)
    _, v, f = R._read_mesh_obj((d / R.PHYSICS).read_text())
    assert np.abs(R.bbox_centre(v)).max() < 1e-7 and np.allclose(v.max(0), HALF, rtol=0, atol=2e-4)
    assert len(f) == 12 and "origin: physics-mesh bounding-box centre" in (d / R.PHYSICS).read_text()
    sha, hulls = objio.read_hulls_obj(d / R.HULLS)
    assert sha == _sha(d / R.PHYSICS) and np.abs(np.vstack([h for h, _ in hulls]) - v).max() < 1e-9
    glb = trimesh.load(d / R.VISUAL)                                       # glTF frame: (x, z, -y) of the body
    assert np.allclose(glb.bounds, [[-HALF[0], -HALF[2], -HALF[1]], [HALF[0], HALF[2], HALF[1]]], rtol=0, atol=2e-4)
    assert R.is_centred(out) and np.allclose(out["origin_shift"], -FAR, rtol=0, atol=2e-4)
    assert np.allclose(out["mass_props"]["center_mass"], 0.0, rtol=0, atol=2e-4)
    assert all(out["files"][k] == _sha(d / k) and out["sizes"][k] == (d / k).stat().st_size
               for k in (R.PHYSICS, R.HULLS, R.VISUAL))
    assert json.loads((d / R.RESULT).read_text()) == out


def test_second_run_changes_nothing(tmp_path):
    d = _folder(tmp_path)
    R.recentre_dir(d)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    R.recentre_dir(d)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_folder_without_a_visual_is_moved(tmp_path):
    d = _folder(tmp_path, with_visual=False)
    out = R.recentre_dir(d)
    assert R.is_centred(out) and not (d / R.VISUAL).exists()


def test_files_changed_since_result_json_are_refused(tmp_path):
    d = _folder(tmp_path)
    (d / R.VISUAL).write_bytes((d / R.VISUAL).read_bytes() + b"\0" * 4)   # as if an earlier run stopped half-way
    with pytest.raises(ImporterError, match="differs from result.json"):
        R.recentre_dir(d)


def test_glb_root_matrix_and_trs_nodes_take_the_shift():
    gltf = {"scenes": [{"nodes": [0, 1]}], "nodes": [
        {"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10.0, 20.0, 30.0, 1]},
        {"translation": [1.0, 2.0, 3.0], "children": [2]}, {"translation": [5.0, 5.0, 5.0]}]}
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    data = b"glTF" + (2).to_bytes(4, "little") + (20 + len(js)).to_bytes(4, "little") + \
        len(js).to_bytes(4, "little") + b"JSON" + js
    out = R.translate_glb(data, np.array([-1.0, -2.0, -3.0]))
    nodes = json.loads(out[20:20 + int.from_bytes(out[12:16], "little")])["nodes"]
    assert nodes[0]["matrix"][12:15] == [9.0, 18.0, 27.0] and nodes[1]["translation"] == [0.0, 0.0, 0.0]
    assert nodes[2]["translation"] == [5.0, 5.0, 5.0] and int.from_bytes(out[8:12], "little") == len(out)


def test_cli_recentres_every_folder_and_writes_result_rows(tmp_path, capsys):
    d = _folder(tmp_path)
    rows = tmp_path / "results.jsonl"
    assert cli.main(["recentre", "--meshes", str(d.parent), "--results-out", str(rows)]) == 0
    assert json.loads(capsys.readouterr().out)["moved"] == 1
    assert [json.loads(r)["files"] for r in rows.read_text().splitlines()] == [
        json.loads((d / R.RESULT).read_text())["files"]]
