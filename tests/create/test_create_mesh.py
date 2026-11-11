# SPDX-License-Identifier: Apache-2.0
"""create end to end on CPU: a mesh in millimetres becomes cards with a watertight mesh, hulls, an ESTIMATE
mass with a range, library friction, a passing drop test and a 'verify' list; a name the library uses is refused;
an MJCF file goes through alloy.io. Needs the importers' extra (trimesh, coacd, fast_simplification)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("coacd")
pytest.importorskip("fast_simplification")

from alloy.cards.geometry import MeshData, write_obj  # noqa: E402
from alloy.cards.library import default_root  # noqa: E402
from alloy.create import cli  # noqa: E402
from alloy.create.build import create  # noqa: E402
from alloy.create.request import CreateError  # noqa: E402


def _box_mm(path, size=(60.0, 40.0, 30.0)):
    hx, hy, hz = (s / 2 for s in size)
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=np.float64)
    f = np.array([[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]], dtype=np.int64)
    write_obj(path, MeshData(v, f))
    return path


def test_mesh_to_checked_card(tmp_path):
    src = _box_mm(tmp_path / "block.obj")
    r = create(mesh=str(src), name="test_block", material="wood", out=str(tmp_path / "lib"), device="cpu")
    d = r.to_dict()
    assert r.ok, r.summary()
    assert (tmp_path / "lib/objects/test_block.yaml").is_file()
    assert (tmp_path / "lib/materials/test_block_eff.yaml").is_file()
    assert (tmp_path / "lib/meshes/test_block/physics.hulls.obj").is_file()
    rows = {v["field"]: v for v in d["values"]}
    assert rows["size"]["value"] == pytest.approx(0.06) and rows["size"]["tag"] == "ESTIMATE"
    assert rows["mass"]["tag"] == "ESTIMATE" and rows["mass"]["lo"] < rows["mass"]["value"] < rows["mass"]["hi"]
    assert rows["mu_static"]["tag"] == "LIBRARY"
    assert "mass" in d["estimates"] and any(s.startswith("mass:") for s in d["verify"])
    assert d["check"]["ok"] and d["check"]["checks"]["rest_drift"] in ("pass", "warn")
    assert "ESTIMATE" in (tmp_path / "lib/materials/test_block_eff.yaml").read_text()


def test_given_mass_and_size_are_kept(tmp_path):
    src = _box_mm(tmp_path / "b.obj")
    r = create(mesh=str(src), size=0.12, mass=0.5, mass_source="kitchen scale", out=str(tmp_path / "lib"),
               check=False)
    rows = {v["field"]: v for v in r.to_dict()["values"]}
    assert rows["size"]["tag"] == "USER" and rows["mass"]["tag"] == "CITED" and rows["mass"]["value"] == 0.5
    assert "fit" in (tmp_path / "lib/materials/b_eff.yaml").read_text()


def test_a_library_name_is_refused(tmp_path):
    with pytest.raises(CreateError) as e:
        create(mesh=str(_box_mm(tmp_path / "x.obj")), name="cube_beech_30", out=str(tmp_path / "lib"), check=False)
    assert "name=" in e.value.fix


def test_mjcf_goes_through_alloy_io(tmp_path):
    r = create(mesh=str(default_root() / "mjcf" / "hinged_box.xml"), name="my_box", out=str(tmp_path / "lib"),
               device="cpu")
    assert r.ok and r.objects and all(o.startswith("my_box") for o in r.objects)


def test_cli_json_and_exit_codes(tmp_path, capfd):
    src = _box_mm(tmp_path / "c.obj")
    assert cli.main(["--mesh", str(src), "--out", str(tmp_path / "lib"), "--no-check"]) == 0
    assert json.loads(capfd.readouterr().out)["name"] == "c"
    assert cli.main(["--mesh", str(src), "--text", "a box"]) == 2
    assert json.loads(capfd.readouterr().out)["error"] == "CreateError"
