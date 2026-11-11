# SPDX-License-Identifier: Apache-2.0
"""Spec -> Alloy cards -> checks, on tiny synthetic meshes (CPU; Newton's inertia routine on Warp CPU)."""

from __future__ import annotations

import json
import math

import pytest

from alloy.cards.library import CardLibrary
from alloy.importers.gymanything.convert import ConvertError, convert_spec
from alloy.importers.gymanything.pipeline import ImportOptions, import_specs

NO_HULLS = ImportOptions(hulls=False)
from alloy.importers.gymanything.spec import load_spec
from alloy.testing.imp_gymanything import box_mesh, cited, ring_mesh, tube_poly, write_spec


def test_box_with_cited_mass_becomes_valid_cards(tmp_path):
    p = write_spec(tmp_path / "specs", box_mesh((0.1, 0.05, 0.02)), key="crate", category="container",
                   mass=cited(0.8, "datasheet", "params.py:7", "maker sheet 800 g"),
                   friction=cited(0.45, "estimate", "params.py:9"), friction_partner="steel table",
                   dims=[{"name": "length", "value_m": 0.2, "measure": "x", "cite": "params.py:7"}])
    rec = convert_spec(load_spec(p), tmp_path / "out")
    assert rec.card == "gar_testsim_crate" and rec.collision == "convex"
    assert rec.mass_kg == pytest.approx(0.8)
    assert rec.defaults == ["surface.restitution"]
    lib = CardLibrary(roots=[tmp_path / "out"], assets=None)
    mat = lib.material("gar_testsim_crate")
    assert mat.density.source == "fit" and mat.density.value == pytest.approx(0.8 / (0.2 * 0.1 * 0.04))
    assert mat.surface.mu_static.value == 0.45 and "steel table" in mat.surface.mu_static.note
    assert [i for i in lib.object_issues("gar_testsim_crate") if i.level == "error"] == []


def test_thin_tube_gets_sdf_collision_and_min_wall(tmp_path):
    p = write_spec(tmp_path / "specs", ring_mesh(tube_poly(0.030, 0.0308, 0.1)), key="sleeve",
                   material={"name": "aluminium", "density": cited(2700.0, "literature", "params.py:3")})
    rec = convert_spec(load_spec(p), tmp_path / "out")
    assert rec.collision == "sdf" and rec.convex_ratio > 10
    assert any("min_wall" in w for w in rec.warnings)
    card = CardLibrary(roots=[tmp_path / "out"], assets=None).object("gar_testsim_sleeve")
    assert card.features.min_wall is not None and card.features.min_wall.value == pytest.approx(0.0008, rel=0.1)


def test_a_probe_reading_on_a_mesh_defect_is_re_measured(tmp_path, monkeypatch):
    """A wall probe reading under 0.3 mm on a solid block (a mesh defect) is re-measured by common.walls: no thin wall,
    so min_wall is declared just under the validator's threshold and the note gives the probe's reading. A cited wall
    (a needle) keeps the probe's value."""
    from alloy.importers.gymanything import meshes
    from alloy.importers.gymanything.convert import THIN_LIMIT

    monkeypatch.setattr(meshes.geo, "wall_thickness", lambda mesh, *a, **k: 1e-6)
    convert_spec(load_spec(write_spec(tmp_path / "specs", box_mesh((0.04, 0.03, 0.02)), key="block")), tmp_path / "out")
    p = write_spec(tmp_path / "specs", box_mesh((0.04, 0.03, 0.02)), key="needle",
                   thickness=cited(1e-6, "datasheet", "params.py:5"))
    convert_spec(load_spec(p), tmp_path / "out")
    lib = CardLibrary(roots=[tmp_path / "out"], assets=None)
    block, needle = (lib.object(f"gar_testsim_{k}").features.min_wall for k in ("block", "needle"))
    assert block.value == math.nextafter(THIN_LIMIT, 0.0) and "probe read 1e-06 m on remesh defects" in block.note
    assert needle.value == 1e-6 and needle.note.startswith("thinnest wall ray-cast on the imported mesh; cited wall")


def test_impossible_density_says_what_to_change(tmp_path):
    p = write_spec(tmp_path / "specs", box_mesh((0.5, 0.5, 0.5)), key="bag", mass=cited(0.005, "estimate", "a.py:1"))
    with pytest.raises(ConvertError, match="export the walls"):
        convert_spec(load_spec(p), tmp_path / "out")


def test_import_specs_writes_index_and_flags_wrong_dims(tmp_path):
    specs = tmp_path / "specs"
    write_spec(specs, box_mesh((0.05, 0.05, 0.05)), key="good", mass=cited(1.0, "datasheet", "a.py:1"),
               dims=[{"name": "edge", "value_m": 0.1, "measure": "z", "cite": "a.py:2"}])
    write_spec(specs, box_mesh((0.05, 0.05, 0.05)), key="badsize", mass=cited(1.0, "datasheet", "a.py:1"),
               dims=[{"name": "edge", "value_m": 0.2, "measure": "z", "cite": "a.py:2"}])
    res = import_specs(specs, tmp_path / "out", opt=NO_HULLS)
    assert res["converted"] == 2 and res["checked_ok"] == 1 and not res["failed"]
    assert "edge" in res["check_failures"]["gar_testsim_badsize"]
    index = json.loads((tmp_path / "out" / "index" / "testsim.json").read_text())
    good = next(o for o in index["objects"] if o["key"] == "good")
    assert good["check"]["mass_ok"] and good["check"]["newton_mass_kg"] == pytest.approx(1.0, rel=1e-3)


def test_rerun_removes_stale_cards(tmp_path):
    specs, out = tmp_path / "specs", tmp_path / "out"
    old = write_spec(specs, box_mesh((0.05, 0.05, 0.05)), key="old", mass=cited(1.0, "datasheet", "a.py:1"))
    import_specs(specs, out, opt=NO_HULLS)
    old.unlink()
    write_spec(specs, box_mesh((0.05, 0.05, 0.05)), key="new", mass=cited(1.0, "datasheet", "a.py:1"))
    import_specs(specs, out, opt=NO_HULLS)
    assert not (out / "materials" / "testsim" / "gar_testsim_old.yaml").exists()
    assert not (out / "meshes" / "gar_testsim_old").exists()
    assert (out / "objects" / "testsim" / "gar_testsim_new.yaml").exists()


def test_import_writes_shared_manifest_queue_and_attribution(tmp_path):
    write_spec(tmp_path / "specs", box_mesh((0.05, 0.05, 0.05)), key="crate", name="Crate",
               mass=cited(1.0, "estimate", "a.py:1", "guess"), friction=cited(0.4, "datasheet", "a.py:2"))
    res = import_specs(tmp_path / "specs", tmp_path / "out", opt=NO_HULLS)
    out = tmp_path / "out"
    man = json.loads((out / "manifest.json").read_text())
    entry = man["objects"]["gar_testsim_crate"]
    assert man["schema"] == "alloy.import_manifest/1" and entry["licence"] == "MIT"
    assert entry["files"]["physics.obj"]["sha256"] and entry["card"] == "objects/testsim/gar_testsim_crate.yaml"
    rows = [json.loads(line) for line in (out / "queue.jsonl").read_text().splitlines()]
    assert {r["field"] for r in rows} == {"material.density", "material.surface.restitution"}
    assert "cmu-l3" in (out / "ATTRIBUTION.md").read_text() and res["queue_rows"] == 2


def test_hull_sidecars_pick_decompose_for_a_tube(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("coacd")
    from alloy.testing.imp_gymanything import ring_mesh, tube_poly

    write_spec(tmp_path / "specs", ring_mesh(tube_poly(0.02, 0.026, 0.06)), key="sleeve",
               mass=cited(0.05, "datasheet", "a.py:1"))
    write_spec(tmp_path / "specs", box_mesh((0.05, 0.05, 0.05)), key="crate", mass=cited(1.0, "datasheet", "a.py:1"))
    res = import_specs(tmp_path / "specs", tmp_path / "out", opt=ImportOptions(workers=1))
    lib = CardLibrary(roots=[tmp_path / "out"], assets=None)
    assert lib.object("gar_testsim_crate").shape.collision == "convex"
    assert lib.object("gar_testsim_sleeve").shape.collision in ("decompose", "sdf")
    assert (tmp_path / "out" / "meshes" / "gar_testsim_crate" / "physics.hulls.obj").is_file()
    assert res["hull_failures"] == {} and res["checked_ok"] == 2
