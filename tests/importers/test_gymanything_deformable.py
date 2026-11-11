# SPDX-License-Identifier: Apache-2.0
"""Cloth sheets, cables, citation grounding and the mesh manifest (CPU)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from alloy.cards import geometry as geo
from alloy.cards import types as ct
from alloy.cards.library import CardLibrary
from alloy.importers.gymanything import citecheck
from alloy.importers.gymanything.deformable import convert_deformable, top_layer
from alloy.importers.gymanything.pipeline import ImportOptions, import_specs
from alloy.importers.gymanything.spec import load_spec
from alloy.testing.imp_gymanything import box_mesh, cited, write_spec

CLOTH = {n: cited(v, "literature-read", "params.py:20") for n, v in
         (("thickness", 3e-4), ("stretch_warp", 900.0), ("stretch_weft", 700.0), ("shear", 20.0),
          ("bend_warp", 2e-6), ("bend_weft", 1e-6))}


def test_flat_slab_cloth_becomes_a_thin_box_with_moduli(tmp_path):
    p = write_spec(tmp_path / "specs", box_mesh((0.2, 0.1, 1.5e-4)), key="sheet", kind="cloth", category="cloth",
                   mass=cited(0.0072, "derived", "params.py:21", "120 g/m2 x 0.06 m2"), cloth=CLOTH)
    rec = convert_deformable(load_spec(p), tmp_path / "out")
    assert rec.collision == "cloth-box" and rec.mass_kg == pytest.approx(0.0072)
    lib = CardLibrary(roots=[tmp_path / "out"], assets=None)
    card, mat = lib.object("gar_testsim_sheet"), lib.material("gar_testsim_sheet")
    assert isinstance(card.shape, ct.Primitive) and card.shape.size == pytest.approx((0.2, 0.1, 1.5e-4))
    assert mat.elastic.youngs.value == pytest.approx(800.0 / 3e-4) and mat.elastic.cloth.shear.value == 20.0
    assert "elastic.poisson" in rec.defaults
    assert [i for i in lib.object_issues("gar_testsim_sheet") if i.level == "error"] == []


def test_top_layer_of_a_slab_is_one_sheet():
    sheet = top_layer(box_mesh((0.2, 0.1, 1e-4)))
    assert len(sheet.faces) == 2 and np.allclose(sheet.vertices[:, 2], 1e-4)


def test_cable_becomes_a_curve_with_ei_and_derived_modulus(tmp_path):
    pts = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    p = write_spec(tmp_path / "specs", box_mesh((0.01, 0.01, 0.01)), key="lead", kind="cable", category="cable",
                   mass=cited(0.02, "datasheet", "params.py:30", "20 g/m"),
                   cable={"points": pts, "radius": cited(0.002, "datasheet", "params.py:31"),
                          "bend_stiffness": cited(1e-4, "estimate", "params.py:32")})
    rec = convert_deformable(load_spec(p), tmp_path / "out")
    lib = CardLibrary(roots=[tmp_path / "out"], assets=None)
    card, mat = lib.object("gar_testsim_lead"), lib.material("gar_testsim_lead")
    assert isinstance(card.shape, ct.CurveSource) and card.features.bend_stiffness.value == 1e-4
    assert rec.mass_kg == pytest.approx(0.02)
    assert mat.elastic.youngs.value == pytest.approx(4e-4 / (math.pi * 0.002**4))


def test_citecheck_finds_numbers_in_scaled_units(tmp_path):
    env = tmp_path / "repo" / "benchmarks/roboworld/environments/testsim"
    env.mkdir(parents=True)
    (env / "params.py").write_text("X = 1\nMUG = dict(mass=.30,  # maker page 300 g\n  h=95)  # mm\n")
    p = write_spec(tmp_path / "specs", box_mesh((0.04, 0.04, 0.0475)), key="mug",
                   mass=cited(0.3, "datasheet", "params.py:2"), friction=cited(0.7, "estimate", "params.py:1"))
    rows = {r["field"]: r for r in citecheck.cite_rows(load_spec(p), tmp_path / "repo")}
    assert rows["mass"]["found"] and rows["mass"]["resolved"]
    assert not rows["friction"]["found"]
    assert citecheck.contains_number(0.095, "h=95  # mm")


def test_import_with_repo_reports_grounding(tmp_path):
    env = tmp_path / "repo" / "benchmarks/roboworld/environments/testsim"
    env.mkdir(parents=True)
    (env / "params.py").write_text("CRATE_KG = 1.2\n")
    write_spec(tmp_path / "specs", box_mesh((0.05, 0.05, 0.05)), key="crate", mass=cited(1.2, "datasheet", "params.py:1"))
    res = import_specs(tmp_path / "specs", tmp_path / "out", opt=ImportOptions(repo=tmp_path / "repo", hulls=False))
    assert res["cites"]["found"] == 1 and res["cites"]["not_found"] == []


def test_open_mesh_falls_back_to_its_convex_hull_with_the_cited_mass(tmp_path):
    from alloy.importers.gymanything.convert import convert_spec

    box = box_mesh((0.05, 0.05, 0.05))
    open_box = geo.MeshData(box.vertices, box.faces[2:])
    p = write_spec(tmp_path / "specs", open_box, key="crate", mass=cited(1.0, "datasheet", "a.py:1"))
    rec = convert_spec(load_spec(p), tmp_path / "out")
    assert rec.collision == "convex" and any("convex hull" in w for w in rec.warnings)
    res = import_specs(tmp_path / "specs", tmp_path / "out2", opt=ImportOptions(hulls=False))
    assert res["checked_ok"] == 1
