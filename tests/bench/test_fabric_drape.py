# SPDX-License-Identifier: Apache-2.0
"""D03M measured-fabric drape (SCR-WP2-14): the pack against the paper's printed values, the pre-registered card
mapping (arm L reproduces the shipped KES card), the scorer, the rig layout, and a CPU build of the rig."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from bench.scenarios import fabric_drape as FD
from bench.scenarios import fabric_drape_score as FS

PACK = json.loads((FD.PACKS / "fabric_drape_petrak2021.json").read_text())
LIBRARY = Path(__file__).resolve().parents[2] / "library" / "materials"


def test_pack_matches_the_papers_printed_values():
    """Every SI value is the printed one times its stated conversion (Petrak et al. 2021, Tables 1-3)."""
    assert PACK["rig"]["specimen_diameter_m"] == 0.30 and PACK["rig"]["support_diameter_m"] == 0.18
    for f in PACK["fabrics"]:
        v = f["verbatim"]
        assert f["areal_mass_kg_m2"] == pytest.approx(v["Weight [g/m2] (Table 1)"] * 1e-3, rel=1e-12)
        assert f["thickness_T0_m"] == pytest.approx(v["T0 [mm]"] * 1e-3, rel=1e-12)
        assert f["bending_warp_N_m"] == pytest.approx(v["B-1 [cNcm]"] * 1e-4, rel=1e-12)
        assert f["bending_weft_N_m"] == pytest.approx(v["B-2 [cNcm]"] * 1e-4, rel=1e-12)
        assert f["shear_G_N_m"] == pytest.approx(v["G [cN/(cm°)−1]"] * 180.0 / math.pi, rel=1e-12)
        assert f["tensile_strain_warp"] == pytest.approx(v["EM-1 [%]"] / 100.0, rel=1e-12)
        assert f["dc_measured"] == v["Kd (Table 3)"]
    assert [f["dc_measured"] for f in PACK["fabrics"]] == [0.222, 0.319, 0.647]


def test_arm_L_reproduces_the_shipped_kes_card():
    """The headline arm is the library's KES convention: cloth_cotton_plain's youngs from its B at its thickness."""
    card = yaml.safe_load((LIBRARY / "cloth_cotton_plain.yaml").read_text())
    b = card["elastic"]["bending"]["value"]
    fab = {"bending_warp_N_m": b, "bending_weft_N_m": b, "thickness_T0_m": 0.975e-3}
    assert FD.youngs_L(fab) == pytest.approx(card["elastic"]["youngs"]["value"], rel=1e-4)
    assert "cloth" not in card["elastic"]          # isotropic: D16 routes it to CLOTH_VBD


@pytest.mark.parametrize("fab", PACK["fabrics"], ids=lambda f: f["id"])
def test_cards_carry_the_inputs(fab):
    h = FD.thickness(fab)
    for arm in FD.ARMS:
        card = FD.material_card(fab, arm)
        el = card["elastic"]
        assert card["density"]["value"] * h == pytest.approx(fab["areal_mass_kg_m2"], rel=1e-12)
        assert el["bending"]["value"] == pytest.approx(0.5 * (fab["bending_warp_N_m"] + fab["bending_weft_N_m"]))
        assert ("cloth" in el) == (arm in ("A", "O"))          # A: Style3D; O: the orthotropic VBD membrane
    eh = {arm: FD.material_card(fab, arm)["elastic"]["youngs"]["value"] * h for arm in "LTS"}
    assert eh["L"] == pytest.approx(FD.youngs_L(fab) * h)
    assert eh["T"] == pytest.approx(math.sqrt(FD.tensile(fab, "warp") * FD.tensile(fab, "weft")))
    assert eh["S"] / (2.0 * (1.0 + FD.NU)) == pytest.approx(fab["shear_G_N_m"])     # VBD's mu is KES G
    assert FD.bending(fab, "geo") <= FD.bending(fab, "mean")


def test_scorer_reads_flat_hanging_and_nodes():
    rest, tris = FD.specimen_mesh(0.15, 0.012, seed=1)
    sc = FS.Scorer(rest, tris, 0.15, 0.09)
    assert sc.dc(rest) == pytest.approx(1.0, abs=1e-9)
    shrunk = rest * np.array([0.09 / 0.15, 0.09 / 0.15, 1.0])
    assert sc.dc(shrunk) == pytest.approx(0.0, abs=0.02)
    ang = np.arctan2(rest[sc.rim, 1], rest[sc.rim, 0])
    lobed = rest.copy()
    lobed[sc.rim, :2] *= (1.0 + 0.1 * np.cos(7 * ang))[:, None]
    assert sc.nodes(lobed) == 7
    assert FS.max_strain_pct(rest, rest * 1.01, sc.edges) == pytest.approx(1.0, rel=1e-6)


def test_the_mesh_guard_refuses_edges_shorter_than_the_sheet():
    """A mesh with an edge shorter than the sheet thickness starts in self-contact and blows up (the R6 mesher's ring
    edges at a 5 mm cell: 1.1 mm against 2 mm); the grid disc and the 8 mm R6 mesh (seed 0) pass."""
    rigs = FD.rigs_for(PACK, "L", "F4")
    with pytest.raises(ValueError, match="shortest edge"):
        FD.check_mesh(FD.specimen_mesh(0.15, 0.005, 0), rigs)
    assert FD.check_mesh(FD.specimen_mesh(0.15, 0.008, 0), rigs) >= FD.H_MIN
    grid = FD.specimen_mesh(0.15, 0.005, 0, "grid")
    assert FD.check_mesh(grid, rigs) >= 0.8 * 0.005
    sc = FS.Scorer(grid[0], grid[1], 0.15, 0.09)
    assert len(sc.rim) > 100 and sc.dc(grid[0]) == pytest.approx(1.0, abs=1e-9)


def test_rigs_for_lays_out_each_fabric_and_arm():
    rigs = FD.rigs_for(PACK, "L,S,L:geo")
    assert [(r.fab["id"][-2:], r.arm, r.bend) for r in rigs[:3]] == [("F1", "L", "mean"), ("F1", "S", "mean"),
                                                                    ("F1", "L", "geo")]
    xy = np.array([r.xy for r in rigs])
    gaps = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 1e9
    assert len(rigs) == 9 and gaps.min() >= FD.RIG_SPACING - 1e-12
    assert len(FD.rigs_for(PACK, "A", "F9")) == 1
    with pytest.raises(ValueError, match="Style3D"):
        FD.rigs_for(PACK, "L,A", "F9")


@pytest.mark.slow  # one CPU build and 0.3 s of drape (about 40 s)
@pytest.mark.timeout(600)
def test_the_rig_builds_on_the_production_route(tmp_path):
    """Two rigs in one CPU world: both route to CLOTH_VBD, the clamp holds each specimen at its sheet thickness, and
    the skirts fall."""
    from bench.scenarios import fabric_drape_run as RUN

    a = RUN.parse(["--out", str(tmp_path), "--fabrics", "F1,F9", "--worlds", "1", "--device", "cpu", "--accuracy",
                   "default", "--edge", "0.012", "--seconds", "0.4"])
    rows = RUN.run(PACK, a)
    assert [r["route"] for r in rows] == ["cloth_vbd", "cloth_vbd"] and all(r["check_ok"] for r in rows)
    for r in rows:
        h = FD.thickness(next(f for f in PACK["fabrics"] if f["id"] == r["fabric"]))
        lo, hi = r["clamp_z_mm"]
        assert 0.0 <= lo and hi <= h * 1e3 + 0.1
        assert r["finite"] and r["dc_world_last"][0] < 0.95
