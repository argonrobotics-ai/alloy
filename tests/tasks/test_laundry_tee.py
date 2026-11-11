# SPDX-License-Identifier: Apache-2.0
"""The fold-a-shirt task (Gym-Anything laundry gap, ``laundry/tee/sleeve_tuck_fold``): the T-shirt mesh
(``alloy.tasks.laundry_garment``) and the cuff swing (``alloy.tasks.laundry_tee``) on their own (CPU, no simulation),
then the sleeve fold through the cloth route (slow)."""

from __future__ import annotations

import numpy as np
import dataclasses

import pytest

from alloy.tasks.fold_checks import self_intersections
from alloy.tasks.laundry_garment import OPEN, TeeShape, make_tee, outline
from alloy.tasks.laundry_tee import TeeRig, _SideFlap

H = 0.000975


@pytest.mark.parametrize("sleeve,cell", [(0.15, 0.02), (0.19, 0.02), (0.12, 0.015)])
def test_the_tee_is_two_panels_sewn_into_one_clean_mesh(sleeve, cell):
    """Every edge in at most two triangles (slivers along the outline dropped), every vertex used, no crossing at
    build; the openings are the only edges with one triangle."""
    t = make_tee(H, cell, s=TeeShape(sleeve=sleeve))
    e = np.sort(np.concatenate([t.triangles[:, [0, 1]], t.triangles[:, [1, 2]], t.triangles[:, [2, 0]]]), axis=1)
    _, uses = np.unique(e, axis=0, return_counts=True)
    assert uses.max() == 2
    assert len(np.unique(t.triangles)) == len(t.vertices)
    assert self_intersections(t.vertices, t.triangles) == 0
    assert (t.panel == 2).sum() > 50 and (t.panel == 0).sum() == (t.panel == 1).sum()
    assert len(t.cuffs) == 2 and all(len(c) >= 4 for c in t.cuffs)


def test_the_panels_lie_two_thicknesses_apart_with_the_seams_between():
    t = make_tee(H, 0.02)
    z = {p: np.unique(np.round(t.vertices[t.panel == p, 2], 9)) for p in (0, 1, 2)}
    assert z[0].tolist() == [round(0.5 * H, 9)] and z[1].tolist() == [round(2.5 * H, 9)]
    assert z[2].tolist() == [round(1.5 * H, 9)]


def test_the_outline_leaves_hem_neck_and_cuffs_open():
    names = {name for _, name in outline(TeeShape())}
    assert set(OPEN) <= names
    assert {"side_l", "side_r", "under_l", "under_r", "top_l", "top_r", "shoulder_l", "shoulder_r"} <= names


def test_the_cuff_swing_is_rigid_and_lands_turned_over():
    """Front and back cuff points (1.95 mm apart) keep their distance through the swing (at 90 deg they sit side by
    side, not merged: the rig's old path merged them and the panels crossed) and land in the opposite order."""
    start = np.array([[0.45, 0.10, 0.5 * H], [0.45, 0.10, 2.5 * H]])
    f = _SideFlap(np.array([0, 1]), start, x_fold=0.25, z_land=0.01, apex=1.0)
    for s in np.linspace(0.0, 1.0, 21):
        p = f.at(s)
        assert np.linalg.norm(p[0] - p[1]) == pytest.approx(2.0 * H, rel=1e-9)
    end = f.at(1.0)
    assert end[0, 2] > end[1, 2]                                  # the bottom panel ends on top
    assert end[:, 0] == pytest.approx([0.05, 0.05], abs=1e-12)


def test_the_rig_defaults_fold_at_the_side_seams():
    rig = TeeRig()
    assert rig.fold_in == 0.0 and rig.sleeve == 0.15



def test_a_shirt_on_another_material_card_builds_with_its_hinges(tmp_path):
    """TeeRig(material=) runs the shirt on an in-memory material card (the object validator asks the library's
    has_material first): a renamed jersey routes with KES hysteretic hinges."""
    import dataclasses

    from alloy.cards.library import CardLibrary
    from alloy.tasks.laundry_tee import build

    jersey = dataclasses.replace(CardLibrary().material("cotton_single_jersey"), name="jersey_test")
    model = build(TeeRig(material=jersey, cell=0.04), tmp_path)[0]
    assert (model.cloth.bend_hb.numpy() > 0).any()

def test_the_rig_holds_self_contact_plies_unless_told_not_to(tmp_path):
    """The rig solves with the cloth default's self-contact hold (the sleeve law is a draw without it);
    ``TeeRig.vbd`` can turn it off, and keeps the route's other settings."""
    from alloy.tasks.laundry_tee import build

    on = build(TeeRig(cell=0.04), tmp_path)[1]
    off = build(TeeRig(cell=0.04, vbd={"alloy_chebyshev_hold_self_contact": False}), tmp_path)[1]
    assert on["alloy_chebyshev_hold_self_contact"] is True and off["alloy_chebyshev_hold_self_contact"] is False
    assert {k: v for k, v in on.items() if k != "alloy_chebyshev_hold_self_contact"} == \
        {k: v for k, v in off.items() if k != "alloy_chebyshev_hold_self_contact"}
    assert on.get("alloy_chebyshev_rho") is not None           # the hold acts on the route's Chebyshev step


def test_the_fine_rig_sets_its_mesh_and_contact_and_the_default_keeps_the_routes(tmp_path):
    from alloy.tasks.laundry_tee import FINE_TEE, build

    assert (FINE_TEE.cell, FINE_TEE.contact_ke, FINE_TEE.contact_kd) == (0.01, 2.0e4, 0.0)
    route = build(TeeRig(cell=0.04), tmp_path)[0]
    soft = build(dataclasses.replace(FINE_TEE, cell=0.04), tmp_path)[0]
    assert route.soft_contact_ke > 1.0e5 and route.soft_contact_kd > 0.0       # the route's: the material's K, kd 10
    assert (soft.soft_contact_ke, soft.soft_contact_kd) == (2.0e4, 0.0)


@pytest.fixture(scope="module")
def fine_fold():
    """The fine rig's fold on the GPU (9,300 particles, about 7 min on an L4)."""
    from alloy.tasks.laundry_tee import FINE_TEE, run

    return run(FINE_TEE, "cuda:0")


@pytest.mark.gpu
@pytest.mark.nightly
@pytest.mark.law
@pytest.mark.timeout(3600)
def test_the_fine_rigs_folded_sleeves_stay_on_the_body(fine_fold):
    """Every gate of the task on the settled shirt (gap #4 slice 2). Rows (one L4, 1 um nudged starts, 2026-11-10,
    [internal note] notes/wp11-gap4): plain weave at 10 mm cells with contact 2e3-2e5 N/m and no contact damping 18 of
    18 starts (6 of 6 at this rig's 2e4): sleeves 100 % inside the body, 2-6 mm RMS from the goal (gate 45), normal-gap p90 14-17 mm (gate
    25), stretch 4 %, 0 crossings, at rest under 1 cm/s. The same shirt at 20 mm cells fails (0 of 2: 76-103 mm)."""
    v = fine_fold
    assert v.success, (v.gates, v.metrics)
    assert v.metrics["crossings_max"] == 0, v.frames


@pytest.fixture(scope="module")
def sleeve_fold():
    """One sleeve fold through the cloth route (CPU, about 2.8 h on one n2 core), shared by the two laws below."""
    from alloy.tasks.laundry_tee import run

    return run(TeeRig())


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(14400)
def test_the_sleeves_reach_the_body_without_crossing(sleeve_fold):
    """Rows (CPU from the suite's start, 2,360 particles, solved as TeeRig does with WP2's hold, 2026-11-09): 0
    crossings in every frame (the swing turns the cuff rigidly: the path that merged the two panels at 90 deg crossed
    them 26-360 times); as the hands let go both sleeves are inside the body's hull and placed within 45 mm RMS (6.3
    and 5.5 mm; main before F32, without the hold, 12.7 and 9.7 mm), stretch 5 %; settled stretch 16 % (gate 30 %).
    Without the hold the law is a draw: F32 there puts the right sleeve 54 mm off; nudged GPU starts pass main 7 of
    12, F32 12 of 18, F32 with the hold 6 of 6."""
    v = sleeve_fold
    assert v.metrics["crossings_max"] == 0 and v.release_metrics["crossings_end"] == 0, v.frames
    for side in ("left", "right"):
        assert v.release_gates[f"{side}_inside"] and v.release_gates[f"{side}_rms"], v.release_metrics
    assert v.gates["stretch"] and v.gates["finite"], v.metrics


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(14400)
@pytest.mark.xfail(strict=True, reason="known gap cloth-sleeve-fold-springback: at the default rig's 20 mm cells and "
                   "the route's contact stiffness the folded sleeves do not stay or lie flat (the fine rig's do: "
                   "test_the_fine_rigs_folded_sleeves_stay_on_the_body)")
def test_the_folded_sleeves_stay_on_the_body(sleeve_fold):
    assert sleeve_fold.success, sleeve_fold.metrics
