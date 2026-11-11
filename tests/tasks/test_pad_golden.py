# SPDX-License-Identifier: Apache-2.0
"""pad-edge and pad-finish verdicts pinned to Gym-Anything's own evaluator (CPU, numpy).

Each state sits on one boundary of their criteria: the lip's 2.5 mm band, its full credit at 0.5 and by the band's
largest mark, the gouge band (3.5-80 mm), the field (past 150 mm) and its 1 mm limit, the 95 % edge share, the line's
100 mm band, the arc's allowed shifts, the dry band (80 mm); for pad-finish the per-side 80 % share, the top past
11.5 mm, the delamination share and the edger track's texture. PINNED holds their evaluator's numbers on these states
(``task.py`` + ``task_set_progress.py``, imported unmodified; harness: [internal note] notes/wp4-reviews/pad-golden), each
criterion after their 5-place rounding (T:155), the edge credit and progress unrounded (pad-edge) or as they round
them (pad-finish terms, 4 places).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import pad_finish_metrics as FM
from alloy.tasks import pad_metrics as PM
from alloy.tasks.concrete_metrics import PasteGrid
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR, pad_fields

HX, HY = INTERIOR[0] / 2, INTERIOR[1] / 2
FINISH_AGE = 4 * 3600.0


def grid(f: dict, **over) -> PasteGrid:
    z = np.zeros(f["h"].shape, np.float32)
    g = PasteGrid(h=f["h"] - FORM_H, t=f["tb"] + f["ts"], cell=float(f["cell"]), paste_mask=f["paste_mask"], D=f["D"],
                  a=f["a"], E=f["E"], S=f["S"], C=f["compact"], q=f["q"], bl=f["bl"], late=z.astype(np.int32),
                  wet_worked=z.astype(np.int32), dwc=z)
    for k, v in over.items():
        setattr(g, k, v)
    return g


def _copy(g: PasteGrid) -> PasteGrid:
    return PasteGrid(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(g).items()})


def edge_states() -> tuple[dict[str, PasteGrid], PasteGrid, np.ndarray, np.ndarray]:
    """Their pad-edge start (seed 0, 2 mm) cut by an ideal edger, then one boundary per state."""
    f = pad_fields("pad-edge", 0)
    x, y = f["x"], f["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    on, d, D = f["paste_mask"], PM.pad_distance(x, y), PM.form_distances(x, y)
    start, ideal = grid(f), grid(f)
    ideal.h = np.where(on & (d < PM.EDGER_RADIUS), np.minimum(ideal.h, PM.arc(np.maximum(d, 0.0))), ideal.h)
    ideal.C = np.where(on & (d < PM.LIP_BAND), 1.0, 0.0).astype(np.float32)
    out = {"ideal": ideal}

    g = _copy(ideal)                                     # the lip's band (south) and its full credit at 0.5 (west)
    south, west = np.abs(X) < HX - 0.02, np.abs(Y) < HY - 0.02
    g.C = np.where(south & (D["s"] < 0.0025), 0.0, g.C)
    g.C = np.where(south & (D["s"] >= 0.0025) & (D["s"] < 0.0045), 1.0, g.C)
    g.C = np.where(west & (D["w"] < 0.0025), 0.6, g.C).astype(np.float32)
    out["lip"] = g

    g = _copy(ideal)                                     # a 3 mm hole 60 mm in from the north form
    hole = (D["n"] >= 0.058) & (D["n"] < 0.062) & (np.abs(X) < 0.01)
    g.h = np.where(hole, np.minimum(g.h, start.h) - 0.003, g.h)
    out["gouge"] = g

    g = _copy(ideal)                                     # the slot's next cell (3 mm in) 2.5 mm under every reference
    slot = (D["n"] >= 0.0025) & (D["n"] < 0.0035) & (np.abs(X) < 0.05)   # (the arc begun a cell later included;
    g.h = np.where(slot, np.minimum(start.h, PM.arc(np.maximum(D["n"] - 0.002, 0.0))) - 0.0025, g.h)   # the north
    # form's cells lie 1, 3, 5 mm in; the east form's face splits a 2 mm cell, so its cells lie 2, 4, 6 mm in)
    out["gouge_slot"] = g

    g = _copy(ideal)                                     # 1.2 mm up 120 mm in: inside the 150 mm the field skips
    g.h = np.where((D["s"] >= 0.118) & (D["s"] < 0.122) & (np.abs(X) < 0.01), g.h + 0.0012, g.h)
    out["field_near"] = g

    g = _copy(ideal)                                     # 1.5 mm up at the centre
    g.h = np.where((np.abs(X) < 0.004) & (np.abs(Y) < 0.004), g.h + 0.0015, g.h)
    out["field_far"] = g

    g = _copy(ideal)                                     # 150 north lines off the arc: share between 0.90 and 0.95
    g.h = np.where((np.abs(X) < 0.15) & (D["n"] < 0.012), g.h + 0.002, g.h)
    out["share"] = g

    g = _copy(ideal)                                     # worked wet 50-70 mm from the south form
    g.wet_worked = ((D["s"] >= 0.05) & (D["s"] < 0.07) & (np.abs(X) < 0.01) & on).astype(np.int32)
    out["dry_band"] = g

    g = _copy(ideal)                                     # worked wet at the centre (past the dry band)
    g.wet_worked = ((np.abs(X) < 0.004) & (np.abs(Y) < 0.004)).astype(np.int32)
    out["wet_field"] = g

    g = _copy(ideal)                                     # a cell 99 mm in on the south line 5 mm from the west form
    i5 = int(np.argmin(np.abs(x + HX - 0.005)))
    j99 = (D["s"][i5] >= 0.098) & (D["s"][i5] < 0.100)
    g.h[i5] = np.where(j99, g.h[i5] + 0.003, g.h[i5])
    out["line_band"] = g

    g = _copy(ideal)                                     # east lines on the arc shifted 3 mm and 0.45 mm low
    east = (D["e"] < 0.0125) & (np.abs(Y) < 0.05) & on
    g.h = np.where(east, PM.arc(np.maximum(D["e"] - 0.003, 0.0)) - 0.00045, g.h)
    out["arc_shift"] = g
    return out, start, x, y


def finish_states() -> tuple[dict[str, PasteGrid], PasteGrid, np.ndarray, np.ndarray]:
    """Their pad-trowel start troweled (D 0.9, texture 0.04 mm) against the pad-finish start, one boundary each."""
    s0, floated = pad_fields("pad-finish", 0), pad_fields("pad-trowel", 0)
    x, y = s0["x"], s0["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    on, d, D = floated["paste_mask"], PM.pad_distance(x, y), PM.form_distances(x, y)
    done = grid(floated, D=np.where(on, 0.9, 0.0).astype(np.float32), a=np.where(on, 4.0e-5, 0.0).astype(np.float32))
    out = {"done": done}

    g = _copy(done)                                      # every third north line off the arc: north share ~0.7
    third = (np.arange(len(x)) % 3 == 0)[:, None] & (np.abs(X) < HX - 0.1)
    g.h = np.where(third & (D["n"] < 0.012) & on, g.h + 0.0025, g.h)
    out["side_share"] = g

    g = _copy(done)                                      # the ring 9.5-11.5 mm in left floated
    ring = on & (d >= PM.EDGER_RADIUS) & (d < FM.TOP_D)
    g.D, g.a = np.where(ring, 0.1, g.D).astype(np.float32), np.where(ring, 2.5e-4, g.a).astype(np.float32)
    out["top_ring"] = g

    g = _copy(done)                                      # 0.02 mm of trapped water on 5 % of the pad
    rng = np.random.default_rng(5)
    g.q = np.where(on & (rng.random(on.shape) < 0.05), 2.0e-5, 0.0).astype(np.float32)
    out["delam"] = g

    g = _copy(done)                                      # the edger's track at 0.055 mm texture
    g.a = np.where(on & (d < FM.EDGER_W), 5.5e-5, g.a).astype(np.float32)
    out["edger_tex"] = g
    return out, grid(s0), x, y


PINNED = {
    "edge": {
        "ideal": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 1.0, "score": 1.0, "success": True},
        "lip": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (0.72373, False),
                         "no_gouge": (0.0, True), "field_untouched": (0.0, True),
                         "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.5147783251231527, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 0.698973944317109, "score": 0.6640252471012535, "success": False},
        "gouge": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (3.0, False),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 0.75, "field": 1.0, "edge_dry": 1.0},
            "progress": 0.75, "score": 0.7124999999999999, "success": False},
        "gouge_slot": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 1.0, "score": 1.0, "success": True},
        "field_near": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 1.0, "score": 1.0, "success": True},
        "field_far": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (1.5, False), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 0.875, "edge_dry": 1.0},
            "progress": 0.875, "score": 0.8312499999999999, "success": False},
        "share": {
            "criteria": {"radius_profile": (0.92683, False), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9853945207847123, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 0.8898567277929892, "score": 0.8453638914033397, "success": False},
        "dry_band": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (100.0, False)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 0.9987261633313377},
            "progress": 0.9987261633313377, "score": 0.9487898551647708, "success": False},
        "wet_field": {
            "criteria": {"radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 1.0, "score": 1.0, "success": True},
        "line_band": {
            "criteria": {"radius_profile": (0.99625, True), "lip_passed": (1.0, True), "no_gouge": (0.0, True),
                         "field_untouched": (0.0, True), "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9980986950133955, "n": 0.9983579638752053, "w": 0.9952781296786826,
                      "e": 0.9956236323851203},
            "gates": {"tools": 1.0, "gouge": 1.0, "field": 1.0, "edge_dry": 1.0},
            "progress": 1.0, "score": 1.0, "success": True},
        "arc_shift": {
            "criteria": {"radius_profile": (0.97326, True), "lip_passed": (1.0, True),
                         "no_gouge": (2.03831, False), "field_untouched": (0.0, True),
                         "edge_not_worked_wet": (0.0, True)},
            "credit": {"s": 0.9983579638752053, "n": 0.9983579638752053, "w": 0.9956236323851203,
                      "e": 0.9893000410931644},
            "gates": {"tools": 1.0, "gouge": 0.9904225, "field": 1.0, "edge_dry": 1.0},
            "progress": 0.9904225, "score": 0.9409013749999999, "success": False},
    },
    "finish": {
        "done": {
            "criteria": {"smooth_hard": (1.0, True), "stone_down": (1.0, True),
                         "flat_straightedge": (2.18402, True), "flat_plane": (1.54485, True),
                         "radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_blister": (0.0, True),
                         "delamination_risk": (0.0, True), "no_trowel_burn": (0.0, True),
                         "no_water_worked": (0.0, True), "surface_wc": (0.0, True)},
            "terms": {"level": 1.0, "edge": 1.0, "stone": 1.0, "finish": 1.0},
            "progress": 1.0},
        "side_share": {
            "criteria": {"smooth_hard": (1.0, True), "stone_down": (1.0, True),
                         "flat_straightedge": (2.18402, True), "flat_plane": (1.54485, True),
                         "radius_profile": (0.91745, False), "lip_passed": (1.0, True), "no_blister": (0.0, True),
                         "delamination_risk": (0.0, True), "no_trowel_burn": (0.0, True),
                         "no_water_worked": (0.0, True), "surface_wc": (0.0, True)},
            "terms": {"level": 1.0, "edge": 0.8512, "stone": 1.0, "finish": 1.0},
            "progress": 0.9628},
        "top_ring": {
            "criteria": {"smooth_hard": (1.0, True), "stone_down": (1.0, True),
                         "flat_straightedge": (2.18402, True), "flat_plane": (1.54485, True),
                         "radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_blister": (0.0, True),
                         "delamination_risk": (0.0, True), "no_trowel_burn": (0.0, True),
                         "no_water_worked": (0.0, True), "surface_wc": (0.0, True)},
            "terms": {"level": 1.0, "edge": 1.0, "stone": 1.0, "finish": 1.0},
            "progress": 1.0},
        "delam": {
            "criteria": {"smooth_hard": (1.0, True), "stone_down": (1.0, True),
                         "flat_straightedge": (2.18402, True), "flat_plane": (1.54485, True),
                         "radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_blister": (0.0, True),
                         "delamination_risk": (0.04985, False), "no_trowel_burn": (0.0, True),
                         "no_water_worked": (0.0, True), "surface_wc": (0.0, True)},
            "terms": {"level": 1.0, "edge": 1.0, "stone": 1.0, "finish": 1.0},
            "progress": 0.9502},
        "edger_tex": {
            "criteria": {"smooth_hard": (1.0, True), "stone_down": (1.0, True),
                         "flat_straightedge": (2.18402, True), "flat_plane": (1.54485, True),
                         "radius_profile": (0.99719, True), "lip_passed": (1.0, True), "no_blister": (0.0, True),
                         "delamination_risk": (0.0, True), "no_trowel_burn": (0.0, True),
                         "no_water_worked": (0.0, True), "surface_wc": (0.0, True)},
            "terms": {"level": 1.0, "edge": 1.0, "stone": 1.0, "finish": 1.0},
            "progress": 1.0},
    },
}


@pytest.fixture(scope="module")
def edge():
    return edge_states()


@pytest.fixture(scope="module")
def finish():
    return finish_states()


@pytest.mark.parametrize("name", list(PINNED["edge"]))
def test_pad_edge_is_their_evaluator(edge, name):
    states, start, x, y = edge
    g, want = states[name], PINNED["edge"][name]
    v = PM.pad_edge(g, start, x, y)
    for k, (value, passed) in want["criteria"].items():
        assert (round(float(v[k].value), 5), v[k].passed) == (value, passed), k
    credit = PM.edge_credit(PM.perimeter(g, start, x, y), g.cell)
    for s, value in want["credit"].items():
        assert credit[s] == pytest.approx(value, rel=1e-12, abs=0), s
    for k, value in want["gates"].items():                      # theirs read the 5-place criterion values
        assert v.progress.gates[k] == pytest.approx(value, rel=1e-6, abs=1e-12), k
    assert v.progress.value == pytest.approx(want["progress"], rel=1e-6, abs=1e-12)
    assert v.score == pytest.approx(want["score"], rel=1e-6, abs=1e-12) and v.success == want["success"]


@pytest.mark.parametrize("name", list(PINNED["finish"]))
def test_pad_finish_is_their_evaluator(finish, name):
    states, start, x, y = finish
    g, want = states[name], PINNED["finish"][name]
    v = FM.pad_finish(g, start, x, y, np.full(g.h.shape, FINISH_AGE))
    for k, (value, passed) in want["criteria"].items():
        assert (round(float(v[k].value), 5), v[k].passed) == (value, passed), k
    for k, value in want["terms"].items():                      # theirs round the terms and progress to 4 places
        assert round(v.terms[k], 4) == value, k
    assert round(v.progress, 4) == want["progress"]
