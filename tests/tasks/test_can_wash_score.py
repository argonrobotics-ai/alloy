# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can verdict (alloy/tasks/can_wash_score.py): gates, free film and score on synthetic readouts,
anchored to Gym-Anything's report numbers (A4 assisted demo: detergent 56.5 -> 0.148 mg, debris 354 mg -> none)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.tasks.can_wash_score import CanReadout, free_film, gates, row, verdict
from alloy.tasks.rinse_score import HANDS_NOT_JUDGED

AREA = 2.4                                        # m², the can's charted area (spec §2.2)


def _readout(**kw) -> CanReadout:
    """One world, every gate passing unless overridden."""
    base = {"soap_kg": 0.148e-6, "soap_gauge": 0.2e-6, "soil_gauge": 0.0, "free_ml": 40.0,
            "tilt_deg": 2.0, "valve_m": 0.05, "shelf_g": 0.0, "hose_gpm": 0.0, "held": False,
            "water_rel": 0.0, "species_rel": 0.0, "species_kg": 0.0, "finite": True}
    base |= kw
    return CanReadout(**{k: np.array([v]) for k, v in base.items()}, area_m2=AREA)


START = _readout(soap_kg=56.5e-6, soap_gauge=40e-6, soil_gauge=100e-5, free_ml=0.0, held=True)


NOT_JUDGED = ["soap_free", "water_off_and_hands_free", "intact_and_not_dropped"]


def test_a4_demo_end_state_passes_every_judged_gate():
    v = verdict(START, _readout(), 0)
    assert v.success and v.score == 1.0, v.failed()
    assert v.terms["rinsed"] == 1.0                  # 56.5 mg -> 0.148 mg is past the 282 µg limit
    assert v.judged == (5, 8) and v.not_judged() == NOT_JUDGED


def test_no_gate_passes_on_a_constant_or_the_scripts_own_flag():
    """Audit AB-gym-3: with every measure at its best, the three gates this port cannot read from the state stay
    not judged (None), never True: hands-free (no finger pads), foam (no model), intact (vacuous in GA too). The
    script's hold fails hands-free; letting go does not pass it. The row says how many criteria were judged."""
    g = gates(_readout(), 0)
    assert [k for k, (ok, _) in g.items() if ok is None] == NOT_JUDGED
    assert g["water_off_and_hands_free"][1] == {"outlet_off": True, "hands": HANDS_NOT_JUDGED, "hose_gpm": 0.0}
    held = gates(_readout(held=True), 0)["water_off_and_hands_free"]
    assert held[0] is False and held[1]["hands"] == "still gripping"
    out = row(verdict(START, _readout(), 0), 3.0)
    assert out["judged"] == [5, 8] and out["not_judged"] == NOT_JUDGED and out["success"] is True
    assert out["summary"].startswith("5 of 8 Gym-Anything criteria judged; not judged: soap_free")
    assert out["gates"]["water_off_and_hands_free"]["ok"] is None and out["settle_s"] == 3.0


def test_the_score_counts_a_gates_judged_parts_only():
    """GA's k = valve_parked x released: a let-go hand (not judged) leaves k to the parked valve and the hose; a
    hand still holding, or a running hose, makes it 0."""
    assert verdict(START, _readout(), 0).terms["docked_released"] == 1.0
    for kw in ({"held": True}, {"hose_gpm": 1e-3}, {"valve_m": 0.16}):
        v = verdict(START, _readout(**kw), 0)
        assert v.terms["docked_released"] == 0.0 and v.score == pytest.approx(0.9 * 2.0 / 3.0), kw


def test_reset_state_scores_zero():
    v = verdict(START, START, 0)
    assert not v.success and v.progress == 0.0 and v.score == 0.0


@pytest.mark.parametrize("scale, ok", [(0.99, None), (1.01, False)])
def test_soap_mean_limit_is_10_ug_per_0085_m2(scale, ok):
    """Under the limit the judged parts pass and the gate stays not judged (foam); over it the gate fails."""
    limit = AREA * 10e-9 / 0.085                     # 282 µg on 2.4 m²
    assert gates(_readout(soap_kg=scale * limit), 0)["soap_free"][0] is ok


@pytest.mark.parametrize("field, value, gate", [
    ("soap_gauge", 1.1e-6, "soap_free"),             # one 1 cm patch over 1 mg/m²
    ("soil_gauge", 1.1e-5, "visibly_clean"),         # over 1 µg/cm²
    ("free_ml", 101.0, "emptied_upright"),
    ("tilt_deg", 20.0, "emptied_upright"),
    ("valve_m", 0.16, "valve_parked"),
    ("shelf_g", 1.1, "contained"),
    ("hose_gpm", 1e-3, "water_off_and_hands_free"),
    ("held", True, "water_off_and_hands_free"),
    ("water_rel", 1e-6, "no_simulation_fault"),
])
def test_each_gate_fails_alone(field, value, gate):
    g = gates(_readout(**{field: value}), 0)
    assert [k for k, (ok, _) in g.items() if ok is False] == [gate]
    assert verdict(START, _readout(**{field: value}), 0).failed() == [gate]


def test_progress_terms():
    """Half the log reduction of detergent and soil, not emptied: c = 0.5, progress c/3; shelf water subtracts."""
    half = _readout(soap_kg=np.sqrt(56.5e-6 * AREA * 10e-9 / 0.085), soil_gauge=np.sqrt(100e-5 * 1e-5),
                    free_ml=500.0)
    v = verdict(START, half, 0)
    assert v.terms["clean"] == pytest.approx(0.5) and v.progress == pytest.approx(0.5 / 3.0)
    wet_shelf = verdict(START, dataclasses.replace(half, shelf_g=np.array([1.05])), 0)
    assert wet_shelf.progress == pytest.approx(0.5 / 3.0 - 0.05)


def test_open_ledger_is_unverifiable():
    v = verdict(START, _readout(species_rel=1e-3, species_kg=1e-9), 0)
    assert v.score is None and v.progress is None


def test_free_film_counts_only_what_would_run():
    """Level floor (polymer h_break 1e-4 m): 0.5 mm clings; a vertical wall: h_break; never under 50 µm."""
    area = np.full(3, 1e-4)
    level, wall = np.array([1.0, 1.0, 1.0]), np.zeros(3)
    assert free_film(np.full(3, 0.4e-3), level, area, np.full(3, 1e-4)) == 0.0
    assert free_film(np.full(3, 1.0e-3), level, area, np.full(3, 1e-4)) == pytest.approx(3 * 0.5e-3 * 1e-4)
    assert free_film(np.full(3, 0.3e-3), wall, area, np.full(3, 1e-4)) == pytest.approx(3 * 0.2e-3 * 1e-4)
    assert free_film(np.full(3, 0.1e-3), wall, area, np.full(3, 1e-6)) == pytest.approx(3 * 0.05e-3 * 1e-4)


def test_score_reads_the_live_state_and_gates_the_settled_one():
    """GA's evaluator reads the score's detergent, debris and shelf before the settle and the gates after it: a can
    clean before the settle scores its reduction even if the settle drains soap back onto it, and its gates fail."""
    live = _readout()
    settled = _readout(soap_kg=5.0e-6, shelf_g=1.5)              # 5 mg back on the can, wet shelf
    v = verdict(START, settled, 0, live=live)
    assert v.terms["rinsed"] == 1.0 and not v.success and v.failed() == ["soap_free", "contained"]
    assert v.progress == pytest.approx(1.0) and v.score == pytest.approx(0.9)
    assert verdict(START, settled, 0).progress < 0.5                # read after the settle: the shelf's excess too
