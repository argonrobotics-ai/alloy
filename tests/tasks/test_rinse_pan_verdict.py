# SPDX-License-Identifier: Apache-2.0
"""rinse_pan's verdict (Gym-Anything's scrub-rinse-pan gates, task.py:714-733, and progress-v1, task.py:369-382) and
its readout: each gate at its boundary on their report's rounding, the pure readout (soap, S, scour face, tilt), the
fault gate's books and the per-world outlets."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.tasks.rinse_pan import (EGG, V_WITHDRAW, PanTask, Readout, books_ok, compose, egg_band, judge,
                                   outlets_running, pan_rinsed, rinse_soap, soap_load, start_egg)
from alloy.tasks.rinse_pan_scene import SPONGE_LIFT, Station
from alloy.tasks.rinse_pan_verdict import SOAP_PER_AREA, PanEnd, clean_board_x, reported, verdict
from alloy.tasks.rinse_sink import GPM, PotSink
from alloy.tasks.rinse_tracker import PAN_SURF_REF_KG, SOAP_REF_AREA_M2, pan_progress, soap_limit_kg
from alloy.tasks.rinse_vessels import Pan

SINK = PotSink()
ST = Station()


def _good(**kw) -> PanEnd:
    x0, x1 = clean_board_x(SINK)
    end = PanEnd(gauge=0.0, soap=1e-9, soap_gauge=0.0, area=0.15, sanitized_s=31.0, into_sanitizer=1e-9,
                 free_water_ml=1.0, pan_pos=np.array([0.5 * (x0 + x1), 0.6, 1.57]), tilt_deg=178.0,
                 sponge_pos=np.array([3.3, 0.66, SINK.rim_z + 0.0005]), outlet_running=False, held=False, books=True)
    for k, v in kw.items():
        setattr(end, k, v)
    return end


def _failed(**kw) -> list[str]:
    return verdict(SINK, _good(**kw), clean=1.0, rinsed=1.0).failed()


def test_a_finished_pan_passes_every_gate():
    v = verdict(SINK, _good(), clean=1.0, rinsed=1.0)
    assert v.success and v.score == 1.0 and len(v.criteria) == 9


def test_mouth_up_on_the_shelf_loses_the_stored_stage():
    """placed fails, so T = 0: progress (C + CR + CRZ)/4 and the score 0.9 x that."""
    v = verdict(SINK, _good(tilt_deg=5.0), clean=1.0, rinsed=1.0)
    assert v.failed() == ["placed"]
    assert v.progress == pytest.approx(0.75) and v.score == pytest.approx(0.9 * 0.75)


def test_soap_over_the_limit_and_a_sponge_off_the_ledge():
    v = verdict(SINK, _good(soap=2.0 * SOAP_PER_AREA * 0.15, sponge_pos=np.array([3.3, 0.3, 0.6])), 1.0, 0.5)
    assert sorted(v.failed()) == ["soap_free", "sponge_on_ledge"]
    assert v.progress == pytest.approx(pan_progress(1.0, 0.5, 31.0, stored=False)[0])


def test_an_open_ledger_is_unverifiable():
    assert verdict(SINK, _good(books=False), 1.0, 1.0).score is None


def test_visibly_clean_at_one_microgram_per_square_centimetre():
    assert _failed(gauge=1.0e-5) == [] and _failed(gauge=1.0001e-5) == ["visibly_clean"]


def test_the_soap_limit_is_per_area_and_the_gauge_counts():
    """At 0.15 m² the limit is 10 µg x 0.15/0.085 = 17.647 µg: at it the pan passes, 2 ng over it fails; a 1 cm
    gauge over 1 mg/m² fails with the mass under the limit."""
    limit = SOAP_PER_AREA * 0.15
    assert _failed(soap=limit) == [] and _failed(soap=limit + 2e-12) == ["soap_free"]
    assert _failed(soap_gauge=1.0e-6) == [] and _failed(soap_gauge=1.0001e-6) == ["soap_free"]


def test_sanitized_needs_thirty_seconds_and_a_clean_entry():
    limit = SOAP_PER_AREA * 0.15
    assert _failed(into_sanitizer=limit + 2e-12) == ["sanitized"]
    assert _failed(sanitized_s=29.99) == ["sanitized"] and _failed(sanitized_s=30.0) == []


def test_drained_at_five_millilitres():
    assert _failed(free_water_ml=5.0) == [] and _failed(free_water_ml=5.001) == ["drained"]


def test_placed_inside_the_shelf_band_only():
    x, y = _good().pan_pos[:2]
    assert _failed(pan_pos=np.array([x, y, 1.49])) == ["placed"]
    assert _failed(pan_pos=np.array([x, y, 1.66])) == ["placed"]


def test_the_ledge_band_floor_is_five_millimetres_under_the_rim():
    x, y = 3.3, 0.66
    assert _failed(sponge_pos=np.array([x, y, SINK.rim_z - 0.0049])) == []
    assert _failed(sponge_pos=np.array([x, y, SINK.rim_z - 0.0051])) == ["sponge_on_ledge"]


def test_a_dropped_sponge_fails_intact_with_the_pan_on_the_shelf():
    assert _failed(sponge_pos=np.array([3.3, 0.66, 0.2])) == ["sponge_on_ledge", "intact_and_not_dropped"]


def test_a_holding_hand_fails_hands_free_with_the_outlets_off():
    assert _failed(held=True) == ["water_off_and_hands_free"]
    assert _failed(outlet_running=True) == ["water_off_and_hands_free"]


def test_the_gates_read_their_reports_rounding():
    """29.996 s reports as 30.00 (sanitized passes); 5.0004 mL as 5.000 (drained passes); 1.00004 µg/cm² as 1.0000."""
    assert _failed(sanitized_s=29.996, free_water_ml=5.0004, gauge=1.00004e-5) == []


def test_the_report_rounds_soap_and_carry_in_to_a_thousandth_of_a_microgram():
    """``task.py:484-504``: 16.0004 µg reports as 16.000, 2.9996 µg carried in as 3.000."""
    r = reported(_good(soap=16.0004e-9, into_sanitizer=2.9996e-9))
    assert r.soap == round(16.0004, 3) * 1e-9 and r.into_sanitizer == round(2.9996, 3) * 1e-9


def _readout(**kw) -> Readout:
    r = Readout(soil_gauge=0.0, faces=5e-9, soap_gauge=0.0, area=0.15, cav_v=30e-6, cav_soap=3e-9, bowl_conc=None,
                sanitized_s=31.0, into_sanitizer=1e-9, free_water_ml=1.0,
                pan_pose=np.array([4.0, 0.6, 1.57, 1.0, 0.0, 0.0, 0.0]),
                sponge_pose=np.array([3.3, 0.66, 0.935, 0.0, 0.0, 0.0, 1.0]), outlet_running=False, held=False)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_the_readout_counts_the_cavity_and_reads_the_scour_face():
    """The pan's soap is its faces' plus its cavity's; the sponge is read at its collider's bottom, SPONGE_LIFT up
    its body's z from the pad's face (down when it lies face up); a pan turned 180° about x is mouth down."""
    end, _ = compose(ST, _readout(), books=True)
    assert end.soap == pytest.approx(8e-9, rel=1e-12, abs=0)
    assert end.sponge_pos.tolist() == pytest.approx([3.3, 0.66, 0.935 + SPONGE_LIFT], rel=0, abs=1e-12)
    assert end.tilt_deg == pytest.approx(180.0, rel=0, abs=1e-9) and end.books
    up = _readout(sponge_pose=np.array([3.3, 0.66, 0.975, 1.0, 0.0, 0.0, 0.0]))
    assert compose(ST, up, books=True)[0].sponge_pos[2] == pytest.approx(0.975 - SPONGE_LIFT, rel=0, abs=1e-12)


def test_the_rinse_term_takes_eighteen_millilitres_of_the_cavity_or_the_bowl():
    """S = faces + the cavity's strength x min(its water, 18 mL); in a bowl, the bowl's strength x 18 mL when more."""
    assert rinse_soap(_readout()) == pytest.approx(5e-9 + 3e-9 * 18.0 / 30.0, rel=1e-12, abs=0)
    assert rinse_soap(_readout(cav_v=10e-6)) == pytest.approx(8e-9, rel=1e-12, abs=0)
    assert rinse_soap(_readout(cav_v=0.0, cav_soap=0.0)) == pytest.approx(5e-9, rel=1e-12, abs=0)
    assert rinse_soap(_readout(bowl_conc=1e-3)) == pytest.approx(5e-9 + 1e-3 * V_WITHDRAW, rel=1e-12, abs=0)
    assert rinse_soap(_readout(bowl_conc=1e-6)) == pytest.approx(5e-9 + 3e-9 * 18.0 / 30.0, rel=1e-12, abs=0)
    _, s = compose(ST, _readout(), books=True)
    assert s == pytest.approx(rinse_soap(_readout()), rel=0, abs=0)


def test_the_fault_gate_needs_every_book_and_no_overflow():
    closed = np.array([True, True, False, True])
    soil = np.array([0.0, 2e-9, 0.0, 0.0])
    over = np.array([False, False, False, True])
    assert books_ok(closed, soil, over).tolist() == [True, False, False, False]


def test_outlets_run_per_world_by_the_faucet_or_the_valves_jets():
    """World 0's faucet runs, world 1's nine jets each under 1e-4 gpm but together over it, world 2 is off."""
    names = [e.name for e in ST.edges()]
    q = np.zeros((3, len(names)))
    q[0, names.index("faucet")] = 2e-4 * GPM
    for k, n in enumerate(names):
        if n.startswith("jet"):
            q[1, k] = 0.2e-4 * GPM
    flow = SimpleNamespace(supply=SimpleNamespace(read=lambda: {"discharge": q}))
    assert sum(n.startswith("jet") for n in names) == 9
    assert outlets_running(flow, ST)().tolist() == [True, True, False]
    assert math.isclose(q[1].sum(), 1.8e-4 * GPM)


def test_a_pan_turned_about_y_reads_its_tilt():
    """The tilt is the pan's axis from up whichever horizontal axis it turned about (the pours roll it about y)."""
    for q, deg in (([0.0, 1.0, 0.0, 0.0], 180.0), ([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)], 90.0)):
        end, _ = compose(ST, _readout(pan_pose=np.array([4.0, 0.6, 1.57, *q])), books=True)
        assert end.tilt_deg == pytest.approx(deg, rel=0, abs=1e-6)


class _Arr:
    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def numpy(self) -> np.ndarray:
        return self.a


def test_the_soap_gauge_counts_the_film_and_the_walls_load():
    """``soap_load``: the pan's cells' film soap plus the soap adsorbed on its walls (kg/m², species 0)."""
    film = np.array([[[1e-6, 9.0], [2e-6, 9.0], [7.0, 9.0]]])
    walls = np.array([[[3e-7, 9.0], [0.0, 9.0], [7.0, 9.0]]])
    flow = SimpleNamespace(tables=SimpleNamespace(cells_of=lambda chart: np.array([0, 1])),
                           sp=SimpleNamespace(mass=_Arr(film)), adsorption=SimpleNamespace(load=_Arr(walls)))
    assert soap_load(flow, 0).tolist() == pytest.approx([1.3e-6, 2e-6], rel=1e-12, abs=0)


def test_the_rinse_term_reads_this_pans_soap_limit():
    """R halfway (log) from the poured pan's soap to this pan's limit is 0.5; on a pan of half GA's reference area
    the limit is half of GA's 16.5 µg."""
    end = _good(area=0.5 * SOAP_REF_AREA_M2)
    s = math.sqrt(PAN_SURF_REF_KG * soap_limit_kg(end.area))
    assert pan_rinsed(end, s) == pytest.approx(0.5, rel=1e-12, abs=0)


def test_the_egg_reaches_eight_millimetres_up_the_wall():
    """Theirs: inner cells under 12 mm over the foot (``system.py:390-400``), 8 mm over this pan's inner floor."""
    z_max = 0.012 - Pan().bottom
    assert EGG["z_max"] == pytest.approx(z_max, rel=0, abs=1e-12)
    z = np.array([z_max - 1e-4, z_max + 1e-4, 0.0])
    assert egg_band(np.array([True, True, False]), z).tolist() == [True, False, False]


def test_a_worlds_verdict_scores_r_on_this_pans_soap_limit():
    """``judge`` (what the row reads per world): faces carrying the log midpoint between the poured pan's soap and
    this pan's limit score R = 0.5 in progress-v1 (soap_free fails, so the pan is not stored)."""
    r = _readout(cav_v=0.0, cav_soap=0.0)
    r.faces = math.sqrt(PAN_SURF_REF_KG * soap_limit_kg(r.area))
    v, end = judge(ST, r, books=True, clean=1.0)
    assert "soap_free" in v.failed() and end.soap == r.faces
    assert v.progress == pytest.approx(pan_progress(1.0, 0.5, 31.0, stored=False)[0], rel=0, abs=1e-12)
    half, _ = judge(ST, r, books=True, clean=0.5)
    assert half.progress == pytest.approx(pan_progress(0.5, 0.5, 31.0, stored=False)[0], rel=0, abs=1e-12)
    assert judge(ST, r, books=False, clean=1.0)[0].score is None


def test_the_egg_is_seeded_on_its_band_only():
    """``start_egg`` loads the inner cells under 8 mm and nothing else, in every world."""
    inner = np.array([True, True, False, True])
    z = np.array([0.0, 0.007, 0.0, 0.02])
    sense = SimpleNamespace(cells=SimpleNamespace(inner=inner, local=np.c_[np.zeros((4, 2)), z]))
    got = {}
    flow = SimpleNamespace(tables=SimpleNamespace(cell_count=6, cells_of=lambda chart: np.arange(1, 5)),
                           process=SimpleNamespace(set_soil=lambda load, hyd: got.update(load=load, hyd=hyd)))
    start_egg(PanTask(), flow, sense, 2)
    assert (got["load"] > 0).tolist() == [[False, True, True, False, False, False]] * 2
