# SPDX-License-Identifier: Apache-2.0
"""rinse_glassware's verdict (Gym-Anything's rinse-glassware gates, task.py:694-711, and score, scoring.py:88-102)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.rinse_glass_verdict import GlassEnd, verdict

CELLS = np.array([[x, y] for y in (np.arange(5) - 2) * 0.1 for x in (np.arange(5) - 2) * 0.1])
FLOOR = 0.017


def _glass(k: int, **kw) -> GlassEnd:
    h = 0.095
    g = GlassEnd(soap_ug0=1600.0, soap_ug=2.0, soap_gauge=0.0, sanitized_s=31.0, into_ug=3.0, free_water_ml=0.5,
                 rack_xyz=np.array([CELLS[k, 0] + 0.005, CELLS[k, 1], FLOOR + h - 0.012]), foot_up=FLOOR + h,
                 height=h, tilt_deg=179.0, peak_impact=0.2, z=1.0)
    for key, val in kw.items():
        setattr(g, key, val)
    return g


def test_six_done_glasses_pass():
    v = verdict([_glass(k) for k in range(6)], CELLS, FLOOR, 0.0, outlet_off=True, gripping=False, books=True)
    assert v.success and v.score == 1.0 and len(v.criteria) == 8


def test_one_glass_mouth_up_loses_its_storage_stage():
    """Its credit is (r + r z)/3 = 2/3; the mean over six is 17/18; the score 0.9 x that."""
    gs = [_glass(k) for k in range(6)]
    gs[2] = _glass(2, tilt_deg=2.0)
    v = verdict(gs, CELLS, FLOOR, 0.0, True, False, True)
    assert v.failed() == ["placed"] and v.score == pytest.approx(0.9 * 17.0 / 18.0)


def test_a_broken_glass_earns_nothing_and_costs_a_unit():
    gs = [_glass(k) for k in range(6)]
    gs[0] = _glass(0, peak_impact=1.5)
    v = verdict(gs, CELLS, FLOOR, 0.0, True, False, True)
    assert v.failed() == ["intact_and_not_dropped"]
    assert v.progress == pytest.approx(5.0 / 6.0 - 1.0 / 6.0)


def test_soap_over_ten_micrograms_and_a_floor_puddle():
    """40 µg left: r = log(1600/40)/log(1600/10), its credit (r + r z + r z t)/3 = r (z = t = 1); 75 mL on the floor
    costs excess 0.5 over six glasses."""
    gs = [_glass(k) for k in range(6)]
    gs[4] = _glass(4, soap_ug=40.0)
    v = verdict(gs, CELLS, FLOOR, floor_ml=75.0, outlet_off=True, gripping=False, books=True)
    assert sorted(v.failed()) == ["contained", "soap_free"]
    r = math.log(1600.0 / 40.0) / math.log(160.0)
    assert v.score == pytest.approx(0.9 * ((5.0 + r) / 6.0 - 0.5 / 6.0), abs=1e-12)


@pytest.mark.parametrize(("field", "value", "gate"), [("soap_ug", 10.5, "soap_free"), ("soap_gauge", 1.01e-6, "soap_free"),
                                                      ("sanitized_s", 29.99, "sanitized"), ("into_ug", 10.5, "sanitized"),
                                                      ("free_water_ml", 2.01, "drained")])
def test_each_limit_is_theirs(field, value, gate):
    """Just over a limit fails its gate: 10 µg, 1 mg/m², 30 s, 10 µg carried in, 2 mL."""
    gs = [_glass(k) for k in range(6)]
    gs[1] = _glass(1, **{field: value})
    assert verdict(gs, CELLS, FLOOR, 0.0, True, False, True).failed() == [gate]


@pytest.mark.parametrize(("kw", "why"), [
    ({"rack_xyz": np.array([CELLS[1, 0] + 0.026, CELLS[1, 1], FLOOR + 0.083])}, "26 mm off its compartment"),
    ({"foot_up": FLOOR + 0.095 - 0.0151}, "foot 15.1 mm low"),
    ({"foot_up": FLOOR + 0.095 + 0.0201}, "foot 20.1 mm high"),
    ({"tilt_deg": 159.9}, "tilted 159.9°")])
def test_placed_is_their_compartment_test(kw, why):
    """Within 25 mm of a compartment's centre, the foot H over the floor (-15 mm..+20 mm), tilt over 160°
    (``task.py:525-538``): just outside each fails ``placed``."""
    gs = [_glass(k) for k in range(6)]
    gs[1] = _glass(1, **kw)
    assert verdict(gs, CELLS, FLOOR, 0.0, True, False, True).failed() == ["placed"], why


def test_holding_a_glass_loses_every_storage_stage():
    """Not released: water_off_and_hands_free fails and t = 0 for every glass, so each credit is (r + r z)/3 = 2/3."""
    v = verdict([_glass(k) for k in range(6)], CELLS, FLOOR, 0.0, outlet_off=True, gripping=True, books=True)
    assert v.failed() == ["water_off_and_hands_free"] and v.score == pytest.approx(0.9 * 2.0 / 3.0, abs=1e-12)


def test_their_report_rounding_decides():
    """1,500 samples of 0.02 s (29.99999999999945 s) report 30.00 s and pass; 10.0004 µg reports 10.000 and passes."""
    t = 0.0
    for _ in range(1500):                 # the tracker's clock adds one period at a time (sum() compensates)
        t += 0.02
    assert t < 30.0
    gs = [_glass(k) for k in range(6)]
    gs[0] = _glass(0, sanitized_s=t, soap_ug=10.0004)
    assert verdict(gs, CELLS, FLOOR, 0.0, True, False, True).success


def test_not_judged_gates_are_none_and_a_judged_failure_still_fails():
    """Foam, the drainboard, contacts while held and the hands are not judged (AB-gym-3): None on the passing glasses,
    success from the judged gates, and the outlet left open fails the gate and the verdict."""
    v = verdict([_glass(k) for k in range(6)], CELLS, FLOOR, 0.0, outlet_off=True, gripping=False, books=True)
    for gate in ("soap_free", "contained", "intact_and_not_dropped", "water_off_and_hands_free"):
        assert v.criteria[gate][0] is None, gate
    assert v.criteria["placed"][0] is True and "static" in v.criteria["placed"][1]["rack"]
    assert v.success and v.judged == (4, 8) and v.not_judged() == [
        "soap_free", "contained", "intact_and_not_dropped", "water_off_and_hands_free"]
    assert v.summary().startswith("4 of 8 Gym-Anything criteria judged; not judged:")
    open_ = verdict([_glass(k) for k in range(6)], CELLS, FLOOR, 0.0, outlet_off=False, gripping=False, books=True)
    assert open_.criteria["water_off_and_hands_free"][0] is False and not open_.success
