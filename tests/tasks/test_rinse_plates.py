# SPDX-License-Identifier: Apache-2.0
"""rinse_plates (Gym-Anything pre-rinse-plates): the 1 cm soil gauge, the rack geometry, the verdict, a short run."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.harness.scene import build_simulation
from alloy.tasks.rinse_plates import VRL, EndState, PreRinse, gauges, placed, run, start_soil, verdict
from alloy.tasks.rinse_plates_scene import Layout, quat_x
from alloy.tasks.rinse_ware import Plate


@pytest.fixture(scope="module")
def one_plate():
    task = PreRinse(plates=1)
    flow, hand, balancer = task.plugins()
    sim = build_simulation(task.scene(2), [flow, hand, balancer], "cpu")
    return task, flow, sim


def test_gauge_is_the_worst_area_weighted_centimetre(one_plate):
    """The gauge bins the face by 1 cm after a 5 mm shift along the normal and takes each bin's area-weighted mean
    (their task.py:74-88): one dirty cell reads its load times its share of its bin's area."""
    task, flow, sim = one_plate
    C = flow.tables.cell_count
    load = np.zeros((2, C))
    assert np.all(gauges(flow, load, 0) == 0.0)
    c = C // 2
    load[1, c] = 1.0e-3
    g = gauges(flow, load, 0)
    t = flow.tables
    p = t.cpos.numpy() + 0.005 * t.cnrm.numpy()
    key = np.floor(p / 0.01).astype(np.int64)
    same = np.all(key == key[c], axis=1)
    assert g[0] == 0.0
    assert g[1] == pytest.approx(1.0e-3 * t.area_np[c] / t.area_np[same].sum(), rel=1e-12)


def test_start_soil_follows_their_recipe(one_plate):
    """0.20 kg/m² x (1 + 0.25 N(0, 1)) on the well within a ragged edge, at least 0.02; none on the rim slope;
    world 0 at the nominal load, others scaled 0.7-1.3 (their system.py:380-388, layout.py:494-496)."""
    task, flow, sim = one_plate
    load = start_soil(task, flow, 2)
    r = np.hypot(*flow.cell_centres(0).reshape(-1, 3)[:, :2].T)
    on = load[0] > 0
    assert r[on].max() <= task.layout.plate.well_radius * 1.06 + 2e-3
    assert np.all(load[:, r > task.layout.plate.well_radius * 1.07] == 0.0)
    assert load[0, on].min() >= 0.02 and load[0, on].mean() == pytest.approx(0.20, rel=0.05)


def test_rack_slots_hold_plates_upright():
    """A plate in its slot counts as placed (rack frame, face normal within 30° of horizontal); a plate lying flat or
    out of the rack does not (their task.py:540-551)."""
    task, lay = PreRinse(), Layout()
    s = lay.slot(2)
    ok, lean = placed(task, np.r_[s.pos, s.quat])
    assert ok and lean == pytest.approx(math.degrees(lay.lean), abs=1e-6)
    assert not placed(task, np.r_[s.pos, quat_x(0.0)])[0]
    assert not placed(task, np.r_[s.pos + [0.0, 0.3, 0.0], s.quat])[0]


def test_verdict_rules():
    """Gates and score of their task.py:734-742 and task_set/scoring.py:103-111."""
    task, lay = PreRinse(plates=2), Layout()
    q = np.stack([np.r_[lay.slot(k).pos, lay.slot(k).quat] for k in range(2)])
    good = EndState(q=q, park_m=0.01, books=True, valve_open=False, held=False, release_speed=0.05)
    g0 = np.full(2, 2.0e-1)
    v = verdict(task, g0, np.full(2, 0.5 * VRL), good)
    assert v.success and v.score == 1.0
    v = verdict(task, g0, np.array([0.5 * VRL, 2.0 * VRL]), good)
    assert v.failed() == ["visibly_clean"] and v.score < 0.9
    v = verdict(task, g0, g0, good)
    assert v.score == 0.0                                                     # untouched: no credit
    v = verdict(task, g0, np.full(2, 0.5 * VRL), EndState(**{**good.__dict__, "park_m": 0.3}))
    assert v.failed() == ["valve_parked"] and v.score == pytest.approx(0.9 * 0.5)
    v = verdict(task, g0, np.full(2, 0.5 * VRL), EndState(**{**good.__dict__, "books": False}))
    assert v.score is None


def test_plate_chart_is_the_face():
    """The chart is the well floor and the rim slope of a 9 in plate, normal out of the face, open at both ends."""
    p = Plate()
    prof = p.profile()
    assert prof[1][0] == p.well_radius and prof[2] == (p.radius, prof[1][1] + p.well_depth)
    ch = p.chart("plate0")
    assert ch.ends == ("open", "open") and ch.contact_angle == 25.0


@pytest.mark.compile_heavy
def test_one_plate_short_rinse():
    """One plate, one short raster pass on a coarse chart: the hand takes the plate out and back into its slot, the
    valve parks, soil comes off where the jets went, and the film, supply and soil ledgers close."""
    task = PreRinse(plates=1, max_passes=1, half=0.03, spacing=0.03, pace=0.5, settle=1.0,
                    layout=Layout(plate=Plate(ds=4.0e-3, n_theta=32)))
    res = run(task, worlds=1)
    assert res.failed == [["visibly_clean"]]                                  # one short pass cannot clean it
    assert res.gauge_ug_cm2[0][0] <= res.start_gauge_ug_cm2[0][0]
    assert sum(res.removed_by.values()) > 0.0
    assert res.soil_ledger_max < 1e-9 and res.film_ledger_max < 1e-9
    assert res.valve_park_m[0] < 0.05
    assert res.clean_after_s == [[None]] and 0.0 < res.release_speed[0] < 0.2  # put back, not dropped
    assert res.valve_track_mm[0] < 2.0                                        # the valve's weight carried once


def test_not_judged_gates_are_none_and_a_judged_failure_still_fails():
    """The centre disc, the clean table, contacts while held and the hands are not judged (AB-gym-3); the floor
    water part of ``contained`` is judged when the run measures it."""
    task, lay = PreRinse(plates=2), Layout()
    q = np.stack([np.r_[lay.slot(k).pos, lay.slot(k).quat] for k in range(2)])
    good = EndState(q=q, park_m=0.01, books=True, valve_open=False, held=False, release_speed=0.05)
    v = verdict(task, np.full(2, 2.0e-1), np.full(2, 0.5 * VRL), good)
    for gate in ("visibly_clean", "contained", "water_off_and_hands_free", "intact_and_not_dropped"):
        assert v.criteria[gate][0] is None, gate
    assert v.success and v.score == 1.0 and v.judged == (3, 7)
    wet = verdict(task, np.full(2, 2.0e-1), np.full(2, 0.5 * VRL), EndState(**{**good.__dict__, "floor_ml": 60.0}))
    assert wet.failed() == ["contained"] and not wet.success
    dry = verdict(task, np.full(2, 2.0e-1), np.full(2, 0.5 * VRL), EndState(**{**good.__dict__, "floor_ml": 0.0}))
    assert dry.criteria["contained"][0] is None and dry.success
    left_open = verdict(task, np.full(2, 2.0e-1), np.full(2, 0.5 * VRL),
                        EndState(**{**good.__dict__, "valve_open": True}))
    assert left_open.failed() == ["water_off_and_hands_free"] and not left_open.success
