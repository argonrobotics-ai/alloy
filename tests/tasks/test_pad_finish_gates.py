# SPDX-License-Identifier: Apache-2.0
"""pad-finish (SC9): what success needs beside the verdict, and the rules a hand-off and a wait keep (DC5's review of
7ef38a4cb: MF1, S1-S3), on the host with fakes, without a simulation: the whole chain's books per world, their 5 h
budget, the waits' re-checks and refusals, the hand-off's refusals and folds, and the watch's report."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.tasks import concrete_pad_start as PS
from alloy.tasks import pad_finish as PF
from alloy.tasks import pad_finish_stage as ST
from alloy.tasks import pad_finish_stages as SG
from alloy.tasks import pad_finish_watch as WT
from alloy.tasks.pad_finish_handoff import START_STATES, Records
from tests.tasks.test_pad_finish_chain import _P, _Arr, _end

W, NXY = 2, (2, 3)
C = NXY[0] * NXY[1]


@pytest.fixture(scope="module")
def start0():
    return PS.pad_fields("pad-finish", 0)


def _closed(f, worlds=2, worked_in=0.0):
    """A chain's end on ``f``'s cells whose paste books close (the rest went off the pad) and whose water books close
    (what bled is the end's film and trapped water, plus ``worked_in``, less the start's)."""
    end = _end(f, worlds=worlds)
    area, paste = float(f["cell"]) ** 2, np.asarray(f["paste_mask"], bool)
    start = float(((f["h"] - f["b"]) * area).sum())
    final = np.array([float(((s - f["b"]) * area).sum()) for s in end.surface])
    water0 = float((np.asarray(f["w"], np.float64) + np.asarray(f["q"], np.float64))[paste].sum()) * area
    water1 = np.array([float((end.states["w"][w] + end.states["q"][w])[paste].sum()) * area for w in range(worlds)])
    end.books = {"off_pad": start - final, "water_bled": water1 + worked_in - water0,
                 "water_worked_in": np.full(worlds, worked_in), "water_evaporated": np.zeros(worlds)}
    return end


def test_a_world_whose_books_leak_fails_alone(start0):
    """MF1: success needs the books closed per world (paste 1e-9, water 1e-12, relative, atol 0). One cell's paste
    gone unbooked in world 1 fails world 1 only; so does 1 mm of bleed water gone in world 0."""
    f0, end = [start0, start0], _closed(start0)
    g = PF.gates(f0, {"margin": end, "trowel": end})
    assert g["books_ok"].tolist() == [True, True]
    end.surface[1, 300, 200] -= 1e-3
    assert PF.gates(f0, {"margin": end, "trowel": end})["books_ok"].tolist() == [True, False]
    end = _closed(start0)
    end.states["w"][0, 300, 200] -= 1e-3
    assert PF.gates(f0, {"margin": end, "trowel": end})["books_ok"].tolist() == [False, True]


def test_the_books_tolerances_are_their_bounds(start0):
    """S7: per book a leak at half its tolerance passes and one at twice fails (paste 1e-9 of the start's paste,
    water 1e-12 of what bled; relative, atol 0)."""
    paste_tol, water_tol = 1e-9, 1e-12                                 # DC5 4 (the L4 runs closed water at 1.3e-14)
    f0, end = [start0, start0], _closed(start0)
    paste = PF.paste_books(f0, end)["start"]
    end.books["off_pad"] = end.books["off_pad"] + np.array([0.5, 2.0]) * paste_tol * paste
    assert PF.gates(f0, {"margin": end, "trowel": end})["books_ok"].tolist() == [True, False]
    end = _closed(start0)
    bled = np.asarray(end.books["water_bled"], np.float64)
    end.books["water_evaporated"] = np.array([0.5, 2.0]) * water_tol * bled
    assert PF.gates(f0, {"margin": end, "trowel": end})["books_ok"].tolist() == [True, False]


def test_the_water_books_count_what_was_worked_in(start0):
    """M18: water worked into the paste is a book (start + bled = end + evaporated + worked in): with it the books
    close; left out, they do not."""
    end = _closed(start0, worlds=1, worked_in=2e-6)
    assert abs(float(PF.water_books([start0], end)["residual"][0])) <= 1e-15
    end.books["water_worked_in"] = np.zeros(1)
    assert abs(float(PF.water_books([start0], end)["residual"][0])) > PF.WATER_TOL


LEDGER_OUTS = ("spilled", "evaporated", "absorbed", "drained_total", "clipped", "dripped", "pooled_total", "to_tools")
# ^ the film's ledger's outs on pad-finish's charts (surface_flow host_api.read "ledger": film + these - supplied)


def test_every_out_leaves_the_paste_books(start0):
    """M15: every out of the film's ledger is paste gone, each booked once; the film's other books (``drained``,
    ``pooled``: running, not totals) are not. Each out a different amount, so dropping any one fails."""
    end = _closed(start0, worlds=1)
    assert set(PF.OUTS) == set(LEDGER_OUTS) and set(PF.OUTS) <= set(ST.FLOW_BOOKS)
    outs = {k: np.array([(i + 1) * 1e-6]) for i, k in enumerate(LEDGER_OUTS)}
    end.books.update(outs, supplied=np.array([sum(float(v[0]) for v in outs.values())]), drained=np.array([5.0]),
                     pooled=np.array([7.0]))
    assert abs(float(PF.paste_books([start0], end)["residual"][0])) <= 1e-15
    for k in LEDGER_OUTS:
        b = PF.paste_books([start0], SG.StageEnd(**{**end.__dict__, "books": {**end.books, k: np.zeros(1)}}))
        assert abs(float(b["residual"][0])) > PF.PASTE_TOL, k


def test_the_clock_and_its_budget_are_per_world(start0):
    """S1, M13, M14: each wait gets the 5 h left (the most any world has used); the material clock is per world and a
    world past 5 h is not within the limit."""
    f0 = [dict(start0), dict(start0)]
    later = [dict(f, age=float(f["age"]) + d) for f, d in zip(f0, (3600.0, 3700.0), strict=True)]
    assert PF.budget(f0, later) == PF.LIMIT_S - 3700.0
    end = _closed(start0)
    end.states["age"][1] = float(start0["age"]) + PF.LIMIT_S + 1.0
    g = PF.gates(f0, {"margin": end, "trowel": end})
    assert np.allclose(g["material_s"], [7200.0 - float(start0["age"]), PF.LIMIT_S + 1.0], rtol=1e-12, atol=0.0)
    assert g["within_limit"].tolist() == [True, False]


def test_success_needs_the_verdict_every_gate_and_no_refusal(start0):
    """M27: a world succeeds only with its verdict, every gate (ride, books, budget) and no stage refused."""
    ok = {"ride_ok": True, "books_ok": True, "within_limit": True}
    assert PF.world_success(True, ok, {})
    assert not PF.world_success(True, ok, {"edge": "pad-finish hand-off: a tool is down on the paste or loaded"})
    assert not PF.world_success(False, ok, {})
    assert not any(PF.world_success(True, {**ok, k: False}, {}) for k in ok)
    end = _closed(start0)
    end.diag["ride_ok"] = [True, False]
    assert PF.gates([start0, start0], {"margin": end, "trowel": _closed(start0)})["ride_ok"].tolist() == [True, False]


def test_the_row_flags_film_over_depth_or_capped():
    """N4: a stage whose film went over its step bound's depth or hit the face cap is named in the row."""
    ends = {"edge": NS(diag={"cells_over_depth": 3, "faces_at_cap": 0}), "wait_edge": NS(diag={})}
    assert PF.flags(ends) == ["edge: cells_over_depth 3"]


def test_the_records_or_the_flags_and_sum_the_distance():
    """M10: the same cell late (or wet worked) in two stages is one cell (their count of cells, T:526), not two."""
    rec = Records()
    one = {"late": np.array([[0, 1]]), "wet_worked": np.array([[1, 0]]), "worked": np.array([[0.1, 0.0]])}
    rec.add(one)
    rec.add(one)
    assert rec.fields["late"].tolist() == [[0, 1]] and rec.fields["wet_worked"].tolist() == [[1, 0]]
    assert np.allclose(rec.fields["worked"], [[0.2, 0.0]], rtol=1e-12, atol=0.0)


# --- a stage's hand-off and waits, on fakes ----------------------------------------------------------------------

class _State(_Arr):
    @property
    def shape(self):
        return self.a.shape


class _Plug:
    """A plug-in's host API the hand-off reads: registered states, inputs, ``read``, the film's cell areas."""

    def __init__(self, states, ins=None, read=None):
        self.states = {k: _State(v) for k, v in states.items()}
        self.ins = {k: _Arr(v) for k, v in (ins or {}).items()}
        self.r, self.tables, self.moving = dict(read or {}), NS(area_np=np.full(C, 4e-6)), False
        self.load, self.h = self.states.get("load"), self.states.get("h")

    def state_arrays(self):
        return self.states

    def inputs(self):
        return self.ins

    def read(self):
        return self.r


def _stage(load=0.0, tool_load=0.0, ceiling=np.inf):
    """A stage's film, layer and carriage on 2 worlds of 6 cells: world 0 supplied 1 uL a cell and bled, world 1
    supplied 2 uL a cell and not; one late cell in world 0."""
    flow = _Plug({"h": np.full((W, C), 1e-3), "supplied": np.array([[1e-6] * C, [2e-6] * C])},
                 {"base": np.full((W, C), 0.08, np.float32)},
                 {"tool_ceiling": np.full((W, C), ceiling), "tool_load": np.full((W, 1), tool_load)})
    late = np.zeros((W, C))
    late[0, 0] = 1
    layer = _Plug({"age": np.full((W, C), 7200.0), **{k: np.zeros((W, C)) for k in START_STATES}, "late": late,
                   "wet_worked": np.zeros((W, C)), "worked": np.full((W, C), 0.1),
                   "bled": np.array([[1e-6] * C, [0.0] * C])}, {"paste": np.ones((W, C), np.int32)})
    return flow, layer, _Plug({"load": np.full((W, 1), load)})


_HELD = {"rest": True, "rule": "bound", "bound": 0.5, "face_max": 0.0, "over": 0, "periods": 10}


def _crop(a):
    return np.asarray(a).reshape(a.shape[0], *NXY)


def test_a_hand_off_refuses_a_tool_down_loaded_or_holding_paste():
    """M01, M02: a store with paste left in it, a tool down on the film, or a carriage load over GA's 0.05 N
    (WAIT_CONTACT_N) refuses the hand-off; a load at it does not."""
    ST.check_rest(*_stage(load=0.05)[::2], _HELD)
    for kw, what in (({"tool_load": 1e-7}, "left on the tools"), ({"ceiling": 0.09}, "down on the paste"),
                     ({"load": 0.06}, "loaded")):
        flow, _, carriage = _stage(**kw)
        with pytest.raises(ValueError, match=what):
            ST.check_rest(flow, carriage, _HELD)


def test_a_hand_off_folds_the_records_books_per_world_and_keeps_a_wait_refusal():
    """M12, M17, S1: the hand-off folds this stage's records into the chain's (a cell flagged before stays flagged,
    the distance adds), sums the film's and the water's books per world, and keeps a wait's refusal as the stage's
    rest error with the run's fields."""
    flow, layer, carriage = _stage()
    before = {k: np.zeros((W, *NXY)) for k in ("late", "wet_worked", "worked")}
    before["wet_worked"][1, 1, 2], before["worked"][:] = 1, 0.2
    watch = SG.Watch(checks=[dict(_HELD)])
    end = SG._finish(None, flow, layer, carriage, Records(before), {"supplied": np.array([1e-5, 0.0])}, _crop, watch,
                     stepped_s=1.0, waited_s=0.0, tools_off=np.zeros(W, int))
    rec = end.records.fields
    assert rec["late"][0, 0, 0] == 1 and rec["wet_worked"][1, 1, 2] == 1 and rec["late"].sum() == 1
    assert np.allclose(rec["worked"], 0.3, rtol=1e-12, atol=0.0)
    assert np.allclose(end.books["supplied"], [1e-5 + 6e-6, 1.2e-5], rtol=1e-12, atol=0.0)
    assert np.allclose(end.books["water_bled"], [6 * 1e-6 * 4e-6, 0.0], rtol=1e-12, atol=0.0)
    assert end.rest_error is None and end.rest["film.checks"] == 1
    late = SG._finish(None, flow, layer, carriage, None, None, _crop, watch, stepped_s=1.0, waited_s=150.0,
                      tools_off=np.zeros(W, int), waits=(SG.Waited(150.0, "pad-finish wait: not ready"),))
    assert late.rest_error == "pad-finish wait: not ready" and late.surface.shape == (W, *NXY)


def _wait_rig(monkeypatch, break_after=None):
    """Their wait command and the watch's 0.2 s check, faked on the film's ``moving``: the command refuses a film
    the watch found moving (as ``material_wait``) and, after ``break_after`` commands, the film's hold breaks."""
    calls = []

    def wait(sim, flow, layer, carriage, seconds, rest=None):
        if not rest(flow):
            raise PluginError("material_wait: the film does not rest by the task's rule; step until it rests")
        calls.append(seconds)
        flow.moving = flow.moving or len(calls) == break_after
        return seconds

    def check(sim, flow, carriage):
        m = flow.moving
        return {"rest": not m, "rule": "none" if m else "bound", "bound": 1.2 if m else 0.5,
                "face_max": 1e-4 if m else 0.0, "over": 4 if m else 0, "periods": 10}
    monkeypatch.setattr(WT.PR, "material_wait", wait)
    monkeypatch.setattr(WT.ST, "rest_after", check)
    monkeypatch.setattr(WT.ST, "seed_yields", lambda flow, layer: None)
    flow, layer, carriage = _stage()
    layer.read = lambda: {"w": np.zeros((W, C)), "age": np.full((W, C), 7200.0)}
    watch = SG.Watch()
    watch.after(None, flow, carriage, layer)
    return calls, flow, layer, carriage, watch


def test_a_wait_stops_at_the_budget_left_and_says_so(monkeypatch):
    """S1, M13: a stage never ready waits exactly the budget left (the last command shortened to it), then hands
    off refused; a ready one does not wait."""
    calls, flow, layer, carriage, watch = _wait_rig(monkeypatch)
    out = SG.wait_for(None, flow, layer, carriage, lambda r: False, 150.0, _crop, watch)
    assert calls == [60.0, 60.0, 30.0] and out.seconds == 150.0 and "not ready within the budget" in out.refusal
    assert len(watch.checks) == 4                                          # the start's, then one per command
    assert SG.wait_for(None, flow, layer, carriage, lambda r: True, 150.0, _crop, watch) == SG.Waited()


def test_a_wait_rechecks_after_every_command_and_stops_on_a_refusal(monkeypatch):
    """M05, N8: the film is watched after every command; a hold that breaks during one refuses the next command, and
    the wait hands that refusal off instead of raising."""
    calls, flow, layer, carriage, watch = _wait_rig(monkeypatch, break_after=2)
    out = SG.wait_for(None, flow, layer, carriage, lambda r: False, 300.0, _crop, watch)
    assert calls == [60.0, 60.0] and out.seconds == 120.0 and "refused" in out.refusal
    assert not watch.last["rest"] and watch.last["periods"] == round(SG.FILM_SETTLE_S / SG.PERIOD)


def test_the_watch_keeps_a_check_it_stepped_on(monkeypatch):
    """S2, M06: a check that found the film moving and stepped it on until it held keeps that first refusal; the
    summary counts it and its bound, faces over 1 and face speed enter the maxima."""
    seq = iter([{"rest": False, "rule": "none", "bound": 1.1, "face_max": 1e-4, "over": 3, "periods": 10},
                {"rest": True, "rule": "held", "bound": 1.05, "face_max": 0.0, "over": 2, "periods": 10}])
    flow = _stage()[0]

    def check(sim, flow, carriage):                     # the second check's 10 periods move 1 mm between 2 cells
        r = next(seq)
        if r["rest"]:
            flow.h.a[1, :2] += (-1e-3, 1e-3)
        return r
    monkeypatch.setattr(WT.ST, "rest_after", check)
    watch = SG.Watch()
    check = watch.after(None, flow, None)
    assert check["rest"] and check["periods"] == 20 == watch.periods
    moved = pytest.approx(4e-9, rel=1e-12)                                # 0.5 x 2 mm x 4e-6 m² (world 1)
    assert check["stepped_on"] == {"bound": 1.1, "face_max": 1e-4, "over": 3, "moved_m3": moved}
    sm = watch.summary()
    assert (sm["film.rule"], sm["film.checks"], sm["film.held_checks"], sm["film.refused_checks"]) == ("held", 1, 1, 0)
    assert (sm["film.stepped_on_checks"], sm["film.max_bound"], sm["film.max_over"]) == (1, 1.1, 3)
    assert sm["film.max_face_before"] == 1e-4 and sm["film.stepped_on_moved_m3"] == pytest.approx(4e-9, rel=1e-12)


def test_a_stage_places_its_other_plug_ins_states():
    """N9: the edger's lip is placed by the stage (its mark carried as compact); a new state of it stops the stage."""
    flow, layer, carriage = _stage()
    ST.require_placed(flow, layer, carriage, (_P("lip"), ("lip",)))
    with pytest.raises(ValueError, match="new_mark"):
        ST.require_placed(flow, layer, carriage, (_P("lip", "new_mark"), ("lip",)))
