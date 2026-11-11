# SPDX-License-Identifier: Apache-2.0
"""pad-finish (SC9): the chain's wiring on the host (DC5's follow-up on 20586cbe7, S6 and the S5 call sites): ``run``
scores six resumed stage ends with every gate in success, ``_chain`` gives each wait the budget left and the edger
the margin band's sheen to wait for, the edger's end wait gets what its program left, and the edger and the ridden
tools refuse an opening that reaches their program. Fakes stand in for the simulations."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from alloy.tasks import concrete_materials as CM
from alloy.tasks import pad_finish as PF
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_finish_stage as ST
from alloy.tasks import pad_finish_stages as SG
from alloy.tasks.concrete_metrics import Criterion
from alloy.tasks.concrete_pad_start import pad_fields
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram
from alloy.tasks.pad_finish_handoff import Records
from tests.tasks.test_pad_finish_gates import _HELD, NXY, W, _closed, _Plug, _stage

STAGES = ("darby", "wait_edge", "edge", "margin", "wait_trowel", "trowel")


@pytest.fixture(scope="module")
def f0():
    return [pad_fields(PF.TASK, w) for w in range(W)]                # run(2, seed 0): world w is seed w


def _ends(fs, age=7200.0):
    """A chain end, world w closed on its own start ``fs[w]`` (``_closed``), the paste ``age`` old."""
    one = [_closed(f, worlds=1) for f in fs]

    def cat(xs):
        return np.concatenate([np.atleast_1d(np.asarray(x)) for x in xs], axis=0)
    end = SG.StageEnd(surface=cat([e.surface for e in one]), states={k: cat([e.states[k] for e in one])
                                                                      for k in one[0].states},
                      records=Records({k: cat([e.records.fields[k] for e in one]) for k in one[0].records.fields}),
                      books={k: cat([e.books[k] for e in one]) for k in one[0].books}, stepped_s=1.0, waited_s=0.0,
                      tools_off=np.zeros(len(fs), int), rest={})
    end.states["age"][:] = age
    return end


def _run(monkeypatch, tmp_path, f0, trowel=None, edge_error=None, ride=(True, True)):
    """``run(2, 0)`` resumed from six saved ends (the trowel's ``trowel``), the verdict faked to pass."""
    passing = [Criterion("smooth_hard", 1.0, 0.95, True)]                # one judged criterion: none is no success
    monkeypatch.setattr(PF.PFM, "pad_finish", lambda *a, **k: NS(success=True, score=1.0, criteria=passing,
                                                                   side_radius={}, side_arc_lip={}))
    for k in STAGES:
        end = trowel if k == "trowel" and trowel is not None else _ends(f0)
        end.rest_error = edge_error if k == "edge" else None
        end.diag = {"ride_ok": list(ride)} if k == "margin" else {}
        PF._save(str(tmp_path), k, end)
    return PF.run(W, 0, resume=str(tmp_path))


def test_run_counts_every_gate_in_success(monkeypatch, tmp_path, f0):
    """S6 (N05-N07): with the verdict passing, a world succeeds only with its books closed, its clock within 5 h, its
    rides held and no stage refused; each gate shows in its verdict."""
    r = _run(monkeypatch, tmp_path, f0)
    assert r.success == W and all(v["books_ok"] and v["within_limit"] and v["ride_ok"] for v in r.verdicts)
    assert all(v["success"] and list(v["alloy_gates"]) == ["ride_ok", "books_ok", "within_limit", "stages_at_rest"]
               for v in r.verdicts)                                    # Alloy's gates, named apart from GA's criteria
    assert r.limit_s == PF.LIMIT_S and r.within_limit == [True, True] and r.flags == []
    end = _ends(f0)
    end.surface[1, 300, 200] -= 1e-3                                   # world 1's paste leaks one cell
    end.states["age"][0] = float(f0[0]["age"]) + PF.LIMIT_S + 1.0     # world 0 past their 5 h
    r = _run(monkeypatch, tmp_path, f0, trowel=end)
    assert [(v["books_ok"], v["within_limit"]) for v in r.verdicts] == [(True, False), (False, True)]
    assert r.success == 0 and not any(v["success"] for v in r.verdicts)
    assert _run(monkeypatch, tmp_path, f0, ride=(True, False)).success == 1
    r = _run(monkeypatch, tmp_path, f0, edge_error="pad-finish wait: not ready within the budget")
    assert r.success == 0 and list(r.rest_errors) == ["edge"]


def test_each_wait_gets_the_budget_left_and_the_edger_waits_for_the_margin_band(monkeypatch, f0):
    """S6 (N10, N11, N14): the waits get their 5 h less the clock used before them; the edger's end wait is the
    margin band's sheen (``MG.band``), not the pad's."""
    ages, calls = iter([645.0, 7845.0, 8000.0, 8300.0, 10200.0, 10500.0]), []

    def stage(name):
        def go(fs, *a, ready=None, limit=None, device="cpu"):
            if name == "wait":
                ready, limit = a[3], a[4]
            calls.append((name, ready, limit))
            return _ends(f0, next(ages))
        return go
    for k in ("darby", "wait", "edge", "margin", "trowel"):
        monkeypatch.setattr(SG, k, stage(k))
    PF._chain(f0, 21.1, "cpu", None, None, None)
    a0 = float(f0[0]["age"])
    assert [c[0] for c in calls] == ["darby", "wait", "edge", "margin", "wait", "trowel"]
    limits = {i: calls[i][2] for i in (1, 2, 4)}
    assert limits == {1: PF.LIMIT_S - (645.0 - a0), 2: PF.LIMIT_S - (7845.0 - a0), 4: PF.LIMIT_S - (8300.0 - a0)}
    ready, band = calls[2][1], MG.band(f0[0]).reshape(-1)
    paste = np.asarray(f0[0]["paste_mask"], bool).reshape(-1)
    w = np.zeros((W, band.size))
    assert ready({"w": w, "age": w})
    w[1, np.flatnonzero(paste & ~band)[0]] = 2 * CM.SHEEN_W                # a sheen off the band: still ready
    assert ready({"w": w, "age": w})
    w[1, np.flatnonzero(band)[0]] = 2 * CM.SHEEN_W
    assert not ready({"w": w, "age": w})


def _program_stage(monkeypatch, opening, start_waited=0.0):
    """The stages' simulations faked: the opening watch steps ``opening`` periods, waits ``start_waited`` s; a settle
    takes 7 periods; every rest check holds after 10. Returns what the program and the end wait were given."""
    got = {}
    flow, layer, carriage = _stage()
    flow.r["ledger"] = [0.0]
    monkeypatch.setattr(SG, "_build", lambda *a: None)
    monkeypatch.setattr(ST, "require_placed", lambda *a: None)
    monkeypatch.setattr(ST, "start_stage", lambda *a: None)
    monkeypatch.setattr(ST, "rest_after", lambda *a: dict(_HELD))
    monkeypatch.setattr(ST, "seed_yields", lambda *a: None)
    monkeypatch.setattr(SG, "_open", lambda *a: (SG.Watch(checks=[dict(_HELD)], periods=opening),
                                                  SG.Waited(start_waited), {}))
    monkeypatch.setattr(SG, "_step", lambda sim, fl, periods, name, hooks=None: got.update(
        periods=periods, hooks=hooks) or (0, 0))
    monkeypatch.setattr(SG, "settle", lambda *a: 7)
    monkeypatch.setattr(SG.PR, "tools_back", lambda *a: np.zeros(W, int))
    monkeypatch.setattr(SG, "snapshot", lambda *a: {})
    monkeypatch.setattr(SG, "wait_for", lambda sim, fl, la, ca, ready, limit, crop, watch: got.update(
        left=limit) or SG.Waited())
    monkeypatch.setattr(SG, "_finish", lambda *a, **kw: kw)
    fs = [{"h": np.zeros(NXY), "compact": np.zeros(NXY), "age": 7200.0} for _ in range(W)]
    return got, (carriage, flow, layer), fs


def _edge(monkeypatch, opening, start_waited=0.0):
    got, (carriage, flow, layer), fs = _program_stage(monkeypatch, opening, start_waited)
    cart = (0.0, 0.8, 0.3)
    monkeypatch.setattr(SG, "PE", NS(PadEdge=lambda: NS(cart=cart, exchange_dt=0.02), build_scene=lambda *a: None,
                                     program=lambda task, f, t0: (ToolProgram("edger", (
                                         Stroke(0.0, cart, mode=HEIGHT),)), 100.0),
                                     plugins=lambda *a: (carriage, flow, layer, NS(marks=lambda: np.zeros((W, 6)))),
                                     chart_fields=None, edger_body=lambda task: None))
    SG.edge(fs, -1.5e-3, Records(), {}, ready=lambda r: True, limit=1000.0, pre_ready=lambda r: True)
    return got


def test_the_edgers_end_wait_gets_what_its_program_left(monkeypatch):
    """S6 (N13): the edger's end wait gets its limit less its start wait and every period it stepped (the opening
    watch, the program, the settle, the end's watch)."""
    got = _edge(monkeypatch, opening=10, start_waited=30.0)
    assert got["periods"] == 5000 - 10
    assert got["left"] == pytest.approx(1000.0 - 30.0 - (5000 - 10 + 7 + 10 + 10) * SG.PERIOD, rel=1e-12)


def test_the_edger_refuses_an_opening_that_reaches_its_program(monkeypatch):
    """S5 (N25): an opening of ``START`` or more refuses the edger before its program steps."""
    with pytest.raises(ValueError, match="first key"):
        _edge(monkeypatch, opening=round(SG.START / SG.PERIOD))


def _ridden(monkeypatch, opening, hooks):
    got, plugins, fs = _program_stage(monkeypatch, opening)
    monkeypatch.setattr(SG.PR, "build_scene", lambda *a: None)
    monkeypatch.setattr(SG, "PT", NS(split=lambda task: None, HOLD=20.0, RideWatch=lambda *a: NS(
        hooks=lambda: dict(hooks), ok=lambda: np.ones(W, bool), need=np.zeros(W), pressed=np.zeros(W), log=[])))
    SG._ride(fs, Records(), {}, SG.Ridden("margin", NS(exchange_dt=0.02), plugins, None, (), 10.0, 1.0), "cpu",
             None, 0.0)
    return got


def test_a_ridden_tool_checks_every_run_after_its_opening_or_refuses(monkeypatch):
    """S5 (N23, N24): the ride checks move by the opening's periods; an opening that reaches the program, or one that
    would swallow a ride check, is refused."""
    check = object()
    got = _ridden(monkeypatch, 10, {300: check})
    assert got["periods"] == 500 - 10 and got["hooks"] == {290: check}
    with pytest.raises(ValueError, match="first key"):
        _ridden(monkeypatch, 270, {300: check})
    with pytest.raises(ValueError, match="inside the opening"):
        _ridden(monkeypatch, 150, {100: check})
