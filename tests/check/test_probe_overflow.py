# SPDX-License-Identifier: Apache-2.0
"""OSS-7 (audit EXAMPLES-8, SCR-OSS22-7): when the check's own contact probe overflows at Newton's automatic size
(64 worlds of scanned meshes on a GPU: 3,330,535 triangle pairs for 2,703,360 slots), it is rebuilt with room for the
counts the overflow reports; when that is not enough either, the report carries a warning instead of a silent pass.
CPU, with Newton's size faked small."""

from __future__ import annotations

import warnings

import pytest

import alloy.check
from alloy.cards import types as ct
from alloy.check import contacts as cc
from alloy.scene.steppers import ContactBudget

TABLE = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))
CUBE = ct.Placement("cube_beech_30", (0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 1.0))
SCENE = ct.SceneCard(robots=(), objects=(TABLE, CUBE))


def _short_newton_size(monkeypatch):
    """Newton's automatic size (budget None) comes out at one contact slot per world; counted budgets stand."""
    real, budgets = cc.ContactProbe._make_layer, []

    def make(self, budget):
        budgets.append(budget)
        return real(self, ContactBudget(rigid_contacts=1) if budget is None else budget)

    monkeypatch.setattr(cc.ContactProbe, "_make_layer", make)
    return budgets


def _check(**kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = alloy.check(SCENE, worlds=2, seconds=0.1, device="cpu", **kw)
    return report, [str(w.message) for w in caught if "contact probe" in str(w.message)]


def test_overflow_at_newton_size_is_rebuilt_from_the_count(monkeypatch):
    clean, _ = _check()
    budgets = _short_newton_size(monkeypatch)
    report, said = _check()
    assert budgets[:2] == [None, None] and 3 <= len(budgets) <= 2 + cc.COUNTED_REBUILDS
    assert budgets[-1].rigid_contacts > budgets[2].rigid_contacts > 1 or len(budgets) == 3
    assert not any("still overflows" in s for s in said)
    assert report.ok and not report.warnings(), report.summary()
    assert report.checks == clean.checks
    assert report.metrics["overlap"] == clean.metrics["overlap"]          # the same pair depths as an unfaked probe
    assert report.metrics["penetration"]["max_pen_m"] == clean.metrics["penetration"]["max_pen_m"]


def test_overflow_that_stays_is_a_warning_not_a_silent_pass(monkeypatch):
    _short_newton_size(monkeypatch)
    monkeypatch.setattr(cc, "COUNTED_REBUILDS", 0)
    report, said = _check()
    assert len([s for s in said if "still overflows" in s]) == 1
    (f,) = [f for f in report.findings if "pair depths are incomplete" in f.message]
    assert f.check == "overlap" and f.level == "warn" and report.checks["overlap"] == "warn" and report.ok
    assert "contacts for 2 slots" in f.message and "not affected" in f.message and "worlds=" in f.fix


def test_grown_sizes():
    assert cc._grown(2000, 800, False) == 2500 and cc._grown(700, 800, False) == 800
    assert cc._grown(700, 800, True) == 1600 and cc._grown(900, 800, True) == 1600


@pytest.mark.parametrize("need, text", [
    ({"triangle_pairs": 9, "triangle_slots": 4, "rigid_contacts": 1, "rigid_slots": 2, "reducer": 1},
     "9 triangle pairs for 4 slots"),
    ({"triangle_pairs": 1, "triangle_slots": 4, "rigid_contacts": 1, "rigid_slots": 2, "reducer": 1},
     "the contact reducer dropped candidates")])
def test_need_text(need, text):
    assert cc._need_text(need) == text
