# SPDX-License-Identifier: Apache-2.0
"""Titration colour change (alloy/tasks/titration_colour.py) on the CPU: the pass test, a small flask titrated drop by
drop (stops on the first drop past equivalence, pink holds, species conserved exactly), and the burette pouring
beside the turning stir bar."""

from __future__ import annotations

import dataclasses

import pytest

from alloy.tasks.titration_colour import Titration, TitrationSpec, run, verdict

SMALL = TitrationSpec(flask=0.016, depth=0.008, rim=0.024, rev_per_s=0.0, run_in=1.0, settle=0.3, hold=0.5,
                      burette=0.6e-6, nozzle=(0.0, 0.0, 0.006), max_seconds=10.0)


def _rows(v_end_ml: float, spill: float = 0.0, err: int = 0) -> list:
    return [{"delivered_ml": 0.0, "spill": 0.0, "conservation_error": 0},
            {"delivered_ml": v_end_ml, "spill": spill, "conservation_error": err}]


@pytest.mark.parametrize("v_end_ml, held, spill, err, ok", [
    (1.70, 3.0, 0.0, 0, True),       # one drop past V_eq 1.64 mL, pink held 3 s
    (1.90, 3.0, 0.0, 0, False),      # four drops over
    (1.70, 2.0, 0.0, 0, False),      # pink not held long enough
    (1.70, 3.0, 0.02, 0, False),     # spilled
    (1.70, 3.0, 0.0, 1, False),      # a species not conserved
])
def test_verdict(v_end_ml, held, spill, err, ok):
    out = verdict(TitrationSpec(), _rows(v_end_ml, spill, err), held, 1.64e-6, 0.064e-6)
    assert out["success"] is ok


@pytest.mark.compile_heavy
def test_small_titration_stops_on_the_first_drop_past_equivalence():
    out = run(SMALL, device="cpu")
    v = out["verdict"]
    assert v["success"], v
    assert v["v_eq_ml"] == pytest.approx(0.2048, rel=0.01)       # 32 particles of 10 mM acid, 0.1 M base
    assert 0.0 <= v["over_drops"] <= 1.0
    assert v["conservation_error"] == 0 and out["health_ok"]
    before = [r["pink"] for r in out["rows"] if r["delivered_ml"] < v["v_eq_ml"]]
    assert max(before) < SMALL.pink_min        # before equivalence only the drop's landing spot flashes pink


@pytest.mark.compile_heavy
def test_burette_pours_beside_the_turning_stir_bar():
    """A pour beside a robot steps SPH on a coupler view (main 444201474 made its flags reach the full model, which
    the plug-in reads): drops arrive, each booked with the burette's composition, while the bar turns."""
    t = Titration(dataclasses.replace(SMALL, rev_per_s=3.0, bar=0.006, rate=1.0e-6), device="cpu")
    n0 = int((t.read()["active"][0] != 0).sum())
    t.burette.set(True)
    t.advance(0.6)
    row = t.observe()
    n = int((t.read()["active"][0] != 0).sum()) - n0
    assert n >= 3
    assert row["delivered_ml"] == pytest.approx(n * t.drop * 1e6, rel=1e-9)
    assert row["conservation_error"] == 0 and row["spill"] == 0.0 and t.sim.check()["ok"]
    assert row["stir_rev"] > 1.0                                     # 1.8 rev at 3 rev/s, less the servo's lag
