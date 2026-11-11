# SPDX-License-Identifier: Apache-2.0
"""pad-edge criteria (alloy/tasks/pad_metrics.py) on Gym-Anything's pad-edge start state (CPU, numpy).

An ideal edger riding on the form tops cuts the start surface down to the 3/8 in arc and marks the lip's slot: it
passes every pad-edge criterion. The square-edged start fails the arc and the lip. Their own pad-float start (the
edger's work as their layout draws it, ``layout.py:299-318``) presses the 76 mm track 1.2 mm down over the arc too,
so on pad-edge's form-top arc (T:291) it is short of the 95 % share: an edger must ride on the forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import pad_metrics as PM
from alloy.tasks.concrete_metrics import PasteGrid
from alloy.tasks.concrete_pad_start import FORM_H, pad_fields


def _grid(f: dict, wet: np.ndarray | None = None) -> PasteGrid:
    mask = f["paste_mask"]
    return PasteGrid(h=f["h"] - FORM_H, t=f["tb"] + f["ts"], cell=float(f["cell"]), paste_mask=mask,
                     C=f["compact"], wet_worked=np.zeros(mask.shape, np.int32) if wet is None else wet)


@pytest.fixture(scope="module")
def states():
    return pad_fields("pad-edge", 0), pad_fields("pad-float", 0)


def _ideal(start: dict) -> PasteGrid:
    """The start surface cut down to the arc about the nearest form, the lip's slot marked (L:312)."""
    g = _grid(start)
    d = PM.pad_distance(start["x"], start["y"])
    g.h = np.where(g.paste_mask & (d < PM.EDGER_RADIUS), np.minimum(g.h, PM.arc(np.maximum(d, 0.0))), g.h)
    g.C = np.where(g.paste_mask & (d < 0.0025), 1.0, 0.0).astype(np.float32)
    return g


def test_an_ideal_edger_passes_pad_edge(states):
    start, _ = states
    v = PM.pad_edge(_ideal(start), _grid(start), start["x"], start["y"])
    assert v.success, [(c.name, c.value) for c in v.criteria if not c.passed]
    assert v["radius_profile"].value > 0.99 and v["lip_passed"].value > 0.99
    assert v["no_gouge"].value == 0.0 and v["field_untouched"].value == 0.0
    assert v.score == 1.0 and v.progress.value == pytest.approx(1.0)


def test_their_edged_state_has_the_track_over_the_arc(states):
    start, edged = states
    v = PM.pad_edge(_grid(edged), _grid(start), start["x"], start["y"])
    assert 0.8 < v["radius_profile"].value < 0.9 and v["lip_passed"].value > 0.99
    assert 1.15 < v["no_gouge"].value < PM.GOUGE_MM                  # the track: 1.2 mm under the field


def test_the_square_start_fails_the_arc_and_the_lip(states):
    start, _ = states
    v = PM.pad_edge(_grid(start), _grid(start), start["x"], start["y"])
    assert v["radius_profile"].value < 0.05 and v["lip_passed"].value == 0.0
    assert not v.success and v.progress.value == 0.0


def test_a_corner_window_reads_its_own_lines_as_the_whole_pad_does(states):
    """A window holding the south-west corner (a light CPU run's region) has the s and w lines the whole pad has
    there, cell for cell: distances come from coordinates, not from the window's edge."""
    start, edged = states
    i, j = slice(0, 120), slice(0, 100)
    sub = {k: (v[i, j] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in edged.items()}
    sub0 = {k: (v[i, j] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in start.items()}
    win = PM.perimeter(_grid(sub), _grid(sub0), start["x"][i], start["y"][j])
    whole = PM.perimeter(_grid(edged), _grid(start), start["x"], start["y"])
    full = [ln for ln in win if len(ln.h)]
    assert {ln.side for ln in full} == {"s", "w"}
    by_cells = {(ln.side, round(float(ln.h.sum()), 12), len(ln.h)) for ln in whole if len(ln.h)}
    for ln in full:
        if len(ln.h) == int(round(PM.LINE_BAND / start["cell"])):        # lines not cut by the window's far edge
            assert (ln.side, round(float(ln.h.sum()), 12), len(ln.h)) in by_cells


def test_gouge_field_and_wet_work_fail_their_criteria(states):
    """A 3 mm hole 40 mm in from the south form, a 2 mm bump in the field, one edge cell worked wet."""
    start, _ = states
    g, x, y = _ideal(start), start["x"], start["y"]
    d = PM.form_distances(x, y)
    hole = (np.abs(d["s"] - 0.040) < 0.002) & (np.abs(x)[:, None] < 0.01)
    g.h = np.where(hole, np.minimum(g.h, start["h"] - FORM_H) - 0.0042, g.h)
    far = PM.pad_distance(x, y) > 0.3
    g.h = np.where(far & (np.abs(x)[:, None] < 0.004) & (np.abs(y)[None, :] < 0.004), g.h + 0.002, g.h)
    g.wet_worked = (np.abs(d["w"] - 0.03) < 0.001) & (np.abs(y)[None, :] < 0.001)
    v = PM.pad_edge(g, _grid(start), x, y)
    assert v["no_gouge"].value == pytest.approx(4.2, abs=0.1) and not v["no_gouge"].passed
    assert v["field_untouched"].value == pytest.approx(2.0, abs=1e-6) and not v["field_untouched"].passed
    assert v["edge_not_worked_wet"].value >= 1 and not v["edge_not_worked_wet"].passed
    assert v["radius_profile"].passed and v["lip_passed"].passed
    gates = v.progress.gates
    assert gates["gouge"] == pytest.approx(PM.falloff(4.2, 2.0, 6.0), abs=0.03) and gates["field"] < 1.0


def test_edge_credit_grades_arc_error_and_lip():
    """TSP:236-246 on one line: on the arc with the lip, 1; 3 mm off with no lip, 0.5·(1 - 1.5/9.5)."""
    d = (np.arange(50) + 0.5) * 0.002
    on = PM.Line("s", PM.arc(d), d, np.where(d < 0.0025, 1.0, 0.0), np.zeros(50), PM.arc(d))
    off = PM.Line("n", PM.arc(d) + 0.003, d, np.zeros(50), np.zeros(50), PM.arc(d))
    c = PM.edge_credit([on, off], 0.002)
    assert c["s"] == pytest.approx(1.0) and c["n"] == pytest.approx(0.5 * (1 - 0.0015 / 0.0095))
    assert c["w"] == 0.0 and c["e"] == 0.0


def test_lip_credit_reads_the_bands_largest_mark_over_half():
    """TSP:242-244 on a 1 mm line with two cells in the lip's 2.5 mm band, marked 0.6 and 0: lip credit
    clip(max / 0.5) = 1 (not the band's mean, 0.3 / 0.5, nor max / 1); on the arc, the line earns 1."""
    d = (np.arange(100) + 0.5) * 0.001
    mark = np.where(d < 0.001, 0.6, 0.0)
    ln = PM.Line("s", PM.arc(d), d, mark, np.zeros(100), PM.arc(d))
    assert PM.edge_credit([ln], 0.001)["s"] == 1.0
    half = PM.Line("s", PM.arc(d), d, mark * 0.25 / 0.6, np.zeros(100), PM.arc(d))     # largest mark 0.25: lip 0.5
    assert PM.edge_credit([half], 0.001)["s"] == 0.75
