# SPDX-License-Identifier: Apache-2.0
"""pad-finish criteria and progress (alloy/tasks/pad_finish_metrics.py) on Gym-Anything's start states (CPU, numpy).

Their pad-trowel start is the pad darbied, edged and floated (``layout.py:271-330``); steel-troweled on top of it
(densification 0.9, texture 0.04 mm) it is a finished pad: every pad-finish criterion passes against the pad-finish
start (struck, 5 min). The start itself scores 0. Each cap and defect fails on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import pad_finish_metrics as FM
from alloy.tasks import pad_metrics as PM
from alloy.tasks.concrete_metrics import PasteGrid
from alloy.tasks.concrete_pad_start import FORM_H, pad_fields


def _grid(f: dict, **over) -> PasteGrid:
    z = np.zeros(f["h"].shape, np.float32)
    g = PasteGrid(h=f["h"] - FORM_H, t=f["tb"] + f["ts"], cell=float(f["cell"]), paste_mask=f["paste_mask"],
                  D=f["D"], a=f["a"], E=f["E"], S=f["S"], C=f["compact"], q=f["q"], bl=f["bl"], late=z.astype(np.int32),
                  wet_worked=z.astype(np.int32), dwc=z)
    for k, v in over.items():
        setattr(g, k, v)
    return g


@pytest.fixture(scope="module")
def pads():
    start, floated = pad_fields("pad-finish", 0), pad_fields("pad-trowel", 0)
    inside = floated["paste_mask"]
    done = _grid(floated, D=np.where(inside, 0.9, 0.0).astype(np.float32),
                 a=np.where(inside, 4.0e-5, 0.0).astype(np.float32))
    return start, done


def _verdict(start, g):
    return FM.pad_finish(g, _grid(start), start["x"], start["y"], np.full(g.h.shape, 4 * 3600.0))


def test_a_troweled_pad_passes_and_the_struck_start_scores_zero(pads):
    start, done = pads
    v = _verdict(start, done)
    assert v.success, [(c.name, c.value) for c in v.criteria if not c.passed]
    assert v.score == 1.0 and v.progress == pytest.approx(1.0)
    assert sorted(v.side_radius) == sorted(v.side_arc_lip) == sorted(PM.SIDES)        # per side, for the rows
    assert min(v.side_radius.values()) >= 0.9 and min(v.side_arc_lip.values()) >= 0.9
    v0 = _verdict(start, _grid(start))
    assert not v0.success and v0.progress < 0.05


def test_an_unfinished_disc_wider_than_75_mm_fails_smooth_hard(pads):
    start, done = pads
    X, Y = np.meshgrid(start["x"], start["y"], indexing="ij")
    disc = np.hypot(X, Y) < 0.041
    v = _verdict(start, _grid(start, **{**vars(done), "D": np.where(disc, 0.1, done.D).astype(np.float32),
                                        "a": np.where(disc, 2.5e-4, done.a).astype(np.float32)}))
    c = v["smooth_hard"]
    assert c.value > FM.TEX_SHARE and not c.passed                  # the share passes; the 82 mm spot does not
    assert 0.8 < v.terms["finish"] <= FM.FAIL_TERM and not v.success       # a failing term earns at most 0.9


def test_a_long_edge_miss_and_a_burn_fail(pads):
    start, done = pads
    d = PM.form_distances(start["x"], start["y"])
    X = np.broadcast_to(start["x"][:, None], done.h.shape)
    miss = (d["s"] < 0.01) & (np.abs(X) < 0.06)                      # 120 mm of the south arc left square
    h = np.where(miss & done.paste_mask, 0.0, done.h)
    burn = np.zeros(done.h.shape, np.int32)
    burn[200:260, 200:260] = 1
    g = _grid(start, **{**vars(done), "h": h, "late": burn, "D": np.where(burn > 0, 0.99, done.D).astype(np.float32)})
    v = _verdict(start, g)
    assert v["radius_profile"].value > FM.EDGE_SHARE and not v["radius_profile"].passed
    assert v["no_trowel_burn"].passed                                # 3600 cells: under the 5 % share
    burn[100:500, 100:400] = 1
    g = _grid(start, **{**vars(done), "late": burn, "D": np.where(burn > 0, 0.99, done.D).astype(np.float32)})
    assert not _verdict(start, g)["no_trowel_burn"].passed


def test_remaining_bleed_is_their_triangle_integral():
    mat = FM.CM.Concrete()
    te, mean = float(mat.bleed_end_time(21.1)), mat.bleed_mean
    assert float(FM.remaining_bleed(0.0)) == pytest.approx(mean * te / 3600.0 / 1000.0, rel=1e-12)
    assert float(FM.remaining_bleed(te)) == 0.0
    ages = np.linspace(0.0, te, 2001)
    rate = np.array([float(mat.bleed_rate(a, 21.1)) for a in ages]) / 1000.0           # m/s
    tail = np.concatenate([np.cumsum(((rate[1:] + rate[:-1]) / 2 * np.diff(ages))[::-1])[::-1], [0.0]])
    np.testing.assert_allclose(FM.remaining_bleed(ages), tail, rtol=0, atol=1e-6 * tail[0])
