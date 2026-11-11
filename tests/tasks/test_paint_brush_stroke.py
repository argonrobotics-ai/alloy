# SPDX-License-Identifier: Apache-2.0
"""brush-stroke with the stand-in brush (CPU): a 40 µm stroke inside the outline passes; one too thin to reach 5 µm dry
fails coverage; one 1.8x too wide fails the 1 mm overrun; the stand-in lays its coat once per cell."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import paint_metrics as pm
from alloy.tasks.paint_brush_stroke import CELL, PAPER, run_standin, stroke_rates


def test_stand_in_lays_the_core_once() -> None:
    x, y = np.meshgrid(np.arange(-PAPER[0] / 2 + CELL / 2, PAPER[0] / 2, CELL),
                       np.arange(-PAPER[1] / 2 + CELL / 2, PAPER[1] / 2, CELL), indexing="ij")  # fmt: skip
    c = np.stack([x, y], -1)
    laid = sum(stroke_rates(c, 40e-6)) * 0.02
    core, outside = pm.template_masks(c)
    assert np.allclose(laid[core], 40e-6) and np.allclose(laid[laid > 0], 40e-6) and not laid[outside].any()


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("coat", "width", "fails"), [(40e-6, 1.0, set()), (5e-6, 1.0, {"coverage"}),
                                                      (40e-6, 1.8, {"within_line"})])
def test_brush_stroke_standin(coat: float, width: float, fails: set[str]) -> None:
    v = run_standin(coat=coat, width_factor=width)
    assert v["dry_film"] == "stage"
    if width == 1.0:                                      # the stage's solids: coat x phi_0 (0.65), not h x phi_0
        assert v["dry_core"]["p50_um"] == pytest.approx(coat * 0.65 * 1e6, rel=1e-4)
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == fails
    assert v["succeeded"] is (not fails)
    # AB-gym-6: one paint, so coverage's thickness alone is not a pass of the criterion; no marks are tracked
    assert v["checks"]["untouched"]["ok"] is None and "0.65" in v["scene"]
    assert "crop" in v["checks"]["within_line"]["note"]
    if "coverage" in fails:
        assert v["judged"] == "2 of 3" and list(v["not_judged"]) == ["untouched"]
    else:
        assert v["checks"]["coverage"]["ok"] is None and v["checks"]["coverage"]["value"] >= pm.STROKE_COVER
        assert v["judged"] == "1 of 3" and list(v["not_judged"]) == ["coverage", "untouched"]
