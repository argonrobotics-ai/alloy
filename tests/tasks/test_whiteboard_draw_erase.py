# SPDX-License-Identifier: Apache-2.0
"""task_whiteboard_draw_erase (deposition plug-in): one short stroke drawn, then wiped in two passes."""

from __future__ import annotations

import pytest

from alloy.tasks.whiteboard_draw_erase import Whiteboard, run

pytestmark = pytest.mark.compile_heavy

SHORT = Whiteboard(strokes=(((-0.05, 0.0), (0.05, 0.0)),), erase_area=(-0.06, 0.06, -0.01, 0.01), erase_rows=0.02,
                   park={"marker": (-0.07, 0.06), "eraser": (0.08, -0.06)})


@pytest.fixture(scope="module")
def short_run():
    return run(SHORT, timing=False)


def test_stroke_lands_on_the_line_at_the_nib_width(short_run):
    res, drawn, _ = short_run
    assert res.coverage > 0.95 and res.precision > 0.95
    assert abs(res.width_mm - res.width_expected_mm) < 1.5 * 1.25     # within about a cell (1.25 mm cells)
    assert res.laid_mg > 0.0


def test_erase_lifts_the_ink_and_keeps_the_ledger(short_run):
    res, _, wiped = short_run
    assert res.erased > 0.9
    assert res.left_after_erase + res.erased == pytest.approx(1.0, abs=1e-9)
    assert res.ledger_rel < 1e-12


@pytest.mark.gpu
def test_gpu_smoke_and_cost_per_step_at_64_worlds():
    """GPU smoke (light-test rule: <= 64 worlds, short): the short task on 64 worlds, with its cost per step."""
    res, _, _ = run(SHORT, worlds=64, device="cuda:0")
    print(f"GPU 64 worlds: {res.ms_per_step:.3f} ms/step with deposition, {res.ms_per_step_without:.3f} without")
    assert res.coverage > 0.95 and res.ledger_rel < 1e-12
    assert res.ms_per_step < 2.0 * res.ms_per_step_without
