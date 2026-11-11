# SPDX-License-Identifier: Apache-2.0
"""task_sand_plate (engine item 10, material removal v1): one raster row of the sanding head over the plate."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.sand_plate import SandPlate, run, target_mask

pytestmark = pytest.mark.compile_heavy


@pytest.fixture(scope="module")
def one_row():
    task = SandPlate(head={"rows": 1, "length": 0.15, "start": (-0.075, 0.0)})
    res, read = run(task, seconds=1.6, timing=False)
    return task, res, read


def test_head_tracks_the_path_and_sands_the_band(one_row):
    task, res, read = one_row
    assert res.track_mm < 3.0
    assert res.mean_depth_um > 0.0
    assert res.coverage > 0.9
    assert res.ledger_rel < 1e-12
    assert res.captured_mg == pytest.approx(0.8 * res.removed_mg, rel=1e-6)    # capture is a float32 parameter


def test_nothing_is_removed_beyond_the_pad(one_row):
    task, res, read = one_row
    removal, _ = task.plugins()
    depth = read["depth"][0]
    nx, ny = depth.shape
    cell = 0.3 / nx
    y = (np.arange(ny) + 0.5) * (0.2 / ny) - 0.1
    reach = removal.config.pad_radius + cell
    assert depth[:, np.abs(y) > reach].max() == 0.0
    assert depth[:, np.abs(y) < 0.5 * removal.config.pad_radius].max() > 0.0


@pytest.mark.gpu
def test_gpu_smoke_and_cost_per_step_at_64_worlds():
    """GPU smoke (light-test rule: <= 64 worlds, short): the task on 64 worlds, and the cost per exchange step with and
    without the removal plug-in (stated in docs/notes/removal-v1.md)."""
    res, _ = run(SandPlate(), worlds=64, device="cuda:0", seconds=0.5)
    print(f"GPU 64 worlds: {res.ms_per_step:.3f} ms/step with removal, {res.ms_per_step_without:.3f} without")
    assert res.ledger_rel < 1e-12 and res.track_mm < 3.0
    assert res.ms_per_step < 2.0 * res.ms_per_step_without
