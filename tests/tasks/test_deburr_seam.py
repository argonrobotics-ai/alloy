# SPDX-License-Identifier: Apache-2.0
"""task_deburr_seam (engine item 10, deburring): a short seam (12 mm) with the task's wheel, dose and grid, on CPU."""

from __future__ import annotations

import pytest

from alloy.tasks.deburr_seam import DeburrSeam, run

pytestmark = pytest.mark.law


@pytest.fixture(scope="module")
def short_seam():
    base = DeburrSeam()
    task = DeburrSeam(seam=0.012, margin=0.002,
                      removal={**base.removal, "size": (0.032, 0.012), "cells": (64, 48)},
                      head={**base.head, "start": (-0.011, 0.0), "length": 0.022})
    res, _ = run(task, timing=False)
    return res


def test_the_burr_wears_first_and_the_base_little(short_seam):
    r = short_seam
    assert r.burr_before_um > 100.0
    assert r.burr_after_max_um < 0.2 * r.burr_before_um, r            # at least 80 % of the burr is gone
    assert r.base_removed_max_um < 0.15 * r.burr_before_um, r         # the base beside it loses far less
    assert r.base_removed_mean_um >= 0.0


def test_the_mass_ledger_is_exact(short_seam):
    r = short_seam
    assert r.removed_mg > 0.0
    assert r.ledger_rel < 1e-12
    assert r.captured_mg == pytest.approx(0.5 * r.removed_mg, rel=1e-7)   # capture 0.5 (a float32 fraction)


@pytest.mark.gpu
def test_gpu_smoke_and_cost_per_step_at_64_worlds():
    """GPU smoke (light-test rule: <= 64 worlds, short): the task on 64 worlds for 0.5 s, and the cost per exchange
    step with and without the removal plug-in (stated in docs/notes/drilling-v1.md)."""
    res, read = run(DeburrSeam(), worlds=64, device="cuda:0", seconds=0.5)
    assert res.ledger_rel < 1e-12 and float(read["removed"].min()) >= 0.0
    print(f"GPU 64 worlds: {res.ms_per_step:.3f} ms/step with removal, {res.ms_per_step_without:.3f} without")
