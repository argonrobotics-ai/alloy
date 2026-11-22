# SPDX-License-Identifier: Apache-2.0
"""Regression tests for finite energy finding selection and non-finite settled pose guards."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.check import events, motion
from alloy.check.run import RunResult


def test_energy_findings_reports_worst_finite_world_despite_nan():
    """A NaN in world 0 must not hide a 90% S1 energy violation in world 1."""
    res = RunResult(monitors={"e_gain_cum": np.array([np.nan, 0.9]), "e_gain": np.array([0.0, 0.0])})
    findings, metrics = events.energy_findings(res, SimpleNamespace())
    assert len(findings) == 1
    assert findings[0].level == "error"
    assert "90 %" in findings[0].message and "world 1" in findings[0].message
    assert metrics["gain_cum_rel"] == pytest.approx(0.9)


def test_settled_pose_returns_none_for_nonfinite_body_pose():
    """_settled_pose must return None rather than a pose containing NaN when the end state is non-finite."""
    idx = SimpleNamespace(
        bodies_of=lambda n: np.array([0]),
        poses={"mug": (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0)},
        world_count=1,
        placement={"mug": 0},
    )
    s0 = SimpleNamespace(body_q=np.array([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]))
    send = SimpleNamespace(body_q=np.array([[np.nan, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]))
    assert motion._settled_pose("mug", None, idx, s0, send) is None
