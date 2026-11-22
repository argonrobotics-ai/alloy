# SPDX-License-Identifier: Apache-2.0
"""Regression test: FitInclineTwin.bind_calib must populate FitValues and preserve _MuStatic."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from alloy.fit.twins.incline import FitInclineTwin, _MuStatic


def test_fit_incline_twin_bind_calib_populates_values_and_keeps_mu_fit_view():
    """FitInclineTwin.bind_calib must bind calibrated mu values into self.values and keep _MuStatic."""
    twin = FitInclineTwin()
    calib = [SimpleNamespace(condition_id="c0", scalars={"slip_angle": math.atan(0.45)})]
    twin.bind_calib(SimpleNamespace(), calib)
    assert isinstance(twin.mu_fit, _MuStatic)
    assert twin.mu_for("c0", kinetic=False) == pytest.approx(0.45)
    assert twin.mu_for("c0", kinetic=True) == pytest.approx(0.45)
