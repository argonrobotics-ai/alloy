# SPDX-License-Identifier: Apache-2.0
"""Regression test: ChainIK.check must reject NaN pose errors and NaN joint coordinates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.arms.ik import ChainIK, IKError, IKTolerance


def test_chain_ik_check_rejects_nan_errors_and_joints():
    """ChainIK.check must raise IKError on NaN pose error or NaN joint coordinates."""
    ik = object.__new__(ChainIK)
    ik.tol = IKTolerance(pos_m=1e-3, ang_rad=1e-2, limit_margin=0.01)
    ik.rig = SimpleNamespace(arms=["left"], specs=[SimpleNamespace(joints=["j0", "j1"])])
    ik.bind = SimpleNamespace(lower=np.array([[-1.0, -1.0]]), upper=np.array([[1.0, 1.0]]))

    with pytest.raises(IKError):
        ik.check(np.zeros((1, 1, 2)), np.array([[[np.nan, 0.0]]]))
    with pytest.raises(IKError):
        ik.check(np.array([[[np.nan, 0.0]]]), np.zeros((1, 1, 2)))
