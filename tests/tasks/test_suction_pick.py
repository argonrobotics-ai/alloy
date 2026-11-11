# SPDX-License-Identifier: Apache-2.0
"""task_suction_pick (alloy/tasks/suction_pick.py): the gantry cup picks, carries and places the box in every world."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import suction_pick as T

pytestmark = pytest.mark.law


def test_targets_follow_the_script():
    assert np.allclose(T.targets(0.0), 0.0)
    assert T.targets(0.5)[2] == pytest.approx(-(T.CLEAR + 5e-4))
    x, _, z = T.targets(1.7)                       # carry done, a quarter into the descent (cosine blend)
    u = 0.5 - 0.5 * np.cos(np.pi * 0.25)
    assert x == pytest.approx(T.PLACE[0]) and z == pytest.approx(0.10 + (-T.CLEAR + 5e-4 - 0.10) * u)
    assert np.allclose(T.targets(3.0), [T.PLACE[0], 0.0, 0.05])


@pytest.mark.parametrize("gripper", T.GRIPPERS)
def test_pick_and_place_in_every_world(gripper):
    row = T.run(3, "cpu", with_plugin=True, gripper=gripper)
    assert row["success"] == 3, row
    assert row["lift_mm_min"] >= 60.0 and row["place_err_mm_max"] <= 1e3 * T.PLACE_TOL, row
    assert row["place_err_mm_max"] < 0.01, row        # critically damped contacts: no chatter walking the box
