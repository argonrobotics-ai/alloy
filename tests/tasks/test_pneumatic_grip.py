# SPDX-License-Identifier: Apache-2.0
"""task_pneumatic_grip (alloy/tasks/pneumatic_grip.py): the valve, the script, the cylinders' stability, and the
pick-and-place in every world (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.fluid import P_ATM
from alloy.tasks import pneumatic_grip as T

pytestmark = pytest.mark.law


def test_valve_ports():
    assert T.valve_openings("open") == {"left_supply_rod": 1.0, "left_exhaust_cap": 1.0, "left_supply_cap": 0.0,
                                        "left_exhaust_rod": 0.0, "right_supply_rod": 1.0, "right_exhaust_cap": 1.0,
                                        "right_supply_cap": 0.0, "right_exhaust_rod": 0.0}
    close = T.valve_openings("close")
    assert {k for k, v in close.items() if v} == {"left_supply_cap", "left_exhaust_rod", "right_supply_cap",
                                                  "right_exhaust_rod"}
    assert not any(T.valve_openings("centre").values())
    with pytest.raises(ValueError):
        T.valve_openings("float")
    assert set(close) == {e.name for e in T.plugin().config.edges}


def test_script():
    assert [T.valve(t) for t in (0.0, 0.39, 0.4, 2.19, 2.2, 3.0)] == ["open", "open", "close", "close", "open", "open"]
    assert np.allclose(T.targets(0.0), 0.0)
    assert T.targets(0.6)[2] == pytest.approx(-T.DESCENT)
    x, _, z = T.targets(1.5)                       # halfway through the carry, lifted
    assert x == pytest.approx(0.5 * T.PLACE[0]) and z == pytest.approx(T.LIFT - T.DESCENT)
    assert np.allclose(T.targets(3.0), [T.PLACE[0], 0.0, 0.05 - T.DESCENT])


def test_grip_force_and_stable_step():
    area = math.pi * 0.01 ** 2 / 4.0
    assert T.grip_force() == pytest.approx((T.SUPPLY - P_ATM) * area, rel=1e-12)
    assert T.grip_force() * 0.8 * 2 > 10.0 * T.BOX_MASS * 9.81       # friction capacity well above the weight
    net, cyl = T.plugin().network(), T.cylinder("left")
    dt = T.scene().control_dt / 16                                    # the scene's exchange step (<= 2 ms)
    for x, p in ((0.0, {"left_cap": P_ATM, "left_rod": T.SUPPLY}), (T.GAP, {"left_cap": T.SUPPLY,
                                                                              "left_rod": P_ATM})):
        assert cyl.check_dt(net, x, T.FINGER_MASS, p) > 2.0 * dt


def test_pick_and_place_in_every_world():
    row = T.run(2, "cpu", with_plugin=True)
    assert row["success"] == 2, row
    assert row["lift_mm_min"] >= 60.0 and row["place_err_mm_max"] <= 1e3 * T.PLACE_TOL, row
    assert row["grip_force_err_n_max"] <= row["grip_force_tol_n"], row
