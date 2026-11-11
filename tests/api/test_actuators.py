# SPDX-License-Identifier: Apache-2.0
"""MJCF actuator table: <general> servos found and named, joint-target actuators kept apart, ctrl rows per world."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from alloy.api._actuators import Actuators


class _A:
    def __init__(self, a):
        self.a = np.asarray(a)

    def numpy(self):
        return self.a


def _model():
    # one robot, 3 joints with 1 dof each, 2 worlds; rows: position servo (general), joint-target position, motor
    gain = [[4500, 0, 0], [100, 0, 0], [1, 0, 0]] * 2
    bias = [[0, -4500, -450], [0, -100, -10], [0, 0, 0]] * 2
    mj = SimpleNamespace(ctrl_source=_A([1, 0, 1] * 2), actuator_trnid=_A([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                                                                            [5, 0]]),
                         actuator_trntype=_A([0] * 6), actuator_gainprm=_A(gain), actuator_biasprm=_A(bias),
                         actuator_world=_A([0, 0, 0, 1, 1, 1]), actuator_label=["a1", "", "m3"] * 2)
    return SimpleNamespace(mujoco=mj, joint_label=["arm/j1", "arm/j2", "arm/j3"] * 2)


def test_table():
    acts = Actuators.from_model(_model(), {"arm": range(0, 3)}, np.array([0, 1, 2, 3]))
    assert acts.stride == 3 and acts.position == {0: 0} and acts.joint_target == {1} and acts.velocity == {}
    assert {"arm/a1", "a1", "arm/j2", "j2", "arm/m3", "m3"} <= set(acts.names)
    assert acts.rows(acts.names["m3"], 2).tolist() == [2, 5]
    assert acts.actuated(2) and not acts.position.get(2)


def test_no_mujoco_attributes():
    acts = Actuators.from_model(SimpleNamespace(), {}, np.array([0, 1]))
    assert acts.stride == 0 and not acts.names
