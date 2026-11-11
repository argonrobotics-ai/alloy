# SPDX-License-Identifier: Apache-2.0
"""The viewer's joint sliders (alloy.viewer._joints) on a one-hinge arm on the CPU, 2 worlds: a slider value is the
position target the joint reaches in every world; untouched sliders leave the simulation bit for bit as without them;
ranges, unactuated joints and the cap; a refused target removes its slider and the run goes on; off by default."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
import alloy.viewer
from alloy.api.errors import SimulationError
from alloy.viewer import _joints

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  <body name="link2" pos="0.3 0 0"><joint name="wrist" type="hinge" axis="0 1 0" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/></body></body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _build(worlds: int = 2) -> alloy.Sim:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    return alloy.build(scene, device="cpu", seed=0)


@pytest.fixture(scope="module")
def sim() -> alloy.Sim:
    return _build()


class _Ui:
    """Stands in for a viewer's widget set: reports the queued (label -> value) answers as changed."""

    def __init__(self, answers: dict[str, float]):
        self.answers, self.ranges = answers, {}

    def text(self, label):
        pass

    def slider_float(self, label, value, lo, hi):
        self.ranges[label] = (lo, hi)
        return (label in self.answers, self.answers.get(label, value))


def test_slider_value_is_the_target_reached_in_every_world(sim):
    with alloy.viewer.launch(sim, backend="null", joints=True) as view:
        view._tools.render(_Ui({"arm/shoulder": 0.8}))
        view.step(150)
        q = sim.state()["arm/shoulder.q"]
        assert q.shape == (2,) and np.allclose(q, 0.8, atol=0.05), q
        print("reached", q)


def test_untouched_sliders_leave_the_simulation_bitwise_equal():
    plain, viewed = _build(), _build()
    plain.step(30)
    with alloy.viewer.launch(viewed, backend="null", joints=True) as view:
        for _ in range(30):
            view._tools.render(_Ui({}))
            view.step(1)
    a, b = plain.state(), viewed.state()
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def test_range_is_the_joint_range_and_unactuated_joints_get_none(sim):
    with alloy.viewer.launch(sim, backend="null", joints=True) as view:
        ui = _Ui({})
        view._tools.render(ui)
    assert ui.ranges == {"arm/shoulder": (-2.0, 2.0)}


def test_unlimited_joint_gets_pi_around_its_start():
    mjcf = ARM.replace('range="-2 2" ', "").replace("<actuator>", '<actuator><position joint="wrist" kp="20"/>')
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", mjcf=mjcf, pos=(0, 0, 0.5))
    found = {s.name: s for s in _joints.find_sliders(alloy.build(scene, device="cpu", seed=0))}
    assert (found["arm/shoulder"].lo, found["arm/shoulder"].hi) == pytest.approx((-np.pi, np.pi))
    assert found["arm/wrist"].value == 0.0 and len(found) == 2


def test_cap_and_no_duplicate_alias(sim, monkeypatch, caplog):
    monkeypatch.setattr(_joints, "MAX_SLIDERS", 1)
    names = sim.names()["joints"]
    assert "shoulder" in names and "arm/shoulder" in names
    with caplog.at_level("INFO", logger="alloy.viewer"):
        assert [s.name for s in _joints.find_sliders(sim)] == ["arm/shoulder"]
    assert not caplog.records                       # one driven joint: under the cap, nothing logged
    monkeypatch.setattr(_joints, "MAX_SLIDERS", 0)
    with caplog.at_level("INFO", logger="alloy.viewer"):
        assert _joints.find_sliders(sim) == []
    assert "showing the first 0 of 1" in caplog.text


def test_refused_target_removes_the_slider_and_the_run_goes_on(monkeypatch, caplog):
    sim = _build()

    def refuse(targets, kind="position"):
        raise SimulationError("refused", "nothing to fix")

    with alloy.viewer.launch(sim, backend="null", joints=True) as view:
        monkeypatch.setattr(sim, "set_targets", refuse)
        view._tools.render(_Ui({"arm/shoulder": 0.5}))
        view.step(3)
        view.step(3)
        assert sim.t == pytest.approx(0.06)
        assert view._tools.tools[0].sliders == []
    assert caplog.text.count("slider for arm/shoulder removed") == 1


def test_off_by_default(sim):
    with alloy.viewer.launch(sim, backend="null") as view:
        assert not view._tools and view._tools.tools == []
    with alloy.viewer.launch(sim, backend="null", joints=True) as view:
        assert len(view._tools.tools) == 1
