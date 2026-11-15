# SPDX-License-Identifier: Apache-2.0
"""A forklift run a policy drove leaves ``time_limit`` not judged, and still needs every other check to succeed
(audit T1-017).

The port judges ``time_limit`` by its scripted plan finishing before the limit, and the plan finishes only when the
truck reaches the script's waypoints (6 mm): a policy that parked the truck in the bay a few cm from the scripted
spot failed ``time_limit`` ("the scripted plan was cut by the limit") with every other check passed. The env's
action, when it differs from the script's, now marks the run as policy-driven; scripted runs are judged as before.
"""
from __future__ import annotations

import dataclasses
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs.ports.forklift import TruckControls, controls_of


class _IO:
    def __init__(self) -> None:
        from alloy.tasks.forklift_control import Controls

        self.last_controls, self.applied = Controls(), []

    def apply(self, c) -> None:  # noqa: ANN001
        self.applied.append(c)
        self.last_controls = c


def test_only_an_action_of_its_own_marks_the_run_as_policy_driven():
    part, io = TruckControls(), _IO()
    part.io = io
    part.write(SimpleNamespace(), controls_of(io.last_controls))       # the expert's action: nothing applied
    assert not io.applied and not getattr(io, "policy_drove", False)
    a = controls_of(io.last_controls)
    a[0] = 0.5                                                         # the policy drives on
    part.write(SimpleNamespace(), a)
    assert len(io.applied) == 1 and io.policy_drove


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_policy_driven_run_does_not_judge_time_limit(monkeypatch):
    from alloy.tasks import forklift_ga_truck as ft
    from alloy.tasks.forklift_ga_run import simulate

    task = dataclasses.replace(importlib.import_module("alloy.tasks.forklift_ga_park").task(), time_limit=1.0)
    scripted = simulate(task)[1].by_name()["time_limit"]
    assert scripted.passed is False                                    # cut by the limit, judged as before
    original = ft.TruckIO.apply

    def driven(self, c) -> None:  # noqa: ANN001
        original(self, c)
        self.policy_drove = True                                       # as the env's write marks it

    monkeypatch.setattr(ft.TruckIO, "apply", driven)
    ep, verdict = simulate(task)
    limit = verdict.by_name()["time_limit"]
    assert not ep.plan_known and limit.passed is None and "policy-driven run" in limit.note
    assert "time_limit" in verdict.not_judged()
    assert not verdict.success and "in_parking_bay" in verdict.failed()   # 1 s in: not parked, so no success
    assert np.isclose(float(ep.t[-1]), 1.0)
