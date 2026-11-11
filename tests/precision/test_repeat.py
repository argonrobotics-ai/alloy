# SPDX-License-Identifier: Apache-2.0
"""Repeat protocol on CPU (SPEC §6.5, §7 WP5: R1/R5 on box_on_plane and pendulum; R2, R4, R8 too; R3/R6 not_run on
CPU). Simulations are WP0's FakeSimulation until WP2's Simulation and WP4's state adapters land."""

from __future__ import annotations

import pytest

from alloy.precision.repeat import repeat_test
from alloy.precision.types import Determinism
from alloy.testing.precision import box_sim, pendulum_sim

PERIODS = 4


@pytest.mark.parametrize("make_sim", [box_sim, pendulum_sim], ids=["box_on_plane", "pendulum"])
@pytest.mark.parametrize("case", ["R1", "R5", "R8"])
def test_repeat_cpu(make_sim, case: str) -> None:
    res = repeat_test(make_sim, case, steps=PERIODS, world_counts=(1, 4))
    assert res.equal == {1: True, 4: True}, res
    assert all(v is None for v in res.first_divergent_step.values())
    assert res.det_mode == Determinism.RUN_TO_RUN  # CPU: serial launches


@pytest.mark.parametrize("make_sim", [box_sim, pendulum_sim], ids=["box_on_plane", "pendulum"])
def test_scenes_move(make_sim) -> None:
    """The repeat scenes are not static: digests change every period."""
    from alloy.precision.repeat import run_digests

    series = run_digests(make_sim(2), PERIODS)
    assert all((a != b).any() for a, b in zip(series, series[1:]))


def test_r4_batch_invariance_cpu() -> None:
    res = repeat_test(box_sim, "R4", steps=PERIODS, world_counts=(1, 3, 8))
    assert res.equal == {1: True, 3: True, 8: True}, res


@pytest.mark.slow
def test_r2_fresh_process_cpu() -> None:
    res = repeat_test(box_sim, "R2", steps=2, world_counts=(2,))
    assert res.equal == {2: True}, res


def test_r3_r6_not_run_on_cpu() -> None:
    r3 = repeat_test(box_sim, "R3", steps=1, world_counts=(1,))
    assert r3.equal == {} and "not_run" in r3.notes["R3@1"]
    r6 = repeat_test(box_sim, "R6", steps=1, world_counts=(1,))
    assert r6.equal == {} and "not_run" in r6.notes["R6"]


def test_unknown_case() -> None:
    with pytest.raises(ValueError):
        repeat_test(box_sim, "R9", steps=1, world_counts=(1,))
