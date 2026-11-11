# SPDX-License-Identifier: Apache-2.0
"""Ladder L3 embodiment rungs (P3-2b, CPU): WP13's scene card, its bully target's start state and B0 actions."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.parallel import ladder_run
from alloy.parallel.rung_tasks import B0_SEED, TASKS, TaskRung


def test_embodiment_tasks_are_registered_as_l3():
    emb = [k for k in TASKS if k.startswith("emb_")]
    assert "emb_e1_fr3" in emb and all(TASKS[k].target == k for k in emb)
    assert ladder_run.is_task("emb_e1_fr3") and ladder_run.rung_of("emb_e1_fr3") == "L3"


@pytest.fixture(scope="module")
def rung():
    return TaskRung("emb_e1_fr3", 2, "cpu")


def test_b0_rows_are_the_campaign_traces_held_per_tick(rung):
    from alloy.bully.actions import legalize
    from alloy.bully.stress import random_legal_traces
    from alloy.bully.targets import get_target

    tg = get_target("emb_e1_fr3")
    sp = tg.actions
    want = legalize(random_legal_traces(tg, 2, 2.0, seed=B0_SEED), sp.low, sp.high, sp.max_delta_per_tick,
                    sp.initial).trace
    hold = int(round(1.0 / (sp.rate_hz * rung.control_dt)))
    src = rung.replayer._plan[0][0].numpy()          # [T * hold, W, A]
    assert src.shape == (want.shape[1] * hold, 2, want.shape[2])
    assert np.array_equal(src[::hold], want.transpose(1, 0, 2)) and np.array_equal(src[hold - 1::hold], src[::hold])


def test_rung_starts_at_the_targets_state_and_passes_its_gate(rung):
    from alloy.bully.targets import get_target

    q1 = get_target("emb_e1_fr3").make_sim(1, "cpu").model.joint_q.numpy()
    rung.reset()
    assert np.allclose(rung.s0.joint_q.numpy()[: q1.size], q1)
    for _ in range(5):
        rung.step()
    rung.counters()
    gate = rung.gate()
    assert gate["finite"] and gate["pass"], gate


def test_rung_steps_like_the_sb7_target(rung):
    from alloy.bully.targets import get_target

    sim = get_target("emb_e1_fr3").make_sim(1, "cpu")
    assert rung.control_dt == sim.control_dt == 0.02
    assert rung.k == 32 and abs(rung.dt - 0.000625) < 1e-12 and rung.periods_per_call == 1
    assert (rung.capacities.nconmax, rung.capacities.njmax, rung.capacities.rigid_contact_max) == (256, 1024, 512)
    assert rung.route == "on"


def test_ground_is_the_lowest_static_plane():
    from alloy.bully.targets import get_target
    from alloy.parallel.rung_tasks import _ground_z

    assert abs(_ground_z(get_target("emb_e3_ur10e_2f85").make_sim(1, "cpu").model) + 0.75) < 1e-6
