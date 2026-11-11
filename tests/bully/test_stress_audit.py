# SPDX-License-Identifier: Apache-2.0
"""B0 phases, random-legal rollouts, the B2 audit brief, and DirectSim (SimulationLike, masked reset)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.bully.actions import legalize
from alloy.bully.brief import ATTACKS, audit_prompt
from alloy.bully.direct import DirectSim
from alloy.bully.stress import named_stress_phases, random_legal_traces, stress_phases, tile_phases
from alloy.bully.types import BullyLimits, BullySeverity, Finding
from alloy.core.errors import BullyError
from alloy.core.protocols import SimulationLike
from alloy.testing.bully import seeded_fake_target


def _legal(trace: np.ndarray, sp) -> bool:
    again = legalize(trace[None], sp.low, sp.high, sp.max_delta_per_tick, sp.initial)
    return bool(np.allclose(again.trace[0], trace, atol=1e-6))


def test_b0_phases_are_legal_and_named():
    t = seeded_fake_target()
    named = named_stress_phases(t)
    names = [n for n, _ in named]
    assert names[:4] == ["val115_1_joints_80pct_2hz", "val115_2_opposed_extremes", "val115_3_squeeze_5hz",
                         "val115_4_drive_high_hold"]
    assert sum(n.startswith("push_actuator_") for n in names) == t.actions.dim
    for _, trace in named:
        assert trace.shape == (100, 2) and _legal(trace, t.actions)
    # phase 1 switches between 80 % of the range at 2 Hz: +-4 N on a +-5 N actuator, 25 ticks per half period
    p1 = dict(named)["val115_1_joints_80pct_2hz"]
    assert np.max(p1) == pytest.approx(4.0) and np.min(p1) == pytest.approx(-4.0)
    assert len(stress_phases(t)) == len(named)
    assert tile_phases(stress_phases(t), 20).shape == (20, 100, 2)


def test_stress_extra_traces_are_legalised_and_fitted():
    t = seeded_fake_target()
    t = type(t)(**{**t.__dict__, "stress_extra": lambda tgt: [np.full((10, 2), 99.0)]})
    extra = dict(named_stress_phases(t))["extra_0"]
    assert extra.shape == (100, 2) and np.all(extra <= 5.0)


def test_random_legal_traces_reproducible_and_legal():
    t = seeded_fake_target()
    a, b = random_legal_traces(t, 32, seed=3), random_legal_traces(t, 32, seed=3)
    np.testing.assert_array_equal(a, b)
    assert all(_legal(x, t.actions) for x in a)
    assert np.std(a) > 0.5  # OU around the centre with stationary std 0.5 of the half-range


def test_audit_brief_lists_rules_box_thresholds_and_findings():
    t = seeded_fake_target()
    f = Finding("BUL-0007", BullySeverity.S1, "max_speed", 25.0, 20.0, 42, "h" * 64, "a" * 40, "n" * 40, "repro")
    text = audit_prompt(t, [f])
    for needle in ("Never edit product code", "BUL-0007", "max_speed", "| 0 | -5 | 5 |", "S1", "nontarget_disp",
                   "python -m alloy.bully try --target fake_seeded", ATTACKS[0]):
        assert needle in text, needle
    assert "None yet" in audit_prompt(t, [], BullyLimits())


def test_direct_sim_is_simulation_like_and_resets_one_world():
    sim = seeded_fake_target().make_sim(3, "cpu")
    assert isinstance(sim, DirectSim) and isinstance(sim, SimulationLike)
    q0 = sim.state.body_q.numpy().copy()
    qd = sim.state.body_qd.numpy()
    qd[:, 0] = 1.0
    sim.state.body_qd.assign(qd)
    sim.step_period()
    assert sim.clock.t == pytest.approx(0.02)
    moved = sim.state.body_q.numpy().copy()
    sim.reset(np.array([0, 1, 0], dtype=np.int32))
    q = sim.state.body_q.numpy()
    np.testing.assert_array_equal(q[1], q0[1])  # world 1 back at its initial pose
    np.testing.assert_array_equal(q[[0, 2]], moved[[0, 2]])  # the others untouched
    with pytest.raises(BullyError):
        DirectSim(sim.model, sim.solver, substeps=3)  # not on the D4 grid
