# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's syringe tasks as finger-rig analogues (alloy/tasks/syringe_tasks.py, SCR-WP8-11 Part A): each task
meets GA's fluid criteria in every world, and the finger pushing a capped and a needled syringe is stable at the tasks'
own step (WP13's review, condition b). CPU."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.fluid import P_ATM
from alloy.tasks import syringe_tasks as st
from alloy.tasks.syringe_rig import FINGER_EFFORT

pytestmark = pytest.mark.compile_heavy                # MuJoCo + the fluid kernels, cold on a gate host

FAST = ["set-dose", "inject", "flush-lock"]
SLOW = ["draw-dose", "vial-to-pad", "fast-bolus", "slow-push"]


@pytest.mark.parametrize("name", FAST)
def test_task_meets_gas_criteria(name):
    row = st.run(name, worlds=2)
    assert row["success_rate"] == 1.0, row


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("name", SLOW)
def test_long_task_meets_gas_criteria(name):
    row = st.run(name, worlds=2)
    assert row["success_rate"] == 1.0, row


def test_a_capped_full_syringe_holds_the_finger():
    """Capped, all liquid: the grip stalls on the water. The plunger moves less than the water's compression at the
    grip's force limit (0.2 mm), the pressure stays within that force over the bore, and nothing oscillates (the last
    half second's pressure spread under 2 % of its level)."""
    r = st.push_check(capped=True)
    x = r["x"][:, 0]
    assert np.all(np.isfinite(r["p"])) and np.ptp(x) < 2e-4, np.ptp(x)
    assert r["p"].max() <= FINGER_EFFORT / r["area"] * 1.05 and r["p"][-1, 0] > 1e4
    tail = r["p"][-25:, 0]
    assert np.ptp(tail) <= 0.02 * tail.mean(), (tail.min(), tail.max())


def test_a_needled_syringe_follows_the_finger_at_poiseuilles_pressure():
    """Through a 25 G needle into tissue, needle down: the plunger follows the finger at 1 mm/s, the pressure settles
    at A·v/G (5 %), and no air comes in."""
    r = st.push_check(capped=False)
    t, x = r["t"], r["x"][:, 0]
    late = slice(len(t) // 2, None)
    v = -np.polyfit(t[late], x[late], 1)[0]
    want = r["area"] * v / st.port("n", "syr", st.N25).conductance
    assert v == pytest.approx(1e-3, rel=5e-2)
    assert float(np.mean(r["p"][late, 0])) == pytest.approx(want, rel=5e-2)
    assert np.all(r["air"] <= 1e-9) and P_ATM > 0
