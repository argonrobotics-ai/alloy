# SPDX-License-Identifier: Apache-2.0
"""SCR-WP11-4: a force jump at first contact re-runs its period at a finer dt_x instead of injecting R^2 dt_x^2 / m.

CPL-4's box_on_cloth (0.2 kg box on the VBD cloth, lagged proxy). At first contact the destination's proxy runs a
force-free step while the box decelerates under the harvested force R, so the power bond books R x gap with
gap = R dt_x^2 / m (WP11's diagnosis, [internal note]): 0.2-27 mJ (up to 5e-3 E_scale)
depending on the contact phase, i.e. the drop height. The watchdog now escalates an exchange that injects more than
``WatchdogPolicy.onset_gain_per_exchange`` (1e-4 E_scale) and flags it as a force jump; the Scheduler re-runs that
period at dt_x / 2, / 4, / 8 (that period only, no escalation ladder). The injection falls about as dt_x^4 (R itself
scales with the penetration reached in one exchange).
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling.onset import OnsetRefiner, escalated_beyond_onset
from alloy.coupling.types import SchedulerPolicy, WatchdogPolicy
from alloy.testing.coupling import ClothSceneSpec, PeriodRunner, cloth_scheduler, coupling_box_on_cloth

LIMIT = WatchdogPolicy().onset_gain_per_exchange
PERIODS = 6                                   # 0.12 s: first contact is at 0.066-0.074 s for dz 0-3.5 mm


def _run(drops: list[float], periods: int = PERIODS, limit: float = LIMIT) -> dict:
    spec = ClothSceneSpec(box_mass=0.2)
    model = coupling_box_on_cloth(len(drops), spec, drop_offsets=drops, device="cpu")
    policy = SchedulerPolicy(watchdog=dataclasses.replace(WatchdogPolicy(), onset_gain_per_exchange=limit))
    sched = cloth_scheduler(model, spec, policy=policy, mode="lagged")
    runner = PeriodRunner(model, sched, 0.02)
    peak, dts, levels = np.zeros(len(drops)), [], []
    for _ in range(periods):
        rep = runner.run_period()
        peak = np.maximum(peak, rep.notes["exchange_peak"])
        dts.append(rep.dt_x)
        levels.append(rep.notes["onset_level"])
    return {"report": rep, "runner": runner, "peak": peak, "dt_x": dts, "levels": levels,
            "body_q": runner.state.body_q.numpy().copy(), "particle_q": runner.state.particle_q.numpy().copy()}


@pytest.mark.timeout(900)
def test_first_contact_jump_reruns_its_period_finer():
    """Mechanics, independent of the contact phase (which moves with every Newton pin): with a tiny limit the first
    contact is always a force jump; its period re-runs at dt_x / 2, / 4, / 8 (the last with the trigger off), the
    ladder is not climbed, and the next period is back on the clock's dt_x."""
    out = _run([0.0], limit=1e-12)
    rep, levels, dts = out["report"], out["levels"], out["dt_x"]
    assert rep.notes["onset_reruns"] == WatchdogPolicy().onset_refine_levels and rep.rollbacks == 0, rep.notes
    hit = levels.index(max(levels))
    assert levels[hit] == WatchdogPolicy().onset_refine_levels and levels[-1] == 0, levels
    assert dts[hit] == dts[0] / 8 and dts[-1] == dts[0], dts                   # that period only


@pytest.mark.timeout(900)
def test_scene_without_a_jump_is_bitwise_unchanged():
    """Phase-proof (the contact phase moves with every Newton pin): with the trigger off, the largest single-exchange
    bond is P; a limit of 2 P never fires, nothing re-runs, and the run is bitwise the trigger-off run."""
    off = _run([1.5e-3], limit=math.inf)
    p = float(off["peak"][0])
    assert p > 0.0 and off["report"].notes["onset_reruns"] == 0
    on = _run([1.5e-3], limit=2.0 * p)
    assert on["report"].notes["onset_reruns"] == 0 and on["runner"].rollbacks == 0
    np.testing.assert_array_equal(on["body_q"], off["body_q"])
    np.testing.assert_array_equal(on["particle_q"], off["particle_q"])


def _report(states, onset, peak):
    return SimpleNamespace(watch_state=np.asarray(states), notes={"onset": np.asarray(onset, bool),
                                                                   "exchange_peak": np.asarray(peak, float)})


def test_refiner_rules():
    """Host rules: a ladder escalation is never a force jump; each accepted re-run halves dt_x once more, the last
    level runs with the trigger off, and the clock's dt_x comes back the next period."""
    ref = OnsetRefiner(WatchdogPolicy())
    ref.new_period()
    ref.attempt(0.02, 5, 0.02 / 32, 10)
    assert ref.armed(2).tolist() == [1, 1]
    assert not escalated_beyond_onset(_report([2, 0], [1, 0], [5e-3, 0])).any()
    assert escalated_beyond_onset(_report([2, 2], [1, 0], [5e-3, 0])).tolist() == [False, True]
    assert not ref.wants_rerun(_report([2, 2], [1, 0], [5e-3, 0]))           # mixed: the ladder decides
    assert not ref.wants_rerun(_report([0, 0], [0, 0], [5e-3, 0]))           # nothing escalated
    ref.notes([True, False], [5e-3, 0.0])                                    # flagged: world 0's cooldown starts
    assert ref.armed(2).tolist() == [1, 1]                                    # ... after its own re-runs
    for level, limit in ((1, LIMIT), (2, LIMIT), (3, math.inf)):            # level 3 = onset_refine_levels
        assert ref.accept(_report([2, 0], [1, 0], [5e-3, 0]))
        assert ref.attempt(0.02, 5, 0.02 / 32, 10) == (0.02 / 2 ** (5 + level), limit)
    assert ref.reruns == 3
    ref.new_period()
    assert ref.attempt(0.02, 5, 0.02 / 32, 10) == (0.02 / 32, LIMIT)
    assert ref.armed(2).tolist() == [0, 1]                                    # world 0: 0.5 s cooldown
    for _ in range(24):
        ref.new_period()
    assert ref.armed(2).tolist() == [0, 1]                                    # 25 periods = 0.5 s later
    ref.new_period()
    assert ref.armed(2).tolist() == [1, 1]                                    # 26 periods: armed again
    ref.new_period()
    assert ref.attempt(0.02, 9, 0.02 / 512, 10) == (0.02 / 512, LIMIT)
    ref.new_period()
    ref.notes([False, True], [0.0, 1e-3])
    assert ref.accept(_report([0, 2], [0, 1], [0, 1e-3]))
    assert ref.attempt(0.02, 9, 0.02 / 512, 10) == (0.02 / 1024, math.inf)  # j_max reached: trigger off


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_eight_heights_first_contact_within_limit():
    """CPL-4's phase sweep (SB-8 protocol), one world per height (an IntegratedSimulation keeps the worlds that did not
    trip; PeriodRunner re-runs every world): with the trigger off some heights inject more than the limit in one
    exchange (Newton ac7f7830: 4.35e-3 at 0.5 mm, 1.78e-3 at 2.5 mm, 4.98e-3 at 3.5 mm); with it on none does."""
    off, on, reruns = [], [], 0
    for k in range(8):
        a, b = _run([0.5e-3 * k], periods=8, limit=math.inf), _run([0.5e-3 * k], periods=8)
        off.append(float(a["peak"][0]))
        on.append(float(b["peak"][0]))
        reruns += b["report"].notes["onset_reruns"]
    print("trigger off", off, "on", on, "re-runs", reruns)
    assert max(on) <= LIMIT, on
    assert reruns >= 1 or max(off) <= LIMIT
